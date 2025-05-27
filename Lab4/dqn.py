
import os
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Imports all our hyperparameters from the other file
from hyperparams import Hyperparameters as params

# stable_baselines3 have wrappers that simplifies 
# the preprocessing a lot, read more about them here:
# https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer


# Creates our gym environment and with all our wrappers.
def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk



class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        # TODO: Deinfe your network (agent)
        # Look at Section 4.1 in the paper for help: https://arxiv.org/pdf/1312.5602v1.pdf
        self.network = nn.Sequential(
        nn.Conv2d(4, 16, kernel_size=8, stride=4),  # first layer convolves 32 filters 20x20 output
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=4, stride=2), #second layer 9x9 output,  increase number of learned features and more edges
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 9 * 9, 256), #final fc layer, input size based on last layer, 
        nn.ReLU(),
        
        nn.Linear(256, env.single_action_space.n)
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    run_name = f"{params.env_id}__{params.exp_name}__{params.seed}__{int(time.time())}"
    best_return = -float('inf')
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = params.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
    
    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(params.env_id, params.seed, 0, params.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=params.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    # We’ll be using experience replay memory for training our DQN. 
    # It stores the transitions that the agent observes, allowing us to reuse this data later. 
    # By sampling from it randomly, the transitions that build up a batch are decorrelated. 
    # It has been shown that this greatly stabilizes and improves the DQN training procedure.
    rb = ReplayBuffer(
        params.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
    )

    obs = envs.reset()


    for global_step in range(params.total_timesteps):
        # Here we get epsilon for our epislon greedy.
        epsilon = linear_schedule(params.start_e, params.end_e, params.exploration_fraction * params.total_timesteps, global_step)

        if random.random() < epsilon:
            actions =  np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)]) # sample a random action from the environment
        else:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device) #Converts observations into proper tensor format. Ensures the model is evaluated on GPU (if available) and in float format.
            q_values = q_network(obs_tensor)  #get q_values from the network you defined, what should the network receive as input?
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # Take a step in the environment
        next_obs, rewards, dones, infos = envs.step(actions)

        # Here we print our reward.
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                break
        for idx, done in enumerate(dones):
            if done:
                episodic_return = infos[idx].get('episode', {}).get('r', 0)
                if episodic_return > best_return and global_step > params.learning_starts:
                    best_return = episodic_return
                    best_model_path = f"Lab4/{run_name}/best_model_{global_step}.pth"
                    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)  # Ensure directory exists
                    torch.save(q_network.state_dict(), best_model_path)
                    print(f"New best model saved with return {best_return} to {best_model_path}")
                

        # Save data to replay buffer
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]

        # Here we store the transitions in D
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        obs = next_obs
        # Training 
        if global_step > params.learning_starts:
            if global_step % params.train_frequency == 0:
                # Sample random minibatch of transitions from D
                data = rb.sample(params.batch_size)
                # You can get data with:
                # data.observation, data.rewards, data.dones, data.actions

                with torch.no_grad():
                    rewards = data.rewards.clone().detach().to(torch.float32)
                    dones = data.dones.clone().detach().to(torch.float32)
                    rewards = rewards.squeeze(1)  # Adjust the number '32' based on your actual batch size
                    dones = dones.squeeze(1)
                    
                    # Now we calculate the y_j for non-terminal phi.
                    target_max = q_network(data.next_observations.to(device=device, dtype=torch.float32)).max(dim=1)[0]   # Calculate max Q
                    td_target = rewards + (params.gamma * target_max * (1-dones)) # actuall Q
                    
                old_val = q_network(data.observations).gather(1, data.actions).squeeze(1)
                loss = F.mse_loss(old_val, td_target) 

                # perform our gradient decent step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % params.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        params.tau * q_network_param.data + (1.0 - params.tau) * target_network_param.data
                    )

    if params.save_model:
        model_path = f"runs/{run_name}/{params.exp_name}_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
