# -*- coding: utf-8 -*-
"""
ProjectNLP Hate Speech Detection Group 5 - VS Code Version (with wandb integration)

This script fine-tunes a BERT model for hate speech detection (e.g., OLID dataset)
using PyTorch and Hugging Face Transformers. It supports command-line arguments
for configuration, includes adversarial training (FGM), evaluation, explainability
(attention viz), saves outputs, and includes commented-out Weights & Biases (wandb)
integration for experiment tracking and hyperparameter optimization.

Usage:
python project_script_name.py --train_file path/to/train.txt --output_dir ./results [options]

Example for OLID Task A:
python project_script_name.py --train_file OLID_Tain_ATUSER_URL_EmojiRemoved_Pedro.txt \
                              --output_dir ./olid_task_a_results \
                              --task a \
                              --epochs 3 \
                              --batch_size 16 \
                              --adversarial_eps 0.01 \
                              --save_plots

To enable wandb (ensure you have wandb installed and logged in):
python project_script_name.py --train_file ... --output_dir ... --use_wandb \
                              --wandb_project "Your Wandb Project Name" \
                              --wandb_entity "Your Wandb Username/Entity"
"""

# --- Core Libraries ---
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import argparse
import json
from typing import List, Dict, Tuple, Optional, Any # For type hinting

# --- Optional: Weights & Biases for Experiment Tracking ---
# Make sure to install wandb: pip install wandb
# import wandb # Uncomment this line to use wandb

# --- Argument Parsing ---
def parse_args():
    """
    Parses command-line arguments for the script.
    Allows configuration of file paths, model, training parameters, and execution environment.
    """
    parser = argparse.ArgumentParser(description="Fine-tune BERT for Hate Speech Detection.")

    # --- File Paths & Directories ---
    parser.add_argument("--train_file", type=str, required=True,
                        help="Required: Path to the training data file (TSV format expected).")
    parser.add_argument("--test_file", type=str, default=None,
                        help="Optional: Path to the labeled test data file (TSV format). If not provided, uses a validation split from the training data.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Required: Directory to save models, results, plots, and logs.")

    # --- Model & Task Configuration ---
    parser.add_argument("--model_name", type=str, default='bert-base-uncased',
                        help="Name of the pre-trained BERT model from Hugging Face Hub (e.g., 'bert-base-uncased', 'roberta-base').")
    parser.add_argument("--task", type=str, default='a', choices=['a', 'b', 'c'],
                        help="Specifies the subtask for OLID dataset ('a': Offensive vs. Not, 'b': Targeted vs. Untargeted, 'c': Target Type - IND/GRP/OTH). Needs adaptation if using different datasets/tasks.")
    parser.add_argument("--max_len", type=int, default=128,
                        help="Maximum sequence length for BERT tokenization. Texts longer than this will be truncated, shorter ones padded.")

    # --- Training Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=4,
                        help="Number of complete passes through the training dataset.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of training examples processed in one iteration (forward/backward pass). Adjust based on GPU memory.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Initial learning rate for the AdamW optimizer. Controls how much model weights are adjusted.")
    parser.add_argument("--adversarial_eps", type=float, default=0.0,
                        help="Epsilon value for FGM adversarial training perturbation strength. Set > 0 to enable FGM.")
    parser.add_argument("--patience", type=int, default=3,
                        help="Number of epochs to wait for validation loss improvement before stopping training early. Set to 0 to disable early stopping.")

    # --- Execution & Environment ---
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initializing random number generators (numpy, torch) for reproducibility.")
    parser.add_argument("--no_gpu", action='store_true',
                        help="Flag to force CPU usage even if a GPU is available.")
    parser.add_argument("--save_plots", action='store_true',
                        help="Flag to save generated plots (history, confusion matrix, attention) to files in the output directory instead of displaying them interactively.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of worker processes for DataLoader to load data in parallel. Set based on CPU cores and system capability.")

    # --- Weights & Biases Arguments (Optional) ---
    parser.add_argument("--use_wandb", action='store_true',
                        help="Flag to enable Weights & Biases logging. Requires 'wandb' library installed and user logged in (`wandb login`).")
    parser.add_argument("--wandb_project", type=str, default="hate-speech-detection",
                        help="Name of the project in Weights & Biases where runs will be logged.")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Your Weights & Biases username or team name (entity) under which the project resides.")
    # Add more wandb args if needed (e.g., --wandb_run_name, --wandb_tags)

    args = parser.parse_args()

    # --- Post-processing/Validation for Arguments ---
    # Ensure output directory exists, creating it if necessary
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory set to: {args.output_dir}")

    # Validate num_workers against available CPU cores
    max_workers = os.cpu_count() if os.cpu_count() else 1
    if args.num_workers > max_workers:
         print(f"Warning: Requested num_workers ({args.num_workers}) exceeds available CPU cores ({max_workers}). Setting num_workers to {max_workers}.")
         args.num_workers = max_workers
    elif args.num_workers < 0:
         print("Warning: num_workers cannot be negative. Setting to 0.")
         args.num_workers = 0
    print(f"Using {args.num_workers} workers for DataLoader.")

    return args

# --- Seed and Device Setup ---
def setup_environment(seed: int, no_gpu: bool) -> torch.device:
    """
    Sets random seeds for reproducibility across libraries (numpy, torch) and
    determines the appropriate compute device (GPU or CPU) based on availability
    and user preference.

    Args:
        seed (int): The random seed value to use.
        no_gpu (bool): If True, forces CPU usage regardless of GPU availability.

    Returns:
        torch.device: The selected compute device ('cuda', 'mps', or 'cpu').
    """
    # Set seeds for numpy and torch CPU/GPU
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Optional: For potentially more deterministic behavior on GPU (can impact performance)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        print(f"PyTorch CUDA seed set to {seed}")
    else:
        print(f"PyTorch CPU seed set to {seed}")

    # --- Device Selection Logic ---
    # Priority: CUDA -> MPS (Apple Silicon) -> CPU
    if not no_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU detected. Using device: {torch.cuda.get_device_name(0)}")
        # Note for collaborators with Intel ARC GPUs:
        # May need Intel Extension for PyTorch (IPEX) installed (`pip install intel_extension_for_pytorch`)
        # and potentially code changes (e.g., `import intel_extension_for_pytorch as ipex`). Consult IPEX docs.
    elif not no_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Check for Apple Silicon Metal Performance Shaders (MPS)
        device = torch.device("mps")
        print("Apple MPS detected. Using device: mps")
    else:
        # Default to CPU
        device = torch.device("cpu")
        if no_gpu:
            print("CPU usage forced via --no_gpu flag.")
        elif not torch.cuda.is_available():
            print("No GPU detected or CUDA not available. Using CPU.")
        else: # Should not happen based on logic, but included for completeness
             print("Using CPU.") # Catch-all, though covered by above conditions
    return device

# --- Data Loading ---
def load_data(filepath: str, task: str = 'a') -> Tuple[Optional[pd.DataFrame], int]:
    """
    Loads and preprocesses data from a TSV file based on the specified OLID subtask.
    Handles label conversion and filtering according to the task requirements.

    Args:
        filepath (str): Path to the TSV data file.
        task (str): The OLID subtask identifier ('a', 'b', or 'c'). Determines required
                    columns, filtering, and label processing.

    Returns:
        Tuple[Optional[pd.DataFrame], int]: A tuple containing:
            - The processed pandas DataFrame with 'id', 'tweet', and numerical 'label'
              columns (or None if loading/processing fails).
            - The number of unique labels expected for the specified task (e.g., 2 for 'a',
              3 for 'c'). Returns 0 if loading fails.
    """
    print(f"Attempting to load data from: {filepath}")
    try:
        # Read TSV using pandas. 'on_bad_lines'='warn' provides info on parsing issues without stopping.
        df = pd.read_csv(filepath, sep='\t', header=0, on_bad_lines='warn')
        print(f"Successfully read {len(df)} rows initially from {filepath}.")

        # --- Data Validation and Cleaning ---
        # Ensure the essential 'tweet' column exists.
        if 'tweet' not in df.columns:
             raise ValueError(f"Critical Error: Required 'tweet' column not found in {filepath}.")
        # Convert 'tweet' column to string type and fill any potential NaN values with empty strings.
        df['tweet'] = df['tweet'].astype(str).fillna('')

        # --- Task-Specific Configuration ---
        # Define configurations for each OLID task, specifying required columns,
        # how to determine the label, and any necessary pre-filtering.
        task_configs = {
            'a': {'required_cols': ['subtask_a'], 'label_col': 'subtask_a', 'positive_label': 'OFF', 'num_labels': 2, 'filter_condition': None},
            'b': {'required_cols': ['subtask_a', 'subtask_b'], 'label_col': 'subtask_b', 'positive_label': 'TIN', 'num_labels': 2, 'filter_condition': ('subtask_a', 'OFF')}, # Filter for OFF tweets first
            'c': {'required_cols': ['subtask_b', 'subtask_c'], 'label_col': 'subtask_c', 'label_map': {'IND': 0, 'GRP': 1, 'OTH': 2}, 'num_labels': 3, 'filter_condition': ('subtask_b', 'TIN')} # Filter for TIN tweets first
        }

        # Validate the provided task identifier.
        if task not in task_configs:
            raise ValueError(f"Invalid task identifier '{task}'. Choose from {list(task_configs.keys())}.")

        config = task_configs[task]

        # Check if all required columns for the specified task are present in the DataFrame.
        missing_cols = [col for col in config['required_cols'] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required column(s) for Task {task.upper()} in {filepath}: {', '.join(missing_cols)}")

        # --- Filtering (for Tasks B and C) ---
        if config['filter_condition']:
            filter_col, filter_val = config['filter_condition']
            original_count = len(df)
            # Apply the filter based on the condition defined in task_configs.
            df = df[df[filter_col] == filter_val].copy() # Use .copy() to avoid SettingWithCopyWarning
            print(f"Filtered data based on '{filter_col}' == '{filter_val}'. Kept {len(df)} out of {original_count} rows.")

        # --- Label Processing ---
        # Select essential columns ('id', 'tweet', and the task-specific label column).
        df = df[['id', 'tweet', config['label_col']]].copy()
        # Rename the task-specific label column to a standard 'label' column name.
        df.rename(columns={config['label_col']: 'label'}, inplace=True)

        # Convert textual labels to numerical format.
        if 'label_map' in config: # Handle multi-class mapping (Task C)
            df['label'] = df['label'].map(config['label_map']).fillna(-1) # Map labels, use -1 for unmappable values
            invalid_labels_count = (df['label'] == -1).sum()
            if invalid_labels_count > 0:
                print(f"Warning: Found {invalid_labels_count} rows with unmappable labels for Task C. These rows will be removed.")
                df = df[df['label'] != -1] # Remove rows with invalid labels
            df['label'] = df['label'].astype(int) # Convert valid labels to integer
        else: # Handle binary mapping (Tasks A and B)
            # Convert labels to 1 if they match the 'positive_label', otherwise 0. Case-insensitive comparison.
            df['label'] = df['label'].apply(lambda x: 1 if str(x).upper() == config['positive_label'] else 0)

        num_labels = config['num_labels'] # Get the expected number of labels for this task.

        # --- Final Checks and Report ---
        if df.empty:
            # If the DataFrame is empty after processing (e.g., due to filtering or invalid labels).
            print(f"Warning: No valid data entries found for Task {task.upper()} after processing {filepath}.")
            return None, 0
        else:
            # Report successful loading and processing.
            print(f"Loaded and processed {len(df)} valid entries for Task {task.upper()}.")
            print("Label distribution:")
            # Display the distribution of the final numerical labels.
            print(df['label'].value_counts(normalize=True).to_string())
            return df, num_labels

    except FileNotFoundError:
        # Handle the case where the specified file does not exist.
        print(f"Error: File not found at specified path: {filepath}")
        return None, 0
    except ValueError as ve:
        # Handle specific errors related to data validation (e.g., missing columns).
        print(f"Data Loading Configuration Error: {ve}")
        return None, 0
    except Exception as e:
        # Catch any other unexpected errors during file reading or processing.
        print(f"An unexpected error occurred while loading/processing file {filepath}: {type(e).__name__} - {e}")
        return None, 0

# --- PyTorch Dataset and DataLoader ---
class TextClassificationDataset(Dataset):
    """
    Custom PyTorch Dataset class for handling text classification data.
    It takes numpy arrays of texts and labels, and uses the provided Hugging Face
    tokenizer to prepare model inputs dynamically during data loading.
    """
    def __init__(self, texts: np.ndarray, labels: np.ndarray, tokenizer: BertTokenizer, max_len: int):
        """
        Initializes the dataset instance.

        Args:
            texts (np.ndarray): A numpy array containing the text samples (strings).
            labels (np.ndarray): A numpy array containing the corresponding numerical labels.
            tokenizer (BertTokenizer): An initialized Hugging Face tokenizer (e.g., BertTokenizer).
            max_len (int): The maximum sequence length for tokenization (padding/truncation).
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        print(f"Initialized Dataset with {len(self.texts)} samples.")

    def __len__(self) -> int:
        """Returns the total number of samples (texts/labels) in the dataset."""
        return len(self.texts)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """
        Retrieves and processes a single sample from the dataset by its index.
        This method is called by the DataLoader.

        Args:
            item (int): The index of the sample to retrieve (0 to len(dataset)-1).

        Returns:
            Dict[str, Any]: A dictionary containing the processed sample:
                'text': The original text string (useful for debugging/analysis).
                'input_ids': Padded/truncated token IDs as a PyTorch tensor.
                'attention_mask': The attention mask as a PyTorch tensor (1 for real tokens, 0 for padding).
                'labels': The numerical label as a PyTorch tensor (dtype=torch.long).
        """
        text = str(self.texts[item]) # Ensure text is a string
        label = self.labels[item]   # Get the corresponding label

        # Use the tokenizer's encode_plus method for comprehensive tokenization:
        # - Adds special tokens ([CLS], [SEP]).
        # - Pads sequences shorter than max_len to max_len.
        # - Truncates sequences longer than max_len to max_len.
        # - Creates an attention mask to differentiate real tokens from padding.
        # - Returns PyTorch tensors ('pt').
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False, # Not needed for standard BERT classification
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # Return the processed data in a dictionary format, ensuring tensors are flattened.
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(), # Shape: [max_len]
            'attention_mask': encoding['attention_mask'].flatten(), # Shape: [max_len]
            'labels': torch.tensor(label, dtype=torch.long) # Ensure label is LongTensor for CrossEntropyLoss
        }

def create_data_loader(df: pd.DataFrame, tokenizer: BertTokenizer, max_len: int, batch_size: int, num_workers: int, shuffle: bool = False) -> DataLoader:
    """
    Utility function to create a PyTorch DataLoader from a pandas DataFrame.
    Instantiates the TextClassificationDataset and wraps it in a DataLoader.

    Args:
        df (pd.DataFrame): The DataFrame containing 'tweet' and 'label' columns.
        tokenizer (BertTokenizer): The Hugging Face tokenizer instance.
        max_len (int): Maximum sequence length for tokenization.
        batch_size (int): The number of samples per batch.
        num_workers (int): The number of subprocesses to use for data loading.
        shuffle (bool): Whether to shuffle the data in each epoch (default: False).
                        Should be True for training, False for validation/testing.

    Returns:
        DataLoader: A configured PyTorch DataLoader instance.
    """
    print(f"Creating DataLoader: batch_size={batch_size}, shuffle={shuffle}, num_workers={num_workers}")
    # Instantiate the custom Dataset using data from the DataFrame
    ds = TextClassificationDataset(
        texts=df.tweet.to_numpy(),   # Convert text column to numpy array
        labels=df.label.to_numpy(), # Convert label column to numpy array
        tokenizer=tokenizer,
        max_len=max_len
    )
    # Create and return the DataLoader
    # pin_memory=True can potentially speed up data transfer to GPU memory
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True
    )

# --- Model Loading ---
def load_model(model_name: str, num_labels: int) -> BertForSequenceClassification:
    """
    Loads a pre-trained BERT model configured for sequence classification from the
    Hugging Face Hub.

    Args:
        model_name (str): The identifier of the pre-trained model (e.g., 'bert-base-uncased').
        num_labels (int): The number of distinct output labels required for the
                          classification task (e.g., 2 for binary, 3 for 3-class).

    Returns:
        BertForSequenceClassification: The loaded Hugging Face model instance, ready for fine-tuning.
                                      Includes a classification head on top of the base BERT model.
    """
    print(f"\nLoading pre-trained model: {model_name} for {num_labels}-way classification.")
    # Load the model using the specified name and number of labels.
    # output_attentions=True is crucial for accessing attention weights later for explainability.
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        output_attentions=True,     # Request attention outputs
        output_hidden_states=False, # Hidden states are usually not needed for basic classification
        # ignore_mismatched_sizes=True # Useful if loading weights into a model with a differently sized classification head (use with caution)
    )
    return model

# --- Training Epoch ---
def train_epoch(model: BertForSequenceClassification, data_loader: DataLoader, loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer, device: torch.device, scheduler: torch.optim.lr_scheduler._LRScheduler,
                adversarial_eps: float, current_epoch: int, use_wandb: bool) -> Tuple[float, float]:
    """
    Performs one full training epoch: iterates through batches, computes loss,
    optionally performs FGM adversarial training, computes gradients, updates weights,
    and logs progress.

    Args:
        model: The PyTorch model (must be in training mode).
        data_loader: DataLoader for the training set.
        loss_fn: The loss function instance.
        optimizer: The optimizer instance.
        device: The target compute device ('cuda' or 'cpu').
        scheduler: The learning rate scheduler instance.
        adversarial_eps (float): Perturbation magnitude for FGM. If 0, FGM is skipped.
        current_epoch (int): The current epoch number (0-indexed).
        use_wandb (bool): Flag to enable/disable Weights & Biases logging.

    Returns:
        Tuple[float, float]: A tuple containing the average training loss and average
                             training accuracy for the completed epoch.
    """
    model = model.train() # Ensure model is in training mode
    total_loss, total_accuracy = 0, 0
    total_samples = 0
    start_time = time.time()
    print(f"  Starting Training Epoch {current_epoch + 1}...")

    # Loop over all batches in the training data loader
    for i, batch in enumerate(data_loader):
        # Move batch tensors to the designated compute device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # --- Standard Forward and Loss Calculation ---
        # Reset gradients from previous iteration
        optimizer.zero_grad()
        # Perform forward pass through the model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # Extract the loss calculated by the model (e.g., CrossEntropyLoss)
        loss = outputs.loss
        # Extract the raw prediction scores (logits)
        logits = outputs.logits

        # --- Metrics Accumulation ---
        batch_loss = loss.item() # Get the scalar loss value for the batch
        total_loss += batch_loss * input_ids.size(0) # Accumulate loss weighted by batch size
        _, preds = torch.max(logits, dim=1) # Get predicted class indices (highest logit)
        batch_accuracy = torch.sum(preds == labels).item() # Count correct predictions
        total_accuracy += batch_accuracy
        total_samples += labels.size(0) # Keep track of total samples processed

        # --- Adversarial Training Step (FGM) ---
        if adversarial_eps > 0:
            # 1. Calculate gradients w.r.t. the original loss.
            # 'retain_graph=True' is essential because we need to reuse parts of the
            # computation graph for the second backward pass (on the adversarial loss).
            loss.backward(retain_graph=True)

            # Access the model's word embedding layer.
            embedding_layer = model.bert.embeddings.word_embeddings
            # Check if this layer has gradients (it might not if frozen).
            if embedding_layer.weight.grad is not None:
                # --- Attempt to perturb embeddings directly ---
                try:
                    # Get the actual embedding vectors for the current input_ids.
                    input_embeddings = embedding_layer(input_ids)
                    # Enable gradient calculation specifically for these embedding vectors.
                    input_embeddings.requires_grad_(True)

                    # 2. Perform a forward pass using the embeddings to get adversarial loss
                    #    and gradients w.r.t. these embeddings.
                    #    Note: This relies on the model's `forward` method supporting `inputs_embeds`.
                    adv_outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask, labels=labels)
                    adv_loss = adv_outputs.loss
                    # Calculate gradients of adv_loss w.r.t. input_embeddings.
                    adv_loss.backward()

                    # Check if gradients were computed for the input embeddings.
                    if input_embeddings.grad is not None:
                        # Calculate the perturbation: epsilon * sign(gradient).
                        perturbation = adversarial_eps * input_embeddings.grad.sign()
                        # Create perturbed embeddings. Detach the original embeddings to avoid
                        # tracking this operation in the computation graph for the original loss.
                        perturbed_embeddings = input_embeddings.detach() + perturbation

                        # Zero gradients before the final adversarial forward/backward pass.
                        optimizer.zero_grad()

                        # 3. Perform a forward pass with the perturbed embeddings.
                        final_outputs = model(inputs_embeds=perturbed_embeddings, attention_mask=attention_mask, labels=labels)
                        final_loss = final_outputs.loss

                        # 4. Perform the backward pass on the adversarial loss. This calculates
                        #    the gradients needed to update the model weights to resist the perturbation.
                        final_loss.backward()
                        # Optional debug print
                        # print(f"Debug: FGM step successful for batch {i}.") if i % 100 == 0 else None

                    else: # Fallback if embedding gradients are None after adv_loss.backward()
                        print(f"Warning (Batch {i}): Could not get embedding gradients for FGM perturbation. Performing standard backprop on original loss.")
                        # Need to ensure gradients for the original loss are used if FGM fails mid-step.
                        # Since loss.backward(retain_graph=True) was already called, the grads should exist.
                        # We just need to zero grads and call backward on the original loss again if the FGM path failed *before* final_loss.backward().
                        # However, if we reached here *after* adv_loss.backward(), the graph might be complex.
                        # Safest fallback: Zero grads and use the original loss's backward pass.
                        optimizer.zero_grad()
                        loss.backward() # Use original loss gradients computed earlier

                except (TypeError, RuntimeError) as e: # Fallback if model doesn't accept 'inputs_embeds' or other runtime error
                    print(f"Warning (Batch {i}): FGM step failed ({type(e).__name__}: {e}). Performing standard backprop on original loss.")
                    optimizer.zero_grad()
                    loss.backward() # Use original loss gradients
            else: # Fallback if embedding layer weights have no gradients (e.g., frozen layer)
                 print(f"Warning (Batch {i}): Embedding layer weights have no gradients. Skipping FGM. Performing standard backprop on original loss.")
                 optimizer.zero_grad()
                 loss.backward() # Use original loss gradients
        else:
            # --- Standard Backward Pass (if FGM is disabled) ---
            # Calculate gradients for the original loss.
            loss.backward()

        # --- Gradient Clipping, Optimizer Step, Scheduler Step ---
        # Clip gradients to prevent them from becoming excessively large (exploding gradients).
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update model parameters using the computed gradients.
        optimizer.step()

        # Update the learning rate based on the scheduler's policy.
        scheduler.step()

        # --- Logging ---
        # Log batch-level metrics to Weights & Biases if enabled
        # if use_wandb:
        #     try:
        #         wandb.log({
        #             "train/batch_loss": batch_loss,
        #             "train/batch_acc": batch_accuracy / labels.size(0) if labels.size(0) > 0 else 0,
        #             "progress/learning_rate": scheduler.get_last_lr()[0], # Log current LR
        #             "progress/epoch": current_epoch + (i / len(data_loader)), # Fractional epoch progress
        #             "progress/batch_num": i
        #         })
        #     except Exception as e:
        #         print(f"Warning: Failed to log to wandb - {e}")


        # Print progress update to console periodically.
        if (i + 1) % 50 == 0 or (i + 1) == len(data_loader): # Log every 50 batches and the last batch
            elapsed_time = time.time() - start_time
            # Calculate running average loss and accuracy for reporting
            current_avg_loss = total_loss / total_samples if total_samples > 0 else 0
            current_avg_acc = total_accuracy / total_samples if total_samples > 0 else 0
            # Display progress information
            print(f'    Batch {i+1:>4}/{len(data_loader)} | Train Loss: {current_avg_loss:.4f} | Train Acc: {current_avg_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed_time:.2f}s')

    # --- End of Epoch Calculation & Reporting ---
    epoch_time = time.time() - start_time
    # Calculate average loss and accuracy for the entire epoch
    avg_epoch_loss = total_loss / total_samples if total_samples > 0 else 0
    epoch_accuracy = total_accuracy / total_samples if total_samples > 0 else 0
    print(f"  Epoch {current_epoch + 1} finished | Avg Train Loss: {avg_epoch_loss:.4f} | Avg Train Acc: {epoch_accuracy:.4f} | Total Time: {epoch_time:.2f}s")

    return avg_epoch_loss, epoch_accuracy

# --- Evaluation ---
def eval_model(model: BertForSequenceClassification, data_loader: DataLoader, loss_fn: torch.nn.Module,
               device: torch.device, num_labels: int, epoch: int, eval_type: str = "Validation") -> Tuple[float, float, float, float, float, np.ndarray, str]:
    """
    Evaluates the model's performance on a given dataset (validation or test set).
    Calculates loss, accuracy, F1, precision, recall, confusion matrix, and classification report.

    Args:
        model: The trained PyTorch model.
        data_loader: DataLoader for the evaluation dataset.
        loss_fn: The loss function instance.
        device: The compute device ('cuda' or 'cpu').
        num_labels (int): The number of unique labels in the dataset (used for metric averaging).
        epoch (int): The current epoch number (for logging purposes).
        eval_type (str): A string indicating the type of evaluation ("Validation" or "Test").

    Returns:
        Tuple containing:
            - avg_loss (float): Average loss over the evaluation dataset.
            - accuracy (float): Overall accuracy.
            - f1 (float): F1 score (binary or weighted average).
            - precision (float): Precision score (binary or weighted average).
            - recall (float): Recall score (binary or weighted average).
            - conf_matrix (np.ndarray): The confusion matrix.
            - class_report (str): A formatted string containing the classification report.
    """
    model.eval() # Set the model to evaluation mode (disables dropout, batch norm updates, etc.)
    total_loss, total_accuracy = 0, 0
    total_samples = 0
    all_labels, all_preds = [], [] # Lists to store all true labels and predictions
    print(f"  Starting {eval_type} Epoch {epoch + 1}...")
    start_time = time.time()

    # Disable gradient calculations as they are not needed for evaluation
    with torch.no_grad():
        # Iterate through all batches in the evaluation data loader
        for batch in data_loader:
            # Move batch data to the compute device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Perform forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            # Get predictions by finding the index with the maximum logit value
            _, preds = torch.max(logits, dim=1)

            # Accumulate loss, accuracy, and store labels/predictions
            total_loss += loss.item() * input_ids.size(0) # Weighted loss
            total_accuracy += torch.sum(preds == labels).item() # Correct predictions
            total_samples += labels.size(0) # Total samples evaluated
            all_labels.extend(labels.cpu().numpy()) # Store true labels (move to CPU)
            all_preds.extend(preds.cpu().numpy()) # Store predictions (move to CPU)

    # Calculate average loss and accuracy
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = total_accuracy / total_samples if total_samples > 0 else 0

    # --- Calculate Detailed Metrics using Scikit-learn ---
    # Determine the averaging strategy for F1, precision, recall based on the number of labels
    # 'binary' is used for 2 classes, 'weighted' for multi-class to account for label imbalance.
    average_strategy = 'binary' if num_labels == 2 else 'weighted'

    # Calculate metrics, handling potential zero division issues (e.g., no instances of a class predicted)
    f1 = f1_score(all_labels, all_preds, average=average_strategy, zero_division=0)
    precision = precision_score(all_labels, all_preds, average=average_strategy, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=average_strategy, zero_division=0)
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    # Generate a formatted classification report string
    class_report = classification_report(all_labels, all_preds, zero_division=0, output_dict=False)

    eval_time = time.time() - start_time
    print(f"  {eval_type} finished | Time: {eval_time:.2f}s")

    # Return all calculated metrics
    return avg_loss, accuracy, f1, precision, recall, conf_matrix, class_report


# --- Plotting and Saving Utilities ---
def plot_history(history: Dict[str, List[float]], output_dir: str, save_plots: bool):
    """
    Generates and displays/saves plots for training and validation loss and accuracy curves.

    Args:
        history (Dict[str, List[float]]): A dictionary containing lists of metrics per epoch
                                          (e.g., history['train_loss'], history['val_loss']).
        output_dir (str): The directory where the plot file should be saved if save_plots is True.
        save_plots (bool): If True, saves the plot to a file; otherwise, displays it.
    """
    # Determine the number of epochs based on the length of the history lists
    epochs_ran = len(history.get('train_loss', []))
    if epochs_ran == 0:
        print("Warning: No history data to plot.")
        return
    epoch_nums = range(1, epochs_ran + 1) # Create x-axis values (epoch numbers)

    plt.figure(figsize=(14, 5)) # Create a figure with two subplots

    # --- Plot Loss ---
    plt.subplot(1, 2, 1) # First subplot for loss
    plt.plot(epoch_nums, history['train_loss'], 'bo-', label='Train Loss') # Blue line with circles
    plt.plot(epoch_nums, history['val_loss'], 'ro-', label='Validation Loss') # Red line with circles
    plt.title('Model Loss During Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend() # Show legend
    plt.grid(True) # Add grid lines
    plt.xticks(epoch_nums) # Ensure x-axis ticks correspond to epoch numbers

    # --- Plot Accuracy ---
    plt.subplot(1, 2, 2) # Second subplot for accuracy
    plt.plot(epoch_nums, history['train_acc'], 'bo-', label='Train Accuracy')
    plt.plot(epoch_nums, history['val_acc'], 'ro-', label='Validation Accuracy')
    plt.title('Model Accuracy During Training')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.xticks(epoch_nums)

    # Adjust layout to prevent overlapping titles/labels
    plt.tight_layout()

    # Save or display the plot
    if save_plots:
        filepath = os.path.join(output_dir, "training_history.png")
        try:
            plt.savefig(filepath)
            print(f"Training history plot saved to {filepath}")
        except Exception as e:
            print(f"Error saving training history plot: {e}")
        plt.close() # Close the plot figure to free up memory
    else:
        plt.show() # Display the plot interactively

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], output_dir: str, filename: str, save_plots: bool):
    """
    Generates and displays/saves a heatmap visualization of the confusion matrix.

    Args:
        cm (np.ndarray): The confusion matrix (output from sklearn.metrics.confusion_matrix).
        labels (List[str]): A list of class names corresponding to the matrix axes.
        output_dir (str): Directory to save the plot file.
        filename (str): Name for the saved plot file (e.g., "validation_cm.png").
        save_plots (bool): If True, saves the plot; otherwise, displays it.
    """
    plt.figure(figsize=(max(6, len(labels)*1.5), max(5, len(labels)*1.2))) # Adjust size based on number of labels
    # Create a DataFrame for better labeling with seaborn
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    # Generate the heatmap
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues') # 'annot=True' shows counts, 'fmt='d'' formats as integers
    plt.title(f'Confusion Matrix ({filename.split(".")[0]})') # Add title
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout() # Adjust layout

    # Save or display
    if save_plots:
        filepath = os.path.join(output_dir, filename)
        try:
            plt.savefig(filepath)
            print(f"Confusion matrix plot saved to {filepath}")
        except Exception as e:
            print(f"Error saving confusion matrix plot: {e}")
        plt.close()
    else:
        plt.show()

# --- Explainability ---
def visualize_attention(model: BertForSequenceClassification, tokenizer: BertTokenizer, text: str, device: torch.device, max_len: int, output_dir: str, filename_prefix: str, save_plots: bool):
    """
    Visualizes the average attention weights from the [CLS] token to all other tokens
    in the input text, averaged across all layers and heads. Helps understand which
    parts of the input the model focused on for classification.

    Args:
        model: The trained model (must output attentions).
        tokenizer: The tokenizer used for the model.
        text (str): The input text string to visualize attention for.
        device: The compute device.
        max_len (int): Max sequence length used during training/tokenization.
        output_dir (str): Directory to save the plot file.
        filename_prefix (str): Prefix for the saved attention plot filename.
        save_plots (bool): If True, saves the plot; otherwise, displays it.
    """
    model.eval() # Ensure model is in evaluation mode
    # Tokenize the input text using the same parameters as during training/evaluation
    inputs = tokenizer.encode_plus(
        text, add_special_tokens=True, max_length=max_len, padding='max_length',
        truncation=True, return_tensors='pt', return_attention_mask=True
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Perform forward pass to get outputs, including attention weights
    with torch.no_grad():
        # Crucially, the model must have been loaded with output_attentions=True
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)

    # Extract attention weights (a tuple, one tensor per layer)
    attentions = outputs.attentions
    # Check if attentions were actually returned
    if attentions is None:
        print("Error: Attention weights not found in model output. Ensure 'output_attentions=True' was set during model loading.")
        return

    # --- Process Attention Weights ---
    # Calculate average attention from the [CLS] token (index 0) across all layers and heads
    num_layers = len(attentions)
    # Shape of one layer's attention: (batch_size=1, num_heads, seq_len, seq_len)
    seq_len = attentions[0].shape[-1] # Get sequence length from attention tensor
    cls_attentions = torch.zeros(seq_len, device=device) # Initialize tensor to accumulate attentions

    # Iterate through each layer's attention tensor
    for layer in range(num_layers):
        # Average attention scores across all heads for the current layer
        # attentions[layer][0] accesses the attention for the first (and only) item in the batch
        avg_head_attention = attentions[layer][0].mean(dim=0) # Shape: [seq_len, seq_len]
        # Extract attention scores originating *from* the [CLS] token (index 0) *to* all other tokens
        cls_attentions += avg_head_attention[0, :]

    # Average the accumulated attention scores over the number of layers
    cls_attentions /= num_layers
    # Move the final attention scores to CPU and convert to numpy array for plotting
    cls_attentions = cls_attentions.cpu().numpy()

    # --- Prepare Tokens for Visualization ---
    # Convert token IDs back to actual tokens/subwords
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())

    # Filter out padding tokens and potentially special tokens ([CLS], [SEP]) for cleaner visualization
    # Use the attention mask to identify real tokens vs padding
    valid_indices = [i for i, mask_val in enumerate(attention_mask[0].cpu().numpy()) if mask_val == 1]
    tokens = [tokens[i] for i in valid_indices] # Keep only tokens corresponding to mask value 1
    cls_attentions = cls_attentions[valid_indices] # Keep only attention scores for these tokens

    # Normalize attention scores to range [0, 1] for consistent color mapping in the heatmap
    min_attn, max_attn = np.min(cls_attentions), np.max(cls_attentions)
    if max_attn > min_attn: # Avoid division by zero if all attentions are the same
         cls_attentions = (cls_attentions - min_attn) / (max_attn - min_attn)
    else:
         cls_attentions = np.zeros_like(cls_attentions) # Set to zero if uniform

    # --- Plotting ---
    # Create the heatmap figure
    plt.figure(figsize=(max(10, len(tokens) * 0.6), 2.5)) # Adjust figure width based on number of tokens
    # Use seaborn's heatmap for visualization. Display attentions as a single row.
    sns.heatmap([cls_attentions], xticklabels=tokens, annot=False, cmap="viridis", cbar=True,
                linewidths=.5, linecolor='lightgray') # Add lines between cells for clarity
    plt.title(f'Avg Attention from [CLS] for: "{text[:60]}..."', fontsize=10) # Show truncated text in title
    plt.xlabel("Tokens", fontsize=9)
    plt.xticks(rotation=45, ha='right', fontsize=8) # Rotate x-axis labels for readability
    plt.yticks([]) # Hide y-axis ticks as it's a single row heatmap
    plt.tight_layout() # Adjust layout

    # Save or display the plot
    if save_plots:
        filepath = os.path.join(output_dir, f"{filename_prefix}_attention_viz.png")
        try:
            plt.savefig(filepath)
            print(f"Attention visualization saved to {filepath}")
        except Exception as e:
            print(f"Error saving attention visualization plot: {e}")
        plt.close()
    else:
        plt.show()


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Setup ---
    # Parse command-line arguments
    args = parse_args()
    # Log the arguments used for this run (useful for reproducibility)
    print("--- Script Arguments ---")
    print(json.dumps(vars(args), indent=2))
    print("-----------------------")

    # Set up random seeds and determine compute device (GPU/CPU)
    device = setup_environment(args.seed, args.no_gpu)

    # --- Weights & Biases Initialization (Optional) ---
    # if args.use_wandb:
    #     try:
    #         # Initialize wandb run
    #         wandb.init(
    #             project=args.wandb_project,
    #             entity=args.wandb_entity, # Can be None if using default entity
    #             config=vars(args) # Log all command-line arguments as hyperparameters
    #             # Optional: Add run name, tags, notes, etc.
    #             # name=f"{args.model_name}-task_{args.task}-eps_{args.adversarial_eps}",
    #             # tags=[args.model_name, f"task_{args.task}", f"fgm_{args.adversarial_eps > 0}"]
    #         )
    #         print("Weights & Biases logging enabled.")
    #     except ImportError:
    #         print("Warning: 'wandb' library not found, but --use_wandb was specified. Please install wandb (`pip install wandb`). Disabling wandb.")
    #         args.use_wandb = False
    #     except Exception as e:
    #         print(f"Warning: Failed to initialize Weights & Biases: {e}. Disabling wandb.")
    #         args.use_wandb = False
    # else:
    #     print("Weights & Biases logging disabled.")
    #     # Ensure args.use_wandb is False if not explicitly set (safety check)
    #     args.use_wandb = False


    # --- Load Tokenizer and Data ---
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    print("\n--- Loading Data ---")
    # Load training data
    train_df, num_labels = load_data(args.train_file, args.task)
    # Exit if training data loading failed
    if train_df is None or train_df.empty:
        print("Critical Error: Failed to load training data. Exiting.")
        # if args.use_wandb: wandb.finish(exit_code=1) # Finish wandb run with error code
        exit(1) # Exit script

    # --- Prepare Validation and Test Sets ---
    val_df = None
    test_df = None
    test_data_loader = None

    # Check if a separate test file is provided
    if args.test_file:
        print(f"Loading provided test file: {args.test_file}")
        test_df, test_num_labels = load_data(args.test_file, args.task)
        # Validate test set loading and label consistency
        if test_df is None or test_df.empty:
            print("Warning: Could not load or process the provided test file. Proceeding without test set evaluation.")
        elif test_num_labels != num_labels:
             print(f"Warning: Number of labels mismatch between train ({num_labels}) and test ({test_num_labels}) sets for task '{args.task}'. Skipping test evaluation.")
             test_df = None # Nullify test_df if labels mismatch
        else:
            # If test set is valid, create a small validation set from the original training data
            val_split_frac = 0.1 # Use 10% of original train data for validation
            val_df = train_df.sample(frac=val_split_frac, random_state=args.seed)
            # Remove validation samples from the training set
            train_df = train_df.drop(val_df.index)
            print(f"Using provided test set ({len(test_df)} samples).")
            print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples.")
            # Create DataLoader for the test set
            test_data_loader = create_data_loader(test_df, tokenizer, args.max_len, args.batch_size, args.num_workers, shuffle=False)
    else:
        # If no test file is provided, split the training data into training and validation sets
        val_split_size = 0.15 # Use 15% for validation
        print(f"No test file provided. Splitting training data ({len(train_df)} samples) into train/validation ({1-val_split_size:.0%}/{val_split_size:.0%}).")
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_split_size,
            random_state=args.seed,
            stratify=train_df['label'] # Ensure proportional split based on labels
        )
        print(f"Resulting split: {len(train_df)} train samples, {len(val_df)} validation samples.")

    # --- Create DataLoaders ---
    # Ensure validation set exists before creating its loader
    if val_df is None or val_df.empty:
         print("Critical Error: No validation data available. Exiting.")
         # if args.use_wandb: wandb.finish(exit_code=1)
         exit(1)

    print("\n--- Creating DataLoaders ---")
    train_data_loader = create_data_loader(train_df, tokenizer, args.max_len, args.batch_size, args.num_workers, shuffle=True)
    val_data_loader = create_data_loader(val_df, tokenizer, args.max_len, args.batch_size, args.num_workers, shuffle=False) # No shuffle for validation

    # --- Initialize Model, Optimizer, Scheduler, Loss ---
    print("\n--- Initializing Model and Training Components ---")
    # Load the pre-trained model
    model = load_model(args.model_name, num_labels)
    # Move model to the designated compute device
    model = model.to(device)

    # Initialize the AdamW optimizer (recommended for Transformers)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8) # eps for numerical stability
    # Calculate total number of training steps for the scheduler
    total_steps = len(train_data_loader) * args.epochs
    # Initialize the learning rate scheduler (linear decay with optional warmup)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps) # No warmup steps here
    # Define the loss function (CrossEntropyLoss is suitable for multi-class classification output from BertForSequenceClassification)
    loss_fn = torch.nn.CrossEntropyLoss().to(device) # Move loss function to device if it has parameters (though CrossEntropyLoss doesn't)

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    # Dictionary to store metrics history per epoch
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    best_val_loss = float('inf') # Initialize best validation loss to infinity
    epochs_no_improve = 0 # Counter for early stopping
    # Path to save the best model based on validation loss
    best_model_path = os.path.join(args.output_dir, f'{args.model_name.replace("/", "_")}_best_val_loss.bin')

    # --- Optional: wandb watch model ---
    # if args.use_wandb:
    #     # Log gradients and model parameters (optional, can increase overhead)
    #     wandb.watch(model, log_freq=100) # Log every 100 batches

    # Loop over the specified number of epochs
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        # --- Training Step ---
        # Run one epoch of training
        train_loss, train_acc = train_epoch(
            model, train_data_loader, loss_fn, optimizer, device, scheduler,
            args.adversarial_eps, epoch, args.use_wandb
        )
        print(f'  Avg Training Epoch {epoch+1}   | Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}')

        # --- Validation Step ---
        # Evaluate the model on the validation set
        val_loss, val_acc, val_f1, val_precision, val_recall, val_cm, val_report = eval_model(
            model, val_data_loader, loss_fn, device, num_labels, epoch, eval_type="Validation"
        )
        print(f'  Avg Validation Epoch {epoch+1} | Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | F1: {val_f1:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f}')
        print(f"  Validation Classification Report (Epoch {epoch+1}):\n{val_report}")

        # --- Store History ---
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1) # Store F1 as it's often a key metric

        # --- Log Epoch Metrics to wandb (Optional) ---
        # if args.use_wandb:
        #     try:
        #         wandb.log({
        #             "epoch": epoch + 1,
        #             "train/epoch_loss": train_loss,
        #             "train/epoch_accuracy": train_acc,
        #             "val/epoch_loss": val_loss,
        #             "val/epoch_accuracy": val_acc,
        #             "val/epoch_f1": val_f1,
        #             "val/epoch_precision": val_precision,
        #             "val/epoch_recall": val_recall,
        #             # Optional: Log confusion matrix as image or table
        #             # "val/confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=all_labels_from_eval, preds=all_preds_from_eval, class_names=cm_labels) # Need to get labels/preds from eval_model
        #         })
        #     except Exception as e:
        #         print(f"Warning: Failed to log epoch metrics to wandb - {e}")


        # --- Early Stopping Check ---
        if args.patience > 0:
            # Check if validation loss has improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss # Update best validation loss
                epochs_no_improve = 0 # Reset patience counter
                # Save the model state dictionary as the best model found so far
                torch.save(model.state_dict(), best_model_path)
                print(f"  Validation loss improved to {best_val_loss:.4f}. Saved best model state to {best_model_path}")
            else:
                epochs_no_improve += 1 # Increment patience counter
                print(f"  Validation loss did not improve for {epochs_no_improve} epoch(s). Patience: {args.patience}.")

            # If patience limit is reached, stop training
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs due to no improvement in validation loss for {args.patience} consecutive epochs.")
                break # Exit the training loop

    print("\n--- Training Finished ---")

    # --- Load Best Model if Early Stopping Occurred ---
    # If early stopping was enabled and a best model was saved, load its weights for final evaluation.
    if args.patience > 0 and os.path.exists(best_model_path):
        print(f"Loading best model state from {best_model_path} for final evaluation.")
        try:
            model.load_state_dict(torch.load(best_model_path))
            model = model.to(device) # Ensure model is on the correct device after loading
        except Exception as e:
            print(f"Error loading best model state: {e}. Proceeding with the model state from the last epoch.")

    # --- Plotting and Final Saving ---
    print("\n--- Generating Plots and Saving Artifacts ---")
    # Plot training history (loss and accuracy curves)
    plot_history(history, args.output_dir, args.save_plots)

    # Save the final model state (either the last epoch's or the best one loaded above)
    final_model_path = os.path.join(args.output_dir, f'{args.model_name.replace("/", "_")}_final_epoch.bin')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model state saved to {final_model_path}")
    # Save the tokenizer configuration (important for reloading the model later)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Tokenizer configuration saved to {args.output_dir}")

    # --- Final Evaluation on Validation Set ---
    print("\n--- Final Evaluation (on Validation Set) ---")
    # Run evaluation one last time using the final (potentially best) model state
    final_val_loss, final_val_acc, final_val_f1, final_val_precision, final_val_recall, final_val_cm, final_val_report = eval_model(
        model, val_data_loader, loss_fn, device, num_labels, epoch=args.epochs, eval_type="Final Validation" # Use args.epochs for logging consistency
    )
    print(f"Final Validation Metrics:")
    print(f"  Loss:      {final_val_loss:.4f}")
    print(f"  Accuracy:  {final_val_acc:.4f}")
    print(f"  F1 Score:  {final_val_f1:.4f}")
    print(f"  Precision: {final_val_precision:.4f}")
    print(f"  Recall:    {final_val_recall:.4f}")
    print("Final Validation Classification Report:")
    print(final_val_report)

    # Define class labels for the confusion matrix based on the task
    cm_labels = ['NOT', 'OFF'] if args.task == 'a' else (['UNT', 'TIN'] if args.task == 'b' else ['IND', 'GRP', 'OTH'])
    # Plot the final validation confusion matrix
    plot_confusion_matrix(final_val_cm, cm_labels, args.output_dir, "validation_confusion_matrix.png", args.save_plots)

    # --- Prepare Results Dictionary ---
    # Store key configuration and final validation results in a dictionary
    results = {
        'args': vars(args), # Store all command line arguments
        'task_details': {
            'task': args.task,
            'num_labels': num_labels,
            'cm_labels': cm_labels
        },
        'final_validation_metrics': {
            'loss': final_val_loss,
            'accuracy': final_val_acc,
            'f1_score': final_val_f1,
            'precision': final_val_precision,
            'recall': final_val_recall,
            'classification_report_str': final_val_report,
            # Also save the classification report as a dictionary for easier parsing
            'classification_report_dict': classification_report(
                 # Need labels and preds from eval_model again, or modify eval_model to return them
                 # For simplicity, recalculating here based on val_df (less ideal)
                 # A better approach: modify eval_model to return all_labels, all_preds
                 val_df['label'].tolist(), # Placeholder: Need actual labels from eval run
                 [], # Placeholder: Need actual predictions from eval run
                 target_names=cm_labels, output_dict=True, zero_division=0
            ) if not val_df.empty else None # Ensure val_df is not empty
        }
    }


    # --- Final Evaluation on Test Set (if provided) ---
    if test_data_loader:
        print("\n--- Final Evaluation (on Test Set) ---")
        # Evaluate the final model on the held-out test set
        test_loss, test_acc, test_f1, test_precision, test_recall, test_cm, test_report = eval_model(
            model, test_data_loader, loss_fn, device, num_labels, epoch=args.epochs, eval_type="Test"
        )
        print(f"Final Test Metrics:")
        print(f"  Loss:      {test_loss:.4f}")
        print(f"  Accuracy:  {test_acc:.4f}")
        print(f"  F1 Score:  {test_f1:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall:    {test_recall:.4f}")
        print("Final Test Classification Report:")
        print(test_report)
        # Plot the test confusion matrix
        plot_confusion_matrix(test_cm, cm_labels, args.output_dir, "test_confusion_matrix.png", args.save_plots)

        # Add test results to the results dictionary
        results['final_test_metrics'] = {
            'loss': test_loss,
            'accuracy': test_acc,
            'f1_score': test_f1,
            'precision': test_precision,
            'recall': test_recall,
            'classification_report_str': test_report,
            # 'classification_report_dict': classification_report(...) # Add dict version if needed
        }

        # --- Log final test metrics to wandb (Optional) ---
        # if args.use_wandb:
        #     try:
        #         wandb.log({
        #             "test/final_loss": test_loss,
        #             "test/final_accuracy": test_acc,
        #             "test/final_f1": test_f1,
        #             "test/final_precision": test_precision,
        #             "test/final_recall": test_recall,
        #             # Optional: Log test confusion matrix
        #             # "test/confusion_matrix": wandb.plot.confusion_matrix(...)
        #         })
        #         # Optional: Log classification report as text or artifact
        #         # wandb.log({"test/classification_report": wandb.Html(f"<pre>{test_report}</pre>")})
        #     except Exception as e:
        #         print(f"Warning: Failed to log test metrics to wandb - {e}")


    # --- Save Final Results ---
    # Save the consolidated results dictionary to a JSON file
    results_path = os.path.join(args.output_dir, "final_run_results.json")
    try:
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2) # Use indent for readability
        print(f"Final run results saved to {results_path}")
    except Exception as e:
        print(f"Error saving final results JSON: {e}")

    # --- Explainability Example ---
    # Run attention visualization on a few examples from the validation set
    print("\n--- Explainability Example (Attention Visualization) ---")
    try:
        # Sample a few texts from the validation DataFrame
        num_examples = min(3, len(val_df)) # Visualize up to 3 examples
        example_texts = val_df.sample(num_examples, random_state=args.seed)['tweet'].tolist()
        print(f"Generating attention visualizations for {num_examples} validation examples...")
        # Generate visualization for each sampled text
        for i, text in enumerate(example_texts):
             print(f"  Visualizing attention for example {i+1}: '{text[:100]}...'") # Print truncated text
             visualize_attention(
                 model, tokenizer, text, device, args.max_len,
                 args.output_dir, f"validation_example_{i+1}", args.save_plots
            )
    except Exception as e:
        # Catch potential errors during sampling or visualization
        print(f"\nWarning: Could not run attention visualization: {type(e).__name__} - {e}")

    # --- Finish wandb Run (Optional) ---
    # if args.use_wandb:
    #     print("Finishing Weights & Biases run...")
    #     wandb.finish()

    print("\n--- Script Finished ---")