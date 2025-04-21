# -*- coding: utf-8 -*-
"""
ProjectNLP Hate Speech Detection Group 5 - VS Code Version

This script fine-tunes a BERT model for hate speech detection (e.g., OLID dataset)
using PyTorch and Hugging Face Transformers. It supports command-line arguments
for configuration, includes adversarial training (FGM), evaluation, explainability
(attention viz), and saves outputs for easier collaboration.

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

"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import argparse
import json
from typing import List, Dict, Tuple, Optional, Any # Added type hinting

# --- Argument Parsing ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune BERT for Hate Speech Detection.")

    # File Paths & Directories
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to the training data file (TSV format expected).")
    parser.add_argument("--test_file", type=str, default=None,
                        help="Optional: Path to the labeled test data file (TSV format). If not provided, uses validation split.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save models, results, plots, and logs.")

    # Model & Task Configuration
    parser.add_argument("--model_name", type=str, default='bert-base-uncased',
                        help="Name of the pre-trained BERT model from Hugging Face.")
    parser.add_argument("--task", type=str, default='a', choices=['a', 'b', 'c'],
                        help="OLID subtask ('a': Offensive, 'b': Targeted, 'c': Target Type). Adjust loading logic if using other datasets.")
    parser.add_argument("--max_len", type=int, default=128,
                        help="Maximum sequence length for BERT.")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=4,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and evaluation.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate for AdamW optimizer.")
    parser.add_argument("--adversarial_eps", type=float, default=0.0,
                        help="Epsilon for FGM adversarial training (set > 0 to enable).")
    parser.add_argument("--patience", type=int, default=3,
                        help="Patience for early stopping (epochs without validation loss improvement). Set 0 to disable.")

    # Execution & Environment
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--no_gpu", action='store_true',
                        help="Force CPU usage even if GPU is available.")
    parser.add_argument("--save_plots", action='store_true',
                        help="Save plots to files instead of displaying them interactively.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for DataLoader.")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    return args

# --- Seed and Device Setup ---
def setup_environment(seed: int, no_gpu: bool) -> torch.device:
    """Sets random seeds and determines the compute device."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Optional: Set deterministic algorithms for reproducibility, might impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # Device Selection
    if not no_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
        # Note for collaborators with Intel ARC GPUs:
        # May need Intel Extension for PyTorch (IPEX) installed (`pip install intel_extension_for_pytorch`)
        # and potentially code changes (e.g., `import intel_extension_for_pytorch as ipex`).
    elif not no_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Fallback for Apple Silicon
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

# --- Data Loading ---
def load_data(filepath: str, task: str = 'a') -> Optional[pd.DataFrame]:
    """Loads and preprocesses data from a TSV file based on the specified OLID task."""
    print(f"Loading data from: {filepath}")
    try:
        df = pd.read_csv(filepath, sep='\t', header=0, on_bad_lines='warn') # Warn about bad lines

        # Ensure 'tweet' column exists and is string
        if 'tweet' not in df.columns:
             raise ValueError(f"'tweet' column not found in {filepath}.")
        df['tweet'] = df['tweet'].astype(str).fillna('')

        # Select columns and process labels based on task
        required_cols = {'a': ['subtask_a'], 'b': ['subtask_a', 'subtask_b'], 'c': ['subtask_b', 'subtask_c']}
        if task not in required_cols:
            raise ValueError(f"Invalid task '{task}'. Choose from {list(required_cols.keys())}.")

        # Check for required columns
        missing_cols = [col for col in required_cols[task] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required column(s) for Task {task.upper()} in {filepath}: {', '.join(missing_cols)}")

        # Process based on task
        if task == 'a':
            df = df[['id', 'tweet', 'subtask_a']].copy()
            df.rename(columns={'subtask_a': 'label'}, inplace=True)
            df['label'] = df['label'].apply(lambda x: 1 if str(x).upper() == 'OFF' else 0)
            num_labels = 2
        elif task == 'b':
            df = df[df['subtask_a'] == 'OFF'][['id', 'tweet', 'subtask_b']].copy()
            df.rename(columns={'subtask_b': 'label'}, inplace=True)
            df['label'] = df['label'].apply(lambda x: 1 if str(x).upper() == 'TIN' else 0)
            num_labels = 2
        elif task == 'c':
            df = df[df['subtask_b'] == 'TIN'][['id', 'tweet', 'subtask_c']].copy()
            df.rename(columns={'subtask_c': 'label'}, inplace=True)
            label_map = {'IND': 0, 'GRP': 1, 'OTH': 2}
            # Handle potential missing labels or unexpected values
            df['label'] = df['label'].map(label_map).fillna(-1) # Use -1 for unknown/missing
            df = df[df['label'] != -1] # Remove rows with invalid labels for Task C
            df['label'] = df['label'].astype(int)
            num_labels = 3 # Multi-class for Task C

        print(f"Loaded {len(df)} valid entries for Task {task.upper()}.")
        if not df.empty:
            print("Label distribution:")
            print(df['label'].value_counts(normalize=True))
        else:
            print("Warning: No valid data loaded after filtering for the task.")
            return None, 0 # Return None and 0 labels if empty

        return df, num_labels

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, 0
    except Exception as e:
        print(f"Error loading or processing file {filepath}: {e}")
        return None, 0

# --- PyTorch Dataset and DataLoader ---
class TextClassificationDataset(Dataset):
    """Custom PyTorch Dataset for text classification."""
    def __init__(self, texts: np.ndarray, labels: np.ndarray, tokenizer: BertTokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            # Adjust dtype based on loss function (Long for CrossEntropy, Float for BCEWithLogitsLoss)
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(df: pd.DataFrame, tokenizer: BertTokenizer, max_len: int, batch_size: int, num_workers: int, shuffle: bool = False) -> DataLoader:
    """Creates a DataLoader for the given DataFrame."""
    ds = TextClassificationDataset(
        texts=df.tweet.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )

# --- Model Loading ---
def load_model(model_name: str, num_labels: int) -> BertForSequenceClassification:
    """Loads the BERT model for sequence classification."""
    print(f"\nLoading model: {model_name}")
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        output_attentions=True, # Needed for visualization
        output_hidden_states=False,
    )
    return model

# --- Training Epoch ---
def train_epoch(model: BertForSequenceClassification, data_loader: DataLoader, loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer, device: torch.device, scheduler: torch.optim.lr_scheduler._LRScheduler,
                adversarial_eps: float) -> Tuple[float, float]:
    """Trains the model for one epoch with optional FGM."""
    model = model.train()
    total_loss, total_accuracy = 0, 0
    total_samples = 0
    start_time = time.time()
    print(f"  Starting training epoch...")

    for i, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # --- Standard Forward Pass ---
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # Accumulate loss and accuracy
        total_loss += loss.item() * input_ids.size(0) # Weighted loss
        _, preds = torch.max(logits, dim=1)
        total_accuracy += torch.sum(preds == labels).item()
        total_samples += labels.size(0)

        # --- Adversarial Step (FGM) ---
        if adversarial_eps > 0:
            loss.backward(retain_graph=True) # Calculate gradients for original loss

            embedding_layer = model.bert.embeddings.word_embeddings
            if embedding_layer.weight.grad is not None:
                try:
                    input_embeddings = embedding_layer(input_ids)
                    input_embeddings.requires_grad_(True)

                    # Pass embeddings through model to get gradients w.r.t. embeddings
                    adv_outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask, labels=labels)
                    adv_loss = adv_outputs.loss
                    adv_loss.backward()

                    if input_embeddings.grad is not None:
                        perturbation = adversarial_eps * input_embeddings.grad.sign()
                        perturbed_embeddings = input_embeddings.detach() + perturbation

                        optimizer.zero_grad() # Zero grads before adversarial forward pass

                        # Forward pass with perturbed embeddings
                        final_outputs = model(inputs_embeds=perturbed_embeddings, attention_mask=attention_mask, labels=labels)
                        final_loss = final_outputs.loss
                        final_loss.backward() # Backward pass on adversarial loss
                        # print("FGM step successful.") # DEBUG

                    else: # Fallback if embedding grads are None
                        print("Warning: Could not get embedding gradients for FGM perturbation. Performing standard backprop on original loss.")
                        optimizer.zero_grad()
                        loss.backward()
                except (TypeError, RuntimeError) as e: # Fallback on error
                    print(f"Warning: FGM step failed ({e}). Performing standard backprop on original loss.")
                    optimizer.zero_grad()
                    loss.backward()
            else: # Fallback if embedding layer weights have no grads
                 print("Warning: Embedding layer weights have no gradients. Performing standard backprop on original loss.")
                 optimizer.zero_grad()
                 loss.backward()
        else:
            # --- Standard Backward Pass ---
            loss.backward()

        # Gradient Clipping, Optimizer Step, Scheduler Step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Print progress
        if (i + 1) % 50 == 0:
            elapsed_time = time.time() - start_time
            current_loss = total_loss / total_samples
            current_acc = total_accuracy / total_samples
            print(f'    Batch {i+1}/{len(data_loader)} | Train Loss: {current_loss:.4f} | Train Acc: {current_acc:.4f} | Time: {elapsed_time:.2f}s')

    # End of epoch calculation
    epoch_time = time.time() - start_time
    avg_epoch_loss = total_loss / total_samples
    epoch_accuracy = total_accuracy / total_samples
    print(f"  Epoch finished | Time: {epoch_time:.2f}s")
    return avg_epoch_loss, epoch_accuracy

# --- Evaluation ---
def eval_model(model: BertForSequenceClassification, data_loader: DataLoader, loss_fn: torch.nn.Module,
               device: torch.device, num_labels: int) -> Tuple[float, float, float, float, float, np.ndarray, str]:
    """Evaluates the model and returns metrics, confusion matrix, and classification report."""
    model = model.eval()
    total_loss, total_accuracy = 0, 0
    total_samples = 0
    all_labels, all_preds = [], []
    print(f"  Starting evaluation...")
    start_time = time.time()

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)

            total_loss += loss.item() * input_ids.size(0)
            total_accuracy += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = total_accuracy / total_samples

    # Determine average strategy based on number of labels
    average_strategy = 'binary' if num_labels == 2 else 'weighted' # Use weighted for multi-class (Task C)

    f1 = f1_score(all_labels, all_preds, average=average_strategy, zero_division=0)
    precision = precision_score(all_labels, all_preds, average=average_strategy, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=average_strategy, zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, zero_division=0, output_dict=False) # Get string report

    eval_time = time.time() - start_time
    print(f"  Evaluation finished | Time: {eval_time:.2f}s")
    return avg_loss, accuracy, f1, precision, recall, conf_matrix, class_report

# --- Plotting and Saving ---
def plot_history(history: Dict[str, List[float]], output_dir: str, save_plots: bool):
    """Plots training/validation loss and accuracy, optionally saves to file."""
    epochs_ran = len(history['train_loss'])
    epoch_nums = range(1, epochs_ran + 1)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epoch_nums, history['train_loss'], 'bo-', label='Train Loss')
    plt.plot(epoch_nums, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.xticks(epoch_nums)

    plt.subplot(1, 2, 2)
    plt.plot(epoch_nums, history['train_acc'], 'bo-', label='Train Accuracy')
    plt.plot(epoch_nums, history['val_acc'], 'ro-', label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.xticks(epoch_nums)

    plt.tight_layout()
    if save_plots:
        filepath = os.path.join(output_dir, "training_history.png")
        plt.savefig(filepath)
        print(f"Training history plot saved to {filepath}")
        plt.close() # Close plot when saving to file
    else:
        plt.show()

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], output_dir: str, filename: str, save_plots: bool):
    """Plots and optionally saves the confusion matrix."""
    plt.figure(figsize=(6, 5))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix ({filename.split(".")[0]})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    if save_plots:
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        print(f"Confusion matrix plot saved to {filepath}")
        plt.close()
    else:
        plt.show()

# --- Explainability ---
def visualize_attention(model: BertForSequenceClassification, tokenizer: BertTokenizer, text: str, device: torch.device, output_dir: str, filename_prefix: str, save_plots: bool):
    """Visualizes average attention from [CLS] token, optionally saves plot."""
    model = model.eval()
    inputs = tokenizer.encode_plus(
        text, add_special_tokens=True, max_length=args.max_len, padding='max_length',
        truncation=True, return_tensors='pt', return_attention_mask=True
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)

    attentions = outputs.attentions
    if attentions is None:
        print("Attention weights not available.")
        return

    num_layers = len(attentions)
    seq_len = attentions[0].shape[2]
    cls_attentions = torch.zeros(seq_len, device=device)
    for layer in range(num_layers):
        avg_head_attention = attentions[layer][0].mean(dim=0)
        cls_attentions += avg_head_attention[0, :]

    cls_attentions /= num_layers
    cls_attentions = cls_attentions.cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    valid_indices = [i for i, mask in enumerate(attention_mask[0].cpu().numpy()) if mask == 1]
    tokens = [tokens[i] for i in valid_indices]
    cls_attentions = cls_attentions[valid_indices]
    if np.max(cls_attentions) > np.min(cls_attentions):
         cls_attentions = (cls_attentions - np.min(cls_attentions)) / (np.max(cls_attentions) - np.min(cls_attentions))
    else:
         cls_attentions = np.zeros_like(cls_attentions)

    plt.figure(figsize=(max(10, len(tokens) * 0.6), 2))
    sns.heatmap([cls_attentions], xticklabels=tokens, annot=False, cmap="viridis", cbar=True, linewidths=.5, linecolor='lightgray')
    plt.title(f'Avg Attention from [CLS] for: "{text[:60]}..."', fontsize=10)
    plt.xlabel("Tokens", fontsize=9)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks([])
    plt.tight_layout()

    if save_plots:
        filepath = os.path.join(output_dir, f"{filename_prefix}_attention_viz.png")
        plt.savefig(filepath)
        print(f"Attention visualization saved to {filepath}")
        plt.close()
    else:
        plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()
    print("Script arguments:")
    print(json.dumps(vars(args), indent=2)) # Log arguments

    device = setup_environment(args.seed, args.no_gpu)

    # --- Load Data ---
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    train_df, num_labels = load_data(args.train_file, args.task)

    if train_df is None or train_df.empty:
        print("Exiting due to data loading issues.")
        exit()

    # Split train/validation or load test set
    val_df = None
    test_df = None
    test_data_loader = None

    if args.test_file:
        test_df, _ = load_data(args.test_file, args.task)
        if test_df is None or test_df.empty:
            print("Warning: Could not load test file, proceeding without test evaluation.")
        else:
            # Use the full training data for training if a separate test set is provided
            val_df = train_df.sample(frac=0.1, random_state=args.seed) # Still keep a small val set from train
            train_df = train_df.drop(val_df.index)
            print(f"Using provided test set ({len(test_df)} samples). Training on {len(train_df)}, validating on {len(val_df)}.")
            test_data_loader = create_data_loader(test_df, tokenizer, args.max_len, args.batch_size, args.num_workers)
    else:
        # Split training data into train and validation
        train_df, val_df = train_test_split(
            train_df,
            test_size=0.15, # Use 15% for validation if no test set
            random_state=args.seed,
            stratify=train_df['label']
        )
        print(f"No test file provided. Splitting training data: {len(train_df)} train, {len(val_df)} validation samples.")

    train_data_loader = create_data_loader(train_df, tokenizer, args.max_len, args.batch_size, args.num_workers, shuffle=True)
    val_data_loader = create_data_loader(val_df, tokenizer, args.max_len, args.batch_size, args.num_workers)


    # --- Load Model, Optimizer, Scheduler, Loss ---
    model = load_model(args.model_name, num_labels)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    total_steps = len(train_data_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # --- Training Loop ---
    print("\nStarting Training...")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = os.path.join(args.output_dir, f'{args.model_name.replace("/", "_")}_best.bin')

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_loss, train_acc = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, args.adversarial_eps)
        print(f'  Training   | Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}')

        val_loss, val_acc, val_f1, val_precision, val_recall, val_cm, val_report = eval_model(model, val_data_loader, loss_fn, device, num_labels)
        print(f'  Validation | Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | F1 ({val_f1:.4f}) | Precision ({val_precision:.4f}) | Recall ({recall:.4f})')
        print("  Validation Classification Report:")
        print(val_report)


        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1) # Store F1 for potential best model selection

        # Early Stopping Check
        if args.patience > 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"  Best validation loss improved. Saved model state to {best_model_path}")
            else:
                epochs_no_improve += 1
                print(f"  Validation loss did not improve for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                break

    print("\nTraining finished.")

    # Load best model if early stopping was enabled and triggered
    if args.patience > 0 and os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} for final evaluation.")
        model.load_state_dict(torch.load(best_model_path))
        model = model.to(device)

    # --- Plotting and Final Saving ---
    plot_history(history, args.output_dir, args.save_plots)

    # Save final model state (could be the last epoch or the best one)
    final_model_path = os.path.join(args.output_dir, f'{args.model_name.replace("/", "_")}_final.bin')
    torch.save(model.state_dict(), final_model_path)
    tokenizer.save_pretrained(args.output_dir) # Save tokenizer config
    print(f"Final model state saved to {final_model_path}")
    print(f"Tokenizer config saved to {args.output_dir}")


    # --- Final Evaluation ---
    print("\n--- Final Evaluation (on Validation Set) ---")
    final_val_loss, final_val_acc, final_val_f1, final_val_precision, final_val_recall, final_val_cm, final_val_report = eval_model(
        model, val_data_loader, loss_fn, device, num_labels
    )
    print(f"Val Loss: {final_val_loss:.4f} | Val Acc: {final_val_acc:.4f} | Val F1: {final_val_f1:.4f}")
    print("Validation Classification Report:")
    print(final_val_report)
    # Determine labels for confusion matrix based on task
    cm_labels = ['NOT', 'OFF'] if args.task == 'a' else (['UNT', 'TIN'] if args.task == 'b' else ['IND', 'GRP', 'OTH'])
    plot_confusion_matrix(final_val_cm, cm_labels, args.output_dir, "validation_confusion_matrix.png", args.save_plots)

    results = {
        'model_name': args.model_name,
        'task': args.task,
        'final_validation_loss': final_val_loss,
        'final_validation_accuracy': final_val_acc,
        'final_validation_f1': final_val_f1,
        'final_validation_precision': final_val_precision,
        'final_validation_recall': final_val_recall,
        'classification_report_val': classification_report( # Save dict version of report
             *np.unique(val_df['label'].values, return_inverse=True)[::-1], # Get labels and preds from eval_model output if needed
             target_names=cm_labels,
             output_dict=True,
             zero_division=0
         ) # Re-calculate report as dict or modify eval_model to return it
    }

    # Evaluate on Test Set if provided
    if test_data_loader:
        print("\n--- Final Evaluation (on Test Set) ---")
        test_loss, test_acc, test_f1, test_precision, test_recall, test_cm, test_report = eval_model(
            model, test_data_loader, loss_fn, device, num_labels
        )
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
        print("Test Classification Report:")
        print(test_report)
        plot_confusion_matrix(test_cm, cm_labels, args.output_dir, "test_confusion_matrix.png", args.save_plots)
        results['final_test_loss'] = test_loss
        results['final_test_accuracy'] = test_acc
        results['final_test_f1'] = test_f1
        results['final_test_precision'] = test_precision
        results['final_test_recall'] = test_recall
        # results['classification_report_test'] = classification_report(...) # Add dict version

    # Save final results
    results_path = os.path.join(args.output_dir, "final_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Final results saved to {results_path}")

    # --- Explainability Example ---
    # Optionally run attention visualization on a few validation examples
    print("\n--- Explainability Example (Attention Visualization) ---")
    try:
        example_texts = val_df.sample(min(3, len(val_df)), random_state=args.seed)['tweet'].tolist()
        for i, text in enumerate(example_texts):
             print(f"\nVisualizing attention for example {i+1}: '{text}'")
             visualize_attention(model, tokenizer, text, device, args.output_dir, f"example_{i+1}", args.save_plots)
    except Exception as e:
        print(f"\nCould not run attention visualization: {e}")


    print("\n--- Script Finished ---")

