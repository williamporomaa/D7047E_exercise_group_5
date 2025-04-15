# Imports
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# Using TfidfVectorizer instead of CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, classification_report, ConfusionMatrixDisplay)

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Using NLTK for more advanced preprocessing
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.corpus import stopwords
from nltk import word_tokenize

import math # For Transformer input dimension adjustment
import re # For preprocessing

# Data loading and preprocessing Function (Using the advanced version)
def preprocess_pandas(data):
    """
    Applies more advanced preprocessing directly to the DataFrame.
    - Lowercase
    - Remove emails, IPs, special characters, numbers
    - Tokenize and remove stopwords
    Returns the modified DataFrame.
    """
    # Ensure 'Sentence' column exists and is string type
    if 'Sentence' not in data.columns:
        raise ValueError("DataFrame must contain a 'Sentence' column.")
    data['Sentence'] = data['Sentence'].astype(str)

    print("Preprocessing: Lowercasing...")
    data['Sentence'] = data['Sentence'].str.lower()
    print("Preprocessing: Removing emails...")
    data['Sentence'] = data['Sentence'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)
    print("Preprocessing: Removing IP addresses...")
    data['Sentence'] = data['Sentence'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\\.|$)){4}', '', regex=True)
    print("Preprocessing: Removing special characters...")
    # Keep alphanumeric and spaces
    data['Sentence'] = data['Sentence'].str.replace(r'[^\w\s]', '', regex=True)
    print("Preprocessing: Removing numbers...")
    data['Sentence'] = data['Sentence'].replace(r'\d+', '', regex=True)  # Added '+' to remove multi-digit numbers

    print("Preprocessing: Tokenizing and removing stopwords...")
    stop_words = set(stopwords.words('english'))
    processed_sentences = []
    for sentence in data['Sentence']:
        word_tokens = word_tokenize(sentence)
        filtered_sent = [w for w in word_tokens if not w in stop_words and len(w) > 1] # Keep words longer than 1 char
        processed_sentences.append(" ".join(filtered_sent))

    data['Sentence'] = processed_sentences
    print("Preprocessing finished.")
    return data

# --- Model Definitions ---

# Task 1.1 ANN MODEL DEFINITION
class SimpleANN(nn.Module):
    """
    A simple feed-forward ANN for binary classification (sentiment analysis).
    Takes TF-IDF features as input.
    """
    def __init__(self, input_dim, hidden_dim=128, output_dim=1):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Ensure input is flattened correctly if needed (should be [batch_size, features])
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# TASK 1.2: TRANSFORMER MODEL DEFINITION
class TransformerClassifier(nn.Module):
    """
    A basic Transformer-based model for binary classification using TF-IDF features.
    Treats the entire TF-IDF vector as a single token input.
    """
    def __init__(self, input_dim, nhead=4, num_layers=2, hidden_dim=128, dropout=0.1): # Added dropout
        super(TransformerClassifier, self).__init__()

        # Adjust input_dim to be divisible by nhead for d_model
        if input_dim % nhead != 0:
            # Ensure d_model is at least nhead, typically larger
            min_d_model = max(nhead, 64) # Example minimum reasonable dimension
            d_model = math.ceil(max(input_dim, min_d_model) / nhead) * nhead # Round up adjusted dim
            print(f"⚠️ Transformer Warning: Adjusting effective input_dim from {input_dim} to {d_model} for compatibility with nhead={nhead}.")
        else:
            d_model = input_dim
            if d_model < nhead: # Ensure d_model is never smaller than nhead
                 d_model = nhead
                 print(f"⚠️ Transformer Warning: Input dim {input_dim} < nhead {nhead}. Adjusting d_model to {d_model}.")


        self.d_model = d_model # Expected embedding dimension after padding/truncation

        # Optional: Linear layer to project input_dim to d_model if they differ significantly
        # Or handle padding/truncation directly in forward pass
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        print(f"Transformer using d_model={self.d_model}, nhead={nhead}")


        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True) # Use batch_first=True
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim // 2), # Added intermediate layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )


    def forward(self, src):
        # src shape: (batch_size, original_features)
        # Ensure input is 2D: [batch_size, features]
        src = src.view(src.size(0), -1)

        # Project or identify map to d_model
        # src = self.input_proj(src) # Project to d_model if needed - let's handle via padding instead for simplicity with TF-IDF

        # --- Padding/Truncation to match d_model ---
        original_dim = src.size(-1)
        if original_dim < self.d_model:
            pad_size = self.d_model - original_dim
            # Pad on the feature dimension (last dimension)
            pad_tensor = torch.zeros(src.size(0), pad_size, device=src.device, dtype=src.dtype)
            src = torch.cat((src, pad_tensor), dim=-1) # Shape: (batch_size, d_model)
        elif original_dim > self.d_model:
            src = src[:, :self.d_model] # Truncate, shape: (batch_size, d_model)
        # Now src shape is (batch_size, d_model)

        # TransformerEncoderLayer expects input shape (N, S, E) or (S, N, E) if batch_first=False
        # N = batch_size, S = sequence_length, E = embedding_dimension (d_model)
        # We treat the whole TF-IDF vector as a single token in the sequence (S=1)
        src = src.unsqueeze(1) # Reshape to (batch_size, 1, d_model) - this matches batch_first=True

        encoded = self.transformer_encoder(src) # Input shape (N, S, E), Output shape (N, S, E)

        # Use the output corresponding to the single token (sequence length is 1)
        encoded_output = encoded.squeeze(1) # Shape: (batch_size, d_model)

        probs = self.classifier(encoded_output) # Shape: (batch_size, 1)
        return probs

# --- Dataset Definition (Takes pre-vectorized tensors) ---
class ReviewsDataset(Dataset):
    """
    Torch Dataset that takes pre-vectorized tensors (e.g., TF-IDF) and labels.
    """
    def __init__(self, features, labels):
        if features.shape[0] != labels.shape[0]:
             raise ValueError("Features and labels must have the same number of samples.")
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Features should already be tensors
        feature_vector = self.features[idx]
        label = self.labels[idx]
        # Return feature vector (potentially needs flattening/reshaping in model) and label
        # Ensure label is FloatTensor for BCELoss
        return feature_vector, label.float().unsqueeze(0) # Return label as [1] tensor

# --- Training Functions (Using DataLoaders) ---

def train_model(model, criterion, optimizer, train_loader, val_loader, model_name="Model", epochs=5):
    """
    Generic training loop for models using DataLoaders.
    Tracks and returns training and validation metrics at each epoch.
    """
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print(f"\n--- Training {model_name} ---")

    for epoch in range(epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (features, labels) in enumerate(train_loader):
            # Move data to appropriate device if using GPU
            # features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features) # Forward pass
            loss = criterion(outputs, labels)
            loss.backward() # Backward pass
            optimizer.step() # Update weights

            running_loss += loss.item() * features.size(0) # Accumulate loss scaled by batch size

            # Calculate training accuracy for the batch
            predicted = (outputs > 0.5).float()
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

            # Optional: Print batch progress
            # if (i + 1) % 5 == 0:
            #     print(f'Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')


        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation phase
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad(): # Disable gradient calculation for validation
            for features, labels in val_loader:
                # Move data to appropriate device if using GPU
                # features, labels = features.to(device), labels.to(device)

                val_outputs = model(features)
                val_loss += criterion(val_outputs, labels).item() * features.size(0) # Accumulate validation loss

                # Calculate validation accuracy
                val_predicted = (val_outputs > 0.5).float()
                correct_val += (val_predicted == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        # Store metrics
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accuracies.append(epoch_train_acc)
        val_accuracies.append(epoch_val_acc)

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, "
              f"Train Acc: {epoch_train_acc:.4f}, Val Acc: {epoch_val_acc:.4f}")

    print(f"--- Finished Training {model_name} ---")
    return train_losses, val_losses, train_accuracies, val_accuracies


# --- Plotting Function ---
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epochs, model_name):
    """
    Plots the training and validation loss and accuracy curves.
    """
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title(f'{model_name} - Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.suptitle(f'{model_name} Training Metrics')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()


# --- Evaluation Functions (Using DataLoaders) ---

def evaluate_model_on_test(model, data_loader, criterion, model_name="Model"):
    """
    Evaluates a trained model using a DataLoader (typically the test_loader).
    Returns accuracy and prints detailed classification report and confusion matrix.
    """
    print(f"\n--- Evaluating {model_name} on Test Set ---")
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0
    total_test = 0

    with torch.no_grad():
        for features, labels in data_loader:
            # features, labels = features.to(device), labels.to(device) # If using GPU
            outputs = model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * features.size(0)

            preds = (outputs > 0.5).float() # Get binary predictions
            all_preds.extend(preds.cpu().numpy().flatten()) # Use flatten()
            all_labels.extend(labels.cpu().numpy().flatten()) # Use flatten()
            total_test += labels.size(0)

    avg_test_loss = test_loss / total_test
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"{model_name} Test Loss: {avg_test_loss:.4f}")
    print(f"{model_name} Test Accuracy: {accuracy:.4f}")
    print(classification_report(all_labels, all_preds, target_names=['Negative (0)', 'Positive (1)']))

    # Display Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative (0)', 'Positive (1)'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

    # Manual Calculation (matches classification_report)
    # confusion_matrix_and_values(all_labels, all_preds, model_name) # Optional: Keep if needed

    return accuracy

# Optional: Keep the manual calculation function if required by assignment
def confusion_matrix_and_values(labels, predictions, modelname):
    print("\nDetailed Metrics Calculation for ", modelname, ":")
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel() # More direct way to get values

    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0 # Also called Sensitivity
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1_score:.4f}\n")


# --- Chatbot Response Functions ---

def preprocess_chatbot_input(text):
    """Applies basic cleaning suitable for chatbot input before vectorization."""
    text = text.lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Optional: remove stopwords (might remove important negations like "not")
    # stop_words = set(stopwords.words('english'))
    # word_tokens = word_tokenize(text)
    # filtered_sent = [w for w in word_tokens if not w in stop_words]
    # text = " ".join(filtered_sent)
    return text.strip()

def chatbot_response(model, user_input, vectorizer, model_name="Model"):
    """
    Takes user input, preprocesses, vectorizes it using the TF-IDF vectorizer,
    feeds it into the specified model (ANN or Transformer), and returns a sentiment response.
    Handles potential padding for Transformer.
    """
    # 1. Preprocess input
    processed_input = preprocess_chatbot_input(user_input)
    if not processed_input:
        return "Input seems empty after preprocessing."

    # 2. Vectorize using the *fitted* TfidfVectorizer
    user_vec = vectorizer.transform([processed_input]).todense()
    user_tensor = torch.FloatTensor(user_vec) # Shape: (1, num_features)

    # 3. Feed to the model (handle Transformer padding)
    model.eval()
    with torch.no_grad():
        # Check if it's the Transformer model based on class or attributes
        if isinstance(model, TransformerClassifier) or hasattr(model, "d_model"):
            # Apply same padding/truncation as in Transformer forward pass
            original_dim = user_tensor.size(-1)
            d_model = model.d_model # Get the required dimension
            if original_dim < d_model:
                pad_size = d_model - original_dim
                pad_tensor = torch.zeros(user_tensor.size(0), pad_size, device=user_tensor.device, dtype=user_tensor.dtype)
                user_tensor = torch.cat((user_tensor, pad_tensor), dim=-1)
            elif original_dim > d_model:
                 user_tensor = user_tensor[:, :d_model]
            # Transformer expects shape (N, S, E) or (S, N, E). Reshape based on batch_first.
            # Since our Transformer uses batch_first=True, needs (N, 1, E)
            # user_tensor = user_tensor.unsqueeze(1) # Model's forward handles this reshape now

        # Get prediction
        output = model(user_tensor)

    # 4. Interpret result
    sentiment_positive = (output.item() > 0.5)
    probability = output.item()

    if sentiment_positive:
        return f"({model_name}): Sounds positive! (Prob: {probability:.2f})"
    else:
        return f"({model_name}): Seems negative. (Prob: {probability:.2f})"


# --- Main Execution Block ---
if __name__ == "__main__":

    # Configuration
    CSV_PATH = r"amazon_cells_labelled.txt" # Adjust path if needed
    TEST_SPLIT_SIZE = 0.20 # 20% for initial test set split
    VALIDATION_SPLIT_SIZE = 0.125 # 12.5% of the remaining 80% -> 10% of total for validation
    RANDOM_STATE = 42
    NUM_EPOCHS = 5 # Adjust epochs as needed
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    HIDDEN_DIM_ANN = 128
    HIDDEN_DIM_TRANSFORMER = 256 # Can be different
    NHEAD_TRANSFORMER = 4 # Number of attention heads
    NUMLAYERS_TRANSFORMER = 2 # Number of transformer encoder layers

    # 1. Load Data
    print("Loading data...")
    try:
        data = pd.read_csv(CSV_PATH, delimiter='\t', header=None, names=['Sentence', 'Class'])
        # Ensure 'Class' is integer type
        data['Class'] = data['Class'].astype(int)
    except FileNotFoundError:
        print(f"Error: Data file not found at {CSV_PATH}")
        exit()
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    print(f"Loaded {len(data)} reviews.")

    # 2. Preprocess Data
    # Add index column needed temporarily by original preprocess_pandas logic if kept
    # data['index'] = data.index
    # columns = ['index', 'Class', 'Sentence'] # Original columns expected by func
    data = preprocess_pandas(data) # Pass the DataFrame directly

    # Separate features (text) and labels
    X = data['Sentence']
    y = data['Class']

    # 3. Split Data (80% Train, 10% Validation, 10% Test)
    print("Splitting data...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE, stratify=y # Stratify helps maintain class balance
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VALIDATION_SPLIT_SIZE, random_state=RANDOM_STATE, stratify=y_train_val # Stratify again
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # 4. Feature Extraction (TF-IDF)
    print("Vectorizing text data using TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        max_features=50000, # Limit feature size
        max_df=0.7, # Adjust max_df
        min_df=3,   # Ignore terms that appear less than 3 times
        use_idf=True,
        norm='l2'
    )

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_val_tfidf = tfidf_vectorizer.transform(X_val)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Convert sparse matrices to dense NumPy arrays, then to Tensors
    # Note: .todense() can be memory intensive for large datasets/features
    try:
        X_train_tensor = torch.FloatTensor(X_train_tfidf.todense())
        X_val_tensor = torch.FloatTensor(X_val_tfidf.todense())
        X_test_tensor = torch.FloatTensor(X_test_tfidf.todense())
    except MemoryError:
        print("Error: Ran out of memory converting TF-IDF to dense tensor.")
        print("Consider reducing max_features or using sparse tensor handling if possible.")
        exit()

    # Convert labels to Tensors
    y_train_tensor = torch.LongTensor(y_train.values) # Use LongTensor or FloatTensor depending on loss
    y_val_tensor = torch.LongTensor(y_val.values)
    y_test_tensor = torch.LongTensor(y_test.values)
    # For BCELoss, labels need to be FloatTensor
    y_train_tensor = y_train_tensor.float()
    y_val_tensor = y_val_tensor.float()
    y_test_tensor = y_test_tensor.float()


    # 5. Create Datasets and DataLoaders
    print("Creating PyTorch Datasets and DataLoaders...")
    train_dataset = ReviewsDataset(X_train_tensor, y_train_tensor)
    val_dataset = ReviewsDataset(X_val_tensor, y_val_tensor)
    test_dataset = ReviewsDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- ANN Model Training and Evaluation ---
    print("\n=== Setting up ANN Model ===")
    input_dim_ann = X_train_tensor.shape[1] # Get feature dimension from TF-IDF
    model_ann = SimpleANN(input_dim=input_dim_ann, hidden_dim=HIDDEN_DIM_ANN, output_dim=1)
    criterion_ann = nn.BCELoss() # Binary Cross-Entropy for Sigmoid output
    optimizer_ann = optim.Adam(model_ann.parameters(), lr=LEARNING_RATE)

    # Train ANN
    ann_train_losses, ann_val_losses, ann_train_accs, ann_val_accs = train_model(
        model_ann, criterion_ann, optimizer_ann, train_loader, val_loader, model_name="ANN", epochs=NUM_EPOCHS
    )

    # Plot ANN Metrics
    plot_metrics(ann_train_losses, ann_val_losses, ann_train_accs, ann_val_accs, NUM_EPOCHS, "ANN")

    # Evaluate ANN on Test Set
    ann_test_accuracy = evaluate_model_on_test(model_ann, test_loader, criterion_ann, model_name="ANN")


    # --- Transformer Model Training and Evaluation ---
    print("\n=== Setting up Transformer Model ===")
    input_dim_transformer = X_train_tensor.shape[1] # Get feature dimension from TF-IDF
    # Note: TransformerClassifier handles d_model adjustment internally
    model_transformer = TransformerClassifier(
        input_dim=input_dim_transformer,
        nhead=NHEAD_TRANSFORMER,
        num_layers=NUMLAYERS_TRANSFORMER,
        hidden_dim=HIDDEN_DIM_TRANSFORMER
    )
    criterion_transformer = nn.BCELoss()
    optimizer_transformer = optim.Adam(model_transformer.parameters(), lr=LEARNING_RATE)

    # Train Transformer
    trans_train_losses, trans_val_losses, trans_train_accs, trans_val_accs = train_model(
        model_transformer, criterion_transformer, optimizer_transformer, train_loader, val_loader, model_name="Transformer", epochs=NUM_EPOCHS
    )

    # Plot Transformer Metrics
    plot_metrics(trans_train_losses, trans_val_losses, trans_train_accs, trans_val_accs, NUM_EPOCHS, "Transformer")

    # Evaluate Transformer on Test Set
    transformer_test_accuracy = evaluate_model_on_test(model_transformer, test_loader, criterion_transformer, model_name="Transformer")


    # --- Final Comparison ---
    print("\n======================")
    print(" FINAL TEST ACCURACIES ")
    print("======================")
    print(f"ANN Test Accuracy:         {ann_test_accuracy:.4f}")
    print(f"Transformer Test Accuracy: {transformer_test_accuracy:.4f}")
    print("======================")

    # --- Chatbot Interaction ---
    print("\n--- Starting Chatbot Interaction (Type 'quit' to exit) ---")
    num_runs = 5 # Limit runs or use a loop
    count = 0
    while count < num_runs:
        try:
            user_input = input(f"\n[{count+1}/{num_runs}] Enter a sentence: ")
            if user_input.lower() == 'quit':
                break

            # Get responses from both models
            response_ann = chatbot_response(model_ann, user_input, tfidf_vectorizer, model_name="ANN")
            response_transformer = chatbot_response(model_transformer, user_input, tfidf_vectorizer, model_name="Transformer")

            print(f"You said: {user_input}")
            print(response_ann)
            print(response_transformer)
            count += 1

        except Exception as e:
            print(f"An error occurred during chatbot interaction: {e}")
            # break # Optionally break on error

    print("\n--- Chatbot interaction finished ---")
    print("--- Script execution complete ---")