import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import mlflow
import mlflow.pytorch
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from constants import *
from models.chess_opening_classifier import ChessOpeningClassifier
from dataset_classes.chess_dataset import ChessDataset

FILENAME = "lichess_games_20K.csv"
NUM_EPOCHS = 50
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
MAX_MOVES = 40
PLUS_MOVES = 0
RARE_OPENINGS = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Assuming your ECO codes are mapped in the file
eco_mapping_df = pd.read_csv("data/ECO_codes.csv")
eco_mapping = {eco: idx for idx, eco in enumerate(eco_mapping_df["code"].unique())}

def load_preprocessed_data(filename):
    with open(filename, 'rb') as f:
        df = pickle.load(f)
    return df

def preprocess_data():
    df = load_preprocessed_data(os.path.join(PREPROCESSED_DIR, FILENAME))
    
    def will_have_zero_moves(matrices, opening_ply, max_moves=MAX_MOVES):
        trim_moves = 2 * opening_ply + PLUS_MOVES
        end_index = min(trim_moves + max_moves, len(matrices))
        return len(matrices[trim_moves:end_index]) == 0
    
    df = df[~df.apply(lambda row: will_have_zero_moves(row['matrices'], row['opening_ply']), axis=1)]
    eco_counts = df["encoded_eco"].value_counts()
    rare_classes = eco_counts[eco_counts < RARE_OPENINGS].index
    df = df[~df["encoded_eco"].isin(rare_classes)]

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["encoded_eco"], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df["encoded_eco"], random_state=42)

    return train_df, val_df, test_df

def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = [inp for inp in inputs]
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_inputs, labels

def train_model():
    # Early stopping parameters
    patience = 5  # How many epochs to wait for improvement
    min_delta = 0.01  # Minimum change to count as an improvement
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    train_df, val_df, test_df = preprocess_data()
    train_dataset = ChessDataset(train_df, max_moves=MAX_MOVES)
    val_dataset = ChessDataset(val_df, max_moves=MAX_MOVES)
    test_dataset = ChessDataset(test_df, max_moves=MAX_MOVES)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)
    
    num_classes = len(eco_mapping)
    model = ChessOpeningClassifier(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # MLflow Setup
    mlflow.set_experiment("Chess Opening Classification")
    
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("num_epochs", NUM_EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("max_moves", MAX_MOVES)
        mlflow.log_param("rare_openings", RARE_OPENINGS)
        mlflow.log_param("file_name", FILENAME)
        
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss, correct, total = 0, 0, 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total += labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
            
            train_acc = correct / total
            train_loss = total_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_correct, val_total, val_loss = 0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item()
                    val_total += labels.size(0)
                    val_correct += (outputs.argmax(1) == labels).sum().item()
            
            val_acc = val_correct / val_total
            val_loss /= len(val_loader)
            
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            
            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0  # Reset counter
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break  # Stop training

        # After training completes, evaluate on the test set
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(outputs.argmax(1).cpu().numpy())        

        # Calculate overall metrics
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        # Log metrics with MLflow        
        mlflow.log_metric("test_acc", acc)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)

        mlflow.pytorch.log_model(model, "model")

        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")

if __name__ == "__main__":
    train_model()