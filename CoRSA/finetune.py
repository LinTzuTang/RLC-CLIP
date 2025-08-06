import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from typing import Optional
from unicore.modules import TransformerEncoderLayer, LayerNorm
import pandas as pd
import os
import numpy as np
from ast import literal_eval


# Define the dataset
class CustomDataset(Dataset):
    def __init__(self, matrix_data, vector_data):
        self.matrix_data = matrix_data
        self.vector_data = vector_data

    def __len__(self):
        return len(self.matrix_data)

    def __getitem__(self, idx):
        return self.matrix_data[idx], self.vector_data[idx]


# Define the 2D matrix encoder (similar to an image encoder)
class MatrixEncoder(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(MatrixEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(
                256 * 38 * 38, output_dim
            ),  # Adjusted for input size 621x621 with pooling
            nn.Dropout(0.3),  # Add Dropout layer
        )

    def forward(self, x):
        # Ensure the input has the correct shape (batch_size, channels, height, width)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension if missing
        return self.cnn(x)


# Define the 1D vector encoder (similar to a text encoder)
class VectorEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VectorEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add Dropout layer
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add Dropout layer
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        return self.fc(x)


# Define the Transformer encoder with pair
class TransformerEncoderWithPair(nn.Module):
    def __init__(
        self,
        encoder_layers: int = 4,
        embed_dim: int = 512,
        ffn_embed_dim: int = 2048,
        attention_heads: int = 4,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 256,
        activation_fn: str = "gelu",
        post_ln: bool = False,
        no_final_head_layer_norm: bool = False,
    ) -> None:

        super().__init__()
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.emb_layer_norm = LayerNorm(self.embed_dim)
        if not post_ln:
            self.final_layer_norm = LayerNorm(self.embed_dim)
        else:
            self.final_layer_norm = None

        if not no_final_head_layer_norm:
            self.final_head_layer_norm = LayerNorm(attention_heads)
        else:
            self.final_head_layer_norm = None

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=ffn_embed_dim,
                    attention_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    post_ln=post_ln,
                )
                for _ in range(encoder_layers)
            ]
        )

    def forward(
        self,
        emb: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        bsz = emb.size(0)
        seq_len = emb.size(1)
        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)

        # account for padding while computing the representation
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        if attn_mask is None:
            attn_mask = (
                torch.zeros((bsz, 1, seq_len, seq_len), device=emb.device)
                .repeat(1, self.attention_heads, 1, 1)
                .view(-1, seq_len, seq_len)
            )

        for i in range(len(self.layers)):
            x, attn_mask, _ = self.layers[i](
                x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True
            )

        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        return x, attn_mask


# Define the CLIP model
class CLIPModel(nn.Module):
    def __init__(self, matrix_encoder, vector_encoder, transformer_encoder):
        super(CLIPModel, self).__init__()
        self.matrix_encoder = matrix_encoder
        self.vector_encoder = vector_encoder
        self.transformer_encoder = transformer_encoder

    def forward(self, matrix, vector):
        matrix_features = self.matrix_encoder(matrix)
        vector_features = self.vector_encoder(vector)
        transformer_input = torch.cat(
            (matrix_features.unsqueeze(1), vector_features.unsqueeze(1)), dim=1
        )
        transformer_output, _ = self.transformer_encoder(transformer_input)
        return transformer_output[:, 0, :], transformer_output[:, 1, :]


# Data preparation function
def prepare_data(data_path_csv, data_path_npy):
    # Load encoded vector data from CSV file
    encoded_df = pd.read_csv(data_path_csv, keep_default_na=False)

    # Load matrix data from a separate .npy file
    matrix_data = np.load(data_path_npy)
    matrix_data = torch.tensor(matrix_data, dtype=torch.float32)

    # Print matrix data size for verification
    print(f"Matrix Data Size: {matrix_data.size()}")

    # Convert the lists in 'rna_embedding' and 'smiles_embedding' columns to NumPy arrays
    loaded_rna_data = np.stack(
        encoded_df["rna_embedding"].apply(lambda x: np.array(eval(x))).values
    )
    loaded_smiles_data = np.stack(
        encoded_df["smiles_embedding"].apply(lambda x: np.array(eval(x))).values
    )

    # Convert loaded data to tensors
    loaded_rna_data = torch.tensor(loaded_rna_data, dtype=torch.float32)
    loaded_smiles_data = torch.tensor(loaded_smiles_data, dtype=torch.float32)

    # Create combined data
    combined_data = torch.cat((loaded_rna_data, loaded_smiles_data), dim=1)

    # Print sizes for verification
    print(f"Loaded RNA Data Size: {loaded_rna_data.size()}")
    print(f"Loaded SMILES Data Size: {loaded_smiles_data.size()}")
    print(f"Combined Data Size: {combined_data.size()}")

    return matrix_data, combined_data



# Get encodings from model


# 1
def get_encodings(model, data_loader, device):
    matrix_encodings = []
    vector_encodings = []
    with torch.no_grad():
        for matrix, vector in data_loader:
            matrix, vector = matrix.to(device), vector.to(device)
            matrix_features, vector_features = model(matrix, vector)
            matrix_encodings.append(matrix_features.cpu())
            vector_encodings.append(vector_features.cpu())

    matrix_encodings = torch.cat(matrix_encodings, dim=0)
    vector_encodings = torch.cat(vector_encodings, dim=0)
    return matrix_encodings, vector_encodings


# 2
def get_encodings(model, data_loader, device):
    matrix_encoder, vector_encoder = model.matrix_encoder, model.vector_encoder
    matrix_encodings = []
    vector_encodings = []
    with torch.no_grad():
        for matrix, vector in data_loader:
            matrix, vector = matrix.to(device), vector.to(device)
            matrix_features_encoder = model.matrix_encoder(matrix)
            vector_features_encoder = model.vector_encoder(vector)
            matrix_features, vector_features = model(matrix, vector)
            # Concatenate both outputs
            matrix_features = torch.cat(
                (matrix_features, matrix_features_encoder), dim=-1
            )
            vector_features = torch.cat(
                (vector_features, vector_features_encoder), dim=-1
            )
            matrix_encodings.append(matrix_features.cpu())
            vector_encodings.append(vector_features.cpu())

    matrix_encodings = torch.cat(matrix_encodings, dim=0)
    vector_encodings = torch.cat(vector_encodings, dim=0)
    return matrix_encodings, vector_encodings


# 3
def get_encodings(model, data_loader, device):
    matrix_encodings = []
    vector_encodings = []
    with torch.no_grad():
        for matrix, vector in data_loader:
            matrix, vector = matrix.to(device), vector.to(device)
            matrix_features = model.matrix_encoder(matrix)
            vector_features = model.vector_encoder(vector)
            matrix_encodings.append(matrix_features.cpu())
            vector_encodings.append(vector_features.cpu())

    matrix_encodings = torch.cat(matrix_encodings, dim=0)
    vector_encodings = torch.cat(vector_encodings, dim=0)
    return matrix_encodings, vector_encodings


# Load and freeze pretrained model
def load_and_freeze_model(model_path, output_dim=768):
    # Load pretrained encoders (MatrixEncoder and VectorEncoder) instead of using CLIPModel's output for training
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    matrix_encoder = MatrixEncoder(input_channels=1, output_dim=output_dim).to(device)
    vector_encoder = VectorEncoder(input_dim=2048, output_dim=output_dim).to(
        device
    )  # Assuming input_dim is 2048 for inference
    transformer_encoder = TransformerEncoderWithPair(embed_dim=output_dim).to(device)
    model = CLIPModel(matrix_encoder, vector_encoder, transformer_encoder).to(device)

    # Load the trained model weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return model


# Define fully connected neural network class
class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FullyConnectedNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 8),
            nn.BatchNorm1d(hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 8, output_dim),
        )

    def forward(self, x):
        return self.fc(x)


# Train fully connected neural network to predict labels and record metrics
def train_fully_connected_nn(
    train_matrix_encodings,
    train_vector_encodings,
    y_train,
    val_matrix_encodings,
    val_vector_encodings,
    y_val,
    output_dim=768,
    learning_rate=3 * 1e-3,
    num_epochs=500,
    batch_size=256,
    early_stop_patience=50,
):
    # Combine matrix and vector encodings
    train_features = torch.cat((train_matrix_encodings, train_vector_encodings), dim=1)
    val_features = torch.cat((val_matrix_encodings, val_vector_encodings), dim=1)

    # Create training and validation datasets
    train_dataset = torch.utils.data.TensorDataset(
        train_features, torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = torch.utils.data.TensorDataset(
        val_features, torch.tensor(y_val, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model, loss function, and optimizer
    input_dim = train_features.size(1)
    hidden_dim = 256
    output_dim = 1  # Assuming regression task
    model = FullyConnectedNN(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.MSELoss() if output_dim == 1 else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50, verbose=True
    )

    # DataFrame to store metrics
    metrics_df = pd.DataFrame(
        columns=["Epoch", "Train Loss", "Train RMSE", "Valid RMSE", "Valid PR"]
    )

    # Early stopping variables
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            print(f"Training Batch Size: {features.size()}")

            # Forward pass
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_rmse = torch.sqrt(torch.tensor(avg_train_loss)).item()
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train RMSE: {train_rmse:.4f}"
        )

        # Validation phase
        model.eval()
        total_val_loss = 0
        val_outputs_list = []
        val_labels_list = []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                print(f"Validation Batch Size: {features.size()}")

                # Forward pass
                outputs = model(features).squeeze()
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()
                val_outputs_list.append(outputs.cpu())
                val_labels_list.append(labels.cpu())

        avg_val_loss = total_val_loss / len(val_loader)
        val_rmse = torch.sqrt(torch.tensor(avg_val_loss)).item()
        val_outputs = torch.cat(val_outputs_list, dim=0)
        val_labels = torch.cat(val_labels_list, dim=0)
        valid_pr = torch.corrcoef(torch.stack([val_outputs, val_labels]))[0, 1].item()
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation RMSE: {val_rmse:.4f}, Valid PR: {valid_pr:.4f}"
        )

        # Record metrics
        metrics_df = pd.concat(
            [
                metrics_df,
                pd.DataFrame(
                    [
                        {
                            "Epoch": epoch + 1,
                            "Train Loss": avg_train_loss,
                            "Train RMSE": train_rmse,
                            "Valid RMSE": val_rmse,
                            "Valid PR": valid_pr,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        # Learning rate decay
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

    # Save the metrics to a CSV file
    metrics_df.to_csv("training_metrics.csv", index=False)

    # Save the best model
    if best_model_state is not None:
        torch.save(best_model_state, "best_finetuned_fully_connected_nn.pth")
        
        

# Load and test the final model and record metrics
def test_fully_connected_nn(test_matrix_encodings, test_vector_encodings, y_test, model_path, batch_size=64):
    # Combine matrix and vector encodings
    test_features = torch.cat((test_matrix_encodings, test_vector_encodings), dim=1)
    test_dataset = torch.utils.data.TensorDataset(test_features, torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained model
    input_dim = test_features.size(1)
    hidden_dim = 256
    output_dim = 1  # Assuming regression task
    model = FullyConnectedNN(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Define loss function
    criterion = nn.MSELoss() if output_dim == 1 else nn.BCEWithLogitsLoss()
    total_test_loss = 0
    test_outputs_list = []
    test_labels_list = []
    
    # Testing loop
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            print(f"Testing Batch Size: {features.size()}")
            
            # Forward pass
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()
            test_outputs_list.append(outputs.cpu())
            test_labels_list.append(labels.cpu())
    
    avg_test_loss = total_test_loss / len(test_loader)
    test_rmse = torch.sqrt(torch.tensor(avg_test_loss)).item()
    test_outputs = torch.cat(test_outputs_list, dim=0)
    test_labels = torch.cat(test_labels_list, dim=0)
    test_pr = torch.corrcoef(torch.stack([test_outputs, test_labels]))[0, 1].item()
    print(f"Test Loss: {avg_test_loss:.4f}, Test RMSE: {test_rmse:.4f}, Test PR: {test_pr:.4f}")
    
    # Save test metrics to a CSV file
    test_metrics_df = pd.DataFrame([{ "Test RMSE": test_rmse, "Test PR": test_pr }])
    test_metrics_df.to_csv("test_metrics.csv", index=False)
