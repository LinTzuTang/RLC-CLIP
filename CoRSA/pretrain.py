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
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
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
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(512 * 38 * 38, output_dim),  # Adjusted for input size 621x621 with pooling
            nn.Dropout(0.3)  # Add Dropout layer
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
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Define the Transformer encoder with pair
class TransformerEncoderWithPair(nn.Module):
    def __init__(
        self,
        encoder_layers: int = 6,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
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
            attn_mask = torch.zeros((bsz, 1, seq_len, seq_len), device=emb.device).repeat(1, self.attention_heads, 1, 1).view(-1, seq_len, seq_len)

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
        transformer_input = torch.cat((matrix_features.unsqueeze(1), vector_features.unsqueeze(1)), dim=1)
        transformer_output, _ = self.transformer_encoder(transformer_input)
        return transformer_output[:, 0, :], transformer_output[:, 1, :]

# Contrastive loss function
def contrastive_loss(matrix_features, vector_features, temperature=0.07):
    # Normalize the features
    matrix_features = F.normalize(matrix_features, dim=-1)
    vector_features = F.normalize(vector_features, dim=-1)

    # Compute the logits
    logits = torch.matmul(matrix_features, vector_features.t()) / temperature
    labels = torch.arange(len(matrix_features)).to(matrix_features.device)

    # Calculate positive pair loss (loss1)
    positive_logit = torch.diag(logits)
    loss1 = -torch.mean(torch.log(torch.exp(positive_logit) / torch.sum(torch.exp(logits), dim=1)))

    # Calculate negative pair loss (loss2)
    negative_logits = logits[~torch.eye(len(matrix_features), dtype=bool)].view(len(matrix_features), -1)
    loss2 = -torch.mean(torch.log(1 - torch.exp(negative_logits) / torch.sum(torch.exp(logits), dim=1, keepdim=True)))

    # Combine loss1 and loss2
    loss = (loss1 + loss2) / 2
    return loss


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
    loaded_rna_data = np.stack(encoded_df['rna_embedding'].apply(lambda x: np.array(eval(x))).values)
    loaded_smiles_data = np.stack(encoded_df['smiles_embedding'].apply(lambda x: np.array(eval(x))).values)

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

# Model training function
def train_model(train_dataset, val_dataset, output_dim=768, batch_size=64, learning_rate=1e-3, num_epochs=100):
    # Create dataloaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, optimizer
    matrix_encoder = MatrixEncoder(input_channels=1, output_dim=output_dim).to(device)  # Adjust input_channels to 1 for grayscale input
    vector_encoder = VectorEncoder(input_dim=train_dataset[0][1].size(0), output_dim=output_dim).to(device)
    transformer_encoder = TransformerEncoderWithPair(embed_dim=output_dim).to(device)
    model = CLIPModel(matrix_encoder, vector_encoder, transformer_encoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Train the model
    train_clip(model, train_loader, val_loader, optimizer, device, scheduler, num_epochs=num_epochs)

    # Save the trained model
    torch.save(model.state_dict(), "clip_model_.pth", _use_new_zipfile_serialization=False)
    