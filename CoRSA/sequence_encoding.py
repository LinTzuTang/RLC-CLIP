import pandas as pd
import numpy as np
import sys
sys.path.append("./materials")
sys.path.append("./materials/models")
from materials.models import fm4m
from multimolecule import RnaTokenizer, RiNALMoModel
import torch


# RNA sequence encoding function with batch processing and memory optimization
# max_length = general_df['rna_sequence'].apply(len).max()
def encode_rna_sequences(rna_sequences, batch_size=64, max_length=2048):
    
    # Load the pretrained tokenizer and model
    rna_tokenizer = RnaTokenizer.from_pretrained('multimolecule/rinalmo')
    rna_model = RiNALMoModel.from_pretrained('multimolecule/rinalmo')

    rna_embeddings = []
    for i in range(0, len(rna_sequences), batch_size):
        batch_sequences = rna_sequences[i:i+batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(rna_sequences) + batch_size - 1) // batch_size}")

        # Tokenize the RNA sequences in the batch
        rna_input = rna_tokenizer(batch_sequences.tolist(), return_tensors='pt', padding=True, truncation=True, max_length=max_length)

        # Make sure to move inputs and model to CPU to reduce GPU memory usage
        rna_input = {key: val.to('cuda' if torch.cuda.is_available() else 'cpu') for key, val in rna_input.items()}
        rna_model.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Get model output and extract [CLS] token embedding
        with torch.no_grad():  # Disable gradient computation to save memory
            rna_output = rna_model(**rna_input)
            last_hidden_state = rna_output['last_hidden_state']
            cls_embeddings = last_hidden_state[:, 0, :]  # Extract CLS token for all sequences in the batch

        # Detach and move to CPU, then convert to NumPy
        cls_embeddings_np = cls_embeddings.detach().cpu().numpy()
        rna_embeddings.extend(cls_embeddings_np)

        # Clear cache to avoid memory overflow
        torch.cuda.empty_cache()

    # Convert all embeddings to a tensor
    return torch.tensor(rna_embeddings, dtype=torch.float32)



# fm4m.avail_models()
def encode_smiles(ligand_smiles, model_type='SMI-TED'):
    smiles_list = list(ligand_smiles.values)
    x_batch = fm4m.get_representation_x(smiles_list, model_type=model_type, return_tensor=False)
    x_batch = np.array(x_batch)
    return torch.tensor(x_batch, dtype=torch.float32)