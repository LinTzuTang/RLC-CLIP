import numpy as np
import torch
from scipy.spatial import distance_matrix
import torch.nn.functional as F
from Bio.PDB import PDBParser, MMCIFParser


def load_structure(filepath):
    """
    Loads atomic coordinates from a PDB or CIF file.

    Args:
        filepath (str): Path to the PDB or CIF file.

    Returns:
        list: A list of atomic coordinates as numpy arrays.
    """
    if filepath.endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    
    structure = parser.get_structure('structure', filepath)
    coordinates = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coordinates.append(atom.coord)
    return np.array(coordinates, dtype=np.float32).reshape(-1, 3)



def get_combined_distance_matrix(mol_pos, pocket_pos):
    """
    Combines distance matrices from pocket and molecule positions into a single matrix.
    Args:
        mol_pos (torch.Tensor or np.ndarray): Coordinates of the molecule (shape: [num_mol_atoms, 3]).
        pocket_pos (torch.Tensor or np.ndarray): Coordinates of the pocket (shape: [num_pocket_atoms, 3]).
    Returns:
        torch.Tensor: Combined distance matrix of shape [(num_mol_atoms + num_pocket_atoms), (num_mol_atoms + num_pocket_atoms)].
    """

    # Convert to numpy arrays if they are PyTorch tensors
    if isinstance(mol_pos, torch.Tensor):
        mol_pos = mol_pos.numpy()
    if isinstance(pocket_pos, torch.Tensor):
        pocket_pos = pocket_pos.numpy()

    # Calculate distance matrices for each part
    pocket_pocket_dist = distance_matrix(pocket_pos, pocket_pos).astype(np.float32)  # Top-left
    mol_mol_dist = distance_matrix(mol_pos, mol_pos).astype(np.float32)              # Bottom-right
    mol_pocket_dist = distance_matrix(mol_pos, pocket_pos).astype(np.float32)        # Cross-distance

    # Get the dimensions of the distance matrices
    num_pocket_atoms = pocket_pos.shape[0]
    num_mol_atoms = mol_pos.shape[0]

    # Create the final combined distance matrix with padding
    combined_size = num_pocket_atoms + num_mol_atoms
    combined_matrix = np.zeros((combined_size, combined_size), dtype=np.float32)

    # Fill in the top-left (pocket-pocket distances)
    combined_matrix[:num_pocket_atoms, :num_pocket_atoms] = pocket_pocket_dist

    # Fill in the bottom-right (mol-mol distances)
    combined_matrix[num_pocket_atoms:, num_pocket_atoms:] = mol_mol_dist

    # Fill in the top-right and bottom-left (mol-pocket distances)
    combined_matrix[:num_pocket_atoms, num_pocket_atoms:] = mol_pocket_dist.T
    combined_matrix[num_pocket_atoms:, :num_pocket_atoms] = mol_pocket_dist

    # Convert to torch tensor and return
    return torch.from_numpy(combined_matrix)
