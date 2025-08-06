import os
import gemmi
import pandas as pd
from openbabel import openbabel
from rdkit import Chem
from rdkit.Chem import rdmolops


def generate_ligand_dataframe(folder_path):
    # Initialize an empty list to store data
    data = []

    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        # Only process files that end with ".cif"
        if filename.endswith(".cif"):
            # Split the filename by underscores to extract pdb_id, ligand_id, and ligand_chain
            parts = filename.split("_")
            if len(parts) == 3:
                pdb_id = parts[0]
                ligand_id = parts[1]
                ligand_chain = parts[2].replace(".cif", "")

                # Append the extracted information to the data list
                data.append([pdb_id, ligand_id, ligand_chain])

    # Create a DataFrame from the data list
    df = pd.DataFrame(data, columns=["pdb_id", "ligand_id", "ligand_chain"])

    return df


# Function to extract RNA chain sequence from CIF file
def extract_rna_chain_sequence(cif_file_path, rna_chain):
    try:
        # Load the CIF file
        doc = gemmi.cif.read(cif_file_path)
        block = doc.sole_block()

        # Extract the list of struct asym ids
        struct_asym_table = block.find(["__struct_asym.id"])
        asym_ids = [row[0] for row in struct_asym_table]

        # Find the order of the specified rna_chain
        given_chain_order = None
        for i, asym_id in enumerate(asym_ids):
            if asym_id == rna_chain:
                given_chain_order = i + 1  # Order starts from 1
                break

        # Set to first chain order if rna_chain is not found
        if given_chain_order is None:
            print(
                f"RNA chain '{rna_chain}' not found in {cif_file_path}. Using the first chain '{asym_ids[0]}' instead."
            )
            given_chain_order = 1
            rna_chain = asym_ids[0]  # Update rna_chain to the first chain id

        # Extract the RNA sequence for the given chain
        entity_poly_seq_table = block.find(
            ["_entity_poly_seq.num", "_entity_poly_seq.mon_id"]
        )
        seq_nums = [row[0] for row in entity_poly_seq_table]
        mon_ids = [row[1] for row in entity_poly_seq_table]

        rna_chains = {}
        chain_order = 0
        sequence = ""

        for seq_num, mon_id in zip(seq_nums, mon_ids):
            if seq_num == "1":
                # Increment chain order for every new chain start
                chain_order += 1
                sequence = ""  # Start a new sequence for the new chain

            # Append monomer ID to the current chain sequence
            sequence += mon_id
            rna_chains[chain_order] = sequence

        # Return the sequence and updated chain id for the specified chain order
        return (
            rna_chains[given_chain_order] if given_chain_order in rna_chains else None
        ), rna_chain

    except Exception as e:
        # Handle cases where the CIF file does not exist or cannot be parsed
        print(f"Error processing {cif_file_path}: {e}")
        return None, rna_chain


# Function to extract SMILES and handle potential issues
def extract_ligand_smiles(row):
    pdb_id = row["pdb_id"]
    ligand_id = row["ligand_id"]
    ligand_chain = row["ligand_chain"]

    pdb_file_path = os.path.join(
        ligand_pdb_folder, f"{pdb_id}_{ligand_id}_{ligand_chain}.pdb"
    )

    if not os.path.exists(pdb_file_path):
        print(f"Warning: PDB file {pdb_file_path} not found.")
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            ligand_smiles = extract_smiles_from_pdb(pdb_file_path)
    except Exception as e:
        print(f"Error extracting SMILES from {pdb_file_path}: {e}")
        ligand_smiles = None

    return ligand_smiles


# Apply the function to the DataFrame
# pdbbind_df_['ligand_smiles'] = pdbbind_df_.apply(extract_ligand_smiles, axis=1)


# Function to extract SMILES and handle potential issues
def extract_ligand_smiles(row):
    pdb_id = row["pdb_id"]
    ligand_id = row["ligand_id"]
    ligand_chain = row["ligand_chain"]

    pdb_file_path = os.path.join(
        ligand_pdb_folder, f"{pdb_id}_{ligand_id}_{ligand_chain}.pdb"
    )

    if not os.path.exists(pdb_file_path):
        print(f"Warning: PDB file {pdb_file_path} not found.")
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            ligand_smiles = extract_smiles_from_pdb(pdb_file_path)
    except Exception as e:
        print(f"Error extracting SMILES from {pdb_file_path}: {e}")
        ligand_smiles = None

    return ligand_smiles


# Apply the function to the DataFrame
# pdbbind_df_['ligand_smiles'] = pdbbind_df_.apply(extract_ligand_smiles, axis=1)


# Fix SMILES

# Set up the path to your PDB files
# ligand_pdb_folder = '/home/tzutang.lin/blue_ufhpc/CoPRA/CoRSA/pdbbind_dataset_rna/parsed_ligand_pdb'


# Function to extract SMILES using RDKit
def extract_ligand_smiles_fix(row):
    pdb_id = row["pdb_id"]
    ligand_id = row["ligand_id"]
    ligand_chain = row["ligand_chain"]
    pdb_file_path = os.path.join(
        ligand_pdb_folder, f"{pdb_id}_{ligand_id}_{ligand_chain}.pdb"
    )

    if not os.path.exists(pdb_file_path):
        print(f"Warning: PDB file {pdb_file_path} not found.")
        return None

    try:
        # Load the PDB file without sanitization
        mol = Chem.MolFromPDBFile(pdb_file_path, sanitize=False)

        if mol is None:
            print(f"Warning: RDKit could not parse {pdb_file_path}")
            return None

        # Split into fragments and attempt to find the ligand-sized fragment
        fragments = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)

        if not fragments:
            print(f"Warning: No fragments found for PDB file {pdb_file_path}")
            return None

        # Find the most suitable fragment to represent the ligand
        for frag in fragments:
            if frag.GetNumAtoms() < 100:  # Adjust the threshold if needed
                ligand_smiles = Chem.MolToSmiles(frag, canonical=True)
                if not ligand_smiles:
                    print(
                        f"Warning: Failed to generate SMILES for fragment in file: {pdb_file_path}"
                    )
                    return None
                return ligand_smiles

        # If no suitable fragment is found
        print(
            f"Warning: No suitable fragment found for SMILES generation in file {pdb_file_path}"
        )
        return None

    except Exception as e:
        print(
            f"Error extracting SMILES from {pdb_file_path} (PDB ID: {pdb_id}, Ligand ID: {ligand_id}, Chain: {ligand_chain}): {e}"
        )
        return None


# Apply the function to the DataFrame
# merged_df['ligand_smiles'] = merged_df.apply(extract_ligand_smiles, axis=1)
