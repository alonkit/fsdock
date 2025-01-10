from datasets.process_sidechains import *


ligand = Chem.MolFromSmiles("N=C(N)SCCN")
core, core_smiles, sidechains ,sidechains_smiles = get_core_and_chains(ligand)
sidechain_indices = get_mask_of_sidechains(ligand,sidechains)