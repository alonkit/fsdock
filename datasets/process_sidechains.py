import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import scaffoldgraph as sg
import re
import itertools
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch
try:
    from utils.logging_utils import get_logger
except:
    import logging
    def get_logger():
        return logging.getLogger(__name__)

'''
very important:
https://github.com/rdkit/rdkit/discussions/5458
https://github.com/rdkit/rdkit/discussions/6844
'''

logger = get_logger()
def get_num_fused_rings(mol):
    """rings that share at least one atom with another ring"""
    num_fused_rings = set()
    atom_rings = [set(cur_ring) for cur_ring in mol.GetRingInfo().AtomRings()]
    for i in range(len(atom_rings)):
        for j in range(i + 1, len(atom_rings)):
            if not atom_rings[i].isdisjoint(atom_rings[j]):
                num_fused_rings.update([i, j])
    return len(num_fused_rings)

def advanced_murcko_scaffold(mol,chains_weight_threshold=0.):
    num_rings = AllChem.CalcNumRings(mol)
    if num_rings == 0:
        return None
    num_fused_rings = get_num_fused_rings(mol)
    frags = sg.tree_frags_from_mol(mol)
    clean_core = None
    m1_weight = AllChem.CalcExactMolWt(mol)
    for i, cur_frag in enumerate(frags):
        
        try:
            # Chem.SanitizeMol(cur_frag)
            cur_frag = Chem.MolFromSmiles(Chem.MolToSmiles(cur_frag))
            frag_weight = AllChem.CalcExactMolWt(cur_frag)
        except:
            logger.warning(f"Failed to sanitize fragment {i}, skipping")
            continue
        if (m1_weight - frag_weight) / m1_weight > chains_weight_threshold or num_fused_rings != get_num_fused_rings(cur_frag):
            clean_core = cur_frag
            break
    return clean_core
    
def get_core_and_chains(m1):
    if  isinstance(m1,str):
        m1 = Chem.MolFromSmiles(m1)
    if m1 is None:
        return None, None, None, None
    for a in m1.GetAtoms():
        a.SetIntProp("__origIdx", a.GetIdx())
    # clean_core = MurckoScaffold.GetScaffoldForMol(m1)
    clean_core = advanced_murcko_scaffold(m1)
    if clean_core is None:
        return None, None, None, None
    core = Chem.ReplaceSidechains(m1, clean_core)
    sidechains = Chem.ReplaceCore(m1, clean_core)
    if core is None or sidechains is None:
        return None, None, None, None
    core_smiles = Chem.MolToSmiles(core)
    sidechains_smiles = Chem.MolToSmiles(sidechains)
    if core_smiles == '' or sidechains_smiles == '':
        return None, None, None, None
    return clean_core, core_smiles, sidechains ,sidechains_smiles

def get_mask_of_sidechains(full_mol,sidechains):
    frags = Chem.GetMolFrags(sidechains, asMols=True)
    mask = np.zeros(full_mol.GetNumAtoms())
    for i, frag in enumerate(frags):
        frag_indices = [a.GetIntProp("__origIdx") for a in frag.GetAtoms() if a.HasProp("__origIdx")]
        mask[frag_indices] = i + 1            
    return mask


if __name__ == '__main__':
    ligand = Chem.MolFromSmiles("O=c1c2ccccc2nc2n1CCCS2")
    core, core_smiles, sidechains ,sidechains_smiles = get_core_and_chains(ligand)
    core_indices = get_mask_of_sidechains(ligand,core)
    sidechain_indices = get_mask_of_sidechains(ligand,sidechains)

def reconstruct_from_core_and_chains(core, chains):
    chains = Chem.MolFromSmiles(chains)
    core_clean = Chem.MolFromSmiles(core)
    if core_clean is None or chains is None:
        return None
    try:
        sidechain_mols = Chem.GetMolFrags(chains, asMols=True)
    except:
        return None
    for mol in sidechain_mols:
        if len([atom.GetSmarts() for atom in mol.GetAtoms() if atom.GetSymbol() == "*"]) == 0:
            return None
    sidechain_tags = [[atom.GetSmarts() for atom in mol.GetAtoms() if atom.GetSymbol() == "*"][0]
                       for mol in sidechain_mols]
    sidechain_indexes = [[atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "*"][0]
                          for mol in sidechain_mols]
    sidechain_dict = dict(zip(sidechain_tags, zip(sidechain_mols, sidechain_indexes)))

    core_sidechain_tags = [atom.GetSmarts() for atom in core_clean.GetAtoms() if atom.GetSymbol() == "*"]
    core_sidechain_tags = [re.sub(r'(\[\d+\*).*\]', r'\1]', x) for x in core_sidechain_tags]
    current_core = core_clean
    for tag in core_sidechain_tags:
        replacement = sidechain_dict.get(tag, None)
        if replacement is None:
            return None
        new_core = Chem.ReplaceSubstructs(current_core,
                                          Chem.MolFromSmiles(tag),
                                          replacement[0],
                                          replacementConnectionPoint=sidechain_dict[tag][1],
                                          useChirality=1)
        if new_core[0] is None:
            return None
        current_core = new_core[0]
    reconstructed_smiles = Chem.MolToSmiles(current_core)
    reconstructed_smiles_clean = re.sub(r'\[\d+\*\]', '', reconstructed_smiles)
    if not smiles_valid(reconstructed_smiles_clean):
        return None
    recon = Chem.MolToSmiles(Chem.MolFromSmiles(reconstructed_smiles_clean))
    canon = Chem.CanonSmiles(recon, useChiral=0)
    return canon


def isotopize_dummies(fragment, isotope):
    for atom in fragment.GetAtoms():
        if atom.GetSymbol() == "*":
            atom.SetIsotope(isotope)
    return fragment


def add_attachment_points(smiles, n, seed=None, fg_weight=0, fg_list=[]):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if seed is not None:
        random.seed(seed)
    if len(fg_list) == 0:
        fg_list = [(1.0, "[c,C][H]", "C*")]
    else:
        epsilon = 1.e-10
        fg_weights = itertools.accumulate([fg_weight / len(fg_list)
                                           for i in range(len(fg_list))]
                                          + [1.0 + epsilon - fg_weight])
        fg_list = list(zip(fg_weights,
                           [x[0] for x in fg_list] + ["[c,C][H]"],
                           [x[1] for x in fg_list] + ["C*"]))

    current_mol = Chem.AddHs(mol)
    current_mol.UpdatePropertyCache()
    current_attachment_index = 1
    for i in range(n):
        next_mol = []
        max_tries = 100
        current_try = 0
        while len(next_mol) == 0:
            the_choice = [x for x in fg_list if x[0] >= random.random()][0]
            the_target = Chem.MolFromSmarts(the_choice[1])
            the_replacement = isotopize_dummies(Chem.MolFromSmiles(the_choice[2]), current_attachment_index)
            next_mol = Chem.ReplaceSubstructs(current_mol, the_target, the_replacement)
            current_try += 1
            if current_try >= max_tries:
                break  # we failed
        if current_try >= max_tries:
            continue  # skip and try again (we will return less than n attachment points)
        current_attachment_index += 1
        current_mol = random.choice(next_mol)
        current_mol.UpdatePropertyCache()

    current_mol = Chem.RemoveHs(current_mol)
    current_mol.UpdatePropertyCache()

    current_smiles = Chem.MolToSmiles(current_mol)
    return current_smiles

