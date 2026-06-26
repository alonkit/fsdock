import torch
from tqdm import tqdm

from datasets.fsmol_dock import FsDockDataset
from collections.abc import Iterable
from copy import deepcopy
import traceback
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from datasets.process_chem.process_mols import read_molecule
from esm import FastaBatchedDataset, pretrained

from datasets.process_chem.process_sidechains import get_core_and_chains_from_scaffold, get_holes, get_mol_smiles
from utils.esm_utils import compute_ESM_embeddings
from utils.logging_utils import get_logger
from utils.map_file_manager import MapFileManager
import os.path as osp
from torch_geometric.data.dataset import files_exist

from utils.protein_utils import get_sequences_from_protein

class FsDockCustomDataset(FsDockDataset):
    @staticmethod
    def process_ligand(args):
        res = {}
        try:
            task_name, idx, ligand_path, scaffold_smi = args
            ligand = read_molecule(ligand_path, sanitize=True)
            if ligand is None:
                return task_name, idx, res
            smiles = get_mol_smiles(ligand)
            res['ligand']=ligand
            res['smiles']=smiles
            core, sidechains, slicing_data = get_core_and_chains_from_scaffold(
                ligand, scaffold_smi
            )
            if core is None:
                get_logger().warning(
                    f"couldnt extract core: {task_name}, {idx}, {Chem.MolToSmiles(ligand)}"
                )
                return task_name, idx, res
            res['core']=core
            res['sidechains']=sidechains
            res.update(slicing_data)
            return (
                task_name,
                idx,
                res
            )
        except Exception as e:
            get_logger().error(
                f"Error processing ligand {task_name}, {idx}, {Chem.MolToSmiles(ligand)}"
            )
            get_logger().error(traceback.format_exc())
            return task_name, idx, res


    def process_ligands(self):
        if files_exist([osp.join(self.processed_dir, self.ligands_file)]):
            self.ligands = torch.load(osp.join(self.processed_dir, self.ligands_file))
            return

        task_groups = self.tasks_df.groupby("assay_id")

        ligand_build_params = []
        tasks_size = {}
        for assay_id, grouped_rows in task_groups:
            tasks_size[assay_id] = len(grouped_rows)
            for idx, (_, row) in enumerate(grouped_rows.iterrows()):
                ligand_build_params.append((assay_id, idx, row["ligand_path"], row['scaffold']))
        ligands = {k: [None] * v for k, v in tasks_size.items()}
        with tqdm(total=len(ligand_build_params), desc="build ligands") as progress_bar:
            with torch.multiprocessing.Pool(self.num_workers) as pool:
                for task_name, idx, chem_data in pool.imap(
                    self.process_ligand, ligand_build_params
                ):
                    ligands[task_name][idx] = chem_data
                    progress_bar.update()
        self.ligands = ligands
        torch.save(self.ligands, osp.join(self.processed_dir, self.ligands_file))

