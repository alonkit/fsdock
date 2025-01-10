import os.path as osp
from collections import defaultdict
import pickle
import time
from typing import Callable, List, Tuple
import concurrent
import pandas as pd
from rdkit import Chem, DataStructs
from multiprocessing import Pool

import torch
from torch_geometric.data import Dataset, HeteroData, makedirs, Batch
from torch_geometric.data.dataset import files_exist
from torch_geometric.nn.pool import radius
import prody as pr
from tqdm import tqdm
from datasets.process_mols import (
    get_lig_graph,
    moad_extract_receptor_structure,
    read_molecule,
    get_binding_pockets,
    hide_sidechains
)
from esm import FastaBatchedDataset, pretrained

from datasets.process_sidechains import get_core_and_chains, get_mask_of_sidechains
from utils.esm_utils import compute_ESM_embeddings
from utils.logging_utils import get_logger
from utils.protein_utils import get_sequences_from_protein


class FsDockDataset(Dataset):
    """
    tasks =
    assay_id, target_id, protein_path, ligand_path, label, type
    """

    saved_protein_file = "proteins.pt"
    saved_esm_file = "esm_embd.pt"

    def __init__(
        self,
        root,
        tasks: pd.DataFrame,
        transform=None,
        receptor_radius=30,
        ligand_radius=20,
        c_alpha_max_neighbors=None,
        remove_hs=False,
        all_atoms=True,
        atom_radius=5,
        atom_max_neighbors=None,
        knn_only_graph=False,
        lazy_load=False,
        num_workers=1,
        tokenizer=None,
        hide_sidechains=False,
        
    ):
        if isinstance(tasks, str):
            tasks = pd.read_csv(tasks)
        self.logger = get_logger()
        self.tasks_df = tasks
        self.num_workers = num_workers
        self.proteins = {}
        self.lm_embeddings = []
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.all_atoms = all_atoms
        self.receptor_radius = receptor_radius
        self.ligand_radius = ligand_radius
        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        self.knn_only_graph = knn_only_graph
        self.graph_dir = f"graphs_rr{receptor_radius}_camn{c_alpha_max_neighbors}_rh{remove_hs}_aa{all_atoms}_ar{atom_radius}_kog{knn_only_graph}"
        self._task_names = self.task_names()
        self.lazy_load = lazy_load
        self.tokenizer = tokenizer
        self.hide_sidechains = hide_sidechains
        super().__init__(root, transform)
        if not lazy_load:
            self.logger.info('loading data for FS-Dock')
            self.complex_graphs = {task:self._load_task(task) for task in tqdm(self._task_names)}
            
    def _load_task(self, idx):
        if isinstance(idx,str):
            f_name = self.get_task_path(idx)
        else:
            f_name = self.get_task_path(self.task_names()[idx])
        task = torch.load(f_name)
        if self.tokenizer:
            for graph in task['graphs']:
                core_tokens = self.tokenizer.encode(graph.core_smiles).ids
                sidechain_tokens = self.tokenizer.encode(graph.sidechains_smiles).ids
                graph.core_tokens = torch.tensor(core_tokens).unsqueeze(0)
                graph.sidechain_tokens = torch.tensor(sidechain_tokens).unsqueeze(0)
        if self.hide_sidechains:
            for graph in task['graphs']:
                hide_sidechains(graph)
        return task

    def len(self):
        return len(self.task_names())
    
    def get(self,name):
        if self.lazy_load:
            return self._load_task(name)
        if not isinstance(name, str):
            name = self.task_names()[name]
        return self.complex_graphs[name]
        
    def task_names(self):
        if hasattr(self,'_task_names'):
            return self._task_names
        return  self.tasks_df["assay_id"].unique()
    
    def get_task_path(self,name):
        return osp.join(self.processed_dir, self.graph_dir, name)

    def processed_file_names(self):
        return [osp.join(self.graph_dir, task) for task in self.task_names()] + [
            self.saved_protein_file,
            self.saved_esm_file,
        ]

    def process_single_task(self, assay_id, grouped_rows):
        full_folder = osp.join(self.processed_dir, self.graph_dir)
        makedirs(full_folder)
        if files_exist([osp.join(full_folder, assay_id)]):
            return
        task = {"name": assay_id, "activity_type": "","graphs": [], "labels": []}
        ligands = []
        for idx, row in (grouped_rows.iterrows()):
            task['activity_type'] = row['type']
            protein_id = row["target_id"]
            ligand_path = row["ligand_path"]
            label = row["label"]
            ligand = read_molecule(ligand_path,sanitize=True)
            ligands.append(ligand)
            task["labels"].append(label)
        graphs, labels = self.build_task_graphs(task, protein_id, ligands)
        task['graphs'] = graphs
        task['labels'] = labels
        torch.save(task, osp.join(full_folder, assay_id))

    def process_multiple_tasks(self, tasks):
        for assay_id, grouped_rows in tqdm(tasks):
            try:
                self.process_single_task(assay_id, grouped_rows)
            except Exception as e:
                self.logger.error(f"failed to process task {assay_id}, {e}")

    def process(self):
        self.process_proteins()
        full_folder = osp.join(self.processed_dir, self.graph_dir)
        makedirs(full_folder)
        tasks = self.tasks_df.groupby("assay_id")
        if self.num_workers == 1:
            self.process_multiple_tasks(tasks)
        else:
            tasks = list(tasks)
            num_tasks = len(tasks)
            num_tasks_in_subtasks = -(-num_tasks // self.num_workers)
            subtasks_groups = [tasks[i:i+num_tasks_in_subtasks] for i in range(0, num_tasks, num_tasks_in_subtasks)]
            with Pool(self.num_workers) as p: 
                p.map(self.process_multiple_tasks, subtasks_groups)




    def build_task_graphs(self, task, protein_id, ligands):
        protein_graph = HeteroData()
        moad_extract_receptor_structure(
            pdb=self.proteins[protein_id],
            complex_graph=protein_graph,
            neighbor_cutoff=self.receptor_radius,
            max_neighbors=self.c_alpha_max_neighbors,
            lm_embeddings=self.lm_embeddings[protein_id],
            knn_only_graph=self.knn_only_graph,
            all_atoms=self.all_atoms,
            atom_cutoff=self.atom_radius,
            atom_max_neighbors=self.atom_max_neighbors,
        )
        ligand_graphs = []
        new_labels = []
        for i, (ligand, label) in enumerate(zip(ligands, task['labels'])):
            ligand_graph = HeteroData()
            ligand_graph['name'] = f'{task["name"]}_{i}'
            get_lig_graph(
                ligand,
                ligand_graph,
            )
            success = self.add_chem(ligand_graph, ligand)
            if not success:
                continue
            ligand_graph.activity_type = task['activity_type']
            ligand_graph.label = label
            
            ligand_graphs.append(ligand_graph)
            new_labels.append(label)
            
        get_binding_pockets(protein_graph, ligand_graphs, self.ligand_radius, self.atom_radius)
        return ligand_graphs, new_labels

    def add_chem(self,data, ligand):
        try:
            data.mol = ligand
            core, core_smiles, sidechains ,sidechains_smiles = get_core_and_chains(ligand)
            if core is None:
                self.logger.debug(f"Could not extract core and side chains for ligand {Chem.MolToSmiles(ligand)}")
                return False
            data.core = core
            data.core_smiles =core_smiles
            data.sidechains = sidechains
            data.sidechains_smiles = sidechains_smiles
            data.sidechains_mask = get_mask_of_sidechains(ligand,sidechains)
            return True
        except Exception as e:
            self.logger.error(f"Error processing ligand {Chem.MolToSmiles(ligand)}")
            self.logger.error(e)
            return False

    
    def process_proteins(self):
        protein_path = osp.join(self.processed_dir, self.saved_protein_file)
        if not files_exist([protein_path]):
            proteins = {}
            tasks = self.tasks_df.groupby("target_id")
            for protein_id, grouped_rows in tqdm(tasks):
                path = grouped_rows.iloc[0]["protein_path"]
                protein = pr.parsePDB(path)
                proteins[protein_id] = protein

            torch.save(proteins, protein_path)
        else:
            proteins = torch.load(protein_path)

        self.generate_ESM(proteins)
        self.proteins = proteins

    def generate_ESM(self, proteins):
        esm_path = osp.join(self.processed_dir, self.saved_esm_file)
        if not files_exist([esm_path]):

            print("Generating ESM language model embeddings")
            model_location = "esm2_t33_650M_UR50D"
            model, alphabet = pretrained.load_model_and_alphabet(model_location)
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()

            protein_sequences = {
                target: get_sequences_from_protein(prot)
                for target, prot in proteins.items()
            }
            labels, sequences = [], []
            for target, sequence in protein_sequences.items():
                s = sequence.split(":")
                sequences.extend(s)
                labels.extend([target + "_chain_" + str(j) for j in range(len(s))])

            lm_embeddings = compute_ESM_embeddings(model, alphabet, labels, sequences)

            self.lm_embeddings = {}
            for target, sequence in protein_sequences.items():
                s = sequence.split(":")
                self.lm_embeddings[target] = [
                    lm_embeddings[f"{target}_chain_{j}"] for j in range(len(s))
                ]
            torch.save(self.lm_embeddings, esm_path)
        else:
            self.lm_embeddings = torch.load(esm_path)
