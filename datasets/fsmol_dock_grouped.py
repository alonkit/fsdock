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


class GFsDockDataset(Dataset):
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
        num_workers=1,
        tokenizer=None,
        hide_sidechains=False,
        
    ):
        if isinstance(tasks, str):
            tasks = pd.read_csv(tasks)
        self.logger = get_logger()
        self.tasks_df = tasks
        self.num_workers = num_workers
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.all_atoms = all_atoms
        self.receptor_radius = receptor_radius
        self.ligand_radius = ligand_radius
        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        self.knn_only_graph = knn_only_graph
        self.graph_dir = f"tasks_rh{remove_hs}"
        self.saved_protein_graph_file = f"protein_graphs_rr{receptor_radius}_camn{c_alpha_max_neighbors}_kog{knn_only_graph}_aa{all_atoms}_ar{atom_radius}.pt"
        self.saved_ligand_receptor_edge_file = f"protein_ligand_edges_lr{ligand_radius}.pt"
        self.saved_ligand_atom_edge_file = f"protein_ligand_edges_la{ligand_radius}.pt"
        self.tokenizer = tokenizer
        self.hide_sidechains = hide_sidechains
        self.tasks = {}
        super().__init__(root, transform)
        if not hasattr(self, "protein_graphs"):
            self.process()
            
            
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
        if not isinstance(name, str):
            name = self.task_names()[name]
        return self.tasks[name]
        
    def task_names(self):
        if hasattr(self,'_task_names'):
            return self._task_names
        return  self.tasks_df["assay_id"].unique()
    
    def get_task_path(self,name):
        return osp.join(self.processed_dir, self.graph_dir, name)

    def processed_file_names(self):
        names = [osp.join(self.graph_dir, task) for task in self.task_names()] + [
            self.saved_protein_file,
            self.saved_protein_graph_file,
            self.saved_esm_file,
            self.saved_ligand_receptor_edge_file,
        ]
        if self.all_atoms:
            names.append(self.saved_ligand_atom_edge_file)
        return names

   
    def process(self):
        self.process_proteins()
        self.process_tasks()
        self.process_ligand_protein_edges()
    
    def process_ligand_protein_edges(self):
        do_receptors = files_exist()
        ligands_receptor_edges = {}
        ligands_atom_edges = {}
        for task_name, task in self.tasks.items():
            protein_graph = self.protein_graphs[task['protein_id']]
            ligand_graphs = task['ligand_graphs']
            ligands_receptor_edges[task_name] = self.get_lig_protein_edges(protein_graph, ligand_graphs, "receptor", self.ligand_radius)
            if self.all_atoms:
                ligands_atom_edges[task_name] = self.get_lig_protein_edges(protein_graph, ligand_graphs, "atom", self.atom_radius)
        
            
    
    def process_tasks(self):
        makedirs(osp.join(self.processed_dir, self.graph_dir))
        tasks = self.tasks_df.groupby("assay_id")
        for assay_id, grouped_rows in tqdm(tasks,desc='processing tasks'):
            try:
                self.process_task(assay_id, grouped_rows)
            except Exception as e:
                self.logger.error(f"failed to process task {assay_id}, {e}")
        
        for task_name in tqdm(self.task_names(), desc='loading tasks'):
            self.tasks[task_name] = self._load_task(task_name)
            

    def process_task(self, assay_id, grouped_rows):
        if files_exist([osp.join(self.processed_dir, self.graph_dir, assay_id)]):
            return
        task = {"name": assay_id,"target":'', "activity_type": "","graphs": [], "labels": []}
        ligands = []
        for idx, row in (grouped_rows.iterrows()):
            task['activity_type'] = row['type']
            protein_id = row["target_id"]
            task['target'] = protein_id
            ligand_path = row["ligand_path"]
            label = row["label"]
            ligand = read_molecule(ligand_path,sanitize=True)
            ligands.append(ligand)
            task["labels"].append(label)
        graphs, labels, lr_edges, la_edges = self.get_task_graphs(task, protein_id, ligands)
        task['graphs'] = graphs
        task['labels'] = labels
        task['lr_edges'] = lr_edges
        task['la_edges'] = la_edges
        torch.save(task, osp.join(full_folder, assay_id))

 

    def get_task_graphs(self, task, protein_id, ligands):
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
        protein_graph = self.protein_graphs[protein_id]

        
        return ligand_graphs, new_labels

    def get_lig_protein_edges(self, protein_graph, ligand_graphs, protein_node_key, cutoff_distance):
        lig_poses = torch.cat([g["ligand"].pos for g in ligand_graphs], dim=0)
        lig_slices = torch.tensor([0,*(len(g["ligand"].pos) for g in ligand_graphs)])
        lig_slices = torch.cumsum(lig_slices, dim=0)
        lig_protein_batch = radius(
            protein_graph["receptor"].pos,
            lig_poses,
            cutoff_distance,
            max_num_neighbors=9999,
        )
        ligands_protein_edges = []
        for lig_i_start, lig_i_end in zip(lig_slices[:-1], lig_slices[1:]):
            curr_edges = lig_protein_batch[:, (lig_i_start <= lig_protein_batch[0]) & (lig_protein_batch[0] < lig_i_end)].clone()
            curr_edges[:0] -= lig_i_start
            ligands_protein_edges.append(curr_edges)
        return ligands_protein_edges
            
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
        if files_exist([osp.join(self.processed_dir, self.saved_protein_graph_file)]):
            self.protein_graphs =  self.generate_protein_graphs(None,None)
            return
        proteins = self.build_proteins_from_pdb()
        lm_embeddings = self.generate_ESM(proteins)
        self.protein_graphs = self.generate_protein_graphs(proteins, lm_embeddings)
    
    def build_proteins_from_pdb(self):
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
        return proteins
    
    def generate_protein_graphs(self, proteins, lm_embeddings):
        protein_graph_path = osp.join(self.processed_dir, self.saved_protein_graph_file)
        if not files_exist([protein_graph_path]):
            protein_graphs = {}
            for protein_id, protein in tqdm(proteins.items(), desc="Generating protein graphs"):
                protein_graph = HeteroData()
                moad_extract_receptor_structure(
                    pdb=protein,
                    complex_graph=protein_graph,
                    neighbor_cutoff=self.receptor_radius,
                    max_neighbors=self.c_alpha_max_neighbors,
                    lm_embeddings=lm_embeddings[protein_id],
                    knn_only_graph=self.knn_only_graph,
                    all_atoms=self.all_atoms,
                    atom_cutoff=self.atom_radius,
                    atom_max_neighbors=self.atom_max_neighbors,
                )
                protein_graphs[protein_id] = protein_graph
            torch.save(protein_graphs, protein_graph_path)
        else:
            protein_graphs = torch.load(protein_graph_path)
        return protein_graphs

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

            for target, sequence in protein_sequences.items():
                s = sequence.split(":")
                lm_embeddings[target] = [
                    lm_embeddings[f"{target}_chain_{j}"] for j in range(len(s))
                ]
            torch.save(lm_embeddings, esm_path)
        else:
            lm_embeddings = torch.load(esm_path)
        return lm_embeddings
