


from copy import deepcopy
import torch
from datasets.fsmol_dock_clf import FsDockClfDataset
from datasets.partitioned_fsmol_dock import FsDockDatasetPartitioned
from torch_geometric.nn.pool import radius


class OneShotDockDatasetPartitioned(FsDockDatasetPartitioned):
    def __getitem__(self, idx_dct):
        if isinstance(idx_dct,list):
            items = [self[s_idx] for s_idx in idx_dct]
            return items
        main_idx = idx_dct['main']
        aux_idx = idx_dct['aux']
        main_data = super().__getitem__(main_idx)
        if isinstance(aux_idx, tuple):
            task_name, aux_idx = aux_idx
        else:
            task_name, aux_idx = self._indices[aux_idx]
        aux_lig = self.tasks[task_name]["graphs"][aux_idx]
        add_aux_lig(
            aux_lig,
            main_data,
            self.ligand_radius,
            self.atom_radius,
            self.all_atoms,
        )
        return main_data

class  OneShotDockClfDataset(FsDockClfDataset):
    def __getitem__(self, idx_dct):
        if isinstance(idx_dct,list):
            items = [self[s_idx] for s_idx in idx_dct]
            return items
        main_idx = idx_dct['main']
        aux_idx = idx_dct['aux']
        main_data = super().__getitem__(main_idx)
        if isinstance(aux_idx, tuple):
            task_name, aux_idx = aux_idx
        else:
            task_name, aux_idx = self._indices[aux_idx]
        aux_lig = self.tasks[task_name]["graphs"][aux_idx]
        add_aux_lig(
            aux_lig,
            main_data,
            self.ligand_radius,
            self.atom_radius,
            self.all_atoms,
        )
        return main_data
    
def add_aux_lig(aux_lig, main_data, ligand_radius, atom_radius, all_atoms):
    orig_lig_num_nodes = main_data['ligand'].num_nodes
    aux_lig_num_nodes = aux_lig['ligand'].num_nodes
    
    main_data['ligand'].x = torch.cat([main_data['ligand'].x, aux_lig['ligand'].x], dim=0)
    main_data['ligand'].pos = torch.cat([main_data['ligand'].pos, aux_lig['ligand'].pos], dim=0)
    
    one_shot_feat = torch.tensor([0]*orig_lig_num_nodes + [1]*aux_lig_num_nodes).unsqueeze(1)
    main_data['ligand'].x = torch.concat([main_data['ligand'].x, one_shot_feat ],dim=1)
    
    main_data.sidechains_mask = torch.cat([main_data.sidechains_mask,torch.zeros(aux_lig_num_nodes)])
    
    
    lig_rec = radius(
        main_data["receptor"].pos,
        aux_lig['ligand'].pos,
        ligand_radius,
        max_num_neighbors=9999,
    )
    lig_rec[0] += orig_lig_num_nodes
    main_data['ligand','receptor'].edge_index = torch.cat([
            main_data['ligand','receptor'].edge_index,
            lig_rec], dim=1)
    
    if all_atoms:
        lig_atom = radius(
            main_data["atom"].pos,
            aux_lig['ligand'].pos,
            atom_radius,
            max_num_neighbors=9999,
        )
        lig_atom[0] += orig_lig_num_nodes
        main_data['ligand','atom'].edge_index = torch.cat([
            main_data['ligand','atom'].edge_index,
            lig_atom], dim=1)
        
        