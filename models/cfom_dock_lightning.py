from collections import defaultdict
import copy
from datetime import datetime
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets.custom_distributed_sampler import CustomDistributedSampler
from datasets.fsmol_dock import FsDockDataset
from datasets.fsmol_dock_clf import FsDockClfDataset
from datasets.partitioned_fsmol_dock import FsDockDatasetPartitioned
from datasets.process_chem.process_sidechains import (
    calc_tani_sim,
    get_fp,
    reconstruct_from_core_and_chains,
)
from models.cfom_dock import CfomDock
from models.tasks.task import AtomNumberTask
from utils.logging_utils import configure_logger, get_logger
from rdkit import Chem


class CfomDockLightning(pl.LightningModule):
    def __init__(
        self,
        cfom_dock_model: CfomDock,
        tokenizer,
        lr,
        weight_decay,
        num_gen_samples,
        loss=None,
        test_clfs=None,
        similarity_threshold=0.4,
        gen_meta_params=None,
        name=None,
        smol=True,
    ):
        super().__init__()
        self.cfom_dock_model = cfom_dock_model
        self.num_gen_samples = num_gen_samples
        self.tokenizer = tokenizer
        if loss is None:
            loss = nn.CrossEntropyLoss(reduction='none')
        self.loss = loss
        self.noise_loss = nn.MSELoss()
        self._reset_eval_step_outputs()
        self.weight_decay = weight_decay
        self.lr = lr
        self.validation_clfs = None
        self.test_clfs = test_clfs
        self.similarity_threshold = similarity_threshold
        self.gen_meta_params = gen_meta_params or {"p":1.}
        # self.save_hyperparameters(
        #     ignore=["cfom_dock_model", "loss", "tokenizer", "validation_clfs", "test_clfs", 'side_']
        # )
        self.name = name or f'{datetime.today().strftime("%Y-%m-%d-%H_%M_%S")}'
        self.name = f'cfom_dock_{self.name}'
        self.test_result_path = f'test_stats/{self.name}'
        self.smol = smol
        self.freeze_layers = self.cfom_dock_model.freeze_layers
        self.unfreeze_start = 4
        self.unfreeze_step = 2

    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.sub_proteins.open()

    def _reset_eval_step_outputs(self):
        self.eval_step_outputs = defaultdict(lambda: defaultdict(list))
    
    def train_dataloader(self):
        if self.smol:
            dst = FsDockDatasetPartitioned('data/fsdock/valid','../docking_cfom/valid_tasks.csv',tokenizer=self.tokenizer, num_workers=torch.get_num_threads())
        else:
            dst = FsDockDatasetPartitioned("data/fsdock/train", "data/fsdock/train_tasks.csv",tokenizer=self.tokenizer)
        dlt = DataLoader(dst, batch_size=2 if self.smol else 20 , sampler=CustomDistributedSampler(dst, shuffle=True), num_workers=torch.get_num_threads(), 
                        worker_init_fn=self.worker_init_fn)
        return dlt
    
    def val_dataloader(self):
        dsv = FsDockClfDataset("data/fsdock/clfs/valid", "data/fsdock/valid_tasks.csv",tokenizer=self.tokenizer, only_inactive=True)
        dlv = DataLoader(dsv, batch_size=32, 
                num_workers=torch.get_num_threads()//2, 
                worker_init_fn=self.worker_init_fn)
        self.validation_clfs=dsv.clfs
        return dlv
    
    def t_to_sigma(self, t):
        return 0.05 ** (1-min(t,1)) * 0.2 ** t
    
    def good_loss_ratio(self, t):
        t = min(t,1)
        return t * 0.9 + (1-t)*0.5
    
    def get_loss(self,data):
        logits = self.cfom_dock_model(
            data.core_tokens,
            data.sidechain_tokens[:, :-1],
            data,
            (data.activity_type, data.label), 
            molecule_sidechain_mask_idx=1
        )
        logits = logits.transpose(1, -1)
        tgt = data.sidechain_tokens[:, 1:]
        return self.loss(logits, tgt)
    
    
    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            for layers in self.freeze_layers:
                if not isinstance(layers,list):
                    layers = [layers]
                for layer in layers:
                    if layer is None:
                        continue
                    for param in layer.parameters():
                        param.requires_grad=False
        if self.current_epoch < self.unfreeze_start:
            return
        elif (self.current_epoch - self.unfreeze_start) % self.unfreeze_step == 0:
            layer_idx = len(self.freeze_layers) - (self.current_epoch - self.unfreeze_start) // self.unfreeze_step - 1
            if layer_idx < 0:
                return
            layers = self.freeze_layers[layer_idx]

            if not isinstance(layers,list):
                layers = [layers]
            for layer in layers:
                if layer is None:
                    continue
                for param in layer.parameters():
                    param.requires_grad=True
    
    def training_step(self, data, batch_idx):
        alpha = self.good_loss_ratio(self.current_epoch / self.trainer.max_epochs)
        loss = self.get_loss(data)
        loss_weights = data.label * alpha + (1- data.label) * (1-alpha)
        loss = loss * loss_weights.unsqueeze(-1)
        loss = loss.mean()
        self.log("train_loss", loss, sync_dist=True)
        self.log("alpha", alpha, sync_dist=True)
        return loss
            
            
    def generate_samples(self, data):
        sidechains_lists = self.cfom_dock_model.generate_samples(
            self.num_gen_samples*3,
            data.core_tokens,
            data.core_smiles,
            data,
            (data.activity_type, [1] * len(data)),
            **self.gen_meta_params
        )
        # we want to genenerate good samples so we give label=1
        new_mols = []
        for sidechains_list in sidechains_lists:
            sidechains_list = self.tokenizer.decode_batch(sidechains_list, skip_special_tokens=True)
            for core, chains, old_smile, task in zip(data.core_smiles, sidechains_list, data.smiles, data.task):
                new_smile = reconstruct_from_core_and_chains(core, chains)
                if new_smile is None:
                    continue
                    # new_mols.append((task, new_smile, old_smile, None))
                else:
                    old_smile = self.removeChirality(old_smile)
                    new_smile = self.removeChirality(new_smile)
                    new_mols.append((task, new_smile, old_smile, get_fp(new_smile)))
        return new_mols
    
    @staticmethod
    def removeChirality(smiles):
        mol = Chem.MolFromSmiles(smiles)
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol)

    def validation_step(self, graph, batch_idx):
        loss = self.get_loss(graph).mean()
        self.log("val_loss", loss,batch_size=len(graph),  sync_dist=True)
        if not self.validation_clfs:
            return
        gen_res = self.generate_samples(graph)
        for (task_name, new_sm, old_sm, new_fp) in gen_res:
            self.eval_step_outputs[task_name][old_sm].append((new_sm, new_fp))

    def test_step(self, graph, batch_idx):
        if not self.test_clfs:
            return
        
        gen_res = self.generate_samples(graph)
        for (task_name, new_sm, old_sm, new_fp) in gen_res:
            self.eval_step_outputs[task_name][old_sm].append((new_sm, new_fp))
        

    def evaluate_single_task(
        self, task_name, opt_molecules, clf, threshold, similarity_threshold
    ):
        all_success_rates, all_diversities, all_similarities = [], [], []
        all_valid_samples, num_molecules = [], 0
        for old_sm in opt_molecules.keys():
            for new_sm, _ in opt_molecules[old_sm]:
                num_molecules += 1
                if new_sm is not None and new_sm != old_sm:
                    all_valid_samples.append(new_sm)
        for _ in range(self.num_gen_samples):
            chosen_mols, similarities, tot_success = [], [], 0
            for old_sm in opt_molecules.keys():
                candidates = [
                        (new_mol,new_fp)
                        for new_mol, new_fp in opt_molecules[old_sm]
                        if new_mol and new_mol != old_sm
                    ]
                if len(candidates) == 0:
                    continue
                candidates, cand_fps = zip(*candidates)
                chosen_candidate_i = random.randint(0, len(candidates)-1)
                chosen_candidate = candidates[chosen_candidate_i]
                cand_fp = cand_fps[chosen_candidate_i]
                chosen_mols.append(chosen_candidate)
                cur_sim = calc_tani_sim(old_sm, chosen_candidate)
                similarities.append(cur_sim)
                cur_score = clf.predict_proba(np.reshape(cand_fp, (1, -1)))
                cur_score = cur_score[0][1]
                if cur_score > threshold and cur_sim > similarity_threshold:
                    tot_success += 1
            all_success_rates.append(tot_success / len(opt_molecules.keys()))
            all_diversities.append(len(set(chosen_mols)) / max(1, len(chosen_mols)))
            all_similarities.append(sum(similarities) / max(1, len(chosen_mols)))
        avg_diversity, std_diversity = np.mean(all_diversities), np.std(all_diversities)
        avg_similarity, std_similarity = np.mean(all_similarities), np.std(
            all_similarities
        )
        avg_success, std_success = np.mean(all_success_rates), np.std(all_success_rates)
        validity = len(all_valid_samples) / num_molecules
        return (
            validity,
            avg_diversity,
            std_diversity,
            avg_similarity,
            std_similarity,
            avg_success,
            std_success,
        )

    def on_test_end(self):
        for name, opt_molecules in self.eval_step_outputs.items():
            with open(f'{self.test_result_path}-{name}', 'w') as f:
                for orig_mol in opt_molecules:
                    for new_mol,_ in opt_molecules[orig_mol]:
                        f.write(f'{orig_mol} {new_mol}\n')
        
        results = defaultdict(list)
        for task_name in sorted(self.eval_step_outputs.keys()):
            opt_molecules = self.eval_step_outputs[task_name]
            (
                validity,
                avg_diversity,
                std_diversity,
                avg_similarity,
                std_similarity,
                avg_success,
                std_success,
            ) = self.evaluate_single_task(
                task_name,
                opt_molecules,
                self.test_clfs[task_name][0],
                self.test_clfs[task_name][1],
                self.similarity_threshold,
            )
            results['task'].append(task_name)
            results['validity'].append(validity)
            results['diversity'].append(avg_diversity)
            results['std_diversity'].append(std_diversity)
            results['similarity'].append(avg_similarity)
            results['std_similarity'].append(std_similarity)
            results['success'].append(avg_success)
            results['std_success'].append(std_success)
            results['total'].append(sum(map(len, opt_molecules.values())))
        pd.DataFrame(results).to_csv(f'{self.test_result_path}.csv')
        self._reset_eval_step_outputs()

    def on_validation_epoch_end(self):
        tot_avg_success = []
        for task_name, opt_molecules in self.eval_step_outputs.items():
            (
                validity,
                avg_diversity,
                std_diversity,
                avg_similarity,
                std_similarity,
                avg_success,
                std_success,
            ) = self.evaluate_single_task(
                task_name,
                opt_molecules,
                self.validation_clfs[task_name][0],
                self.validation_clfs[task_name][1],
                self.similarity_threshold,
            )
            self.log(f"{task_name}_validity", validity, sync_dist=True)
            self.log(f"{task_name}_diversity", avg_diversity, sync_dist=True)
            self.log(f"{task_name}_std_diversity", std_diversity, sync_dist=True)
            self.log(f"{task_name}_similarity", avg_similarity, sync_dist=True)
            self.log(f"{task_name}_std_similarity", std_similarity, sync_dist=True)
            self.log(f"{task_name}_success", avg_success, sync_dist=True)
            self.log(f"{task_name}_std_success", std_success, sync_dist=True)
            tot_avg_success.append(avg_success)
        self.log("validation_avg_success", sum(tot_avg_success) / (len(tot_avg_success)+1), sync_dist=True)
        self._reset_eval_step_outputs()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, min_lr=self.lr / 100)
        return {
                        "optimizer": optimizer,
                        "lr_scheduler": sched,
                        "monitor": "train_loss"
                    }
