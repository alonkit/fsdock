import scipy.spatial # very important, does not work without it, i don't know why
import resource

from datasets.custom_distributed_sampler import CustomDistributedSampler, CustomTaskDistributedSampler
from datasets.partitioned_fsmol_dock import FsDockDatasetPartitioned
from models.dock_lightning import DockLightning
from models.fs_dock_lightning import FSDockLightning
from models.tasks.task import AtomNumberTask, LabelTask
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import pytorch_lightning as pl
from tokenizers import Tokenizer
import torch
from datasets.fsmol_dock import FsDockDataset
from datasets.fsmol_dock_clf import FsDockClfDataset
from datasets.samplers import TaskSequentialSampler
from datasets.task_data_loader import TaskDataLoader
from models.cfom_dock import CfomDock
from models.cfom_dock_lightning import CfomDockLightning
from models.graph_embedder import GraphEmbedder
from models.graph_encoder import GraphEncoder
from models.interaction_encoder import InteractionEncoder
import datasets.process_chem.features as features
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from models.transformer import TransformerDecoder, TransformerEncoder
from pytorch_lightning.loggers import WandbLogger
import os
from pytorch_lightning.tuner import Tuner
from utils.logging_utils import get_logger

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.manual_seed(0)

def get_model(tokenizer):
    graph_embedder = GraphEmbedder(
        distance_embed_dim=16,
        cross_distance_embed_dim=16,
        lig_max_radius=5,
        rec_max_radius=10,
        cross_max_distance=20,
        lig_feature_dims=features.lig_feature_dims,
        lig_edge_feature_dim=4,
        lig_emb_dim=48,
        rec_feature_dims=features.rec_residue_feature_dims,
        atom_feature_dims=features.rec_atom_feature_dims,
        prot_emd_dim=48,
        dropout=0.1,
        lm_embedding_dim=1280,
    )
    graph_encoder = GraphEncoder(
        in_channels=48,
        edge_channels=48,
        hidden_channels=[48,48,48,48,48,48,48, 48,64],
        out_channels=128,
        attention_groups=8,
        graph_embedder=graph_embedder,
        dropout=0.1,
        max_length=128
    )
    smiles_encoder = TransformerEncoder(
        len(tokenizer.get_vocab()),
        embedding_dim=128,
        hidden_size=128,
        nhead=4,
        n_layers=2,
        max_length=128,
        pad_token=tokenizer.token_to_id("<pad>"),
    )
    sidechain_decoder = TransformerDecoder(len(tokenizer.get_vocab()), embedding_dim=128,
                                            hidden_size=128, nhead=4,
                                            n_layers=2, max_length=128, pad_token=tokenizer.token_to_id("<pad>"),
                                            start_token=tokenizer.token_to_id("<bos>"),
                                            end_token=tokenizer.token_to_id("<eos>"))
    interaction_encoder = InteractionEncoder(128)
    model = CfomDock(None, sidechain_decoder, interaction_encoder, graph_encoder)
    return model

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.sub_proteins.open()   


def test_model(path):
    get_logger().info("Testing model")
    tokenizer = Tokenizer.from_file('models/configs/smiles_tokenizer.json')
    model = get_model(tokenizer)


    dstest = FsDockClfDataset("data/fsdock/clfs/test", "data/fsdock/test_tasks.csv",tokenizer=tokenizer, only_inactive=True, min_roc_auc=0.70)
    dltest = DataLoader(dstest, batch_size=64, 
                         num_workers=torch.get_num_threads()//2, 
                        worker_init_fn=worker_init_fn)
    
    
    lit_model = CfomDockLightning(model, tokenizer, lr=1e-4, weight_decay=1e-4, num_gen_samples=20, test_clfs=dstest.clfs)
    trainer = pl.Trainer(
        max_epochs=100, 
        check_val_every_n_epoch=10,
        strategy='ddp_find_unused_parameters_true')
    trainer.test(lit_model, dltest, ckpt_path=path)

def pretrain_model(full_model, wandb_logger,smol):
    model = full_model.graph_encoder
    wandb_logger.watch(model, log='all')

    
    dock_lit_model = DockLightning(model, lr=1e-4, weight_decay=1e-4, smol=smol)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        monitor="val_noise_loss",
        mode="max",
        dirpath=f"checkpoints/{dock_lit_model.name}/",
        filename= "{epoch:02d}-{val_noise_loss:.5f}",
    )
    trainer = pl.Trainer(
        # num_nodes=2,
        # devices=10,
        max_epochs=200, 
        callbacks=[checkpoint_callback], 
        check_val_every_n_epoch=10,
        strategy='ddp_find_unused_parameters_true',
        logger=wandb_logger)
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(lit_model, mode="binsearch")
    trainer.fit(dock_lit_model)
    
    wandb_logger.experiment.unwatch(model)

def load_finedtuned_graph_encoder(full_model, path):
    model = full_model.graph_encoder

    
    dock_lit_model = DockLightning.load_from_checkpoint(path, graph_encoder_model=model, lr=1e-4, weight_decay=1e-4)

def train_model(smol=False):
    wandb_logger = WandbLogger(project="CfomDockLightning", offline=smol)

    tokenizer = Tokenizer.from_file('models/configs/smiles_tokenizer.json')
    model = get_model(tokenizer)
    
    # load finetuned
    load_finedtuned_graph_encoder(model, '/home/alon.kitin/fs-dock/checkpoints/dock_2025-02-17-19_55_19/epoch=199-val_noise_loss=0.01078.ckpt')
    #pretrain
    # pretrain_model(model, wandb_logger, smol)

    wandb_logger.watch(model, log='all')

    
    cfom_dock_lit_model = CfomDockLightning(model, tokenizer, lr=1e-4, weight_decay=1e-4, num_gen_samples=10, smol=smol)
    # cfom_dock_lit_model = CfomDockLightning.load_from_checkpoint('checkpoints/cfom_dock_2025-02-14-21_05_43/epoch=54-validation_avg_success=0.22468.ckpt',cfom_dock_model=model, tokenizer=tokenizer, lr=1e-4, weight_decay=1e-4, num_gen_samples=10, smol=smol)
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="validation_avg_success",
        mode="max",
        dirpath=f"checkpoints/{cfom_dock_lit_model.name}/",
        filename= "{epoch:02d}-{validation_avg_success:.5f}",
    )
    trainer = pl.Trainer(
        # num_nodes=2,
        num_sanity_val_steps=0,
        devices=1 if smol else 16,
        max_epochs=150, 
        callbacks=[checkpoint_callback], 
        check_val_every_n_epoch=5,
        strategy='ddp_find_unused_parameters_true',
        logger=wandb_logger)
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(lit_model, mode="binsearch")
    trainer.fit(cfom_dock_lit_model)
    
    wandb_logger.experiment.unwatch(model)
    dstest = FsDockClfDataset("data/fsdock/test", "data/fsdock/test_tasks.csv",tokenizer=tokenizer, only_inactive=True, min_roc_auc=0.7)
    dltest = DataLoader(dstest, batch_size=64, 
                         num_workers=torch.get_num_threads()//2, 
                        worker_init_fn=worker_init_fn)
    trainer.test(cfom_dock_lit_model, dltest, ckpt_path="best")

    
def train_fs_model(smol=False):
    wandb_logger = WandbLogger(project="FsDockLightning", offline=smol)

    tokenizer = Tokenizer.from_file('models/configs/smiles_tokenizer.json')
    model = get_model(tokenizer)
    
    # load finetuned
    load_finedtuned_graph_encoder(model, '/home/alon.kitin/fs-dock/checkpoints/dock_2025-02-17-19_55_19/epoch=199-val_noise_loss=0.01078.ckpt')
    #pretrain
    # pretrain_model(model, wandb_logger, smol)
    model = model.graph_encoder
    wandb_logger.watch(model, log='all')

    
    fs_dock_lit_model = FSDockLightning(model, lr=1e-4, weight_decay=1e-4, num_examples=10, smol=smol)
    # cfom_dock_lit_model = CfomDockLightning.load_from_checkpoint('checkpoints/cfom_dock_2025-02-14-21_05_43/epoch=54-validation_avg_success=0.22468.ckpt',cfom_dock_model=model, tokenizer=tokenizer, lr=1e-4, weight_decay=1e-4, num_gen_samples=10, smol=smol)
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="val_roc_auc",
        mode="max",
        dirpath=f"checkpoints/{fs_dock_lit_model.name}/",
        filename= "{epoch:02d}-{validation_avg_success:.5f}",
    )
    trainer = pl.Trainer(
        # num_nodes=2,
        num_sanity_val_steps=0,
        # devices=1 if smol else 16,
        max_epochs=150, 
        callbacks=[checkpoint_callback], 
        # check_val_every_n_epoch=5,
        strategy='ddp_find_unused_parameters_true',
        logger=wandb_logger)
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(lit_model, mode="binsearch")
    trainer.fit(fs_dock_lit_model)
    
    wandb_logger.experiment.unwatch(model)
    # dstest = FsDockClfDataset("data/fsdock/test", "data/fsdock/test_tasks.csv",tokenizer=tokenizer, only_inactive=True, min_roc_auc=0.7)
    # dltest = DataLoader(dstest, batch_size=64, 
    #                      num_workers=torch.get_num_threads()//2, 
    #                     worker_init_fn=worker_init_fn)
    # trainer.test(fs_dock_lit_model, dltest, ckpt_path="best")


# dsv = FsDockDatasetPartitioned(
#                 'data/fsdock/valid',
#                 '../docking_cfom/valid_tasks.csv',
#                         )
# for i in range(16):
#     dlv = DataLoader(dsv, 
#                         sampler=CustomTaskDistributedSampler(dsv, shuffle=True,
#                                         task_size=18, num_replicas=16, rank=i ))
#     for j,b in enumerate(dlv):
#         pass
#     print(len(dlv), i , j)

if __name__ == "__main__":
    # train_model(smol=bool(os.environ.get("SMOL")))
    train_fs_model(smol=bool(os.environ.get("SMOL")))
    # test_model('checkpoints/cfom_dock_2025-02-19-00_15_46/epoch=84-validation_avg_success=0.23056.ckpt')
    