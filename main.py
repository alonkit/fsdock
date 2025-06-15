import scipy.spatial # very important, does not work without it, i don't know why
import resource

from datasets.custom_distributed_sampler import CustomDistributedSampler, CustomTaskDistributedSampler
from datasets.partitioned_fsmol_dock import FsDockDatasetPartitioned
from models.cfom_dock_ablation import CfomDockAblation
from models.dock_lightning import DockLightning
from models.fs_dock_lightning import FSDockLightning
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
from hydra.utils import instantiate, get_class
from utils.omega_utils import load_config


os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.manual_seed(0)

ABLATION = True

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
        dropout=0.3,
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
        tokenizer,
        embedding_dim=128,
        hidden_size=128,
        nhead=4,
        n_layers=2,
        max_length=128,
    )
    sidechain_decoder = TransformerDecoder(tokenizer, embedding_dim=304,
                                            hidden_size=128, nhead=4,
                                            n_layers=2, max_length=128)
    interaction_encoder = InteractionEncoder(304)
    if ABLATION:
        model = CfomDockAblation(None, sidechain_decoder, interaction_encoder, graph_encoder)
    else:
        model = CfomDock(None, sidechain_decoder, interaction_encoder, graph_encoder)
        
    return model

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.sub_proteins.open()   


def pretrain_model(full_model,config, wandb_logger,smol):
    model = full_model.graph_encoder
    # wandb_logger.watch(model, log='all')

    dock_lit_model = instantiate(config.pretrain.lightning, graph_encoder_model=model, smol=smol)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        monitor="val_noise_loss",
        mode="max",
        dirpath=f'{config.metadata.experiment_folder}/checkpoints/{type(dock_lit_model).__name__}/',
        filename= "{val_noise_loss:.5f}-{epoch:02d}",
    )
    trainer = instantiate(config.pretrain.trainer, logger=wandb_logger, callbacks=[checkpoint_callback])


    dst = instantiate(config.pretrain.train_dataset)
    if "sampler" in config.pretrain:
        sampler = instantiate(config.pretrain.sampler, dataset=dst)
    else:
        sampler = None
    dlt = DataLoader(dst, batch_size=config.pretrain.batch_size, 
                        sampler=sampler,
                        num_workers=torch.get_num_threads(), 
                        worker_init_fn=worker_init_fn)
    
    dsv = instantiate(config.pretrain.val_dataset)
    dlv = DataLoader(dsv, batch_size=config.pretrain.batch_size,
                num_workers=torch.get_num_threads()//2, 
                worker_init_fn=worker_init_fn)


    trainer.fit(dock_lit_model, 
                train_dataloaders=dlt, 
                val_dataloaders=dlv)
    
    # wandb_logger.experiment.unwatch(model)

def load_pretrained_graph_encoder(full_model, config):
    model = full_model.graph_encoder

    cls = get_class(config.load_pretrained.lightning._target_)
    dock_lit_model = cls.load_from_checkpoint(config.load_pretrained.path, graph_encoder_model=model, lr=1e-4, weight_decay=1e-4)

def train_model(config, smol=False):
    metadata = config.metadata
    wandb_logger = instantiate(config.logger)

    if 'tokenizer' in config:
        tokenizer = Tokenizer.from_file(config.tokenizer.path)
    else:
        tokenizer = Tokenizer.from_file('models/configs/smiles_tokenizer.json') 
    model = instantiate(config.model)
    
    assert not ("load_pretrained" in config and "pretrain" in config), "Cannot load pretrained and pretrain at the same time, choose one of them"
    if "load_pretrained" in config:
    # load finetuned
    # '/home/alon.kitin/fs-dock/checkpoints/dock_2025-04-14-22_58_32/epoch=99-val_noise_loss=0.01746.ckpt'
        load_pretrained_graph_encoder(model, config)
    elif 'pretrain' in config:
        pretrain_model(model,config, wandb_logger, smol)

    # wandb_logger.watch(model, log='all')

    lit_model = instantiate(config.lightning, model=model, tokenizer=tokenizer, smol=smol, name=metadata.name)
    lit_model.test_result_path = f'{metadata.experiment_folder}/test_results/{type(lit_model).__name__}/'
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="validation_avg_success",
        mode="max",
        dirpath=f'{config.metadata.experiment_folder}/checkpoints/{type(lit_model).__name__}/',
        filename= "{validation_avg_success:.5f}_{epoch:02d}",
    )
    trainer = instantiate(config.train.trainer, logger=wandb_logger, callbacks=[checkpoint_callback])

    dst = instantiate(config.train.train_dataset, tokenizer=tokenizer
                      )
    
    if "sampler" in config.train:
        trainer.strategy.setup_environment()
        sampler = instantiate(config.train.sampler, dataset=dst)
    else:
        sampler = None
    dlt = DataLoader(dst, batch_size=config.train.batch_size, 
                        sampler=sampler,
                        num_workers=torch.get_num_threads(), 
                        worker_init_fn=worker_init_fn)
    
    dsv = instantiate(config.train.val_dataset, tokenizer=tokenizer)
    dlv = DataLoader(dsv, batch_size=config.train.batch_size,
                num_workers=torch.get_num_threads()//2, 
                worker_init_fn=worker_init_fn)
    
    lit_model.validation_clfs=dsv.clfs
    trainer.fit(lit_model, 
                train_dataloaders=dlt, 
                val_dataloaders=dlv)
    

if __name__ == "__main__":
    config= load_config()
    train_model(smol=bool(os.environ.get("SMOL")), config=config)
    # train_fs_model(smol=bool(os.environ.get("SMOL")))
    # test_model('/home/alon.kitin/fs-dock/checkpoints/cfom_dock_2025-03-09-07_36_09/epoch=49-validation_avg_success=0.20919.ckpt')
    