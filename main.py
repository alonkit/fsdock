import random
import sys
import scipy.spatial # very important, does not work without it, i don't know why
import resource

import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from pytorch_lightning.tuner import Tuner
from utils.logging_utils import get_logger
from hydra.utils import instantiate, get_class
from utils.omega_utils import load_config
import torch.multiprocessing as mp

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.manual_seed(0)
random.seed(0)

torch.multiprocessing.set_sharing_strategy('file_system')
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
# mp.set_start_method("spawn", force=True)


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # dataset.sub_proteins.open()   


def pretrain_model(full_model,config, wandb_logger,smol):
    model = full_model.graph_encoder
    # wandb_logger.watch(model, log='all')

    dock_lit_model = instantiate(config.pretrain.lightning, graph_encoder_model=model, smol=smol)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="valid_loss/dataloader_idx_0",
        mode="min",
        dirpath=f'{config.metadata.experiment_folder}/checkpoints/{type(dock_lit_model).__name__}/',
        filename= "{valid_loss:.5f}-{epoch:02d}",
    )
    trainer = instantiate(config.pretrain.trainer, logger=wandb_logger, callbacks=[checkpoint_callback])


    dst = instantiate(config.pretrain.train_dataset)
    if "sampler" in config.pretrain:
        sampler = instantiate(config.pretrain.sampler, dataset=dst)
    else:
        sampler = None
    dlt = DataLoader(dst, batch_size=config.pretrain.batch_size, 
                        sampler=sampler,
                        num_workers=torch.get_num_threads())
    
    dsv = instantiate(config.pretrain.val_dataset)
    dlv = DataLoader(dsv, batch_size=config.pretrain.batch_size,
                num_workers=torch.get_num_threads()//2)


    trainer.fit(dock_lit_model, 
                train_dataloaders=dlt, 
                val_dataloaders=dlv)
    
    # wandb_logger.experiment.unwatch(model)

def load_pretrained_graph_encoder(full_model, config):
    model = full_model.graph_encoder

    cls = get_class(config.load_pretrained.lightning._target_)
    dock_lit_model = cls.load_from_checkpoint(config.load_pretrained.path, graph_encoder_model=model, lr=1e-4, weight_decay=1e-4)

def train_model(config, smol=False, ckpt=None):
    metadata = config.metadata
    wandb_logger = instantiate(config.logger)

    tokenizer = instantiate(config.tokenizer)
    model = instantiate(config.model)
    
    assert not ("load_pretrained" in config and "pretrain" in config), "Cannot load pretrained and pretrain at the same time, choose one of them"
    if "load_pretrained" in config:
    # load finetuned
    # '/home/alon.kitin/fs-dock/checkpoints/dock_2025-04-14-22_58_32/epoch=99-val_noise_loss=0.01746.ckpt'
        load_pretrained_graph_encoder(model, config)
    elif 'pretrain' in config:
        pretrain_model(model,config, wandb_logger, smol)

    # wandb_logger.watch(model, log='all')

    lit_model = instantiate(config.lightning, model=model, name=metadata.name)
    lit_model.test_result_path = f'{metadata.experiment_folder}/test_results/{type(lit_model).__name__}/'
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        monitor="valid_loss/dataloader_idx_0",
        mode="min",
        save_on_train_epoch_end=True,
        dirpath=f'{config.metadata.experiment_folder}/checkpoints/{type(lit_model).__name__}/',
        filename= "{valid_loss/dataloader_idx_0:.5f}_{epoch:02d}",
    )
    trainer : pl.Trainer = instantiate(config.train.trainer, logger=wandb_logger, callbacks=[checkpoint_callback])

    dst = instantiate(config.train.train_dataset, tokenizer=tokenizer
                      )
    
    if "sampler" in config.train:
        trainer.strategy.setup_environment()
        sampler = instantiate(config.train.sampler, dataset=dst)
    else:
        sampler = None
    dlt = DataLoader(dst, batch_size=config.train.batch_size, 
                        sampler=sampler,
                        num_workers=min(torch.get_num_threads(),5), 
                        worker_init_fn=worker_init_fn)
    
    dsv = instantiate(config.train.val_dataset, tokenizer=tokenizer)
    dlv = DataLoader(dsv, batch_size=config.train.batch_size,
                # num_workers=torch.get_num_threads()//2, 
                worker_init_fn=worker_init_fn)
    
    dsv_corrupt = instantiate(config.train.corrupt_val_dataset, tokenizer=tokenizer)
    dlv_corrupt = DataLoader(dsv_corrupt, batch_size=config.train.batch_size,
                # num_workers=torch.get_num_threads()//2, 
                worker_init_fn=worker_init_fn)
    
    
    if isinstance(ckpt,str) and len(ckpt)==0:
        ckpt = None
    print('ckpt=',ckpt)
    try:
        lit_model.validation_clfs=dsv.clfs
    except:
        lit_model.validation_clfs=None
    trainer.fit(lit_model, 
                train_dataloaders=dlt, 
                val_dataloaders=[dlv, dlv_corrupt],
                ckpt_path=ckpt)
    

if __name__ == "__main__":
    config= load_config(config_path=sys.argv[1] if len(sys.argv) > 1 else 'config.yaml')
    ckpt = sys.argv[2] if len(sys.argv) > 2 else None
    train_model(smol=bool(os.environ.get("SMOL")), config=config, ckpt=ckpt)
    # train_fs_model(smol=bool(os.environ.get("SMOL")))
    # test_model('/home/alon.kitin/fs-dock/checkpoints/cfom_dock_2025-03-09-07_36_09/epoch=49-validation_avg_success=0.20919.ckpt')
    