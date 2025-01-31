import scipy.spatial # very important, does not work without it, i don't know why
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

from utils.logging_utils import get_logger

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.manual_seed(0)

def get_model(tokenizer):
    graph_embedder = GraphEmbedder(
        distance_embed_dim=16,
        cross_distance_embed_dim=16,
        lig_max_radius=5,
        rec_max_radius=10,
        cross_max_distance=10,
        lig_feature_dims=features.lig_feature_dims,
        lig_edge_feature_dim=4,
        lig_emb_dim=16,
        rec_feature_dims=features.rec_residue_feature_dims,
        atom_feature_dims=features.rec_atom_feature_dims,
        prot_emd_dim=16,
        dropout=0.1,
        lm_embedding_dim=1280,
    )
    graph_encoder = GraphEncoder(
        in_channels=16,
        edge_channels=16,
        hidden_channels=[32,64],
        out_channels=128,
        attention_groups=4,
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
    model = CfomDock(smiles_encoder, sidechain_decoder, interaction_encoder, graph_encoder)
    return model

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.sub_proteins.open()   


def test_model():
    get_logger().info("Testing model")
    tokenizer = Tokenizer.from_file('models/configs/smiles_tokenizer.json')
    model = get_model(tokenizer)


    dst = FsDockClfDataset("data/fsdock/test", "data/fsdock/test_tasks.csv",tokenizer=tokenizer, only_inactive=True, min_roc_auc=0.7)
    dlt = DataLoader(dst, batch_size=64, 
                         num_workers=torch.get_num_threads()//2, 
                        worker_init_fn=worker_init_fn)
    
    
    lit_model = CfomDockLightning(model, tokenizer, lr=1e-4, weight_decay=1e-4, num_gen_samples=10, test_clfs=dst.clfs)
    trainer = pl.Trainer(
        max_epochs=100, 
        check_val_every_n_epoch=10,
        strategy='ddp_find_unused_parameters_true')
    trainer.test(lit_model, dlt, ckpt_path="checkpoints/cfom-dock-epoch=79-validation_loss=0.00000.ckpt")

def train_model():
    wandb_logger = WandbLogger(project="CfomDockLightning", offline=False)

    tokenizer = Tokenizer.from_file('models/configs/smiles_tokenizer.json')
    model = get_model(tokenizer)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="validation_avg_success",
        mode="max",
        dirpath="checkpoints/",
        filename="cfom-dock-{epoch:02d}-{validation_avg_success:.5f}",
    )

    # dst = FsDockDataset("data/fsdock/valid", "data/fsdock/train_tasks.csv",tokenizer=tokenizer)
    dst = FsDockDataset("data/fsdock/train", "data/fsdock/train_tasks.csv",tokenizer=tokenizer)
    dlt = DataLoader(dst, batch_size=64, shuffle=True, num_workers=torch.get_num_threads(), 
                    worker_init_fn=worker_init_fn)

    dsv = FsDockClfDataset("data/fsdock/valid", "data/fsdock/valid_tasks.csv",tokenizer=tokenizer, only_inactive=True)
    dlv = DataLoader(dsv, batch_size=64, 
                         num_workers=torch.get_num_threads()//2, 
                        worker_init_fn=worker_init_fn)
    
    
    lit_model = CfomDockLightning(model, tokenizer, lr=1e-4, weight_decay=1e-4, num_gen_samples=10, validation_clfs=dsv.clfs)
    trainer = pl.Trainer(
        max_epochs=100, 
        callbacks=[checkpoint_callback], 
        check_val_every_n_epoch=10,
        strategy='ddp_find_unused_parameters_true',
        logger=wandb_logger)
    trainer.fit(lit_model, dlt, dlv)


if __name__ == "__main__":
    # train_model()
    test_model()
    