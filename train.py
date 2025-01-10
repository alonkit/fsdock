import scipy.spatial # very important, does not work without it, i don't know why
import pytorch_lightning as pl
from tokenizers import Tokenizer
from datasets.untasked_fsmol_dock import UntaskedFsDockDataset
from models.cfom_dock import CfomDock
from models.cfom_dock_lightning import CfomDockLightning
from models.graph_embedder import GraphEmbedder
from models.graph_encoder import GraphEncoder
from models.interaction_encoder import InteractionEncoder
import datasets.features as features
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from models.transformer import TransformerDecoder, TransformerEncoder
def train_model():

    tokenizer = Tokenizer.from_file('models/configs/smiles_tokenizer.json')
    ds = UntaskedFsDockDataset("data/fsdock/smol", "data/tasks_smol.csv",tokenizer=tokenizer, hide_sidechains=True)
    dl = DataLoader(ds, batch_size=4, shuffle=True)
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
        hidden_channels=[32, 64, 64],
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
                                            n_layers=2, max_length=128, pad_token=tokenizer.token_to_id("<pad>"))
    interaction_encoder = InteractionEncoder(128)
    model = CfomDock(smiles_encoder, sidechain_decoder, interaction_encoder, graph_encoder)
    
    lit_model = CfomDockLightning(model, lr=1e-4, weight_decay=1e-4)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="val_loss",
        mode="min",
        dirpath="checkpoints/",
        filename="cfom-dock-{epoch:02d}-{val_loss:.2f}",
    )
    trainer = pl.Trainer(max_epochs=100, callbacks=[checkpoint_callback])
    

    trainer.fit(lit_model, dl, dl)


if __name__ == "__main__":
    train_model()
# Example usage:
# train_texts = ["example sentence 1", "example sentence 2"]
# train_labels = [0, 1]
# val_texts = ["example validation sentence 1", "example validation sentence 2"]
# val_labels = [0, 1]
# train_model(train_texts, train_labels, val_texts, val_labels)