import scipy.spatial # very important, does not work without it, i don't know why
from tokenizers import Tokenizer
from torch_geometric.transforms import ToUndirected
from torch_geometric.data import collate
from datasets.fsmol_dock_grouped import GFsDockDataset
from datasets.untasked_fsmol_dock import UntaskedFsDockDataset
from models.cfom_dock import CfomDock
from models.graph_embedder import GraphEmbedder
from models.graph_encoder import GraphEncoder
from models.interaction_encoder import InteractionEncoder
from models.layers.point_graph_transformer_conv import PGHTConv
import datasets.features as features
from torch_geometric.loader import DataLoader

from models.transformer import TransformerDecoder, TransformerEncoder

tokenizer = Tokenizer.from_file('models/configs/smiles_tokenizer.json')
ds = GFsDockDataset('data/fsdock/valid','../docking_cfom/valid_tasks.csv', tokenizer=tokenizer)
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
model.to('cuda')
for data in dl:
    data = data['graphs'][0]
    data = data.to('cuda')
    y = model(data.core_tokens, data.sidechain_tokens, data, (data.activity_type, data.label))
    print(y)

