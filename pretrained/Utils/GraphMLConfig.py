import torch

class GraphMLConfig:
    debug = True
    project_path = "./"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    # Parameters for Build Dataset Loader
    batch_size = 128
    random_seed = 42
    shuffle = True
    drop_last = True # It has to be true
    graphMetric_method = 'smiles'
    dataset_file = '/home/zhengyjo/M3-KMGCL-ZZ/Utils/dataset_nmrshiftdb2_modified_1204.csv'
    graphs_path = "/home/zhengyjo/../../scratch0/zhengyjo/nmrshiftdb2/graph_hyb/"
    surface_path = '/home/zhengyjo/../../scratch0/zhengyjo/nmrshiftdb2/fingerprint/'
    smiles_path= '/home/zhengyjo/../../scratch0/zhengyjo/nmrshiftdb2/smiles/'

    # Parameters for training
    lr = 1e-3
    weight_decay = 1e-3
    patience = 2
    factor = 0.5
    epochs = 200

    # alpha * mr_loss + (1-alpha) * element_loss
    alpha = 0.5

    # Parameters for 3d graph Model
    graph3d_embedding = 128
    graph3d_pretrained = False
    graph3d_trainable = True

    # Parameters for 2d graph Model
    graph2d_pretrained = True
    graph2d_trainable = False

    # Parameters for Projection
    num_projection_layers = 1
    projection_dim = [300]
    dropout = 0.1

