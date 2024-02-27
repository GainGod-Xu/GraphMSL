from DatasetModels.GraphMLDataset import GraphMLDataset
from DatasetModels.GraphDataset import GraphDataset
from DatasetModels.SurfaceDataset import SurfaceDataset
from torch.utils.data import DataLoader

from chemprop.args import TrainArgs
from chemprop.constants import TEST_SCORES_FILE_NAME, TRAIN_LOGGER_NAME
from chemprop.data.utils import get_data_cl
from chemprop.data import get_data,get_task_names, MoleculeDataset, validate_dataset_type,MoleculeDataLoader
from chemprop.utils import create_logger, makedirs, timeit, multitask_mean
from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim, set_explicit_h, set_adding_hs, set_keeping_atom_map, set_reaction, reset_featurization_parameters

import pandas as pd
import os

def build_dataset_loader(config,pass_args):

    dataframe = pd.read_csv(config.dataset_file)
    # Zhengyang's modification to use chemprop Molecule dataset
    
    smile_input_dataset = get_data_cl(path=pass_args.data_path,
        args=pass_args,smiles_columns= ['smiles']) 
    # Zhengyang's modification end
    
    graph_dataset = GraphDataset(dataframe['graph'], config)
    surface_dataset = SurfaceDataset(dataframe['surface'], config)

    # Create CGIPDataset instance
    dataset = GraphMLDataset(graph3d_dataset=graph_dataset, surface_dataset=surface_dataset, smile_input_dataset=smile_input_dataset, config=config)
    dataset_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle, drop_last=config.drop_last, collate_fn=dataset.collate_fn)

    return dataset_loader

