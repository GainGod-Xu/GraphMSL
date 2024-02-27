import pandas as pd
from Utils.GraphMLConfig import GraphMLConfig
from Utils.BuildDatasetLoader import build_dataset_loader
from GraphMLModels.GraphMLModel import GraphMLModel
import torch
from Utils.TrainEpoch import train_epoch

from chemprop.args import TrainArgs
from chemprop.constants import TEST_SCORES_FILE_NAME, TRAIN_LOGGER_NAME
from chemprop.data.utils import get_data_cl
from chemprop.data import get_data,get_task_names, MoleculeDataset, validate_dataset_type,MoleculeDataLoader
from chemprop.utils import create_logger, makedirs, timeit, multitask_mean
from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim, set_explicit_h, set_adding_hs, set_keeping_atom_map, set_reaction, reset_featurization_parameters
from chemprop.models import MoleculeModel,mpn
from dig.threedgraph.method import SphereNet

import os
import argparse
# Create the parser and add arguments
parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, default='gin', help="Model type.")
parser.add_argument("--num_layer", type=int, default=5, help="Number of layers.")
parser.add_argument("--embed_dim", type=int, default=128, help="Embed dimension.")
parser.add_argument("--path", type=str, default='/home/zhengyjo/M3-KMGCL-ZZ/Utils/dataset_nmrshiftdb2_smiles_modified_1204.pkl', help="smiles file")
parser.add_argument("--graphMetric", type=str, default='smiles', help="graphMetric")
parser.add_argument("--alpha", type=float, default=0.5, help="alpha")
# Parse and print the results
args = parser.parse_args()

def main():
    # Create the parser and add arguments
    out_name = "best_" + args.graphMetric + "_alpha_" + str(args.alpha) + "_chemprop"
    print('Output name:%s' % out_name)
    
    GraphMLConfig.graphMetric_method = args.graphMetric
    GraphMLConfig.alpha = args.alpha
    
    arguments = [
    '--data_path', args.path,
    '--dataset_type', 'kmgcl',
    '--smiles_columns','smiles',
    ]
    
    pass_args=TrainArgs().parse_args(arguments)

    # Generate train_dataset_loader and valid_dataset_loader
    train_dataset_loader = build_dataset_loader(GraphMLConfig, pass_args)

    # graph_model
    graph2d_model = mpn.MPN(pass_args)
    graph3d_model = SphereNet(energy_and_force=False,
                              cutoff=5.0,
                              num_layers=4,
                              hidden_channels=128,
                              out_channels=128, # graph Embedding dim
                              int_emb_size=64,
                              basis_emb_size_dist=8,
                              basis_emb_size_angle=8,
                              basis_emb_size_torsion=8,
                              out_emb_channels=256,
                              num_spherical=3,
                              num_radial=6,
                              envelope_exponent=5,
                              num_before_skip=1,
                              num_after_skip=2,
                              num_output_layers=3)

    device = GraphMLConfig.device
    model = GraphMLModel(graph3d_model=graph3d_model,
                         graph2d_model=graph2d_model,
                         config=GraphMLConfig).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=GraphMLConfig.lr, weight_decay=GraphMLConfig.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=GraphMLConfig.patience, factor=GraphMLConfig.factor
    )

    step = "epoch"

    best_loss = float('inf')
    for epoch in range(GraphMLConfig.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_dataset_loader, optimizer, lr_scheduler, step, GraphMLConfig.accuracies_req)

        if train_loss.avg < best_loss:
            best_loss = train_loss.avg
            torch.save(model.state_dict(), './' + out_name + ".pt")
            print("Saved Best Model!")

        print("\n")



if __name__ == "__main__":
    main()



