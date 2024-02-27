from torch import nn
import torch.nn.functional as F
from GraphMLModels.Encoder import Encoder
from GraphMLModels.Projection import Projection
from torchmetrics.functional import pairwise_cosine_similarity
import torch

class GraphMLModel(nn.Module):
    def __init__(
        self,
        graph3d_model,
        graph2d_model,
        config,
    ):
        super().__init__()
        self.graph2d_encoder = Encoder(model=graph2d_model, trainable=False)
        self.graph3d_encoder = Encoder(model=graph3d_model, trainable=True)
        self.projection_graph = Projection(embedding_dim=1, projection_dim=config.projection_dim, projection_dropout=config.dropout)
        self.projection_surface = Projection(embedding_dim=1, projection_dim=config.projection_dim, projection_dropout=config.dropout)
        self.decomposed_MahMatrix = nn.Parameter(torch.eye(input_dim=config.graph3d_emb_dim)) #denoted as W in our manuscirpts
        self.perturb_matrix = config.perturb_factor * torch.eye(input_dim=config.graph3d_emb_dim) #denoted as alpha * I in our manuscirpts
        self.alpha = config.alpha
        self.device = config.device

    def forward(self, batch):

        # GraphEmbedding & NodeEmbedding
        graph3dEmbedding = self.graph3d_encoder(batch['graph3d'])

        # Compute xi - xj
        xi_minus_xj = graph3dEmbedding.unsqueeze(1) - graph3dEmbedding.unsqueeze(0)
        # Compute the transpose of xi - xj
        xi_minus_xj_transpose = xi_minus_xj.transpose(1, 0)
        # Compute determinant of W to ensure its non-singular matrix
        if torch.det(self.decomposed_MahMatrix):
            self.decomposed_MahMatrix = self.decomposed_MahMatrix + self.perturb_matrix
        # Compute the outer product of W and its transpose
        WW_transpose = torch.matmul(self.decomposed_MahMatrix, self.decomposed_MahMatrix.transpose(1, 0))
        # Compute the expression (xi - xj)^T · (W W^T) · (xi - xj)
        graph3dMeric  = torch.einsum('ijk,kl,ijl->ij', xi_minus_xj_transpose, WW_transpose, xi_minus_xj)
        graph3dMeric  =  F.softmax(graph3dMeric, dim=-1)

        # compute Logits
        graph2dLogits = self.projection_graph(graph3dMeric)
        surfaceLogits = self.projection_surface(graph3dMeric)

        # compute Labels
        graph2d = batch['smiles_input'].batch_graph()
        graph2dLabels = self.compute_graph2d_metric(graph2d)
        surfaceLabels = self.compute_surface_metric(batch['surface'])

        #compute loss
        graph2d_loss = F.cross_entropy(graph2dLogits, graph2dLabels)
        surface_loss = F.cross_entropy(surfaceLogits, surfaceLabels)
        loss = graph2d_loss + surface_loss
        return loss, graph2d_loss, surface_loss

    def compute_surface_metric(surface):
        with torch.no_grad():
            # Compute the mean of the data over the entire dataset
            mean = torch.mean(surface, dim=0, keepdim=True)

            # Compute the deviations from the mean
            deviations = surface - mean

            # Compute the covariance matrix
            covariance_matrix = torch.matmul(deviations.t(), deviations) / (surface.shape[0] - 1)

            # Compute the inverse covariance matrix
            inv_covariance_matrix = torch.inverse(covariance_matrix)

            # Compute xi - xj
            xi_minus_xj = surface.unsqueeze(1) - surface.unsqueeze(0)

            # Compute the squared Mahalanobis distance
            surface_metric = torch.matmul(torch.matmul(xi_minus_xj, inv_covariance_matrix),
                                          xi_minus_xj.permute(0, 2, 1))
            surface_metric = F.softmax(surface_metric, dim=-1)
        return surface_metric

    def compute_graph2d_metric(self, graph):
        with torch.no_grad():
            graph2dEmbedding = self.graph2d_encoder(graph)
            #graph2d_metric = pairwise_cosine_similarity(graph2dEmbedding,graph2dEmbedding)
            # Compute the mean of the data over the entire dataset
            mean = torch.mean(graph2dEmbedding, dim=0, keepdim=True)

            # Compute the deviations from the mean
            deviations = graph2dEmbedding - mean

            # Compute the covariance matrix
            covariance_matrix = torch.matmul(deviations.t(), deviations) / (graph2dEmbedding.shape[0] - 1)

            # Compute the inverse covariance matrix
            inv_covariance_matrix = torch.inverse(covariance_matrix)

            # Compute xi - xj
            xi_minus_xj = graph2dEmbedding.unsqueeze(1) - graph2dEmbedding.unsqueeze(0)

            # Compute the squared Mahalanobis distance
            graph2d_metric = torch.matmul(torch.matmul(xi_minus_xj, inv_covariance_matrix),
                                          xi_minus_xj.permute(0, 2, 1))
            graph2d_metric = F.softmax(graph2d_metric, dim=-1)
        return graph2d_metric



