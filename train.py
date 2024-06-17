import h5py
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets
import torchvision.transforms as transforms
import time
import os
from vectornet_argoverse_dataset import *

class PolylineSubgraphLayer(nn.Module):
    def __init__(self, node_dims, max_agent_polyline_count=16, max_road_polyline_count=150):
        super().__init__()
        
        self.max_agent_polyline_count=max_agent_polyline_count
        self.max_road_polyline_count=max_road_polyline_count

        self.encoder = nn.Linear(node_dims, node_dims)
        nn.init.xavier_uniform_(self.encoder.weight)

        self.aggregate = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, x: torch.Tensor, road_polyline_length=8, agent_polyline_length=None):
        N = x.shape[0]
        agent_polyline_length = agent_polyline_length if agent_polyline_length is not None else x.shape[-2]
        nonzero_vectors_mask = ~torch.all((torch.abs(x) < 1e-6), dim=-1)
        # nonzero_polyline_mask = torch.any(nonzero_vectors_mask, dim=-1)
        # agent_polyline_count = torch.sum(nonzero_polyline_mask[:,:self.max_agent_polyline_count], dim=-1)
        # road_polyline_count = torch.sum(nonzero_polyline_mask[:,self.max_agent_polyline_count:], dim=-1)

        x = self.encoder(x)
        x = F.relu(x)

        agent_polylines = x[:,:self.max_agent_polyline_count,:agent_polyline_length]
        road_polylines = x[:,self.max_agent_polyline_count:,:road_polyline_length]

        agent_aggregate = torch.unflatten(torch.movedim(self.aggregate(torch.movedim(torch.flatten(agent_polylines, 0,1), -2, -1)), -1, -2), 0, (N,-1))
        road_aggregate = torch.unflatten(torch.movedim(self.aggregate(torch.movedim(torch.flatten(road_polylines, 0,1), -2, -1)), -1, -2), 0, (N,-1))
        aggregate = torch.cat((agent_aggregate, road_aggregate), -3)
        x = torch.cat((x, torch.broadcast_to(aggregate, x.shape)), axis=-1)
        
        x[~nonzero_vectors_mask] = 0
        return x
        
# class GlobalInteractionGraph(nn.Module):

class VectorNet(nn.Module):
    def __init__(self, initial_node_dims, max_agent_polyline_count=16, max_road_polyline_count=150):
        super().__init__()

        self.max_agent_polyline_count=max_agent_polyline_count
        self.max_road_polyline_count=max_road_polyline_count

        self.subgraph_layer_1 = PolylineSubgraphLayer(initial_node_dims*1, max_agent_polyline_count, max_road_polyline_count)
        self.subgraph_layer_2 = PolylineSubgraphLayer(initial_node_dims*2, max_agent_polyline_count, max_road_polyline_count)
        self.subgraph_layer_3 = PolylineSubgraphLayer(initial_node_dims*4, max_agent_polyline_count, max_road_polyline_count)

    def forward(self, x: torch.Tensor, road_polyline_length=8, agent_polyline_length=None)->torch.Tensor:
        x = self.subgraph_layer_1(x, road_polyline_length=road_polyline_length, agent_polyline_length=agent_polyline_length)
        x = self.subgraph_layer_2(x, road_polyline_length=road_polyline_length, agent_polyline_length=agent_polyline_length)
        x = self.subgraph_layer_3(x, road_polyline_length=road_polyline_length, agent_polyline_length=agent_polyline_length)


if __name__ == "__main__":
    torch.manual_seed(42)
    H5_FOLDER_PATH = "../AutoBots/h5_files"

    training_set = ArgoverseVectornetDataset(os.path.join(H5_FOLDER_PATH, "train_dataset.hdf5"))
    test_set = ArgoverseVectornetDataset(os.path.join(H5_FOLDER_PATH, "test_dataset.hdf5"))

    print(len(training_set))
    print(len(test_set))

    training_loader = DataLoader(training_set, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64)
    model = PolylineSubgraphLayer(6)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(1):
        model.train()
        print("-"*5 + f"EPOCH: {epoch}" + "-"*5)

        for i, (x, y) in enumerate(test_loader):
            model.train()
            y_pred = model(x)
            break
        #     loss = loss_fn(y_pred, y)

        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        # with torch.inference_mode():
        #     for i, (x, y) in enumerate(test_loader):
        #         model.train()
        #         y_pred = model(x)
        


