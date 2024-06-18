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

class ArgoverseVectornetDataset(Dataset):
    def __init__(self, hd5_file_path):
        self.f = h5py.File(hd5_file_path, "r")

    def __len__(self):
        return len(self.f["ego_trajectories"])
    
    def __getitem__(self, index) -> torch.Tensor:
        #handle agents
        raw_agent_trajectory = torch.Tensor(self.f["agents_trajectories"][index])
        raw_agent_trajectory = torch.movedim(raw_agent_trajectory, -2, -3)

        #since ego is treated as another agent, append it to the start of the agents
        raw_ego_trajectory = torch.unsqueeze(torch.Tensor(self.f["ego_trajectories"][index]), 0)
        raw_agent_trajectory = torch.cat((raw_ego_trajectory, raw_agent_trajectory), dim=0)
        agent_polyline_mask = ~(torch.all(torch.flatten(torch.abs(raw_agent_trajectory)<1e-6, -2,-1), dim=-1))
        agent_polyline_count = torch.sum(agent_polyline_mask, dim=-1)
        agent_start_pts = raw_agent_trajectory[:,1:,:2]
        agent_end_pts = raw_agent_trajectory[:,:-1,:2]

        agent_polyline = torch.cat((agent_start_pts, agent_end_pts), dim=-1)
        agent_zero_vector_mask = torch.all(torch.abs(agent_polyline)<1e-6, dim=-1)

        #handle road
        raw_road_pts = torch.Tensor(self.f["road_pts"][index])

        road_polyline_mask = ~(torch.all(torch.flatten(torch.abs(raw_road_pts)<1e-6, -2,-1), dim=-1))
        road_polyline_count = torch.sum(road_polyline_mask, dim=-1)
        road_start_pts = raw_road_pts[:,1:,:2]
        road_end_pts = raw_road_pts[:,:-1,:2]
        
        road_polyline = torch.cat((road_start_pts, road_end_pts), dim=-1)
        road_zero_vector_mask = torch.all(torch.abs(road_polyline)<1e-6, dim=-1)

        #indices
        agent_polyline_indices = torch.arange(0, agent_polyline_count)
        road_polyline_indices = torch.arange(agent_polyline_count, agent_polyline_count + road_polyline_count)

        #concat the indices to the agent polylines
        agent_polyline = torch.cat((agent_polyline, torch.zeros((*agent_polyline.shape[:-1], 2))), dim=-1)
        temp = agent_polyline[agent_polyline_mask]
        temp[:,:,-1] = torch.unsqueeze(agent_polyline_indices, dim=-1)
        agent_polyline[agent_polyline_mask] = temp
        agent_polyline[agent_zero_vector_mask] = 0

        #concat the indices to the road polylines
        road_polyline = torch.cat((road_polyline, torch.zeros((*road_polyline.shape[:-1], 2))), dim=-1)
        temp = road_polyline[road_polyline_mask]
        temp[:,:,-1] = torch.unsqueeze(road_polyline_indices, dim=-1)
        road_polyline[road_polyline_mask] = temp
        road_polyline[road_zero_vector_mask] = 0

        #get the mask for the transition point from nonzero to zero vector
        agent_last_item_mask = ~agent_zero_vector_mask & torch.roll(agent_zero_vector_mask, shifts=-1, dims=-1)
        #since all-nonzero polylines dont have a transition, add a true value regardless 
        temp = agent_last_item_mask[agent_polyline_mask ^ torch.any(agent_last_item_mask, dim=-1)]
        temp[:,-1] = True
        agent_last_item_mask[agent_polyline_mask ^ torch.any(agent_last_item_mask, dim=-1)] = temp
        #for empty polylines, denote the first element as the y index
        temp = agent_last_item_mask[~agent_polyline_mask]
        temp[:,0] = True
        agent_last_item_mask[~agent_polyline_mask] = temp
        assert torch.all(torch.sum(agent_last_item_mask, dim=-1) == 1), f"There must be exactly 1 final element for each polyline, got {torch.sum(agent_last_item_mask, dim=-1)}"
        
        agent_x = torch.unflatten(agent_polyline[~agent_last_item_mask], dim=0, sizes=(agent_polyline.shape[0], -1))
        agent_y = agent_polyline[agent_last_item_mask]

        road_x = road_polyline

        #pad road polyline with zeros so it fits the max size of the array
        road_x = torch.cat((road_x, torch.zeros(road_x.shape[0], agent_x.shape[1] - road_x.shape[1], road_x.shape[2])), dim=1)

        ret_x = torch.cat((agent_x, road_x))
        ret_y = torch.cat((agent_y, torch.zeros(road_polyline.shape[0], road_polyline.shape[-1])))

        return ret_x.clone(), ret_y.clone()

if __name__ == "__main__":
    # FILE_PATH = "../AutoBots/h5_files/test_dataset.hdf5"
    FILE_PATH = "../AutoBots/h5_files/train_dataset.hdf5"

    dataset = ArgoverseVectornetDataset(FILE_PATH)
    # # for i in range(1):
    x, y = dataset[0]

    # print(x.shape)
    # print(y.shape)
    print(torch.sum(x, dim=(-2, -1)))
    print(torch.sum(y, dim=-1))
    # print(y[:5])
    # print(torch.all(abs(x) < 1e-6, dim=-1)[:17])

    # loader = DataLoader(dataset, 64, shuffle=True, num_workers=8)
    # device = torch.device("cpu")
    # print(len(dataset))
    # for epoch in range(10):
    #     print(epoch)
    #     n = 0
    #     start = time.time()
    #     for i, (x,y) in enumerate(loader):
    #         if n % 100 == 0:
    #             print((i/(len(loader)))*100)
    #         n += 1
    #     print(time.time()-start)
    #     print(n)