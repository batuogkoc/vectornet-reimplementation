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

        #handle road
        raw_road_pts = torch.Tensor(self.f["road_pts"][index])
        
        road_polyline_mask = ~(torch.all(torch.flatten(torch.abs(raw_road_pts)<1e-6, -2,-1), dim=-1))
        road_polyline_count = torch.sum(road_polyline_mask, dim=-1)
        road_start_pts = raw_road_pts[:,1:,:2]
        road_end_pts = raw_road_pts[:,:-1,:2]
        
        road_polyline = torch.cat((road_start_pts, road_end_pts), dim=-1)

        # #handle ego
        # raw_ego_trajectory = torch.unsqueeze(torch.Tensor(hd5_file["ego_trajectories"][index]), 0)
        
        # ego_start_pts = raw_ego_trajectory[:,1:,:2]
        # ego_end_pts = raw_ego_trajectory[:,:-1,:2]

        # ego_polyline = torch.cat((ego_start_pts, ego_end_pts), dim=-1)
        
        agent_polyline_indices = torch.arange(0, agent_polyline_count)
        road_polyline_indices = torch.arange(agent_polyline_count, agent_polyline_count + road_polyline_count)

        #concat the indices to the agent polylines
        agent_polyline = torch.cat((agent_polyline, torch.zeros((*agent_polyline.shape[:-1], 2))), dim=-1)
        temp = agent_polyline[agent_polyline_mask]
        temp[:,:,-1] = torch.unsqueeze(agent_polyline_indices, dim=-1)
        agent_polyline[agent_polyline_mask] = temp

        #concat the indices to the road polylines
        road_polyline = torch.cat((road_polyline, torch.zeros((*road_polyline.shape[:-1], 2))), dim=-1)
        temp = road_polyline[road_polyline_mask]
        temp[:,:,-1] = torch.unsqueeze(road_polyline_indices, dim=-1)
        road_polyline[road_polyline_mask] = temp
        agent_x = agent_polyline[:,:-1,:]
        agent_y = agent_polyline[:,-1,:] 

        road_x = road_polyline[:,:-1,:]
        road_y = road_polyline[:,-1,:]

        #pad road polyline with zeros so it fits the max size of the array
        road_x = torch.cat((road_x, torch.zeros(road_x.shape[0], agent_x.shape[1] - road_x.shape[1], road_x.shape[2])), dim=1)

        ret_x = torch.cat((agent_x, road_x))
        ret_y = torch.cat((agent_y, road_y))

        return ret_x.clone(), ret_y.clone()

if __name__ == "__main__":
    # FILE_PATH = "../AutoBots/h5_files/test_dataset.hdf5"
    FILE_PATH = "../AutoBots/h5_files/train_dataset.hdf5"

    dataset = ArgoverseVectornetDataset(FILE_PATH)
    # # for i in range(1):
    x, y = dataset[1]

    print(x.shape)
    print(y.shape)

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