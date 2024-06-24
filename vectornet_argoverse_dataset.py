import h5py
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time

class ArgoverseVectornetDataset(Dataset):
    def __init__(self, hd5_file_path, target_prediction_horizon=30):
        self.f = h5py.File(hd5_file_path, "r")
        self.target_prediction_horizon = 30

    def __len__(self):
        return len(self.f["ego_trajectories"])
    
    def __getitem__(self, index) -> torch.Tensor:
        #handle agents
        raw_agent_trajectory = torch.Tensor(self.f["agents_trajectories"][index])
        raw_agent_trajectory = torch.movedim(raw_agent_trajectory, -2, -3)

        #since ego is treated as another agent, append it to the start of the agents
        raw_ego_trajectory = torch.unsqueeze(torch.Tensor(self.f["ego_trajectories"][index]), 0)
        raw_ego_trajectory[abs(raw_ego_trajectory)<1e-6] = 2e-6
        raw_agent_trajectory = torch.cat((raw_ego_trajectory, raw_agent_trajectory), dim=0)
        agent_polyline_mask = ~(torch.all(torch.flatten(torch.abs(raw_agent_trajectory)<1e-6, -2,-1), dim=-1))
        agent_polyline_count = torch.sum(agent_polyline_mask, dim=-1)
        agent_start_pts = raw_agent_trajectory[:,:-1,:2]
        agent_end_pts = raw_agent_trajectory[:,1:,:2]

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
        assert torch.sum(torch.all(abs(agent_polyline[0]) < 1e-6, dim=-1), dim=-1) == 0, f"The target vehicle must have all elems populated. Instead, it has {torch.sum(torch.all(abs(agent_polyline[0]) < 1e-6, dim=-1), dim=-1)} empty vectors"

        # agent_x = torch.unflatten(agent_polyline[~agent_last_item_mask], dim=0, sizes=(agent_polyline.shape[0], -1))
        # agent_y = agent_polyline[agent_last_item_mask]
        agent_x = agent_polyline.clone()
        agent_y = agent_polyline[0,-self.target_prediction_horizon:]
        agent_x[0,-self.target_prediction_horizon:] = 0        


        road_x = road_polyline

        #pad road polyline with zeros so it fits the max size of the array
        road_x = torch.cat((road_x, torch.zeros(road_x.shape[0], agent_x.shape[1] - road_x.shape[1], road_x.shape[2])), dim=1)
        

        ret_x = torch.cat((agent_x, road_x))
        ret_y = agent_y[:,2:4]

        assert torch.sum(torch.all(abs(ret_x) < 1e-6, dim=-1), dim=-1)[0] == self.target_prediction_horizon, "The prediction points for target aren't masked correctly"
        assert ret_y.shape[0] == self.target_prediction_horizon, "The y size doesnt match the target prediction horizon"

        return ret_x.clone(), ret_y.clone()

class CachedArgoverseVectornetDataset():
    pass

def WIP_cache_dataset(dataset):
    X = []
    Y = []
    start = time.time()

    with h5py.File("cached_train_dataset.hdf5", "w") as f:
        X = f.create_dataset("X", (len(dataset), *dataset[0][0].shape), dtype=float)
        Y = f.create_dataset("Y", (len(dataset), *dataset[0][1].shape), dtype=float)
        for i, (x, y) in enumerate(dataset):
            # x = torch.unsqueeze(x, dim=0)
            # y = torch.unsqueeze(y, dim=0)
            # if X is None:
            #     X = x
            #     Y = y
            #     continue
            X[i] = x
            Y[i] = y
            if i%100==0 and i!=0:
                fraction_done = i/len(dataset)
                time_taken = time.time()-start
                print(f"{i} | {len(dataset)} | {fraction_done*100}% | time taken: {time_taken} | eta: {time_taken/fraction_done*(1-fraction_done)} | total est: {time_taken/fraction_done}")
                print(X.shape, Y.shape)
            # X = torch.cat((X, x), dim=0)
            # Y = torch.cat((Y, y), dim=0)
        
    

if __name__ == "__main__":
    FILE_PATH = "../AutoBots/h5_files/train_dataset.hdf5"
    # FILE_PATH = "../h5_files/train_dataset.hdf5"
    dataset = ArgoverseVectornetDataset(FILE_PATH)

    x,y = dataset[2]

    print(x.shape)
    print(y.shape)
