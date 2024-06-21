import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time
from datetime import datetime
import os
from vectornet_argoverse_dataset import *
from torch.utils.tensorboard import SummaryWriter

class PolylineSubgraphLayer(nn.Module):
    def __init__(self, input_node_dim, hidden_layer_dim):
        super().__init__()

        self.encoder = nn.Linear(input_node_dim, hidden_layer_dim)
        nn.init.xavier_uniform_(self.encoder.weight)
        #TODO: check if the correct number of dimentions is goven
        #i.e. shall we only normalize over embedding vectors or should we do it over polylines as well?
        self.layer_norm = nn.LayerNorm(hidden_layer_dim)

        self.aggregate = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, x: torch.Tensor):
        N = x.shape[0]
        nonzero_vectors_mask = ~torch.all((torch.abs(x) < 1e-6), dim=-1)

        x = self.encoder(x)
        x = self.layer_norm(x)
        x = F.relu(x)

        #NOTE: Important hotfix: if you dont call clone, the gradient calculations fail. (MAY BE KILLING GRADIENT FLOW!!)
        x = x.clone()
        x[~nonzero_vectors_mask] = float("-inf")

        aggregate = torch.unflatten(torch.movedim(self.aggregate(torch.movedim(torch.flatten(x, 0,1), -2, -1)), -1, -2), 0, (N,-1))

        x = torch.cat((x, torch.broadcast_to(aggregate, x.shape)), axis=-1)
        
        x[~nonzero_vectors_mask] = 0
        return x
        

class VectorNet(nn.Module):
    def __init__(self, initial_node_dims, max_agent_polyline_count=16, max_road_polyline_count=150):
        super().__init__()

        self.max_agent_polyline_count=max_agent_polyline_count
        self.max_road_polyline_count=max_road_polyline_count

        self.subgraph_layer_1 = PolylineSubgraphLayer(initial_node_dims, 64)
        self.subgraph_layer_2 = PolylineSubgraphLayer(128, 64)
        self.subgraph_layer_3 = PolylineSubgraphLayer(128, 64)

        self.aggregate = nn.AdaptiveMaxPool1d(1)
        self.attention = nn.MultiheadAttention(128, 1, batch_first=True)
        self.trajectory_decoder = nn.Linear(128, initial_node_dims)

    def forward(self, x: torch.Tensor, agent_polyline_length=None)->torch.Tensor:
        N = x.shape[0]

        x = self.subgraph_layer_1(x)
        x = self.subgraph_layer_2(x)
        x = self.subgraph_layer_3(x)

        x_zero_vector_mask = torch.all(torch.abs(x)<1e-6, dim=-1)
        x[x_zero_vector_mask] = float("-inf")
        x = torch.unflatten(torch.movedim(self.aggregate(torch.movedim(torch.flatten(x, 0,1), -2, -1)), -1, -2), 0, (N,-1))
        x = torch.squeeze(x)
        x[torch.all(x_zero_vector_mask, dim=-1)] = 0

        agent_polyline_length = agent_polyline_length if agent_polyline_length is not None else x.shape[-2]
        nonzero_polyline_mask = ~torch.all((torch.abs(x) < 1e-6), dim=-1)
        agent_polyline_count = torch.sum(nonzero_polyline_mask[:,:self.max_agent_polyline_count], dim=-1)
        road_polyline_count = torch.sum(nonzero_polyline_mask[:,self.max_agent_polyline_count:], dim=-1)
        
        agent_range_tensor = torch.arange(self.max_agent_polyline_count, device=x.device)
        agent_mask = (agent_range_tensor.unsqueeze(0) < agent_polyline_count.unsqueeze(-1))

        road_range_tensor = torch.arange(self.max_road_polyline_count, device=x.device)
        road_mask = (road_range_tensor.unsqueeze(0) < road_polyline_count.unsqueeze(-1))

        mask_1d = torch.cat((agent_mask, road_mask), dim=-1)
        mask = mask_1d.unsqueeze(-1) & mask_1d.unsqueeze(-2)

        #NOTE: Important hotfix: let the unused outputs attend to whatever they want. If you dont allow an output
        # to attend to anything it outputs nan which messes up gradient calculations
        mask[~mask_1d] = True
        x, _ = self.attention(x,x,x, attn_mask=(~mask))

        output_mask = torch.broadcast_to(torch.cat((agent_mask, torch.zeros_like(road_mask)), dim=-1).unsqueeze(-1), x.shape)
        x[~output_mask] = 0
        x = self.trajectory_decoder(x)
        x[~torch.any(output_mask, dim=-1)] = 0
        return x
    
if __name__ == "__main__":
    # torch.manual_seed(42)
    # torch.autograd.anomaly_mode.set_detect_anomaly(True)
    # device = torch.device("cpu")
    device = torch.device("cuda")

    H5_FOLDER_PATH = "../h5_files"

    train_set = ArgoverseVectornetDataset(os.path.join(H5_FOLDER_PATH, "train_dataset.hdf5"))
    test_set = ArgoverseVectornetDataset(os.path.join(H5_FOLDER_PATH, "test_dataset.hdf5"))

    print(len(train_set))
    print(len(test_set))
    NUM_WORKERS = 2
    train_loader = DataLoader(train_set, batch_size=64, num_workers=NUM_WORKERS, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, num_workers=NUM_WORKERS, shuffle=True)

    START_EPOCH = 0

    model = VectorNet(6).to(device)
    
    loss_fn = nn.MSELoss(reduction="sum")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.3)
    
    LOAD_PROGRESS_PATH = "runs/2024-06-20T15:23:33/e-8-train_l-813.9820446734548-test_loss-869.9771018757278.pt"
    if LOAD_PROGRESS_PATH:
        CHECKPOINT_FOLDER, _ = os.path.split(LOAD_PROGRESS_PATH)
        path_components = os.path.normpath(LOAD_PROGRESS_PATH).split(os.sep)
        EXPERIMENT_DATE_TIME = path_components[1]
        state = torch.load(LOAD_PROGRESS_PATH, map_location=device)
        if "epoch" in state and not "epoch_progress" in state:
            START_EPOCH = state["epoch"] + 1
        else:
            assert False, "Must start progress from a finished epoch"
        
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optim_state_dict"])
        print(state.keys())
    else:
        EXPERIMENT_DATE_TIME = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        CHECKPOINT_FOLDER = f"runs/{EXPERIMENT_DATE_TIME}"

    writer = SummaryWriter(f'runs_tensorboard/{EXPERIMENT_DATE_TIME}')
    os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

    for epoch in range(START_EPOCH, 25):
        print("-"*5 + f"EPOCH: {epoch}" + "-"*5)
        start = time.time()
        train_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            model.train()
            y_pred = model(x)

            loss = loss_fn(y_pred, y)
            a = loss_fn(y_pred, y).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            running_average_training_loss = train_loss/max((i*train_loader.batch_size), 1)
            if i % 10 == 0 and i != 0:
                writer.add_scalar("running_average_training_loss", running_average_training_loss, epoch*len(train_loader) + i)
                fraction_done = max(i/len(train_loader), 1e-6)
                time_taken = (time.time()-start)
                print(f"i: {i}| loss: {loss} | ratl: {running_average_training_loss}")
                print(f"{fraction_done*100}% | est time left: {time_taken*(1-fraction_done)/fraction_done} s | est total: {time_taken/fraction_done} s")
            if i%100 == 0 and i!=0:
                torch.save({
                    "epoch": epoch,
                    "epoch_progress": i,
                    "epoch_size": len(train_loader),
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "running_average_training_loss": running_average_training_loss ,
                }, os.path.join(CHECKPOINT_FOLDER, f"e-{epoch}-i-{i}-mbtl-{loss.item()}-ratl-{running_average_training_loss}.pt"))

        train_loss /= len(train_loader.dataset)

        test_loss = 0
        with torch.inference_mode():
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)

                model.eval()
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                test_loss += loss.item()

        test_loss /= len(test_loader.dataset)

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("test_loss", test_loss, epoch)

        print(f"train loss: {train_loss} test loss: {test_loss}")
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "training_loss": train_loss,
            "test_loss": test_loss,
        }, os.path.join(CHECKPOINT_FOLDER, f"e-{epoch}-train_l-{train_loss}-test_loss-{test_loss}.pt"))

        scheduler.step()
        


