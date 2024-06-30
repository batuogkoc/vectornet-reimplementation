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
    def __init__(self, initial_node_dims, hidden_layer_size, prediction_horizon_size, prediction_vector_size, max_agent_polyline_count=16, max_road_polyline_count=150):
        super().__init__()

        self.max_agent_polyline_count=max_agent_polyline_count
        self.max_road_polyline_count=max_road_polyline_count

        self.prediction_horizon_size = prediction_horizon_size
        self.prediction_vector_size = prediction_vector_size

        self.subgraph_layer_1 = PolylineSubgraphLayer(initial_node_dims, hidden_layer_size)
        self.subgraph_layer_2 = PolylineSubgraphLayer(hidden_layer_size*2, hidden_layer_size)
        self.subgraph_layer_3 = PolylineSubgraphLayer(hidden_layer_size*2, hidden_layer_size)

        self.aggregate = nn.AdaptiveMaxPool1d(1)
        self.attention = nn.MultiheadAttention(hidden_layer_size*2, 1, batch_first=True)
        self.trajectory_decoder = nn.Linear(hidden_layer_size*2, prediction_horizon_size*prediction_vector_size)

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

        # output_mask = torch.broadcast_to(torch.cat((agent_mask, torch.zeros_like(road_mask)), dim=-1).unsqueeze(-1), x.shape)
        # x[~output_mask] = 0
        x = self.trajectory_decoder(x[:,0])
        # x[~torch.any(output_mask, dim=-1)] = 0
        x = torch.unflatten(x, -1, (self.prediction_horizon_size, self.prediction_vector_size))
        return x

def calculate_metrics(y, y_pred):
    with torch.no_grad():
        diff = y-y_pred

        displacement = torch.sqrt(torch.sum(diff**2, axis=-1))

        return {
            "ade": torch.mean(displacement).item(),
            "de_1": torch.mean(displacement[:, 9]).item(),
            "de_2": torch.mean(displacement[:, 19]).item(),
            "de_3": torch.mean(displacement[:, 29]).item(),
        }



if __name__ == "__main__":
    KUACC=True
    RECORD=True
    if KUACC:
        print("-"*10 + "~KUACC~" + "-"*10)
        device = torch.device("cuda")
        H5_FOLDER_PATH = "../h5_files"

    else:
        torch.manual_seed(42)
        device = torch.device("cpu")
        H5_FOLDER_PATH = "../AutoBots/h5_files"
    # torch.autograd.anomaly_mode.set_detect_anomaly(True)

    train_set = ArgoverseVectornetDataset(os.path.join(H5_FOLDER_PATH, "train_dataset.hdf5"))
    test_set = ArgoverseVectornetDataset(os.path.join(H5_FOLDER_PATH, "val_dataset.hdf5"))

    print(len(train_set))
    print(len(test_set))
    NUM_WORKERS = 1
    SHUFFLE = False
    train_loader = DataLoader(train_set, batch_size=64, num_workers=NUM_WORKERS, shuffle=SHUFFLE)
    test_loader = DataLoader(test_set, batch_size=64, num_workers=NUM_WORKERS, shuffle=SHUFFLE)

    START_EPOCH = 0

    model = VectorNet(6, 64, 30, 2).to(device)
    
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.3)
    
    # LOAD_PROGRESS_PATH = "runs/2024-06-20T15:23:33/e-8-train_l-813.9820446734548-test_loss-869.9771018757278.pt"
    LOAD_PROGRESS_PATH = None
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
        scheduler.load_state_dict(state["scheduler_state_dict"])
        print(state.keys())
    else:
        EXPERIMENT_DATE_TIME = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        CHECKPOINT_FOLDER = f"runs/{EXPERIMENT_DATE_TIME}"

    if RECORD:
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            running_average_training_loss = train_loss/max((i), 1)

            if i % 20 == 0 and i != 0:
                fraction_done = max(i/len(train_loader), 1e-6)
                time_taken = (time.time()-start)
                print(f"i: {i}| loss: {loss} | ratl: {running_average_training_loss}")
                print(f"{fraction_done*100}% | est time left: {time_taken*(1-fraction_done)/fraction_done} s | est total: {time_taken/fraction_done} s")
                if RECORD:
                    writer.add_scalar("running_average_training_loss", running_average_training_loss, epoch*len(train_loader) + i)
                
            if i%500 == 0 and i!=0 and RECORD:
                torch.save({
                    "epoch": epoch,
                    "epoch_progress": i,
                    "epoch_size": len(train_loader),
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "running_average_training_loss": running_average_training_loss,
                }, os.path.join(CHECKPOINT_FOLDER, f"e-{epoch}-i-{i}-mbtl-{loss.item()}-ratl-{running_average_training_loss}.pt"))

        train_loss /= len(train_loader)

        test_loss = 0
        metrics = {
            "ade": 0,
            "de_1": 0,
            "de_2": 0,
            "de_3": 0,
        }
        with torch.inference_mode():
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)

                model.eval()
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                test_loss += loss.item()

                batch_metrics = calculate_metrics(y, y_pred)
                metrics["ade"] += batch_metrics["ade"]
                metrics["de_1"] += batch_metrics["de_1"]
                metrics["de_2"] += batch_metrics["de_2"]
                metrics["de_3"] += batch_metrics["de_3"]

        test_loss /= len(test_loader)
        metrics["ade"] /= len(test_loader)
        metrics["de_1"] /= len(test_loader)
        metrics["de_2"] /= len(test_loader)
        metrics["de_3"] /= len(test_loader)

        print(f"train loss: {train_loss} test loss: {test_loss}")
        print(f"metrics: {metrics}")
        scheduler.step()

        if RECORD:
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("test_loss", test_loss, epoch)
            writer.add_scalar("ade", metrics["ade"], epoch)
            writer.add_scalar("de_1", metrics["de_1"], epoch)
            writer.add_scalar("de_2", metrics["de_2"], epoch)
            writer.add_scalar("de_3", metrics["de_3"], epoch)

            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "training_loss": train_loss,
                "test_loss": test_loss,
                "metrics": metrics,
            }, os.path.join(CHECKPOINT_FOLDER, f"e-{epoch}-train_l-{train_loss}-test_l-{test_loss}-ade-{metrics['ade']}.pt"))
        


