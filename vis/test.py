import h5py
import matplotlib.pyplot as plt

FILE_PATH = "../AutoBots/h5_files/test_dataset.hdf5"

with h5py.File(FILE_PATH, "r") as hd5_file:
    print(list(hd5_file.keys()))
    print(hd5_file["agents_trajectories"].shape)
    print(hd5_file["agents_trajectories"].dtype)
