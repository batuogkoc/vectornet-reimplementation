import torch

print("a")
PATH = "runs/2024-06-24T22:33:53/e-24-train_l-3.702771568053846-test_l-3.018788269891538-ade-1.5461550773252737.pt"
state = torch.load(PATH)
print(state.keys())
print(state["metrics"])
print(state["training_loss"])
print(state["test_loss"])