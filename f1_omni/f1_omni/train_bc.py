import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import ast
import torch.nn as nn
import torch.optim as optim


class CarDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scan_data = self.data.iloc[idx, 0]  # First column: data_x (LiDAR scan)
        action_raw = self.data.iloc[idx, 1]  # Second column: data_y (steering, speed)

        # Convert scan data from string to list of floats
        scan_data = torch.tensor(eval(scan_data), dtype=torch.float32)

        # Convert action data from string to tensor
        action = torch.tensor(eval(action_raw), dtype=torch.float32)

        return scan_data, action


# Load dataset
dataset = CarDataset("dataset.csv")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



class BCModel(nn.Module):
    def __init__(self, input_size, output_size=2):
        super(BCModel, self).__init__()
        self.fc = nn.Sequential(
        nn.Linear(input_size, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, output_size)  # Output: [steering_angle, velocity]
)
    def forward(self, x):
        return self.fc(x)

# Initialize model
input_size = len(dataset[0][0])  # Length of scan data
model = BCModel(input_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train model
for epoch in range(10):
    for scan, action in dataloader:
        optimizer.zero_grad()
        predicted_action = model(scan)
        loss = criterion(predicted_action, action)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

torch.save(model.state_dict(), "bc_pretrained.pth")
print("model saved!")
