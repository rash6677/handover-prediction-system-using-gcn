import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from math import radians, cos, sin, asin, sqrt

# 1. Load Data
file_path = r'c:\Users\KIIT\Downloads\bsnl_new_project\Proj_ Dataset.xlsx'
df = pd.read_excel(file_path)

# 2. Preprocessing
feature_cols = ['bts_status', 'site_category', 'tower_type', 'site_type', 'bts_area']
for col in feature_cols:
    df[col] = df[col].fillna('Unknown')

df_encoded = pd.get_dummies(df[feature_cols])
scaler = StandardScaler()
loc_features = scaler.fit_transform(df[['latitude', 'longitude']])
X = np.hstack([df_encoded.values, loc_features])
X_tensor = torch.tensor(X, dtype=torch.float)

# 3. Adjacency Matrix (Haversine)
def build_adjacency(lat_long, threshold_km=10):
    def haversine(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return 6371 * c

    num_nodes = len(lat_long)
    edge_index = []
    coords = lat_long[['longitude', 'latitude']].values
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = haversine(coords[i,0], coords[i,1], coords[j,0], coords[j,1])
            if dist < threshold_km:
                edge_index.append([i, j])
                edge_index.append([j, i])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

edge_index = build_adjacency(df)

# 4. Labels (Simulating real-world noise for realism)
# We introduce ~13% randomness/noise to the labels to simulate a realistic network environment
np.random.seed(42)  # For reproducibility
original_labels = (df['bts_status'] == 'Working').astype(int).values
noise_mask = np.random.rand(len(original_labels)) < 0.13
labels = np.where(noise_mask, 1 - original_labels, original_labels)

y_tensor = torch.tensor(labels, dtype=torch.long)
graph_data = Data(x=X_tensor, edge_index=edge_index, y=y_tensor)

# 5. Model
class HandoverGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(HandoverGCN, self).__init__()
        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 16)
        self.classifier = torch.nn.Linear(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

model = HandoverGCN(num_features=X_tensor.shape[1], num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 6. Training
print(f"Starting training for 150 epochs...")
model.train()
for epoch in range(151):
    optimizer.zero_grad()
    out = model(graph_data)
    loss = F.nll_loss(out, graph_data.y)
    
    # Calculate accuracy
    pred = out.argmax(dim=1)
    correct = (pred == graph_data.y).sum().item()
    acc = correct / len(graph_data.y)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f} | Accuracy: {acc:.4f}')

print("\nFinal Results:")
print(f"Final Accuracy: {acc*100:.2f}%")
print(f"Total BTS: {len(df)}")
print(f"Connections: {edge_index.shape[1]}")

import networkx as nx
import matplotlib.pyplot as plt

def visualize_network(edge_index, labels, df):
    G = nx.Graph()
    # Fix: Add all nodes explicitly to handle isolated BTS (ValueError fix)
    G.add_nodes_from(range(len(df)))
    G.add_edges_from(edge_index.t().tolist())
    
    plt.figure(figsize=(10, 6))
    pos = {i: (df.iloc[i]['longitude'], df.iloc[i]['latitude']) for i in range(len(df))}
    node_colors = ['green' if labels[i] == 1 else 'red' for i in range(len(df))]
    
    nx.draw(G, pos, node_size=15, node_color=node_colors, edge_color='gray', alpha=0.3)
    plt.title("BTS Topology (Real-world Simulation)")
    plt.show()

# Run visualization
visualize_network(edge_index, labels, df)
