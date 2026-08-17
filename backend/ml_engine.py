import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from captum.attr import IntegratedGradients
import numpy as np
import joblib

class WindAdvectionGNN(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels):
        # "add" aggregation to sum incoming pollution advected by wind
        super(WindAdvectionGNN, self).__init__(aggr='add') 
        
        self.lin_node = torch.nn.Linear(in_channels, 16)
        self.lin_edge = torch.nn.Linear(edge_channels, 16)
        self.lin_msg = torch.nn.Linear(16 + 16, 16)
        self.lin_update = torch.nn.Linear(16 + 16, 16)
        self.final_lin = torch.nn.Linear(16, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x1 = F.relu(self.lin_node(x))
        x1 = self.propagate(edge_index, x=x1, edge_attr=edge_attr)
        x2 = self.propagate(edge_index, x=x1, edge_attr=edge_attr)
        return self.final_lin(x2)

    def message(self, x_j, edge_attr):
        # Modulate the hidden state passed from node j to node i based on wind
        edge_emb = self.lin_edge(edge_attr)
        msg_input = torch.cat([x_j, edge_emb], dim=1)
        return F.relu(self.lin_msg(msg_input))
        
    def update(self, aggr_out, x):
        update_input = torch.cat([x, aggr_out], dim=1)
        return F.relu(self.lin_update(update_input))


class AeroTwinMLEngine:
    def __init__(self, model_path="gnn_model.pth", scaler_path="feature_scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        # Force CPU if MPS isn't fully supported for MessagePassing or captum.
        # Captum's IntegratedGradients sometimes has issues with MPS gradients.
        self.device = torch.device("cpu")
        self.model = WindAdvectionGNN(in_channels=3, out_channels=1, edge_channels=2).to(self.device)
        self.feature_names = ["PM2.5_Lag", "Temp_Inversion", "CO"]
        
        self.train_or_load_model()
        self.load_scaler()

    def load_scaler(self):
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            print(f"Loaded feature scaler from {self.scaler_path}")
        else:
            self.scaler = None
            print("No feature scaler found, will use raw scaling.")

    def train_or_load_model(self):
        if os.path.exists(self.model_path):
            print(f"Loading ST-GNN from {self.model_path}...")
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
        else:
            print("Training new PyTorch Geometric Wind Advection GNN...")
            num_nodes = 5
            edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j], dtype=torch.long).t().contiguous()
            num_edges = edge_index.size(1)
            
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
            criterion = torch.nn.MSELoss()
            
            self.model.train()
            for epoch in range(50):
                optimizer.zero_grad()
                x = torch.rand((num_nodes, 3)).to(self.device)
                edge_attr = torch.rand((num_edges, 2)).to(self.device)
                # Synthetic target calculation
                y = (x[:, 0] * 50 + x[:, 1] * 10 + x[:, 2] * 5 + torch.randn(num_nodes)).view(-1, 1).to(self.device)
                
                out = self.model(x, edge_index, edge_attr)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                
            torch.save(self.model.state_dict(), self.model_path)
            print(f"ST-GNN saved to {self.model_path}")
            self.model.eval()

    def build_live_graph(self, live_node_data: list) -> Data:
        num_nodes = len(live_node_data)
        x_list = []
        for n in live_node_data:
            raw_co = n.get("live_co", 1.5)
            raw_pm25 = n["pm25"]
            raw_temp = n.get("live_temp", 25.0)
            
            if self.scaler:
                # Scaler expects ['co', 'no2', 'o3', 'pm25', 'so2']
                # Create a dummy row to transform just CO and PM2.5
                dummy = np.array([[raw_co, 0, 0, raw_pm25, 0]])
                scaled = self.scaler.transform(dummy)[0]
                scaled_co = max(0, min(1, scaled[0]))
                scaled_pm25 = max(0, min(1, scaled[3]))
            else:
                scaled_co = raw_co
                scaled_pm25 = raw_pm25 * 0.9
                
            x_list.append([scaled_pm25, raw_temp / 50.0, scaled_co])
        x = torch.tensor(x_list, dtype=torch.float)
        
        edge_list, edge_attr_list = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_list.append([i, j])
                    dist = ((live_node_data[j]["lat"] - live_node_data[i]["lat"])**2 + 
                            (live_node_data[j]["lon"] - live_node_data[i]["lon"])**2)**0.5
                    avg_wind = (live_node_data[i]["live_wind"] + live_node_data[j]["live_wind"]) / 2.0
                    edge_attr_list.append([avg_wind, dist])
                    
        return Data(
            x=x, 
            edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous(), 
            edge_attr=torch.tensor(edge_attr_list, dtype=torch.float)
        ).to(self.device)

    def predict_with_xai(self, live_node_data: list) -> dict:
        data = self.build_live_graph(live_node_data)
        
        # 1. Forward pass
        with torch.no_grad():
            preds = self.model(data.x, data.edge_index, data.edge_attr)
        
        # 2. XAI with Captum
        def model_forward(node_features):
            return self.model(node_features, data.edge_index, data.edge_attr)
            
        ig = IntegratedGradients(model_forward)
        
        # Calculate attribution (target=0 because out_channels=1 for PM2.5 prediction)
        attr, _ = ig.attribute(data.x, target=0, return_convergence_delta=True)
        total_attribs = torch.abs(attr)
            
        mean_attribs = torch.mean(total_attribs, dim=0).detach().cpu().numpy()
        total_impact = mean_attribs.sum()
        
        if total_impact > 0:
            percentages = (mean_attribs / total_impact) * 100
        else:
            percentages = np.zeros_like(mean_attribs)
            
        # 3. Scale predictions to 1.2x - 1.5x of baseline (Live Data)
        raw_preds = preds.view(-1).cpu().numpy()
        min_pred, max_pred = raw_preds.min(), raw_preds.max()
        
        scaled_preds = []
        for i, node in enumerate(live_node_data):
            baseline = node["pm25"]
            if max_pred > min_pred:
                # Normalize raw GNN prediction between 0 and 1
                norm = (raw_preds[i] - min_pred) / (max_pred - min_pred)
                # Map to a multiplier between 1.2x and 1.5x
                multiplier = 1.2 + (norm * 0.3)
            else:
                multiplier = 1.35
            
            scaled_preds.append(round(baseline * float(multiplier), 1))
            
        return {
            "predictions": scaled_preds,
            "xai_attributions": {feat: round(float(pct), 1) for feat, pct in zip(self.feature_names, percentages)}
        }

if __name__ == "__main__":
    engine = AeroTwinMLEngine(model_path="/Users/alfayez/Desktop/AeroTwin/backend/gnn_model.pth")
