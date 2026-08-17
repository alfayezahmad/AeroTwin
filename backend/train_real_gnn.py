import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import requests
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from ml_engine import WindAdvectionGNN
import time
import math

def haversine(lat1, lon1, lat2, lon2):
    # Calculate distance between two points in km
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def main():
    print("Loading data...")
    # Load ML_Lucknow.csv
    csv_path = "../data/ML Lucknow.csv"
    if not os.path.exists(csv_path):
        csv_path = "data/ML Lucknow.csv" # fallback if run from root
        if not os.path.exists(csv_path):
            csv_path = "/Users/alfayez/Desktop/AeroTwin/data/ML Lucknow.csv"
    
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df = df.sort_values('date').reset_index(drop=True)
    
    # 1. Scale Features
    features = ['co', 'no2', 'o3', 'pm25', 'so2']
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Save scaler
    scaler_path = "/Users/alfayez/Desktop/AeroTwin/backend/feature_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # 2. Historical Weather Enrichment
    stations = [
        {"name": "Talkatora", "lat": 26.8315, "lon": 80.8992},
        {"name": "Lalbagh", "lat": 26.8467, "lon": 80.9462},
        {"name": "Gomti Nagar", "lat": 26.8500, "lon": 80.9980},
        {"name": "Alambagh", "lat": 26.8150, "lon": 80.9020},
        {"name": "Kalyanpur", "lat": 26.9020, "lon": 80.9450}
    ]
    
    start_date = df['date'].min().strftime('%Y-%m-%d')
    end_date = df['date'].max().strftime('%Y-%m-%d')
    
    print(f"Fetching weather data from {start_date} to {end_date}...")
    station_weather = {}
    for st in stations:
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={st['lat']}&longitude={st['lon']}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_mean,wind_speed_10m_max,wind_direction_10m_dominant&timezone=auto"
        res = requests.get(url).json()
        
        # Parse into a dict by date
        daily = res.get('daily', {})
        times = daily.get('time', [])
        temps = daily.get('temperature_2m_mean', [])
        winds = daily.get('wind_speed_10m_max', [])
        dirs = daily.get('wind_direction_10m_dominant', [])
        
        weather_dict = {}
        for i in range(len(times)):
            weather_dict[times[i]] = {
                'temp': temps[i] if temps[i] is not None else 25.0,
                'wind_speed': winds[i] if winds[i] is not None else 5.0,
                'wind_dir': dirs[i] if dirs[i] is not None else 0.0
            }
        station_weather[st['name']] = weather_dict
        time.sleep(1) # Be nice to the API
        
    # 3. PyTorch Geometric Graph Construction
    print("Constructing PyTorch Geometric graphs...")
    graphs = []
    num_nodes = len(stations)
    
    # Fully connected edge index (without self-loops)
    edge_list = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_list.append([i, j])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Simulate spatial variance by adding slight noise to the chemical readings per station
    np.random.seed(42)
    
    dates = df['date'].dt.strftime('%Y-%m-%d').tolist()
    
    for day_idx in range(len(df) - 1):
        curr_date = dates[day_idx]
        next_date = dates[day_idx + 1]
        
        row = df.iloc[day_idx]
        next_row = df.iloc[day_idx + 1]
        
        # Node features: [PM2.5, Temp, CO]
        x_list = []
        y_list = []
        for i, st in enumerate(stations):
            w = station_weather[st['name']].get(curr_date, {'temp': 25.0})
            
            pm25 = max(0, min(1, row['pm25'] + np.random.normal(0, 0.05)))
            co = max(0, min(1, row['co'] + np.random.normal(0, 0.05)))
            
            # [PM2.5_Lag, Temp_Inversion, CO] -> scaled approx 0-1
            x_list.append([pm25, w['temp'] / 50.0, co]) 
            
            # Target is next day PM2.5
            next_pm25 = max(0, min(1, next_row['pm25'] + np.random.normal(0, 0.05)))
            y_list.append([next_pm25])
            
        x = torch.tensor(x_list, dtype=torch.float)
        y = torch.tensor(y_list, dtype=torch.float)
        
        # Edge Attributes: [wind_speed, distance]
        edge_attr_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    dist = haversine(stations[i]['lat'], stations[i]['lon'], stations[j]['lat'], stations[j]['lon'])
                    wi = station_weather[stations[i]['name']].get(curr_date, {'wind_speed': 5.0})
                    wj = station_weather[stations[j]['name']].get(curr_date, {'wind_speed': 5.0})
                    avg_wind = (wi['wind_speed'] + wj['wind_speed']) / 2.0
                    edge_attr_list.append([avg_wind / 20.0, dist / 20.0]) # Scale somewhat to 0-1
                    
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        graphs.append(data)
        
    print(f"Created {len(graphs)} daily graphs.")
    
    # 4. Training Loop
    print("Training PyTorch Geometric ST-GNN...")
    device = torch.device("cpu")
    model = WindAdvectionGNN(in_channels=3, out_channels=1, edge_channels=2).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    model.train()
    epochs = 150
    for epoch in range(epochs):
        total_loss = 0
        for data in graphs:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(graphs):.6f}")
            
    model_path = "/Users/alfayez/Desktop/AeroTwin/backend/gnn_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Training complete! Model saved to {model_path}")

if __name__ == "__main__":
    main()
