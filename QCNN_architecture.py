import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import brevitas.nn as qnn
from brevitas.export import QONNXManager

class EdgeIIoT_FINN_QCNN(nn.Module):
    def __init__(self, num_features, num_classes, bit_width=4):
        super(EdgeIIoT_FINN_QCNN, self).__init__()
        self.num_features = int(num_features)
        
        self.quant_inp = qnn.QuantIdentity(bit_width=8, signed=True, return_quant_tensor=True)
        
        self.conv1 = qnn.QuantConv2d(1, 16, kernel_size=(1, 3), padding=(0, 1), weight_bit_width=bit_width, bias=False, return_quant_tensor=True)
        self.relu1 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.pool1 = nn.MaxPool2d((1, 2))
        
        self.conv2 = qnn.QuantConv2d(16, 32, kernel_size=(1, 3), padding=(0, 1), weight_bit_width=bit_width, bias=False, return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.pool2 = nn.MaxPool2d((1, 2))
        
        self.post_pool_ident = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        with torch.no_grad():
            dummy_x = torch.zeros((1, 1, 1, self.num_features))
            dummy_x = self.pool1(self.conv1(dummy_x))
            dummy_x = self.pool2(self.conv2(dummy_x))
            self.flattened_size = int(dummy_x.numel())
        
        self.flatten = nn.Flatten()
        self.fc1 = qnn.QuantLinear(self.flattened_size, 16, weight_bit_width=bit_width, bias=False, return_quant_tensor=True)
        self.relu3 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        
        self.fc2 = qnn.QuantLinear(16, num_classes, weight_bit_width=8, bias=False, return_quant_tensor=False)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.post_pool_ident(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = self.post_pool_ident(x)
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def load_and_preprocess_edge_iiot(csv_path, batch_size=2048):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    target_col = 'Attack_type'
    df = df.dropna(subset=[target_col])
    
    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])
    X_numeric = X_raw.select_dtypes(include=[np.number]).fillna(0)
    
    raw_shape = X_numeric.shape
    num_features = raw_shape[1]
    
    print(f"Data Loaded. Shape: {raw_shape}. Features detected: {num_features}")
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    
    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y_encoded, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).unsqueeze(1), 
        torch.tensor(y_train, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).unsqueeze(1), 
        torch.tensor(y_test, dtype=torch.long)
    )
    
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), \
           DataLoader(test_ds, batch_size=batch_size, shuffle=False), \
           num_features, num_classes

def run_workflow():
    DATASET_PATH = "DNN-EdgeIIoT-dataset.csv"
    WEIGHTS_PATH = "edge_iiot_qcnn_weights.pth"
    EXPORT_PATH = "edge_iiot_qcnn_finn.onnx"
    
    train_loader, test_loader, n_feat, n_class = load_and_preprocess_edge_iiot(DATASET_PATH)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = EdgeIIoT_FINN_QCNN(n_feat, n_class).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("\n[INFO] Training for 20 epochs...")
    for epoch in range(20):
        model.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if i % 200 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), WEIGHTS_PATH)
    
    print("\n[INFO] Exporting to FINN-compatible ONNX...")
    model.cpu().eval()
    model.fc2.return_quant_tensor = True 
    
    dummy_input = torch.randn(1, 1, 1, n_feat)
    QONNXManager.export(model, args=(dummy_input,), export_path=EXPORT_PATH)
    print(f"Success! Model saved as: {EXPORT_PATH}")

if __name__ == "__main__":
    run_workflow()