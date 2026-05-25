import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import brevitas.nn as qnn
from brevitas.export import QONNXManager

class EdgeIIoT_FINN_QCAE(nn.Module):
    def __init__(self, num_features, bit_width=4, latent_dim=16):
        super(EdgeIIoT_FINN_QCAE, self).__init__()
        self.num_features = int(num_features)
        
        # --- ENCODER ---
        self.quant_inp = qnn.QuantIdentity(bit_width=8, signed=True, return_quant_tensor=True)
        
        self.conv1 = qnn.QuantConv2d(1, 16, kernel_size=(1, 3), padding=(0, 1), weight_bit_width=bit_width, bias=False, return_quant_tensor=True)
        self.relu1 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.pool1 = nn.MaxPool2d((1, 2))
        
        self.conv2 = qnn.QuantConv2d(16, 32, kernel_size=(1, 3), padding=(0, 1), weight_bit_width=bit_width, bias=False, return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.pool2 = nn.MaxPool2d((1, 2))
        
        self.post_pool_ident = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Calculate flattened size: 32 channels * 1 height * (features / 4) width
        self.flattened_size = 32 * 1 * (self.num_features // 4)
        
        self.flatten = nn.Flatten()
        self.fc_enc = qnn.QuantLinear(self.flattened_size, latent_dim, weight_bit_width=bit_width, bias=False, return_quant_tensor=True)
        self.relu_enc = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        
        # --- DECODER ---
        self.fc_dec = qnn.QuantLinear(latent_dim, self.flattened_size, weight_bit_width=bit_width, bias=False, return_quant_tensor=True)
        self.relu_dec = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        
        # Transposed Convolutions to perfectly reverse MaxPool2d
        self.deconv1 = qnn.QuantConvTranspose2d(32, 16, kernel_size=(1, 2), stride=(1, 2), weight_bit_width=bit_width, bias=False, return_quant_tensor=True)
        self.relu_dec1 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        
        # Final layer reconstructs the original 1-channel output
        self.deconv2 = qnn.QuantConvTranspose2d(16, 1, kernel_size=(1, 2), stride=(1, 2), weight_bit_width=8, bias=False, return_quant_tensor=False)

    def forward(self, x):
        # Encoder
        x = self.quant_inp(x)
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = self.post_pool_ident(x)
        
        x = self.flatten(x)
        latent = self.relu_enc(self.fc_enc(x))
        
        # Decoder
        x = self.relu_dec(self.fc_dec(latent))
        
        # Reshape back to 4D tensor for Transposed Convolutions: (Batch, Channels, Height, Width)
        x = x.view(-1, 32, 1, self.num_features // 4)
        
        x = self.relu_dec1(self.deconv1(x))
        reconstructed = self.deconv2(x)
        
        return reconstructed

def load_and_preprocess_edge_iiot_cae(csv_path, batch_size=2048):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    target_col = 'Attack_type'
    df = df.dropna(subset=[target_col])
    
    X_raw = df.drop(columns=[target_col])
    X_numeric = X_raw.select_dtypes(include=[np.number]).fillna(0)
    
    raw_shape = X_numeric.shape
    num_features = raw_shape[1]
    
    # AE bottleneck logic requires features to be divisible by 4 (due to two MaxPool2d layers)
    pad_amount = (4 - (num_features % 4)) % 4
    if pad_amount > 0:
        print(f"Padding {pad_amount} zeros to features. Original: {num_features}, New: {num_features + pad_amount}")
        X_padded = np.pad(X_numeric.values, ((0, 0), (0, pad_amount)), mode='constant', constant_values=0)
    else:
        X_padded = X_numeric.values

    num_features_padded = X_padded.shape[1]
    
    # We only need X for an Autoencoder (unsupervised)
    X_train, X_test = train_test_split(X_padded, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Format to (Batch, Channel=1, Height=1, Width=Features)
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).unsqueeze(1))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).unsqueeze(1))
    
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), \
           DataLoader(test_ds, batch_size=batch_size, shuffle=False), \
           num_features_padded

def run_workflow():
    DATASET_PATH = "DNN-EdgeIIoT-dataset.csv"
    WEIGHTS_PATH = "edge_iiot_qcae_weights.pth"
    EXPORT_PATH = "edge_iiot_qcae_finn.onnx"
    
    # 1. Prep Data
    train_loader, test_loader, n_feat_padded = load_and_preprocess_edge_iiot_cae(DATASET_PATH)
    
    # 2. Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = EdgeIIoT_FINN_QCAE(n_feat_padded).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Use MSELoss for Autoencoder reconstruction
    criterion = nn.MSELoss() 
    
    print("\n[INFO] Training QCAE for 20 epochs...")
    for epoch in range(20):
        model.train()
        for i, (data,) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            
            # The target is the original input data
            loss = criterion(output, data) 
            loss.backward()
            optimizer.step()
            
            if i % 200 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | MSE Loss: {loss.item():.4f}")
    
    # Save weights
    torch.save(model.state_dict(), WEIGHTS_PATH)
    
    # 3. Export to FINN
    print("\n[INFO] Exporting QCAE to FINN-compatible ONNX...")
    model.cpu().eval()
    model.deconv2.return_quant_tensor = True 
    
    dummy_input = torch.randn(1, 1, 1, n_feat_padded)
    QONNXManager.export(model, args=(dummy_input,), export_path=EXPORT_PATH)
    print(f"Success! Model saved as: {EXPORT_PATH}")

if __name__ == "__main__":
    run_workflow()