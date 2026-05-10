import numpy as np
import vector
import matplotlib.pyplot as plt
import torch
# from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

def load_data_to_4mom(data):
    """Convert data into 4-momentum vectors using the vector library.\\
    
    Format:\\
        0:3   - lepton 1 (E, px, py, pz)\\
        4:7   - lepton 2 (E, px, py, pz)\\
        8:11  - jet 1 (E, px, py, pz)\\
        12:15 - jet 2 (E, px, py, pz)\\
        16:19 - MET (E, px, py, pz)
    
    Args:
        data (np.ndarray): The input data array.

    Returns:
        tuple: 4-momentum vectors (lepton1, lepton2, jet1, jet2, MET).
    """
    vec_lepton1 = vector.array({"E": data[:, 0], "px": data[:, 1], "py": data[:, 2], "pz": data[:, 3]})
    vec_lepton2 = vector.array({"E": data[:, 4], "px": data[:, 5], "py": data[:, 6], "pz": data[:, 7]})
    vec_jet1 = vector.array({"E": data[:, 8], "px": data[:, 9], "py": data[:, 10], "pz": data[:, 11]})
    vec_jet2 = vector.array({"E": data[:, 12], "px": data[:, 13], "py": data[:, 14], "pz": data[:, 15]})
    vec_MET = vector.array({"E": data[:, 16], "px": data[:, 17], "py": data[:, 18], "pz": data[:, 19]})

    return vec_lepton1, vec_lepton2, vec_jet1, vec_jet2, vec_MET

def build_features(lep1, lep2, jet1, jet2, MET):
    """

    """
    return np.column_stack([
        lep1.pt, lep1.eta, lep1.phi, 
        lep2.pt, lep2.eta, lep2.phi,
        jet1.pt, jet1.mass, jet1.eta, jet1.phi,
        jet2.pt, jet2.mass, jet2.eta, jet2.phi,
    ])


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20):
    """Train the model and evaluate on validation set.\
    
    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        device: Device to run the training on (e.g., 'cuda' or 'cpu').
        num_epochs (int): Number of epochs to train.

    Returns:
        tuple: Lists of training and validation losses per epoch.
    """

    from tqdm import tqdm
    from sklearn.metrics import roc_auc_score, roc_curve

    train_losses    = []
    val_losses      = []
    val_aucs        = []
    roc_data        = None

    for epoch in range(num_epochs):

        # Training
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        val_running_loss = 0.0
        all_probs  = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
                val_running_loss += loss.item() * inputs.size(0)

                all_probs.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        auc = roc_auc_score(all_labels, all_probs)
        val_aucs.append(auc)
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

        roc_data = (fpr, tpr, thresholds)

        print(f'Epoch {epoch+1}/{num_epochs} — '
              f'Train Loss: {epoch_train_loss:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, '
              f'Val AUC: {auc:.4f}', flush=True)

    return train_losses, val_losses, val_aucs, roc_data


class Red_Sea3(nn.Module):
    def __init__(self, input_len):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_len, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)