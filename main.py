from functions import load_data_to_4mom, Red_Sea3, train_model, build_features
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():

    epochs = 10
    lr = 0.0001
    batch_size = 256

    # Load data from .npy files
    HWW_Data = np.load("Data/reco-gen_hww_rec.npy")
    ttbar_Data = np.load("Data/reco-gen_ttbar_nobjet_nocjv_rec.npy")
    wwjj_Data = np.load("Data/reco-gen_wwjj_rec.npy")

    # Convert data to 4-momentum vectors
    HWW_lep1, HWW_lep2, HWW_jet1, HWW_jet2, HWW_MET = load_data_to_4mom(HWW_Data)
    ttbar_lep1, ttbar_lep2, ttbar_jet1, ttbar_jet2, ttbar_MET = load_data_to_4mom(ttbar_Data)
    wwjj_lep1, wwjj_lep2, wwjj_jet1, wwjj_jet2, wwjj_MET = load_data_to_4mom(wwjj_Data) 

    # Build feature matrices for each dataset
    X_hww  = build_features(HWW_lep1,   HWW_lep2,   HWW_jet1,   HWW_jet2,   HWW_MET)
    X_ttbar = build_features(ttbar_lep1, ttbar_lep2, ttbar_jet1, ttbar_jet2, ttbar_MET)
    X_wwjj  = build_features(wwjj_lep1,  wwjj_lep2,  wwjj_jet1,  wwjj_jet2,  wwjj_MET)


    # Split ttbar and wwjj into two non-overlapping halves
    ttbar_data, ttbar_bg = train_test_split(X_ttbar, test_size=0.5, random_state=33)
    wwjj_data,  wwjj_bg  = train_test_split(X_wwjj,  test_size=0.5, random_state=33)

    X_data = np.concatenate([X_hww, ttbar_data, wwjj_data], axis=0)
    X_bg   = np.concatenate([ttbar_bg, wwjj_bg], axis=0)

    # Combine signal and background data, and create labels
    X_label1 = np.concatenate([X_data, X_bg], axis=0)
    Y_label1 = np.ones(len(X_label1))

    X_label0 = np.copy(X_data)
    Y_label0 = np.zeros(len(X_label0))

    X = np.concatenate([X_label1, X_label0], axis=0)
    Y = np.concatenate([Y_label1, Y_label0], axis=0)

    # Target ratio
    ratio = {'hww': 0.144, 'ttbar': 0.536, 'wwjj': 0.319}

    # positive weights for data
    w_hww   = ratio['hww']   / len(X_hww)
    w_ttbar_d = ratio['ttbar'] / len(ttbar_data)
    w_wwjj_d  = ratio['wwjj']  / len(wwjj_data)
    # negative weights for background
    w_ttbar_bg = ratio['ttbar'] / len(ttbar_bg) * -1
    w_wwjj_bg  = ratio['wwjj']  / len(wwjj_bg) * -1

    Weights = np.concatenate([
        # Label 1: X_data (positive weights)
        np.full(len(X_hww),      w_hww),
        np.full(len(ttbar_data), w_ttbar_d),
        np.full(len(wwjj_data),  w_wwjj_d),
        # Label 1: X_bg (negative weights)
        np.full(len(ttbar_bg),   w_ttbar_bg),
        np.full(len(wwjj_bg),    w_wwjj_bg),
        # Label 0: X_data copy (positive weights)
        np.full(len(X_hww),      w_hww),
        np.full(len(ttbar_data), w_ttbar_d),
        np.full(len(wwjj_data),  w_wwjj_d),
    ], axis=0)


    # Split into training and validation sets
    X_train, X_val, Y_train, Y_val, W_train, W_val = train_test_split(X, Y, Weights, test_size=0.2, random_state=33)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Convert to PyTorch tensors and create DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(Y_train, dtype=torch.float32), 
        torch.tensor(W_train, dtype=torch.float32))
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), 
        torch.tensor(Y_val, dtype=torch.float32), 
        torch.tensor(W_val, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    model = Red_Sea3(input_len=X_train.shape[1]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    
    train_losses, val_losses = train_model(
    model, train_loader, val_loader, criterion, optimizer, device, num_epochs=epochs
    )

    # Loss plot
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_plot_1.png')


    # Apply weights to data and compare to pure HWW
    model.eval()

    X_val_unscaled = scaler.inverse_transform(X_val)
    X_val_data = X_val_unscaled[(Y_val == 1) & (W_val > 0)]  # Original features for data samples
    X_val_data_scaled = X_val[(Y_val == 1) & (W_val > 0)]   # Scaled features for data samples
    W_val_data = W_val[(Y_val == 1) & (W_val > 0)]       # Weights for data samples

    with torch.no_grad():
        X_tensor = torch.tensor(X_val_data_scaled, dtype=torch.float32).to(device)
        logits = model(X_tensor).squeeze()
        C = torch.sigmoid(logits).cpu().numpy()
        nu = C / (1 - C + 1e-8)

    # Plot each feature reweighted vs pure HWW
    for i, feature_name in enumerate(['lep1_pt', 'lep1_eta', 'lep1_phi',
                                        'lep2_pt', 'lep2_eta', 'lep2_phi',
                                        'jet1_pt', 'jet1_mass', 'jet1_eta', 'jet1_phi',
                                        'jet2_pt', 'jet2_mass', 'jet2_eta', 'jet2_phi']):
        plt.figure()
        plt.hist(X_val_data[:, i], bins=50, weights=nu * W_val_data,
                density=True, histtype='step', label='Reweighted data')
        plt.hist(X_hww[:, i], bins=50,
                density=True, histtype='step', label='Pure HWW')
        plt.hist(X_val_data[:, i], bins=50, weights=W_val_data,
                density=True, histtype='step', label='Data')
        plt.xlabel(feature_name)
        plt.ylabel('Density')
        plt.title(f'Closure test: {feature_name}')
        plt.legend()
        plt.savefig(f'Closure_tests/closure_{feature_name}.png')
        plt.close()

    torch.save(model.state_dict(), 'model_1.pth')
    print("Model saved.", flush=True)




    return

if __name__ == "__main__":
    main()