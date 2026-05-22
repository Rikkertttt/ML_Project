from functions import load_data_to_4mom, Red_Sea3, train_model_1stmeth, build_features
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():

    epochs = 50
    lr = 0.001
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

    # Concatenate and create labels
    X = np.concatenate([X_hww, X_ttbar, X_wwjj], axis=0)
    Y = np.concatenate([np.ones(len(X_hww)), np.zeros(len(X_ttbar)), np.zeros(len(X_wwjj))], axis=0)

    # Split into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Convert to PyTorch tensors and create DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    model = Red_Sea3(input_len=X_train.shape[1], use_sigmoid=True).to(device)
    criterion = torch.nn.BCELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    
    train_losses, val_losses, val_aucs, (fpr, tpr, thresholds) = train_model_1stmeth(
    model, train_loader, val_loader, criterion, optimizer, device, num_epochs=epochs
    )

    Youden_index = tpr - fpr
    optimal_idx = np.argmax(Youden_index)
    optimal_threshold = thresholds[optimal_idx]

    # Loss plot
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('First_method/loss_plot.png')

    # AUC over epochs
    plt.figure()
    plt.plot(val_aucs)
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Validation AUC over Epochs')
    plt.savefig('First_method/auc_plot.png')

    # ROC curve (final epoch)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {val_aucs[-1]:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Final Epoch)')
    plt.axvline(x=fpr[optimal_idx], color='r', linestyle='--', label=f'Optimal Threshold = {optimal_threshold:.4f}')
    plt.legend()
    plt.savefig('First_method/roc_curve.png')

    print(f"Final AUC: {val_aucs[-1]:.4f}", flush=True)

    torch.save(model.state_dict(), 'First_method/model.pth')
    print("Model saved.", flush=True)


    model.eval()
    X_val_unscaled = scaler.inverse_transform(X_val)

    with torch.no_grad():
        X_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        probs = model(X_tensor).cpu().numpy()
        y_pred = (probs >= optimal_threshold).astype(int).ravel()
        


    for i, feature_name in enumerate(['lep1_pt', 'lep1_eta', 'lep1_phi',
                                        'lep2_pt', 'lep2_eta', 'lep2_phi',
                                        'jet1_pt', 'jet1_mass', 'jet1_eta', 'jet1_phi',
                                        'jet2_pt', 'jet2_mass', 'jet2_eta', 'jet2_phi']):
        plt.figure()
        plt.hist(X_val_unscaled[:, i], bins=50,
                density=True, histtype='step', label='Validation data')
        plt.hist(X_val_unscaled[:, i][y_pred == 1], bins=50,
                density=True, histtype='step', label='Cut validation data')
        plt.hist(X_hww[:, i], bins=50,
                density=True, histtype='step', label='Pure HWW')
        plt.xlabel(feature_name)
        plt.ylabel('Density')
        plt.title(f'Closure test: {feature_name}')
        plt.legend()
        plt.savefig(f'First_method/{feature_name}.png')
        plt.close()




    return

if __name__ == "__main__":
    main()