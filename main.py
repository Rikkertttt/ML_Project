from functions import load_data_to_4mom, Red_Sea3, train_model, build_features
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def main():

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
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    model = Red_Sea3(input_len=X_train.shape[1]).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    
    train_losses, val_losses, val_aucs, (fpr, tpr, thresholds) = train_model(
    model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20
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

    # AUC over epochs
    plt.figure()
    plt.plot(val_aucs)
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Validation AUC over Epochs')
    plt.savefig('auc_plot_1.png')

    # ROC curve (final epoch)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {val_aucs[-1]:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Final Epoch)')
    plt.legend()
    plt.savefig('roc_curve_1.png')

    print(f"Final AUC: {val_aucs[-1]:.4f}", flush=True)

    torch.save(model.state_dict(), 'model_1.pth')
    print("Model saved.", flush=True)




    return

if __name__ == "__main__":
    main()