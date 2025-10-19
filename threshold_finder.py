import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split
from tqdm import tqdm
import os
import numpy as np

from architectures import VAT_Anomaly
from data_loader import AudioDataset

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# PARAMETRI
# =========================================================
batch_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "pesi"
model_path = os.path.join(save_dir, "best_vae_model_1s.pth")

# =========================================================
# FUNZIONE PER CALCOLO THRESHOLD
# =========================================================
def compute_threshold(model, val_loader, k=3.0):
    """
    Calcola la soglia di anomalia sul validation set come:
        threshold = mean + k * std
    dove k ≈ 3 è un moltiplicatore tipico (valori più alti → meno falsi positivi).
    """
    model.eval()
    model.to(device)

    reconstruction_errors = []

    print("Calcolo reconstruction error su validation set...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            batch = batch.to(device)
            original, recon, mu, logvar = model(batch)
            errors = F.mse_loss(recon, original, reduction='none')
            errors = errors.mean(dim=(1, 2))  # errore medio per sample
            reconstruction_errors.extend(errors.cpu().numpy())

    reconstruction_errors = np.array(reconstruction_errors)
    mean_err = reconstruction_errors.mean()
    std_err = reconstruction_errors.std()
    threshold = mean_err + k * std_err

    print("\n=== RISULTATI THRESHOLD ===")
    print(f"Media errori ricostruzione: {mean_err:.8f}")
    print(f"Deviazione standard errori: {std_err:.8f}")
    print(f"Soglia stimata (k={k}):     {threshold:.8f}")
    print("============================\n")

    return threshold, reconstruction_errors


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    # Dataset
    ae_dataset = AudioDataset(audio_dir="audio_anomali/dataset_1s", supervised=False)
    fan_dataset = AudioDataset(audio_dir="audio_anomali/ventola_1s", supervised=False)
    combined_dataset = ConcatDataset([ae_dataset, fan_dataset])

    # Split
    train_size = int(0.75 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    _, ae_val = random_split(combined_dataset, [train_size, val_size])
    ae_val_loader = DataLoader(ae_val, batch_size=batch_size, shuffle=False)

    # Caricamento modello
    model = VAT_Anomaly()
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Calcolo soglia
    threshold, errors = compute_threshold(model, ae_val_loader, k=3.0)

    # Salvataggio soglia su file
    os.makedirs(save_dir, exist_ok=True)
    threshold_file = os.path.join(save_dir, "threshold.txt")
    with open(threshold_file, "w") as f:
        f.write(f"{threshold:.8f}\n")

    print(f"Soglia salvata in: {threshold_file}")
