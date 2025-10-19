import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import torchaudio
from tqdm import tqdm
import os

from data_loader import AudioDataset
from trainer import trainer_VAE
from architectures import VAT_Anomaly

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)



### RUNNER ###
if __name__ == "__main__":
    batch_size = 512
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ae_dataset = AudioDataset(audio_dir="data/doy/283", segment_duration=1, 
                              sample_rate=192000, supervised=False)
    fan_dataset = AudioDataset(audio_dir="data/doy/284", segment_duration=1, 
                               sample_rate=192000, supervised=False)
    combined_dataset = ConcatDataset([ae_dataset, fan_dataset])

    train_size = int(0.75 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    ae_train, ae_val = random_split(combined_dataset, [train_size, val_size])

    ae_train_loader = DataLoader(ae_train, batch_size=batch_size, shuffle=True)
    ae_val_loader = DataLoader(ae_val, batch_size=batch_size, shuffle=False)

    model = VAT_Anomaly(sr=192000)

    # Ora puoi specificare dove e come salvare i pesi
    trainer_VAE(
        model=model,
        train_loader=ae_train_loader,
        val_loader=ae_val_loader,
        epochs=30,
        lr=1e-3,
        patience=5,
        device=device,
        save_dir="192_vae_results_1s",
        save_name="192_best_vae_model_1s.pth",
        resume=True 
    )

