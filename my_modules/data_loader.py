import os
import csv
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

# def generate_labels_csv(audio_root, output_csv="labels.csv"):
#     """Genera un file CSV con i percorsi dei file audio e le relative etichette."""
#     rows = []
#     label_map = {}
#
#     for label_idx, class_name in enumerate(sorted(os.listdir(audio_root))):
#         class_path = os.path.join(audio_root, class_name)
#         if not os.path.isdir(class_path):
#             continue
#
#         label_map[class_name] = label_idx
#         for filename in os.listdir(class_path):
#             if filename.endswith(".wav"):
#                 file_path = os.path.join(class_path, filename)
#                 rows.append((file_path, label_idx))
#
#     with open(output_csv, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['filepath', 'label'])  # header
#         writer.writerows(rows)
#
#     print(f" Mappa delle classi: {label_map}")
#     return label_map  # Restituisce la mappa delle etichette per riferimento

def generate_labels_csv(audio_root, output_csv="labels.csv"):
    rows = []
    label_map = {}

    for label_idx, class_name in enumerate(sorted(os.listdir(audio_root))):
        class_path = os.path.join(audio_root, class_name)
        if not os.path.isdir(class_path):
            continue

        label_map[class_name] = label_idx
        for filename in os.listdir(class_path):
            if filename.endswith(".flac"):
                relative_path = os.path.join(class_name, filename)
                rows.append((relative_path, label_idx))

    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'label'])  # header
        writer.writerows(rows)

    print(f" Mappa delle classi v12: {label_map}")
    return label_map


class AudioDataset(Dataset):
    def __init__(self, audio_dir, segment_duration=10, sample_rate=16000,
                 supervised=False, label_csv_path=None):
        """Dataset per file audio con supporto per modalità supervisionata e non supervisionata.

        Args:
            audio_dir (str): Percorso alla directory contenente i file audio
            segment_duration (int): Durata di ogni segmento audio in secondi
            sample_rate (int): Frequenza di campionamento desiderata
            supervised (bool): Se True, utilizza le etichette dal file CSV
            label_csv_path (str): Percorso al file CSV con le etichette (obbligatorio se supervised=True)
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_duration * sample_rate
        self.supervised = supervised

        if supervised:
            assert label_csv_path is not None, "label_csv_path is required in supervised mode"
            self.data = pd.read_csv(label_csv_path)
            self.file_paths = [os.path.join(audio_dir, row['filepath']) for _, row in self.data.iterrows()]
            self.labels = self.data['label'].tolist()
        else:
            self.file_paths = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".flac")]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Trim or pad to fixed length
        if waveform.shape[1] > self.segment_length:
            waveform = waveform[:, :self.segment_length]
        else:
            pad_len = self.segment_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        waveform = waveform.squeeze(0)  # [160000]

        if self.supervised:
            label = int(self.labels[idx])
            return waveform, label,  torch.tensor(0), torch.tensor(0)#da verificare
        else:
            return waveform

