import os
import csv
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split


class AudioDataset(Dataset):
    def __init__(self, audio_dir, segment_duration=10, sample_rate=16000, mode="train"):
        """
        Dataset audio flessibile, stile MIMII.

        Args:
            audio_dir (str): percorso alla cartella contenente i file audio
            segment_duration (float): durata del segmento in secondi
            sample_rate (int): frequenza di campionamento desiderata
            mode (str): 'train' o 'test'
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.segment_len = int(segment_duration * sample_rate)
        self.mode = mode

        self.files = [
            os.path.join(audio_dir, f)
            for f in sorted(os.listdir(audio_dir))
            if f.endswith(".wav") or f.endswith(".flac")
        ]

        print(f"[{mode}] trovati {len(self.files)} file in {audio_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        wav, sr = torchaudio.load(file_path)

        # Resample se necessario
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # Converti in mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Trim o pad a lunghezza fissa
        if wav.size(1) > self.segment_len:
            wav = wav[:, :self.segment_len]
        else:
            pad_len = self.segment_len - wav.size(1)
            wav = torch.nn.functional.pad(wav, (0, pad_len))

        wav = wav.squeeze(0)  # [segment_len]

        # opzionale: etichetta per test
        if self.mode == "test":
            label = 1 if "anomaly" in file_path else 0
            return wav, label
        else:
            return wav


class MIMIIDataset(Dataset):
    def __init__(self, audio_dir, sample_rate=16000, segment_duration=1.0, mode="train"):
        """
        Dataset per MIMII DUE (train: normal / test: normal + anomaly)

        Args:
            audio_dir (str): percorso alla cartella (es. 'mimii_due/fan/id_00/train')
            sample_rate (int): frequenza di campionamento desiderata
            segment_duration (float): durata del segmento in secondi
            mode (str): 'train' o 'test'
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.segment_len = int(sample_rate * segment_duration)
        self.mode = mode

        # raccogli file audio
        self.files = [
            os.path.join(audio_dir, f)
            for f in sorted(os.listdir(audio_dir))
            if f.endswith(".wav")
        ]

        print(f"[{mode}] trovati {len(self.files)} file in {audio_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        wav, sr = torchaudio.load(file_path)

        # resample se necessario
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.mean(dim=0)  # mono

        # trim o pad a lunghezza fissa
        if wav.size(0) > self.segment_len:
            wav = wav[:self.segment_len]
        else:
            wav = torch.nn.functional.pad(wav, (0, self.segment_len - wav.size(0)))

        # etichetta (solo per test)
        if self.mode == "test":
            label = 1 if "anomaly" in file_path else 0
            return wav, label
        else:
            return wav
