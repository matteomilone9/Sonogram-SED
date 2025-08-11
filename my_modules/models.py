import torch
import torch.nn as nn
import torchaudio.transforms as T

class NS_1(nn.Module):
    def __init__(self) -> None:
        super(NS_1, self).__init__()
        self.frames = 5
        self.n_mels = 128
        self.duration_seconds = 1
        self.t_bins = 1 + (self.duration_seconds * 16000) // 512
        self.vector_array_size = self.t_bins - self.frames + 1
        self.transform_tf = T.MelSpectrogram(sample_rate=16000,
                                                n_fft=1024,
                                                win_length=1024,
                                                hop_length=512,
                                                center=True,
                                                pad_mode="reflect",
                                                power=2.0,
                                                norm="slaney",
                                                n_mels=self.n_mels,
                                                mel_scale="htk",)

        self.encoder = nn.Sequential(
            nn.Linear(in_features = 640, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU()
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(in_features = 128, out_features = 8),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features = 8, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 640)
        )

        # WEIGHTS INIT LIKE KERAS
        nn.init.xavier_uniform_(self.encoder[0].weight)
        nn.init.xavier_uniform_(self.encoder[3].weight)
        nn.init.xavier_uniform_(self.encoder[6].weight)
        nn.init.xavier_uniform_(self.encoder[9].weight)
        nn.init.xavier_uniform_(self.bottleneck[0].weight)
        nn.init.xavier_uniform_(self.decoder[0].weight)
        nn.init.xavier_uniform_(self.decoder[3].weight)
        nn.init.xavier_uniform_(self.decoder[6].weight)
        nn.init.xavier_uniform_(self.decoder[9].weight)
        nn.init.xavier_uniform_(self.decoder[-1].weight)

        # BIAS INIT LIKE KERAS
        nn.init.zeros_(self.encoder[0].bias)
        nn.init.zeros_(self.encoder[3].bias)
        nn.init.zeros_(self.encoder[6].bias)
        nn.init.zeros_(self.encoder[9].bias)
        nn.init.zeros_(self.bottleneck[0].bias)
        nn.init.zeros_(self.decoder[0].bias)
        nn.init.zeros_(self.decoder[3].bias)
        nn.init.zeros_(self.decoder[6].bias)
        nn.init.zeros_(self.decoder[9].bias)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, x):
        x = self.preprocessing(x)
        original = x
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return original, x

    def preprocessing(self, x):
        # compute mel spectrogram
        batch_size = x.size()[0]
        x = self.transform_tf(x)  # shape: (B, n_mels, t_bins)

        # Convert to decibels (dB)
        x = 10 * torch.log10(x + 1e-8)

        # Normalize dB range from [-80, 0] → [0, 1]
        x = (x + 80) / 80
        x = torch.clamp(x, 0.0, 1.0)

        # Frame stacking (vectorize mel frames into context windows)
        vector_dim = self.frames * self.n_mels
        feature_vector = torch.zeros((batch_size, self.vector_array_size, vector_dim), device=x.device)

        for batch in range(batch_size):
            for i in range(self.frames):
                feature_vector[batch, :, self.n_mels * i: self.n_mels * (i + 1)] = \
                    x[batch, :, i: i + self.vector_array_size].T

        return feature_vector

class NS_5(nn.Module):
    def __init__(self) -> None:
        super(NS_5, self).__init__()
        self.frames = 5
        self.n_mels = 128
        self.duration_seconds = 5
        self.t_bins = 1 + (self.duration_seconds * 16000) // 512
        self.vector_array_size = self.t_bins - self.frames + 1

        self.transform_tf = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=512,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=self.n_mels,
            mel_scale="htk",
        )

        self.encoder = nn.Sequential(
            nn.Linear(640, 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU()
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(128, 8),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

class NS_10(nn.Module):
    def __init__(self) -> None:
        super(NS_10, self).__init__()
        self.frames = 10
        self.n_mels = 128
        self.duration_seconds = 5
        self.t_bins = 1 + (self.duration_seconds * 16000) // 512
        self.vector_array_size = self.t_bins - self.frames + 1

        self.transform_tf = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=512,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=self.n_mels,
            mel_scale="htk",
        )

        self.encoder = nn.Sequential(
            nn.Linear(1280, 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU()
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(128, 8),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x
