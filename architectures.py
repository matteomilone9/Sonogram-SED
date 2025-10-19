import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.nn.functional as F

### 1S ###
class NS_1(nn.Module):
    def __init__(self, sample_rate=16000, segment_duration=1):
        super(NS_1, self).__init__()
        self.sample_rate = sample_rate
        self.duration_seconds = segment_duration
        self.frames = 5
        self.n_mels = 128
        self.t_bins = 1 + (self.duration_seconds * self.sample_rate) // 512
        self.vector_array_size = self.t_bins - self.frames + 1
        self.transform_tf = T.MelSpectrogram(sample_rate=self.sample_rate,
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

### VAR AUTOENCODER + TRANSFORMER ###
class VAT_Anomaly(nn.Module):
    def __init__(self, sample_rate=16000, segment_duration=1):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration_seconds = segment_duration
        self.frames=5
        self.n_mels = 128
        self.t_bins = 1 + (self.duration_seconds * self.sample_rate) // 512
        self.vector_array_size = self.t_bins - self.frames + 1
        self.vector_dim = self.frames * self.n_mels

        ###=== MEL SPECTROGRAM ===###
        # -----------------
        self.transform_tf = T.MelSpectrogram(
            sample_rate=self.sample_rate,
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

        ###=== ENCODER ===###
        # -----------------
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.vector_dim, 256),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
        )

    
        ###=== TRANSFORMER BLOCK ===###
        # -----------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)


        ###=== VAE bottleneck ===###
        # -----------------
        latent_dim = 16
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        
        ###=== DECODER  ===###
        # -----------------
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.vector_dim)
        )

    ###=== Forward ===###
    def forward(self, x):
        x = self.preprocessing(x)
        original = x  # for reconstruction loss

        # Encoder
        x = self.encoder_fc(x)               # (B, T, 128)
        x = self.transformer(x)              # (B, T, 128)

        # Bottleneck (mean & logvar per frame)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std                   # (B, T, latent_dim)

        # Decode
        recon = self.decoder_fc(z)           # (B, T, vector_dim)
        return original, recon, mu, logvar

    def preprocessing(self, x):
        batch_size = x.size(0)
        x = self.transform_tf(x)  # (B, n_mels, t_bins)
        x = 10 * torch.log10(x + 1e-8)
        x = (x + 80) / 80
        x = torch.clamp(x, 0.0, 1.0)

        # Frame stacking
        feature_vector = torch.zeros(
            (batch_size, self.vector_array_size, self.vector_dim), device=x.device
        )
        for b in range(batch_size):
            for i in range(self.frames):
                feature_vector[b, :, self.n_mels * i:self.n_mels * (i + 1)] = \
                    x[b, :, i:i + self.vector_array_size].T
        return feature_vector

### DA PROVARE ###
class GMVAE_Anomaly(nn.Module):
    def __init__(self, num_clusters=5):
        super().__init__()
        self.frames = 5
        self.n_mels = 128
        self.duration_seconds = 1
        self.t_bins = 1 + (self.duration_seconds * 16000) // 512
        self.vector_array_size = self.t_bins - self.frames + 1
        self.vector_dim = self.frames * self.n_mels
        self.latent_dim = 16
        self.num_clusters = num_clusters

        # === MEL SPECTROGRAM ===
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

        # === ENCODER ===
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.vector_dim, 256),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(self.vector_array_size),
            nn.ReLU(),
        )

        # === TRANSFORMER ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # === VAE bottleneck ===
        self.fc_mu = nn.Linear(128, self.latent_dim)
        self.fc_logvar = nn.Linear(128, self.latent_dim)

        # === DECODER ===
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.vector_dim)
        )

        # === GM Prior parameters ===
        self.prior_mu = nn.Parameter(torch.randn(num_clusters, self.latent_dim))
        self.prior_logvar = nn.Parameter(torch.zeros(num_clusters, self.latent_dim))
        self.prior_logits = nn.Parameter(torch.zeros(num_clusters))  # logit delle probabilità a priori

    # === Forward ===
    def forward(self, x):
        x = self.preprocessing(x)
        original = x

        # Encoder
        x = self.encoder_fc(x)
        x = self.transformer(x)

        # Bottleneck
        mu = self.fc_mu(x)        # (B, T, latent_dim)
        logvar = self.fc_logvar(x)

        # Reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std        # (B, T, latent_dim)

        # === GM Prior responsibilities ===
        # Calcolo p(c|z) per ogni cluster
        B, T, D = z.shape
        z_expand = z.unsqueeze(2)          # (B, T, 1, D)
        prior_mu = self.prior_mu.unsqueeze(0).unsqueeze(0)  # (1, 1, K, D)
        prior_var = torch.exp(self.prior_logvar).unsqueeze(0).unsqueeze(0)  # (1,1,K,D)

        log_prob = -0.5 * torch.sum(((z_expand - prior_mu)**2) / prior_var + torch.log(2*torch.pi*prior_var), dim=-1)
        # aggiungo log pi_k
        log_prob += F.log_softmax(self.prior_logits, dim=0)  # (B, T, K)
        cluster_prob = F.softmax(log_prob, dim=-1)           # p(c|z) (B, T, K)

        # Decode
        recon = self.decoder_fc(z)           # (B, T, vector_dim)
        return original, recon, mu, logvar, z, cluster_prob

    # === Preprocessing audio ===
    def preprocessing(self, x):
        batch_size = x.size(0)
        x = self.transform_tf(x)
        x = 10 * torch.log10(x + 1e-8)
        x = (x + 80) / 80
        x = torch.clamp(x, 0.0, 1.0)

        feature_vector = torch.zeros(
            (batch_size, self.vector_array_size, self.vector_dim), device=x.device
        )
        for b in range(batch_size):
            for i in range(self.frames):
                feature_vector[b, :, self.n_mels * i:self.n_mels * (i + 1)] = \
                    x[b, :, i:i + self.vector_array_size].T
        return feature_vector
