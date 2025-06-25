import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import torchaudio
import numpy as np
import pandas as pd
import os
from pydub import AudioSegment
from torch.utils.data import DataLoader
from itertools import cycle
import argparse

# === DATASET CLASSES ===
class AudioDatasetMixture(torch.utils.data.Dataset):
    def __init__(self, csv_file, target_duration=10000, target_sample_rate=16000,
                 target_channels=1):
        self.df = pd.read_csv(csv_file)
        self.target_duration = target_duration
        self.target_channels = target_channels
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        mixture_path = os.path.join(self.df.loc[idx, 'mixture'])
        mixture_audio = AudioSegment.from_file(mixture_path).set_channels(self.target_channels).set_frame_rate(self.target_sample_rate)
        mixture, _ = self._pydub_to_array(mixture_audio)
        mixture_tensor = torch.Tensor(mixture)
        return mixture_tensor

    def _pydub_to_array(self, audio: AudioSegment) -> (np.ndarray, int):
        return np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((audio.channels, -1)) / (
                1 << (8 * audio.sample_width - 1)), audio.frame_rate

class AudioDatasetBackground(torch.utils.data.Dataset):
    def __init__(self, csv_file, target_duration=10000, target_sample_rate=16000,
                 target_channels=1):
        self.df = pd.read_csv(csv_file)
        self.target_duration = target_duration
        self.target_channels = target_channels
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        background_path = os.path.join(self.df.loc[idx, 'background'])
        background_audio = AudioSegment.from_file(background_path).set_channels(self.target_channels).set_frame_rate(self.target_sample_rate)
        background, _ = self._pydub_to_array(background_audio)
        background_tensor = torch.Tensor(background)
        return background_tensor

    def _pydub_to_array(self, audio: AudioSegment) -> (np.ndarray, int):
        return np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((audio.channels, -1)) / (
                1 << (8 * audio.sample_width - 1)), audio.frame_rate

# === CUSTOM DATALOADER WRAPPER ===
class UnpairedDataLoader:
    """
    Wrapper per combinare due dataloader separati in modo non accoppiato.
    """
    def __init__(self, mixture_loader, background_loader):
        self.mixture_loader = mixture_loader
        self.background_loader = background_loader
        # Usa cycle per il dataset più piccolo per evitare che finisca prima
        self.background_cycle = cycle(background_loader)
        
    def __iter__(self):
        for mixture_batch in self.mixture_loader:
            try:
                background_batch = next(self.background_cycle)
            except StopIteration:
                # Ricrea il cycle se necessario
                self.background_cycle = cycle(self.background_loader)
                background_batch = next(self.background_cycle)
            
            yield mixture_batch, background_batch
    
    def __len__(self):
        return len(self.mixture_loader)

# === MODEL COMPONENTS ===
class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, norm=True, activation=True):
        super(SeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, bias=bias)

        self.bn = nn.BatchNorm1d(out_channels) if norm else nn.Identity()
        self.act = nn.LeakyReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class UNet1D(nn.Module):
    def __init__(self, input_channels=1, base_channels=32):
        super(UNet1D, self).__init__()
        # Encoder
        self.enc1 = SeparableConv1d(input_channels, base_channels, kernel_size=9, padding=4)
        self.enc2 = SeparableConv1d(base_channels, base_channels*2, kernel_size=9, stride=2, padding=4)
        self.enc3 = SeparableConv1d(base_channels*2, base_channels*4, kernel_size=9, stride=2, padding=4)
        self.enc4 = SeparableConv1d(base_channels*4, base_channels*8, kernel_size=9, stride=2, padding=4)
        
        # Bottleneck
        self.bottleneck = SeparableConv1d(base_channels*8, base_channels*8, kernel_size=9, padding=4)
        
        # Decoder
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(base_channels*8, base_channels*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels*4),
            nn.LeakyReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(base_channels*8, base_channels*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels*2),
            nn.LeakyReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(base_channels*4, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            SeparableConv1d(base_channels*2, input_channels, kernel_size=9, padding=4, norm=False, activation=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoder
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e3], dim=1)
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        out = self.dec1(d2)
        
        return out

class Discriminator1D(nn.Module):
    def __init__(self, input_channels=1, base_channels=32):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels*2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(base_channels*2, base_channels*4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv1d(base_channels*4, base_channels*8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.final_conv = nn.Conv1d(base_channels*8, 1, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final_conv(x)
        x = self.avgpool(x)
        return x.view(-1)

# === LIGHTNING MODULE ===
class AudioGAN(pl.LightningModule):
    def __init__(
        self,
        train_mixture_csv: str,
        train_background_csv: str,
        val_mixture_csv: str,
        val_background_csv: str,
        test_mixture_csv: str = None,
        test_background_csv: str = None,
        batch_size: int = 5,
        lr_g: float = 2e-4,
        lr_d: float = 1e-4,
        lambda_l1: float = 10.0,
        d_train_freq: int = 2,
        d_accuracy_threshold: float = 0.8,
        g_loss_threshold: float = 2.0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Models
        self.generator = UNet1D()
        self.discriminator = Discriminator1D()
        
        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        
        # Training parameters
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.lambda_l1 = lambda_l1
        self.d_train_freq = d_train_freq
        self.d_accuracy_threshold = d_accuracy_threshold
        self.g_loss_threshold = g_loss_threshold
        
        # Data paths
        self.train_mixture_csv = train_mixture_csv
        self.train_background_csv = train_background_csv
        self.val_mixture_csv = val_mixture_csv
        self.val_background_csv = val_background_csv
        self.test_mixture_csv = test_mixture_csv
        self.test_background_csv = test_background_csv
        self.batch_size = batch_size
        
        # Training state
        self.g_steps = 0
        self.automatic_optimization = False  # Manual optimization for GAN
        
        # Metrics tracking
        self.train_metrics = {
            'loss_d': [], 'loss_g': [], 'loss_g_gan': [], 'loss_g_l1': [],
            'd_accuracy': [], 'd_train_steps': [], 'g_train_steps': []
        }

    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=self.lr_g, betas=(0.5, 0.999))
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=self.lr_d, betas=(0.5, 0.999))
        return [opt_g, opt_d]

    def training_step(self, batch, batch_idx):
        mixture, background = batch
        
        # Ensure correct dimensions
        if mixture.ndim == 2:
            mixture = mixture.unsqueeze(1)
        if background.ndim == 2:
            background = background.unsqueeze(1)
        
        # Align sequence lengths by padding/truncating to match
        min_length = min(mixture.shape[-1], background.shape[-1])
        mixture = mixture[..., :min_length]
        background = background[..., :min_length]
        
        opt_g, opt_d = self.optimizers()
        
        # Generate fake samples
        fake_background = self.generator(mixture)
        
        # Calculate discriminator accuracy
        with torch.no_grad():
            real_output = self.discriminator(background)
            fake_output_detached = self.discriminator(fake_background.detach())
            
            real_preds = torch.sigmoid(real_output) > 0.5
            fake_preds = torch.sigmoid(fake_output_detached) < 0.5
            
            d_correct = (real_preds.sum() + fake_preds.sum()).item()
            total_preds = real_preds.numel() + fake_preds.numel()
            d_accuracy = d_correct / max(total_preds, 1)
        
        # Decide whether to train discriminator
        should_train_d = True
        if self.g_steps % self.d_train_freq != 0:
            should_train_d = False
        if d_accuracy > self.d_accuracy_threshold:
            should_train_d = False
        
        # Train Discriminator
        d_train_steps = 0
        if should_train_d:
            opt_d.zero_grad()
            
            real_output = self.discriminator(background)
            fake_output = self.discriminator(fake_background.detach())
            
            real_labels = torch.ones_like(real_output)
            fake_labels = torch.zeros_like(fake_output)
            
            loss_d_real = self.adversarial_loss(real_output, real_labels)
            loss_d_fake = self.adversarial_loss(fake_output, fake_labels)
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            
            self.manual_backward(loss_d)
            opt_d.step()
            d_train_steps = 1
        else:
            # Calculate loss for logging without training
            with torch.no_grad():
                real_output = self.discriminator(background)
                fake_output = self.discriminator(fake_background.detach())
                real_labels = torch.ones_like(real_output)
                fake_labels = torch.zeros_like(fake_output)
                loss_d_real = self.adversarial_loss(real_output, real_labels)
                loss_d_fake = self.adversarial_loss(fake_output, fake_labels)
                loss_d = 0.5 * (loss_d_real + loss_d_fake)
        
        # Train Generator
        opt_g.zero_grad()
        fake_background = self.generator(mixture)
        fake_output = self.discriminator(fake_background)
        
        real_labels = torch.ones_like(fake_output)
        loss_g_gan = self.adversarial_loss(fake_output, real_labels)
        loss_g_l1 = self.l1_loss(fake_background, background)
        
        # Adaptive lambda based on generator performance
        if loss_g_gan.item() > self.g_loss_threshold:
            adaptive_lambda = self.lambda_l1 * 1.5
            loss_g = 0.5 * loss_g_gan + adaptive_lambda * loss_g_l1
        else:
            loss_g = loss_g_gan + self.lambda_l1 * loss_g_l1
        
        self.manual_backward(loss_g)
        opt_g.step()
        
        self.g_steps += 1
        
        # Dynamic learning rate adjustment
        if d_accuracy < 0.6:
            for param_group in opt_d.param_groups:
                param_group['lr'] = min(param_group['lr'] * 1.1, 5e-4)
        elif d_accuracy > 0.85:
            for param_group in opt_d.param_groups:
                param_group['lr'] = max(param_group['lr'] * 0.9, 1e-5)
        
        # Logging
        self.log_dict({
            'train/loss_d': loss_d,
            'train/loss_g': loss_g,
            'train/loss_g_gan': loss_g_gan,
            'train/loss_g_l1': loss_g_l1,
            'train/d_accuracy': d_accuracy,
            'train/d_train_steps': d_train_steps,
            'train/lr_g': opt_g.param_groups[0]['lr'],
            'train/lr_d': opt_d.param_groups[0]['lr']
        }, prog_bar=True, on_step=True, on_epoch=True)
        
        return {
            'loss': loss_g,  # Required for progress bar
            'loss_d': loss_d,
            'loss_g': loss_g,
            'loss_g_gan': loss_g_gan,
            'loss_g_l1': loss_g_l1,
            'd_accuracy': d_accuracy
        }

    def validation_step(self, batch, batch_idx):
        mixture, background = batch
        
        if mixture.ndim == 2:
            mixture = mixture.unsqueeze(1)
        if background.ndim == 2:
            background = background.unsqueeze(1)
        
        # Align lengths
        min_length = min(mixture.shape[-1], background.shape[-1])
        mixture = mixture[..., :min_length]
        background = background[..., :min_length]
        
        pred_background = self.generator(mixture)
        val_loss_l1 = self.l1_loss(pred_background, background)
        
        self.log('val/loss_l1', val_loss_l1, prog_bar=True, on_step=False, on_epoch=True)
        return {'val_loss_l1': val_loss_l1}

    def test_step(self, batch, batch_idx):
        mixture, background = batch
        
        if mixture.ndim == 2:
            mixture = mixture.unsqueeze(1)
        if background.ndim == 2:
            background = background.unsqueeze(1)
        
        # Align lengths
        min_length = min(mixture.shape[-1], background.shape[-1])
        mixture = mixture[..., :min_length]
        background = background[..., :min_length]
        
        pred_background = self.generator(mixture)
        test_loss_l1 = self.l1_loss(pred_background, background)
        
        self.log('test/loss_l1', test_loss_l1, on_step=False, on_epoch=True)
        return {'test_loss_l1': test_loss_l1}

    def train_dataloader(self):
        mixture_dataset = AudioDatasetMixture(self.train_mixture_csv)
        background_dataset = AudioDatasetBackground(self.train_background_csv)
        
        mixture_loader = DataLoader(
            mixture_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4,
            persistent_workers=True,
            pin_memory=True
        )
        
        background_loader = DataLoader(
            background_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4,
            persistent_workers=True,
            pin_memory=True
        )
        
        return UnpairedDataLoader(mixture_loader, background_loader)

    def val_dataloader(self):
        mixture_dataset = AudioDatasetMixture(self.val_mixture_csv)
        background_dataset = AudioDatasetBackground(self.val_background_csv)
        
        mixture_loader = DataLoader(
            mixture_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4,
            persistent_workers=True,
            pin_memory=True
        )
        
        background_loader = DataLoader(
            background_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4,
            persistent_workers=True,
            pin_memory=True
        )
        
        return UnpairedDataLoader(mixture_loader, background_loader)

    def test_dataloader(self):
        if self.test_mixture_csv and self.test_background_csv:
            mixture_dataset = AudioDatasetMixture(self.test_mixture_csv)
            background_dataset = AudioDatasetBackground(self.test_background_csv)
            
            mixture_loader = DataLoader(
                mixture_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=4,
                persistent_workers=True,
                pin_memory=True
            )
            
            background_loader = DataLoader(
                background_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=4,
                persistent_workers=True,
                pin_memory=True
            )
            
            return UnpairedDataLoader(mixture_loader, background_loader)
        return None

# === INFERENCE UTILITIES ===
def pydub_to_array(audio: AudioSegment) -> (np.ndarray, int):
    return np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((audio.channels, -1)) / (
            1 << (8 * audio.sample_width - 1)), audio.frame_rate

def array_to_pydub(audio_np_array: np.ndarray, sample_rate: int = 16000, sample_width: int = 2, channels: int = 1) -> AudioSegment:
    return AudioSegment((audio_np_array * (2 ** (8 * sample_width - 1))).astype(np.int16).tobytes(),
                        frame_rate=sample_rate, sample_width=sample_width, channels=channels)

def preprocess_audio(filepath, sample_rate=16000, channels=1):
    audio = AudioSegment.from_file(filepath).set_frame_rate(sample_rate).set_channels(channels)
    audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32)
    audio_array /= (1 << (8 * audio.sample_width - 1))  # normalize [-1, 1]
    audio_tensor = torch.tensor(audio_array).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, T)
    return audio_tensor, audio.frame_rate

def postprocess_and_export(tensor, filename, sample_rate=16000):
    audio_np = tensor.squeeze().cpu().numpy()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio = array_to_pydub(audio_np, sample_rate=sample_rate)
    audio.export(filename, format="wav")

def infer_from_path(model, path_to_wav, output_event_path="event_output.wav", device='cuda'):
    model.eval()
    with torch.no_grad():
        mixture_tensor, sr = preprocess_audio(path_to_wav)
        mixture_tensor = mixture_tensor.to(device)
        output_background = model(mixture_tensor)

        # Allinea dimensioni se serve
        min_len = min(mixture_tensor.shape[-1], output_background.shape[-1])
        mixture_tensor = mixture_tensor[..., :min_len]
        output_background = output_background[..., :min_len]

        # Residuo = Evento
        estimated_event = mixture_tensor - output_background

        postprocess_and_export(estimated_event, output_event_path, sample_rate=sr)
        print(f"Evento stimato salvato in: {output_event_path}")

# === TRAINING SCRIPT ===
def main():
    parser = argparse.ArgumentParser(description='Train Audio GAN with Lightning')
    parser.add_argument('--train_mixture_csv', type=str, default='csv/train_mixture.csv')
    parser.add_argument('--train_background_csv', type=str, default='csv/train_background.csv')
    parser.add_argument('--val_mixture_csv', type=str, default='csv/val_mixture.csv')
    parser.add_argument('--val_background_csv', type=str, default='csv/val_background.csv')
    parser.add_argument('--test_mixture_csv', type=str, default='csv/test_mixture.csv')
    parser.add_argument('--test_background_csv', type=str, default='csv/test_background.csv')
    parser.add_argument('--batch_size', type=int, default=18)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--precision', type=str, default='16-mixed')
    parser.add_argument('--results_dir', type=str, default='results-lightning')
    parser.add_argument('--lr_g', type=float, default=2e-4)
    parser.add_argument('--lr_d', type=float, default=1e-4)
    parser.add_argument('--lambda_l1', type=float, default=10.0)
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Initialize model
    model = AudioGAN(
        train_mixture_csv=args.train_mixture_csv,
        train_background_csv=args.train_background_csv,
        val_mixture_csv=args.val_mixture_csv,
        val_background_csv=args.val_background_csv,
        test_mixture_csv=args.test_mixture_csv,
        test_background_csv=args.test_background_csv,
        batch_size=args.batch_size,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        lambda_l1=args.lambda_l1
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.results_dir,
        filename='audio-gan-{epoch:02d}-{val/loss_l1:.4f}',
        monitor='val/loss_l1',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val/loss_l1',
        patience=10,
        mode='min'
    )
    
    # Logger
    csv_logger = CSVLogger(args.results_dir, name='logs')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.gpus,
        num_nodes=args.num_nodes,
        precision=args.precision,
        callbacks=[checkpoint_callback, early_stopping],
        logger=csv_logger,
        enable_progress_bar=True,
        log_every_n_steps=50,
        val_check_interval=0.5,  # Validate twice per epoch
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        strategy='ddp_find_unused_parameters_true' if args.gpus > 1 or args.num_nodes > 1 else 'auto'
    )
    
    # Training
    trainer.fit(model)
    
    # Test
    if args.test_mixture_csv and args.test_background_csv:
        trainer.test(model, ckpt_path='best')
    
    print(f"Training completed! Results saved in: {args.results_dir}")
    
    # Example inference
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Loading best model from: {best_model_path}")
        best_model = AudioGAN.load_from_checkpoint(best_model_path)
        best_model.eval()
        
        # Example inference (uncomment and modify path as needed)
        # infer_from_path(
        #     best_model.generator, 
        #     "audio_sources/train_set/mix_4962/mixture.wav", 
        #     f"{args.results_dir}/lightning_output_event.wav"
        # )

if __name__ == '__main__':
    main()