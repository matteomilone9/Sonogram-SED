import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


### TRAINER AUTOENCODER ###
def trainer_NS1(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20,
                      results_dir="results", csv_name="training_log.csv", model_name="best_model.pth"):
    """
    Train an autoencoder model and save training logs and best model.

    Args:
        model: Autoencoder model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda or cpu)
        num_epochs: Number of training epochs
        results_dir: Directory to save results
        csv_name: Name of training log file
        model_name: Name of best model
    """

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # File CSV per il log delle metriche
    csv_filename = os.path.join(results_dir, csv_name)

    # Crea un DataFrame vuoto per il log
    if os.path.exists(csv_filename):
        log_df = pd.read_csv(csv_filename)
    else:
        log_df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Train", unit="batch") as train_bar:
            for input in train_bar:
                input = input.to(device)

                optimizer.zero_grad()

                original, reconstructed = model(input)
                loss = criterion(reconstructed, original)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                train_bar.set_postfix(loss=train_loss / (train_bar.n + 1))

        train_loss /= len(train_loader)

        # Validazione
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation", unit="batch") as val_bar:
                for input in val_bar:
                    input = input.to(device)
                    original, reconstructed = model(input)

                    loss = criterion(reconstructed, original)
                    val_loss += loss.item()

                    val_bar.set_postfix(loss=val_loss / (val_bar.n + 1))

            val_loss /= len(val_loader)

        # Stampa risultati
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print("-" * 70)

        # Aggiorna il log
        new_row = pd.DataFrame({
            "epoch": [epoch + 1],
            "train_loss": [train_loss],
            "val_loss": [val_loss]
        })
        log_df = pd.concat([log_df, new_row], ignore_index=True)
        log_df.to_csv(csv_filename, index=False)

        # Salvataggio modello se migliora
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(results_dir, model_name)
            torch.save(model.state_dict(), model_path)
            print(f"Best Model Saved (Epoch {epoch + 1}, Val Loss: {val_loss:.4f})")
            print("-" * 70)

    print(f"Training log salvato in {csv_filename}")


### VAE/VAT LOSS AND TRAINER LOOP ###
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + 0.001 * kl_loss
    return total_loss, recon_loss, kl_loss


def trainer_VAE(model, train_loader, val_loader,
                epochs=50, lr=1e-3, device='cuda',
                patience=10,
                save_dir="models",
                save_name="vae_best_model.pth",
                resume=False):

    # Crea la cartella se non esiste
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, save_name)
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
    threshold_path = os.path.join(save_dir, "threshold_info.txt")

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)

    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Resume checkpoint
    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        print(f"🔄 Resume training da epoca {start_epoch}, best_val_loss={best_val_loss:.6f}")

    for epoch in range(start_epoch, epochs):
        # ---------------- TRAIN ----------------
        model.train()
        total_loss, total_recon, total_kl = 0, 0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            batch = batch.to(device)
            original, recon, mu, logvar = model(batch)
            loss, recon_l, kl_l = vae_loss(recon, original, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_l.item()
            total_kl += kl_l.item()

        scheduler.step()
        n = len(train_loader)
        print(f"[Train] Epoch {epoch+1}/{epochs} | Total: {total_loss/n:.6f} | Recon: {total_recon/n:.6f} | KL: {total_kl/n:.6f}")

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss, val_recon, val_kl = 0, 0, 0
        recon_losses = []  # per threshold

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]"):
                batch = batch.to(device)
                orig, recon, mu, logvar = model(batch)
                loss, recon_l, kl_l = vae_loss(recon, orig, mu, logvar)

                val_loss += loss.item()
                val_recon += recon_l.item()
                val_kl += kl_l.item()

                # MSE per threshold
                mse = F.mse_loss(recon, orig, reduction='mean').item()
                recon_losses.append(mse)

        n_val = len(val_loader)
        val_loss_mean = val_loss / n_val
        print(f"[Val] Epoch {epoch+1}/{epochs} | Total: {val_loss_mean:.6f} | Recon: {val_recon/n_val:.6f} | KL: {val_kl/n_val:.6f}")

        # Calcolo soglia direttamente
        recon_losses = np.array(recon_losses)
        mean_recon = float(recon_losses.mean())
        std_recon = float(recon_losses.std())
        threshold = mean_recon + 3 * std_recon
        print(f"Threshold (3σ): {threshold:.6f} | Mean: {mean_recon:.6f} | Std: {std_recon:.6f}")

        # Salvataggio solo se migliora
        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"MIGLIOR modello salvato in: {best_model_path} (val_loss={best_val_loss:.6f})")

            # Salva anche la soglia nella stessa cartella
            with open(threshold_path, "w") as f:
                f.write(f"Epoch: {epoch+1}\n")
                f.write(f"Mean_recon_loss: {mean_recon:.8f}\n")
                f.write(f"Std_recon_loss: {std_recon:.8f}\n")
                f.write(f"Threshold_3sigma: {threshold:.8f}\n")
            print(f"Soglia salvata in: {threshold_path}")
        else:
            epochs_no_improve += 1
            print(f"NESSUN MIGLIORAMENTO ({epochs_no_improve}/{patience})")

        # Salva checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'epochs_no_improve': epochs_no_improve
        }, checkpoint_path)

        # Early stopping
        if epochs_no_improve >= patience:
            print("Early stopping: nessun miglioramento nelle ultime epoche.")
            break

    print(f"TRAINING TERMINATO. Miglior modello salvato in: {best_model_path}")


### GMVAE ###
### TRAIN LOOP GMVAE ###
def trainer_GMVAE(model, train_loader, val_loader,
                  epochs=50, lr=1e-3, device='cuda',
                  patience=10,
                  save_dir="models",
                  save_name="gmvae_best_model.pth",
                  resume=False):

  
    # === Crea cartelle e percorsi ===
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, save_name)
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)

    # === Variabili per resume ===
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # === Resume training se richiesto ===
    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        print(f"🔄 Resume training da epoca {start_epoch}, best_val_loss={best_val_loss:.6f}")

    # === Funzione di perdita GMVAE ===
    def gmv_loss(original, recon, mu, logvar, z, cluster_prob,
                 prior_mu, prior_logvar, prior_logits):
        """
        Combina:
         - ricostruzione (MSE)
         - KL_z tra posterior q(z|x) e mixture prior p(z|c)
         - KL_c tra q(c|z) e prior p(c)
        """
        # 1️⃣ Reconstruction loss
        recon_loss = F.mse_loss(recon, original, reduction='mean')

        # 2️⃣ KL_z: dipende dal mixture
        mu_expand = mu.unsqueeze(2)         # (B, T, 1, D)
        logvar_expand = logvar.unsqueeze(2) # (B, T, 1, D)
        prior_mu = prior_mu.unsqueeze(0).unsqueeze(0)          # (1, 1, K, D)
        prior_logvar = prior_logvar.unsqueeze(0).unsqueeze(0)  # (1, 1, K, D)

        kl_z = 0.5 * torch.sum(
            cluster_prob.unsqueeze(-1) * (
                torch.exp(logvar_expand - prior_logvar)
                + (mu_expand - prior_mu)**2 / torch.exp(prior_logvar)
                - 1 + prior_logvar - logvar_expand
            ),
            dim=[2, 3]
        ).mean()

        # 3️⃣ KL_c: differenza tra distribuzioni sui cluster
        kl_c = torch.sum(
            cluster_prob * (
                torch.log(cluster_prob + 1e-8)
                - F.log_softmax(prior_logits, dim=0)
            ),
            dim=-1
        ).mean()

        # 4️⃣ Total loss
        total_loss = recon_loss + kl_z + kl_c
        return total_loss, recon_loss, kl_z, kl_c

    # === Training loop ===
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss, total_recon, total_kl_z, total_kl_c = 0, 0, 0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            batch = batch.to(device)
            original, recon, mu, logvar, z, cluster_prob = model(batch)
            loss, recon_l, kl_z, kl_c = gmv_loss(
                original, recon, mu, logvar, z, cluster_prob,
                model.prior_mu, model.prior_logvar, model.prior_logits
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_l.item()
            total_kl_z += kl_z.item()
            total_kl_c += kl_c.item()

        scheduler.step()
        n = len(train_loader)
        print(f"[Train] Epoch {epoch+1}/{epochs} | Total: {total_loss/n:.6f} | Recon: {total_recon/n:.6f} | KL_z: {total_kl_z/n:.6f} | KL_c: {total_kl_c/n:.6f}")

        # === Validation ===
        model.eval()
        val_loss, val_recon, val_kl_z, val_kl_c = 0, 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]"):
                batch = batch.to(device)
                orig, recon, mu, logvar, z, cluster_prob = model(batch)
                loss, recon_l, kl_z, kl_c = gmv_loss(
                    orig, recon, mu, logvar, z, cluster_prob,
                    model.prior_mu, model.prior_logvar, model.prior_logits
                )
                val_loss += loss.item()
                val_recon += recon_l.item()
                val_kl_z += kl_z.item()
                val_kl_c += kl_c.item()

        n_val = len(val_loader)
        val_loss_mean = val_loss / n_val
        print(f"[Val] Epoch {epoch+1}/{epochs} | Total: {val_loss_mean:.6f} | Recon: {val_recon/n_val:.6f} | KL_z: {val_kl_z/n_val:.6f} | KL_c: {val_kl_c/n_val:.6f}")

        # === Salvataggio miglior modello ===
        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ MIGLIOR modello salvato in: {best_model_path} (val_loss={best_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"NESSUN MIGLIORAMENTO ({epochs_no_improve}/{patience})")

        # === Checkpoint ===
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'epochs_no_improve': epochs_no_improve
        }, checkpoint_path)

        # === Early stopping ===
        if epochs_no_improve >= patience:
            print("⏹️ Early stopping: nessun miglioramento nelle ultime epoche.")
            break

    print(f"🏁 TRAINING TERMINATO. Miglior modello salvato in: {best_model_path}")
