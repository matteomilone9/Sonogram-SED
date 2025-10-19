import os
import pandas as pd
import torch
import torch.nn as nn
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

### LOSS ###
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + 0.001 * kl_loss
    return total_loss, recon_loss, kl_loss

### TRAIN LOOP ###
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

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)

    # Inizializza variabili per il resume
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Carica checkpoint se resume=True e file esiste
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

        # ---------------- Validation ----------------
        model.eval()
        val_loss, val_recon, val_kl = 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]"):
                batch = batch.to(device)
                orig, recon, mu, logvar = model(batch)
                loss, recon_l, kl_l = vae_loss(recon, orig, mu, logvar)
                val_loss += loss.item()
                val_recon += recon_l.item()
                val_kl += kl_l.item()

        n_val = len(val_loader)
        val_loss_mean = val_loss / n_val
        print(f"[Val] Epoch {epoch+1}/{epochs} | Total: {val_loss_mean:.6f} | Recon: {val_recon/n_val:.6f} | KL: {val_kl/n_val:.6f}")

        # Salvataggio solo se migliora
        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"MIGLIOR modello salvato in: {best_model_path} (val_loss={best_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"NESSUN MIGLIORAMENTO ({epochs_no_improve}/{patience})")

        # Salva checkpoint ogni epoca per resume
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

    print(f"TRAINING TEMRINATO. Miglior modello salvato in: {best_model_path}")