import os
import pandas as pd
import torch
from tqdm import tqdm

def trainer_(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20,
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
