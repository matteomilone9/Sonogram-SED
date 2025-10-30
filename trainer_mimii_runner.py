import os
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import warnings

from architectures import CNN_VAT_Anomaly_ADV
from trainer import trainer_MIMII_Optimized
from data_loader import MIMIIDataset

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def test_model(model, test_loader, device, threshold_path):
    model.eval()
    model.to(device)

    if not os.path.exists(threshold_path):
        print(f"⚠️ Threshold non trovata in {threshold_path}")
        return None

    # Legge la soglia percentile
    with open(threshold_path, "r") as f:
        lines = f.readlines()
        threshold_line = [l for l in lines if "Threshold_percentile" in l]
        if not threshold_line:
            print(f"⚠️ Nessuna soglia percentile trovata in {threshold_path}")
            return None
        threshold = float(threshold_line[0].split(":")[1].strip())

    print(f"📏 Threshold caricata (percentile): {threshold:.12f}")

    mse_all, labels_all, preds_all = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            wavs, labels = batch[:2] if isinstance(batch, (list, tuple)) and len(batch) >= 2 else (batch, None)
            wavs = wavs.to(device)
            if labels is not None:
                labels = labels.to(device)

            orig, recon, mu, logvar = model(wavs)
            mse = F.mse_loss(recon, orig, reduction='none').mean(dim=(1, 2))
            mse_all.extend(mse.cpu().numpy())

            preds = (mse.cpu().numpy() >= threshold).astype(int)
            preds_all.extend(preds)

            if labels is not None:
                labels_all.extend(labels.cpu().numpy())

    mse_all = np.array(mse_all)
    preds_all = np.array(preds_all)
    labels_all = np.array(labels_all) if labels_all else None

    if labels_all is not None and len(labels_all) > 0:
        tp = int(np.sum((preds_all == 1) & (labels_all == 1)))
        tn = int(np.sum((preds_all == 0) & (labels_all == 0)))
        fp = int(np.sum((preds_all == 1) & (labels_all == 0)))
        fn = int(np.sum((preds_all == 0) & (labels_all == 1)))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        try:
            auc = roc_auc_score(labels_all, mse_all)
            auc_str = f"{auc:.4f}"
        except:
            auc = None
            auc_str = "NA"
    else:
        # Se label non disponibili
        tp = tn = fp = fn = precision = recall = f1 = auc = None
        auc_str = "NA"

    print(f"\n📊 RISULTATI TEST:")
    if labels_all is not None and len(labels_all) > 0:
        print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        if auc is not None:
            print(f"AUC-ROC: {auc:.4f}")
    print(f"🚨 Percentuale clip anomale rilevate: {preds_all.mean()*100:.2f}%")

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "auc_roc": auc, "auc_str": auc_str,
        "threshold": threshold,
        "mean_mse": mse_all.mean()
    }

# === MAIN LOOP ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device in uso: {device}")

    root_dir = "data/mimii_due"
    machine_types = ["fan", "gearbox", "pump", "slider", "valve"]
    sample_rate = 16_000
    segment_duration = 10
    batch_size = 128
    epochs = 300
    percentile_threshold = 95  # Percentile per anomaly detection

    for machine in machine_types:
        print(f"\n{'='*80}")
        print(f"⚙️ Addestramento modello per macchina: {machine.upper()}")
        print(f"{'='*80}\n")

        save_dir = os.path.join("models_pth/mimii_due", machine)
        os.makedirs(save_dir, exist_ok=True)
        threshold_path = os.path.join(save_dir, "threshold_info.txt")
        results_file = os.path.join(save_dir, "metrics_results.txt")

        # Scrive header solo se file non esiste
        if not os.path.exists(results_file):
            with open(results_file, "w") as f:
                f.write("Machine,TP,FP,TN,FN,Precision,Recall,F1,AUC,Threshold,Mean_MSE\n")

        train_dir = os.path.join(root_dir, f"dev_data_{machine}", machine, "train")
        val_dir = os.path.join(root_dir, f"dev_data_{machine}", machine, "source_test")
        test_dir = os.path.join(root_dir, f"dev_data_{machine}", machine, "target_test")

        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            print(f"❌ Cartella mancante per {machine}, salto macchina.")
            continue

        train_dataset = MIMIIDataset(train_dir, sample_rate=sample_rate, 
                                     segment_duration=segment_duration, mode="train")
        val_dataset   = MIMIIDataset(val_dir, sample_rate=sample_rate, 
                                     segment_duration=segment_duration, mode="train")
        test_dataset  = MIMIIDataset(test_dir, sample_rate=sample_rate, 
                                     segment_duration=segment_duration, mode="test")


        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        model = CNN_VAT_Anomaly_ADV(sample_rate=sample_rate, 
                                    segment_duration=segment_duration, n_fft=1024, hop_length=512)
        save_name = f"vae_mimii_{machine}_cnn.pth"

        # --- TRAIN ---
        trainer_MIMII_Optimized(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=1e-3,
            percentile=percentile_threshold,
            patience=5,
            device=device,
            save_dir=save_dir,
            save_name=save_name,
            resume=True
        )

        # --- TEST ---
        print(f"\n🧪 Avvio test per {machine.upper()}...")
        metrics = test_model(model, test_loader, device, threshold_path)

        # --- SALVATAGGIO METRICHE ---
        if metrics is not None:
            with open(results_file, "a") as f:
                f.write(
                    f"{machine},{metrics['TP']},{metrics['FP']},{metrics['TN']},{metrics['FN']},"
                    f"{metrics['precision']:.4f},{metrics['recall']:.4f},{metrics['f1']:.4f},"
                    f"{metrics['auc_str']},{metrics['threshold']:.12f},{metrics['mean_mse']:.12f}\n"
                )

    print(f"\n✅ Tutti i modelli MIMII DUE addestrati e testati! Metriche salvate nei rispettivi file per ogni macchina.")
