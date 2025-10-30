import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import warnings
from sklearn.metrics import roc_auc_score

from data_loader import AudioDataset
from architectures import CNN_VAT_Anomaly_ADV
from trainer import CNN_trainer_VAE_Optimized  

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
            # batch può essere tuple (wavs, labels) oppure solo wavs
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                wavs, labels = batch
                labels = labels.to(device)
            else:
                wavs, labels = batch, None

            wavs = wavs.to(device)

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


# === MAIN ===
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device in uso: {device}")

    batch_size = 128
    sample_rate = 192000
    segment_duration = 1
    epochs = 30
    percentile_threshold = 95  # Percentile per anomaly detection

    # --- PATH DATASET ---
    train_dir = "data/dataset_192k"
    test_dir  = "data/testset_192k"   

    # --- DATASET ---
    ae_dataset = AudioDataset(
        audio_dir=train_dir,
        segment_duration=segment_duration,
        sample_rate=sample_rate,
        mode="train"
    )

    # --- Creiamo un subset del training dataset ---
    subset_fraction = 1
    subset_size = int(len(ae_dataset) * subset_fraction)

    generator = torch.Generator().manual_seed(42)
    subset_dataset, _ = random_split(ae_dataset, [subset_size, len(ae_dataset) - subset_size], generator=generator)

    # --- Split train/val sul subset ---
    train_size = int(0.75 * len(subset_dataset))
    val_size = len(subset_dataset) - train_size
    ae_train, ae_val = random_split(subset_dataset, [train_size, val_size], generator=generator)

    # --- DataLoader ---
    train_loader = DataLoader(ae_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ae_val, batch_size=batch_size, shuffle=False)

    # --- Dataset e DataLoader test ---
    test_dataset = AudioDataset(
        audio_dir=test_dir,
        segment_duration=segment_duration,
        sample_rate=sample_rate,
        mode="test"
    )
    subset_testset, _ = random_split(test_dataset, [subset_size, len(test_dataset) - subset_size], generator=generator)
    test_loader = DataLoader(subset_testset, batch_size=batch_size, shuffle=False)

    # --- MODEL ---
    model = CNN_VAT_Anomaly_ADV(sample_rate=sample_rate)

    save_dir = "models_pth/CNN_NS_V2_192K_ADV"
    os.makedirs(save_dir, exist_ok=True)
    save_name = "CNN_192K_best_vae_model_1s.pth"
    threshold_path = os.path.join(save_dir, "threshold_info.txt")
    metrics_file = os.path.join(save_dir, "metrics_results.txt")

    if not os.path.exists(metrics_file):
        with open(metrics_file, "w") as f:
            f.write("TP,FP,TN,FN,Precision,Recall,F1,AUC,Threshold,Mean_MSE\n")

    # --- TRAIN ---
    CNN_trainer_VAE_Optimized(
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
    print(f"\n🧪 Avvio test...")
    metrics = test_model(model, test_loader, device, threshold_path)

    # --- SALVATAGGIO METRICHE ---
    if metrics is not None:
        with open(metrics_file, "a") as f:
            f.write(
                f"{metrics['TP']},{metrics['FP']},{metrics['TN']},{metrics['FN']},"
                f"{metrics['precision']:.4f},{metrics['recall']:.4f},{metrics['f1']:.4f},"
                f"{metrics['auc_str']},{metrics['threshold']:.12f},{metrics['mean_mse']:.12f}\n"
            )

    print(f"\n✅ Addestramento e test completati! Metriche salvate in {metrics_file}")
