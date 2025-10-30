import os
import io
import csv
import shlex
import tempfile
import torch
import torch.nn.functional as F
import torchaudio
import paramiko
from datetime import datetime, timedelta

from typing import List, Optional
from arch_2 import VAT_Anomaly_Linear_ADV
from arch_3 import VAT_Anomaly_CNN_ADV

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# === CONFIGURAZIONE REMOTA ===
# =========================================================
REMOTE_HOST = "151.14.46.173"
REMOTE_USER = "milone"
SSH_PORT = 2200
SSH_KEY = "lys-ai_keys/id_rsa"
UUID_2 ="8c69c0b3-ae12-4552-9b11-0aaa0304a06d"
REMOTE_BASE = f"/mnt/BB-S_STORAGE/{UUID_2}/recordings"


# =========================================================
# === PARAMETRI MODELLO ===
# =========================================================
sample_rate = 192_000
duration = 1
seg_len = sample_rate * duration
threshold = 0.000000001968


# =========================================================
# === FUNZIONI AUDIO ===
# =========================================================
def load_audio(file_like, target_length: int = seg_len):
    waveform, sr = torchaudio.load(file_like)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    original_length = waveform.shape[1]
    if original_length > target_length:
        waveform = waveform[:, :target_length]
    else:
        pad_len = target_length - original_length
        waveform = F.pad(waveform, (0, pad_len))

    return waveform.squeeze(0), original_length


def run_inference(model, audio_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    model.transform_tf = model.transform_tf.to(device)

    audio_tensor = audio_tensor.unsqueeze(0).to(device)  # [1, 16000]
    with torch.no_grad():
        x, recon, mu, logvar = model(audio_tensor)
    loss = F.mse_loss(recon, x).item()
    return x.cpu(), recon.cpu(), loss


# =========================================================
# === SSH / SFTP UTILITY ===
# =========================================================
def _open_sftp():
    transport = paramiko.Transport((REMOTE_HOST, SSH_PORT))
    transport.connect(username=REMOTE_USER, pkey=paramiko.RSAKey.from_private_key_file(SSH_KEY))
    sftp = paramiko.SFTPClient.from_transport(transport)
    return sftp, transport


def _ssh_exec(remote_cmd: str, check: bool = True):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(REMOTE_HOST, port=SSH_PORT, username=REMOTE_USER, key_filename=SSH_KEY)
    stdin, stdout, stderr = ssh.exec_command(remote_cmd)
    out = stdout.read().decode("utf-8", errors="ignore")
    err = stderr.read().decode("utf-8", errors="ignore")
    code = stdout.channel.recv_exit_status()
    ssh.close()
    if check and code != 0:
        raise RuntimeError(f"SSH error ({code}): {err.strip()}")
    return out


# =========================================================
# === ANALISI FILE REMOTI ===
# =========================================================
def analyze_remote_file_sftp(sftp: paramiko.SFTPClient, remote_path: str, model) -> float:
    """
    Legge il file remoto in memoria e calcola la perdita MSE con VAT_Anomaly.
    """
    try:
        with sftp.open(remote_path, "rb") as f_remote:
            data = f_remote.read()

        bio = io.BytesIO(data)
        waveform, _ = load_audio(bio)
        _, _, loss = run_inference(model, waveform)
        return loss

    except Exception as e:
        print(f"❌ Errore durante l'analisi di {remote_path}: {e}")
        raise


def analyze_day_full_ssh(giorno: str,
                         device_da_analizzare: str,
                         tipo: str,
                         model,
                         model_name: str,
                         batch_size: int = 1000):
    """
    Analizza tutti i file audio (WAV/FLAC) di un giorno e device remoto,
    salvando solo le anomalie nel CSV remoto.
    Se il CSV esiste già, riprende dal file successivo all'ultimo elaborato.
    """
    try:
        sftp, transport = _open_sftp()
    except Exception as e:
        print(f"❌ Errore apertura SFTP: {e}")
        return

    if tipo == "WAV":
        base_dir = f"{REMOTE_BASE}/{giorno}/device/{device_da_analizzare}"
        out_dir = f"/mnt/BB-S_STORAGE/{UUID_2}/output/{giorno}/WAV/{device_da_analizzare}"
    else:
        base_dir = f"{REMOTE_BASE}/{giorno}/FLAC/device/{device_da_analizzare}"
        out_dir = f"/mnt/BB-S_STORAGE/{UUID_2}/output/{giorno}/FLAC/{device_da_analizzare}"

    out_csv = f"{out_dir}/{giorno}_{model_name}_{device_da_analizzare}_{tipo}_AI_alert.csv"

    print(f"📖 Analisi {tipo} per giorno {giorno}, device {device_da_analizzare}")
    print(f"👉 Cartella input: {base_dir}")
    print(f"👉 Output CSV: {out_csv}")

    # 🔹 Crea cartella output se manca
    _ssh_exec(f"mkdir -p -m 2775 {shlex.quote(out_dir)}", check=False)

    # 🔹 Lista file ordinati
    find_cmd = (
        f"test -d {shlex.quote(base_dir)} && "
        f"find {shlex.quote(base_dir)} -type f -iname '*.{tipo.lower()}' || true"
    )
    out = _ssh_exec(find_cmd, check=False)
    files = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
    files.sort()

    if not files:
        print(f"⚠️ Nessun file trovato in {base_dir}")
        sftp.close()
        transport.close()
        return

    print(f"[SSH] Trovati {len(files)} file in {base_dir}")

    # =====================================================
    # === Resume: riprende dal primo file successivo a quello già processato
    # =====================================================
    last_processed = None
    try:
        sftp.stat(out_csv)
        print(f"📂 CSV già esistente, controllo ultimo file elaborato...")

        with sftp.open(out_csv, "r") as remote_f:
            with io.TextIOWrapper(remote_f, encoding="utf-8") as wrapper:
                for row in wrapper:
                    last_row = row.strip()
                if last_row and not last_row.startswith(("modello", "path")):
                    last_processed = last_row.split(",")[0]
                    print(f"🔁 Ultimo file registrato nel CSV: {last_processed}")
    except FileNotFoundError:
        # Se non esiste il CSV, crealo
        with sftp.open(out_csv, "w") as remote_f:
            with io.TextIOWrapper(remote_f, encoding="utf-8", newline="") as wrapper:
                writer = csv.writer(wrapper)
                writer.writerow(["modello", "tipo_file", "durata_segmento", "verificato"])
                writer.writerow([model_name, tipo, "1", "0"])
                writer.writerow(["path", "status", "loss"])
        sftp.chmod(out_csv, 0o664)
        print(f"✅ Creato nuovo CSV con intestazione: {out_csv}")

    # 🔹 Se trovato ultimo file, salta tutti i precedenti
    if last_processed and last_processed in files:
        idx = files.index(last_processed)
        files = files[idx + 1:]
        print(f"⏩ Riparto dal file successivo ({len(files)} rimanenti).")
    elif last_processed:
        print(f"⚠️ Ultimo file {last_processed} non trovato nella lista. Analizzo tutto da capo.")

    if not files:
        print(f"✅ Tutti i file già analizzati. Nessun nuovo file da elaborare.")
        sftp.close()
        transport.close()
        return

    # =====================================================
    # === Analisi in batch ===
    # =====================================================
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        print(f"👉 Analisi batch {i // batch_size + 1}: {len(batch)} file")

        rows = []
        for fpath in batch:
            try:
                loss = analyze_remote_file_sftp(sftp, fpath, model)
                if loss > threshold:
                    status = "Anomalo"
                    print(f"  {os.path.basename(fpath)} → {status} (MSE: {loss:.12f})")
                    rows.append([fpath, status, f"{loss:.12f}"])
                else:
                    print(f"  {os.path.basename(fpath)} → Normale (MSE: {loss:.12f})")
            except Exception as e:
                print(f"  ❌ Errore su {fpath}: {e}")

        # 🔹 Scrive solo le anomalie trovate
        if rows:
            with sftp.open(out_csv, "a") as remote_f:
                with io.TextIOWrapper(remote_f, encoding="utf-8", newline="") as wrapper:
                    writer = csv.writer(wrapper)
                    writer.writerows(rows)
            print(f"✅ Salvato batch {i // batch_size + 1} ({len(rows)} anomalie) su {out_csv}")

    sftp.close()
    transport.close()
    print("🏁 Analisi completata.")



# =========================================================
# === MAIN ===
# =========================================================
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Caricamento modello
    model_path = "models_pth/CNN_NS_V2_192K_ADV/CNN_192K_best_vae_model_1s.pth"
    model = VAT_Anomaly_CNN_ADV(sample_rate=sample_rate, segment_duration=duration)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model_name = "TEST_NS_V2_CNN_LINEAR"

    # === Range DOY da analizzare ===
    start_doy = "2025-283"   # Giorno iniziale
    end_doy   = "2025-284"   # Giorno finale

    # === Device da analizzare 
    devices = ["RSP2-MIC1"]

    # === Formato
    tipo = "FLAC"   # oppure "WAV"

    # === Conversione e ciclo ===
    start_date = datetime.strptime(start_doy, "%Y-%j")
    end_date = datetime.strptime(end_doy, "%Y-%j")

    total_days = (end_date - start_date).days + 1
    print(f"📅 Analisi da {start_doy} a {end_doy} → {total_days} giorni")

    for i in range(total_days):
        giorno = (start_date + timedelta(days=i)).strftime("%Y-%j")
        print(f"\n🗓️  Giorno {giorno}")
        for device_da_analizzare in devices:
            analyze_day_full_ssh(
                giorno=giorno,
                device_da_analizzare=device_da_analizzare,
                tipo=tipo,
                model=model,
                model_name=model_name,
                batch_size=1000
            )
