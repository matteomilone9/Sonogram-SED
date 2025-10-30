import warnings
import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup
import io
import os
import csv
from urllib.parse import urljoin
import torch
import torch.nn.functional as F
import torchaudio

from architectures import VAT_Anomaly
from arch_2 import VAT_Anomaly_Linear_ADV

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =========================================================
# === PARAMETRI MODELLO ===
# =========================================================
sample_rate = 192000
segment_duration = 1
seg_len = sample_rate * segment_duration

threshold = 0.00175

# =========================================================
# === FUNZIONI DI SUPPORTO ===
# =========================================================
def load_audio(file_like, target_length=seg_len):
    waveform, sr = torchaudio.load(file_like)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    original_length = waveform.shape[1]  # Lunghezza in campioni

    # Padding o trimming
    if original_length > target_length:
        waveform = waveform[:, :target_length]
        original_length = target_length
    else:
        pad_len = target_length - original_length
        waveform = F.pad(waveform, (0, pad_len))

    return waveform.squeeze(0), original_length  # [16000], valore reale


def run_inference(model, audio_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    model.transform_tf = model.transform_tf.to(device)

    audio_tensor = audio_tensor.unsqueeze(0).to(device)  # [1, 16000]

    with torch.no_grad():
        x, recon, mu, logvar = model(audio_tensor)

    # Calcolo MSE sulla ricostruzione
    loss = F.mse_loss(recon, x).item()
    return x.cpu(), recon.cpu(), loss


def analyze_in_memory(url, username, password, model):
    """
    Scarica e analizza un singolo file audio con il modello VAE_Anomaly.
    """
    resp = requests.get(url, auth=HTTPBasicAuth(username, password), stream=True)
    resp.raise_for_status()
    audio_data = io.BytesIO(resp.content)

    waveform, _ = load_audio(audio_data)
    _, _, loss = run_inference(model, waveform)
    return loss


def list_links(url, username, password):
    resp = requests.get(url, auth=HTTPBasicAuth(username, password))
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    links = [a['href'] for a in soup.find_all('a', href=True)]
    return links


def gather_audio_urls(base_url, username, password):
    urls = []
    links = list_links(base_url, username, password)

    # Cerca direttamente file WAV o FLAC
    for href in links:
        if href.lower().endswith(('.wav', '.flac')):
            urls.append(urljoin(base_url + "/", href))

    # Cerca anche nelle sottocartelle
    for href in links:
        if href.endswith('/'):
            sub_url = urljoin(base_url + "/", href)
            sub_links = list_links(sub_url, username, password)
            for sub_href in sub_links:
                if sub_href.lower().endswith(('.wav', '.flac')):
                    urls.append(urljoin(sub_url, sub_href))
    return urls


def analyze_all(urls, username, password, model, model_type, results_folder, model_name, threshold=threshold):

    os.makedirs(results_folder, exist_ok=True)

    for base_url in urls:
        print(f"\n=== Analisi per: {base_url} ===")
        try:
            audio_urls = gather_audio_urls(base_url, username, password)
        except Exception as e:
            print(f"Errore nel raccogliere URL audio da {base_url}: {e}")
            continue

        if not audio_urls:
            print(f"Nessun file audio trovato in {base_url}")
            continue

        results = []
        for url in audio_urls:
            try:
                loss = analyze_in_memory(url, username, password, model)
                status = "Anomalo" if loss > threshold else "Normale"
                results.append((url, status, loss))
                print(f"{url} --> {status} (MSE: {loss:.6f})")
            except Exception as e:
                print(f"Errore su {url}: {e}")
                results.append((url, "Errore", None))

        # Filtra solo anomalie
        anomalies = [r for r in results if r[1] == "Anomalo"]

        try:
            giorno = base_url.rsplit("/recordings/", 1)[1].split("/")[0]
        except IndexError:
            giorno = "unknown"

        # Salvataggio CSV
        # csv_filename = os.path.join(results_folder, f"{model_type}_esito_{giorno}.csv")

        # Estrai anno_giorno (es. 2025-290) e device (es. RSP2-MIC1) dal path
        try:
            giorno = base_url.rsplit("/recordings/", 1)[1].split("/")[0]  # 2025-290
            giorno_fmt = giorno.replace("-", "_")  # diventa 2025_290
        except IndexError:
            giorno_fmt = "unknown"

        # Estrai il device (ultimo elemento dopo /device/)
        try:
            device_name = base_url.split("/device/")[1].split("/")[0]
        except IndexError:
            device_name = "unknown_device"

        # Crea il nome dinamico del file CSV
        csv_filename = os.path.join(
            results_folder,
            f"{giorno_fmt}_{model_name}_{device_name}.csv"
        )

        def save_csv(csv_path, data):
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["url", "status", "loss"])
                for url, status, loss in data:
                    loss_str = f"{loss:.6f}" if loss is not None else ""
                    writer.writerow([url, status, loss_str])

        save_csv(csv_filename, anomalies)
        print(f"Salvate {len(anomalies)} anomalie in '{csv_filename}'")

        prepend_metadata_to_csv(csv_filename, modello=model_name, tipo_file="FLAC",
                               durata_segmento="1", verificato1="0", verificato2="1")


def prepend_metadata_to_csv(csv_filename, modello, tipo_file, durata_segmento, verificato1, verificato2):
    header = ["modello", "tipo_file", "durata_segmento", "verificato", "verificato"]
    metadata = [modello, tipo_file, durata_segmento, verificato1, verificato2]
    try:
        with open(csv_filename, mode='r') as original:
            original_lines = original.readlines()
        with open(csv_filename, mode='w', newline='') as modified:
            writer = csv.writer(modified)
            writer.writerow(header)
            writer.writerow(metadata)
            modified.writelines(original_lines)
    except Exception as e:
        print(f"Errore inserimento meta-intestazione in {csv_filename}: {e}")


# =========================================================
# === MAIN ===
# =========================================================
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Carica modello
    model_path = "models_pth/NS_V2_192K_ADV/192K_best_vae_model_1s.pth"
    model = VAT_Anomaly_Linear_ADV(sample_rate=192000)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model_type = '1s'
    model_name = 'NS_V2_192K_ADV'

    # Lista cartelle audio da analizzare
    base_urls = [
        "https://lys-ai.it/recordings/8c69c0b3-ae12-4552-9b11-0aaa0304a06d/recordings/2025-292/FLAC/device/RSP2-MIC1",
        ]

    username = "milone"
    password = "Neurone-pc"

    # Cartella di salvataggio
    results_folder = "esiti/esiti_NS_V2_192k_ADV"

    # Analizza tutti i file
    analyze_all(base_urls, username, password, model, model_type, results_folder, 
                model_name=model_name, threshold=threshold)
