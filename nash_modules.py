import warnings
import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup
import io
import os
import csv
from urllib.parse import urljoin

from aed_models import *

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# === PARAMETRI MODELLO ===
sample_rate = 16000
duration = 1
seg_len = sample_rate * duration
threshold = 0.0020
tolerance = 0.0002

# === FUNZIONI ===
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

    return waveform.squeeze(0), original_length  # [160000], valore reale


def run_inference(model, audio_tensor, original_length, threshold=threshold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    model.transform_tf = model.transform_tf.to(device)

    audio_tensor = audio_tensor.unsqueeze(0).to(device)  # [1, 160000]
    feature_vector = model.preprocessing(audio_tensor)   # [1, T, 640]

    # Calcolo numero di frame reali
    hop_length = model.transform_tf.hop_length
    real_t_bins = 1 + (original_length // hop_length)
    real_vector_array_size = real_t_bins - model.frames + 1

    with torch.no_grad():
        encoded = model.encoder(feature_vector)
        bottleneck = model.bottleneck(encoded)
        reconstructed = model.decoder(bottleneck)

    # Taglio alla lunghezza effettiva in feature (T)
    feature_vector = feature_vector[:, :real_vector_array_size, :]
    reconstructed = reconstructed[:, :real_vector_array_size, :]

    # Calcolo errore
    loss = F.mse_loss(reconstructed, feature_vector).item()
    return feature_vector.cpu(), reconstructed.cpu(), loss


def analyze_in_memory(url, username, password, model):
    resp = requests.get(url, auth=HTTPBasicAuth(username, password), stream=True)
    resp.raise_for_status()
    audio_data = io.BytesIO(resp.content)

    waveform, real_samples = load_audio(audio_data)
    _, _, loss = run_inference(model, waveform, real_samples)

    return loss

def list_links(url, username, password):
    print(f"Listo i link da: {url}")
    resp = requests.get(url, auth=HTTPBasicAuth(username, password))
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    links = [a['href'] for a in soup.find_all('a', href=True)]
    print(f"Link trovati: {links}")
    return links

def gather_audio_urls(base_url, username, password):
    urls = []
    links = list_links(base_url, username, password)

    # Aggiunta 1: cerca direttamente i file FLAC in base_url
    for href in links:
        if href.lower().endswith(('.wav', '.flac')):
            urls.append(urljoin(base_url + "/", href))

    # Aggiunta 2: cerca anche nelle eventuali sottocartelle
    for href in links:
        if href.endswith('/'):  # è una cartella
            sub_url = urljoin(base_url + "/", href)
            sub_links = list_links(sub_url, username, password)
            for sub_href in sub_links:
                if sub_href.lower().endswith(('.wav', '.flac')):
                    urls.append(urljoin(sub_url, sub_href))

    return urls


def analyze_all(urls, username, password, model, model_type, results_folder,
                threshold=threshold, tolerance=tolerance):
    """
    Analizza tutti i file audio nei base_url, salva i risultati in CSV:
    - anomalie oltre la soglia + tolleranza
    - falsi positivi nella zona di tolleranza ±0.0002
    """

    lower_bound = threshold - tolerance
    upper_bound = threshold + tolerance

    # Creazione cartella risultati se non esiste
    os.makedirs(results_folder, exist_ok=True)

    for base_url in urls:
        print(f"\n=== Analisi per: {base_url} ===")

        try:
            audio_urls = gather_audio_urls(base_url, username, password)
        except Exception as e:
            print(f"Errore nel raccogliere URL audio da {base_url}: {e}")
            continue

        if not audio_urls:
            print(f"Attenzione: nessun file audio trovato in {base_url}")
            continue

        results = []
        for url in audio_urls:
            print(f"\nAnalizzo: {url}")
            try:
                loss = analyze_in_memory(url, username, password, model)

                if loss > upper_bound:
                    status = "Anomalo"
                elif lower_bound < loss <= upper_bound:
                    status = "FalsoPositivo"
                else:
                    status = "Normale"

                print(f" --> Risultato: {status} (MSE: {loss:.6f})")
                results.append((url, status, loss))
            except Exception as e:
                print(f"Errore su {url}: {e}")
                results.append((url, "Errore", None))

        # Separiamo anomalie e falsi positivi
        anomalies = [r for r in results if r[1] == "Anomalo"]

        # includo anche i falsi positivi che superano la soglia
        anomalies.extend([r for r in results if r[1] == "FalsoPositivo" and r[2] is not None and r[2] > threshold])

        falsi_positivi = [r for r in results if r[1] == "FalsoPositivo"]

        try:
            giorno = base_url.rsplit("/recordings/", 1)[1].split("/")[0]
        except IndexError:
            print(f"Impossibile estrarre il giorno dal path: {base_url}")
            giorno = "unknown"

        # === CSV anomalie ===
        csv_filename = os.path.join(results_folder, f"{model_type}_esito_{giorno}.csv")
        try:
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["url", "status", "loss"])
                for url, status, loss in anomalies:
                    loss_str = f"{loss:.6f}" if loss is not None else ""
                    writer.writerow([url, status, loss_str])
            print(f"\nSalvati {len(anomalies)} file anomali in '{csv_filename}'")

            # rilevo tipo file (WAV o FLAC) dal primo risultato
            tipo_file = "FLAC"
            if anomalies:
                if anomalies[0][0].lower().endswith(".wav"):
                    tipo_file = "WAV"

            prepend_metadata_to_csv(
                csv_filename=csv_filename,
                modello="NS_V1.01S_V6_2",
                tipo_file=tipo_file,
                durata_segmento="1",
                verificato1="0",
                verificato2="1"
            )

        except Exception as e:
            print(f"Errore durante il salvataggio di {csv_filename}: {e}")

        # === CSV falsi positivi ===
        csv_fp_filename = os.path.join(results_folder, f"{model_type}_falsipositivi_{giorno}.csv")
        try:
            with open(csv_fp_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["url", "status", "loss"])
                for url, status, loss in falsi_positivi:
                    loss_str = f"{loss:.6f}" if loss is not None else ""
                    writer.writerow([url, status, loss_str])
            print(f"\nSalvati {len(falsi_positivi)} file falsi positivi in '{csv_fp_filename}'")

            # rilevo tipo file (WAV o FLAC) dal primo risultato
            tipo_file = "FLAC"
            if falsi_positivi:
                if falsi_positivi[0][0].lower().endswith(".wav"):
                    tipo_file = "WAV"

            prepend_metadata_to_csv(
                csv_filename=csv_fp_filename,
                modello="NS V1.01S_V6",
                tipo_file=tipo_file,
                durata_segmento="1",
                verificato1="0",
                verificato2="1"
            )

        except Exception as e:
            print(f"Errore durante il salvataggio di {csv_fp_filename}: {e}")


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

        print(f"Meta-intestazione inserita in '{csv_filename}'")
    except Exception as e:
        print(f"Errore durante l'inserimento della meta-intestazione in {csv_filename}: {e}")
