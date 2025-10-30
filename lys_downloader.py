import os
import requests
import csv
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from requests.auth import HTTPBasicAuth


def download_flac_from_url(base_url, output_folder, username, password):
    """Scarica tutti i file .flac presenti in una directory HTTP indicizzata"""
    os.makedirs(output_folder, exist_ok=True)

    print(f"🔍 Esploro la directory: {base_url}")

    try:
        # Ottieni la pagina HTML che elenca i file
        resp = requests.get(base_url, auth=HTTPBasicAuth(username, password))
        resp.raise_for_status()
    except Exception as e:
        print(f"❌ Errore durante l'accesso a {base_url}: {e}")
        return

    # Analizza i link nella pagina
    soup = BeautifulSoup(resp.text, "html.parser")
    links = [a.get("href") for a in soup.find_all("a") if a.get("href")]

    flac_links = [l for l in links if l.lower().endswith(".flac")]

    if not flac_links:
        print("⚠️  Nessun file .flac trovato nella directory.")
        return

    print(f"🎧 Trovati {len(flac_links)} file FLAC. Inizio il download...")

    for link in flac_links:
        full_url = urljoin(base_url, link)
        filename = os.path.basename(link)
        out_path = os.path.join(output_folder, filename)

        print(f"⬇️  Scarico {filename} ...")

        try:
            file_resp = requests.get(full_url, auth=HTTPBasicAuth(username, password))
            file_resp.raise_for_status()

            with open(out_path, "wb") as f:
                f.write(file_resp.content)

        except Exception as e:
            print(f"❌ Errore scaricando {full_url}: {e}")

    print("✅ Download completato!")


def download_flac_from_csv(csv_paths, output_folder, username=None, password=None, only_anomalies=True):
    """
    Scarica i file .flac dalle URL contenute in uno o più CSV.
    
    csv_paths: lista di percorsi a file CSV
    output_folder: cartella dove salvare i file
    username, password: credenziali per autenticazione HTTP (opzionale)
    only_anomalies: se True, scarica solo le righe con status "Anomalo"
    """
    os.makedirs(output_folder, exist_ok=True)
    
    headers_skipped = False

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"⚠️  CSV non trovato: {csv_path}")
            continue
        
        print(f"📄 Leggo CSV: {csv_path}")

        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Salta eventuali righe di metadata iniziali fino a trovare l'intestazione url,status,loss
        start_idx = 0
        for i, row in enumerate(rows):
            if len(row) >= 3 and row[0].lower() == 'url' and row[1].lower() == 'status':
                start_idx = i + 1
                break

        flac_rows = rows[start_idx:]

        if not flac_rows:
            print(f"⚠️  Nessuna riga utile trovata in {csv_path}")
            continue

        for row in flac_rows:
            if len(row) < 2:
                continue
            url, status = row[0], row[1]

            if only_anomalies and status.lower() != "anomalo":
                continue

            filename = os.path.basename(urlparse(url).path)
            out_path = os.path.join(output_folder, filename)

            if os.path.exists(out_path):
                print(f"✅ File già presente: {filename}")
                continue

            print(f"⬇️  Scarico {filename} ...")
            try:
                if username and password:
                    resp = requests.get(url, auth=HTTPBasicAuth(username, password))
                else:
                    resp = requests.get(url)
                resp.raise_for_status()

                with open(out_path, "wb") as f:
                    f.write(resp.content)
            except Exception as e:
                print(f"❌ Errore scaricando {url}: {e}")

    print("✅ Download da CSV completato!")

download_flac_from_url(
    base_url="https://lys-ai.it/recordings/8c69c0b3-ae12-4552-9b11-0aaa0304a06d/recordings/2025-297/FLAC/device/RSP2-MIC1/",
    output_folder = "data/test_297",
    username = "milone",
    password = "Neurone-pc"
    )


# csv_files = ["esiti/esiti_192/1s_esito_2025-286.csv", 
#              "esiti/esiti_192/1s_esito_2025-287.csv"]

# download_flac_from_csv(csv_files, 
#                        output_folder="data/anomali_download", 
#                        username="milone", 
#                        password="Neurone-pc")
