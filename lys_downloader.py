import os
import requests
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

download_flac_from_url(
    base_url="https://lys-ai.it/recordings/8c69c0b3-ae12-4552-9b11-0aaa0304a06d/recordings/2025-286/FLAC/device/RSP2-MIC1/",
    output_folder = "data/doy/286",
    username = "milone",
    password = "Neurone-pc"
    )
