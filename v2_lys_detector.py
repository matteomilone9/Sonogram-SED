import warnings
import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup
import io
import os
import csv
from urllib.parse import urljoin

from nash_modules import *
from aed_models import *

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Carica il modello
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "results_1s_v5/best_model_1s_v5.pth"
    model = NS_1()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model_type = '1s'

    # Lista cartelle audio da analizzare
    base_urls = [
        "https://lys-ai.it/recordings/8c69c0b3-ae12-4552-9b11-0aaa0304a06d/recordings/2025-255/device/RSP1-MIC1",
    ]

    username = "milone"
    password = "Neurone-pc"

    # Cartella di salvataggio
    results_folder = "results_pc12"

    # Analizza tutti i file
    analyze_all(base_urls, username, password, model, model_type, results_folder)
