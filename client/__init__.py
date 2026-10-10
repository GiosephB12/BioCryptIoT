""""
Pacchetto client

Contiene:
- PopulateDB: legge il dataset, estrae embedding, li cifra e pubblica via MQTT
- ClientTest: invia embedding di test per verificare l'autenticazione lato server
"""

from .populateDb import populateDb
from .clientTest import clientTest

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
USERS_JSON = DATA_DIR / "users.json"

def start_client():
    # Avvia PopulateDB solo se users.json non esiste.

    if not USERS_JSON.exists():
        print("[CLIENT INIT] users.json non trovato, avvio PopulateDB...")
        # istanza della classe populateDb
        populator = populateDb()
        populator.process_dataset()
        populator.stop()
    else:
        print("[CLIENT INIT] users.json trovato, puoi avviare ClientTest per autenticazione.")