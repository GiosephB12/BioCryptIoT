"""
Pacchetto server

Contiene:
- ServerDB: popola il database
- ServerAccess: gestisce richieste di autenticazione
"""

from .serverDb import serverDb
from .serverAccess import serverAccess

import os
from pathlib import Path

# Percorso database
DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "facenetIoT.db"

def start_server():
    # Avvia il server DB se il DB non esiste.
    
    if not DB_PATH.exists():
        print("[SERVER INIT] DB non trovato, avvio ServerDB...")
        server_db = serverDb()
        server_db.run()
    else:
        print("[SERVER INIT] DB trovato, puoi avviare ServerAccess quando vuoi.")