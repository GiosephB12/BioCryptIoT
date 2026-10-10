"""
Modulo server per l'archiviazione.
Ascolta il topic 'facenet/embeddings', valida i payload cifrati 
e li persiste in un database SQLite locale.
"""

import paho.mqtt.client as mqtt
import sqlite3
import json
from pathlib import Path
import os
import logging

class serverDb:
    
    def __init__(self, broker="mqtt-broker", port=8883, topic="facenet/embeddings"):
        # Configurazione percorsi
        self.DATA_DIR = Path(__file__).parent.parent / "data"
        self.DB_PATH = self.DATA_DIR / "facenetIoT.db"
        os.makedirs(self.DATA_DIR, exist_ok=True)
        self.topic = topic
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

        # Inizializzazione client MQTT con TLS
        self.client = mqtt.Client(client_id="serverDb")

        # Certificato CA utilizzato per verificare il broker
        self.client.tls_set(
        ca_certs="/app/certs/ca.crt",
        certfile="/app/certs/serverDb.crt",
        keyfile="/app/certs/serverDb.key"
        )

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.init_db()

    def init_db(self):
        # Crea tabella Utenti se non esiste.
        conn = sqlite3.connect(self.DB_PATH)
        conn.cursor().execute("""
            CREATE TABLE IF NOT EXISTS Utenti (
                Id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                embedding TEXT NOT NULL,
                nonce TEXT NOT NULL,
                helper_data TEXT NOT NULL)
            """)
        
        conn.commit(); conn.close()
        self.logger.info(f"[DB] Database pronto in {self.DB_PATH}")

    def on_connect(self, client, userdata, flags, rc, properties=None):
        #Callback connessione MQTT
        if rc == 0:
            self.logger.info("[MQTT] Connesso al broker")
            client.subscribe(self.topic, qos=1)
        else: self.logger.error(f"[MQTT] Connessione fallita (rc={rc})")

    def on_message(self, client, userdata, msg):
        try:
            # Validazione chiavi obbligatorie per integrità payload
            data = json.loads(msg.payload.decode())
            if not all(k in data for k in ["id", "name", "embedding", "nonce", "helper_data"]):
                return
            
            # Persistenza atomica: INSERT OR REPLACE
            conn = sqlite3.connect(self.DB_PATH)
            conn.cursor().execute("INSERT OR REPLACE INTO Utenti VALUES (?,?,?,?,?)",
                (data["id"], data["name"], data["embedding"], data["nonce"], json.dumps(data["helper_data"])))
            conn.commit(); conn.close()
            self.logger.info(f"[DB] Utente salvato: {data['name']} (ID: {data['id']})")
        except Exception as e: self.logger.error(f"[MQTT] Errore: {e}")

    def run(self):
        # Avvia connessione MQTT sicura tramite TLS sulla porta 8883.
        self.client.connect("mqtt-broker",8883,60)
        self.logger.info("[MQTT] Avvio loop...")
        self.client.loop_forever()

if __name__ == "__main__":
    serverDb().run()