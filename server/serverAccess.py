"""
Modulo server per la verifica biometrica 1:1 via MQTT.
Ascolta il topic 'facenet/auth', recupera i template cifrati dal DB,
e li confronta con il nuovo embedding in ingresso tramite distanza L2 e Fuzzy Extractor.
"""

import sqlite3
import json
import base64
import logging
import numpy as np
from pathlib import Path
import paho.mqtt.client as mqtt
from utils import cryptoUtils

class serverAccess:

    def __init__(self, broker="mqtt-broker", port=8883, topic="facenet/auth"):

        # Configurazione percorsi
        self.DB_PATH = Path(__file__).parent.parent / "data" / "facenetIoT.db"
        self.topic = topic
        self.result_topic = "facenet/auth/result"

        # Configurazione del sistema di logging strutturato
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)

        # Inizializzazione client MQTT TLS
        self.client = mqtt.Client(client_id="serverAccess")

        # Certificato CA per autenticazione broker
        self.client.tls_set(
        ca_certs="/app/certs/ca.crt",
        certfile="/app/certs/serverAccess.crt",
        keyfile="/app/certs/serverAccess.key"
        )

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(broker,port,60)

    def on_connect(self, client, userdata, flags, rc, properties=None):
        # Callback connessione MQTT
        if rc == 0:
            self.logger.info("[MQTT] ServerAccess connesso")
            client.subscribe(self.topic, qos=1)

        else:
            self.logger.error(f"[MQTT] Connessione fallita (rc={rc})")

    def on_message(self, client, userdata, msg):
        try:
            # Decodifica payload MQTT
            data = json.loads(msg.payload.decode())
            # Estrae ID
            uid = data["id"]

            """""
            Ricostruzione embedding biometrico da payload MQTT:
            1. Riparazione padding Base64: aggiunge '=' mancanti per garantire decodifica robusta
               in reti IoT dove i parser possono troncare caratteri finali
            2. base64.b64decode(): converte la stringa ASCII nel buffer binario originale (2048 byte)
            3. np.frombuffer(dtype=np.float32): interpreta i byte come 512 float IEEE 754,
               ricostruendo esattamente il vettore feature estratto da InceptionResnetV1

               questo flusso garantisce che l'embedding di test sia numericamente identico
            """
            b64_emb = data["embedding"] + '=' * (-len(data["embedding"]) % 4)
            emb = np.frombuffer(base64.b64decode(b64_emb), dtype=np.float32)

            if emb.shape[0] != 512:
                raise ValueError(f"Embedding dimensione errata: {emb.shape}")

            # Esegue la verifica biometrica 1:1
            res = self.authenticate_user(uid, emb)

            # Log esito dell'autenticazione
            self.logger.info(f"[ACCESSO] Utente {uid} -> {' AUTORIZZATO' if res else ' NEGATO'}")

            # Invio risultato al clientTest
            response = {"id": uid,"result": res}
            client.publish(self.result_topic,json.dumps(response),qos=1,retain=False)

        except Exception as e:
            self.logger.error(f"[ERRORE] {e}")


    def authenticate_user(self, user_id, new_emb):
        with sqlite3.connect(self.DB_PATH) as conn:

            row = conn.cursor().execute(
                "SELECT embedding, nonce, helper_data FROM Utenti WHERE Id=?",(user_id,)).fetchone()
            if not row:
                return False
            
            # Tuple del risultato SQL.
            enc, nonce, hdata_json = row

            # Parsing di helper_data
            hdata = json.loads(hdata_json) if isinstance(hdata_json, str) else hdata_json
            return cryptoUtils.decrypt_embedding_with_fuzzy(enc,nonce,hdata,new_emb) is not None

    def run(self):
        # Avvia connessione al broker e mantiene il loop di ascolto attivo.
        self.logger.info("[MQTT] Avvio loop...")
        self.client.loop_forever()

if __name__ == "__main__":
    serverAccess().run()