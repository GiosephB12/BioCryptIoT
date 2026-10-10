"""
Modulo client per la validazione end-to-end.
Attesa sincronizzazione enrollment, estrazione embedding da dataset LFW,
pubblicazione su MQTT per verifica 1:1 tramite serverAccess.
"""

import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import cv2
import os
import numpy as np
import paho.mqtt.client as mqtt
import json
from pathlib import Path
import base64
import time

class clientTest:

    def __init__(self, dataset_path="/app/lfw", broker="mqtt-broker", port=8883, topic="facenet/auth"):

        # Configurazione percorsi
        self.dataset_path = Path(dataset_path)
        self.topic = topic
        self.result_topic = "facenet/auth/result"

        data_dir = Path(__file__).parent.parent / "data"
        self.users_json = data_dir / "users.json"
        self.done_file = data_dir / "populate.done"

        # Contatori risultati autenticazione
        self.success = 0
        self.failure = 0
        self.total = 0

        # Sincronizzazione con fase di enrollment
        timeout, start_time = 1800, time.time()

        while not self.done_file.exists():

            if time.time() - start_time > timeout:
                raise TimeoutError("populate.done non trovato")
            print(f"[INFO] Attesa popolamento... ({int(time.time()-start_time)}s)")
            time.sleep(5)
        print("[OK] Populate completato")

        # Caricamento mapping Nome -> ID Utente
        with open(self.users_json, "r") as f:
            self.user_dict = json.load(f)
        print(f"[OK] users.json caricato ({len(self.user_dict)} utenti)")

        # Inizializzazione client MQTT con TLS
        self.client = mqtt.Client(client_id="clientTest")
        self.client.tls_set(
        ca_certs="/app/certs/ca.crt",
        certfile="/app/certs/clientTest.crt",
        keyfile="/app/certs/clientTest.key"
        )

        self.client.on_message = self.on_result
        self.client.connect(broker,port,60)
        self.client.subscribe(self.result_topic,qos=1)
        self.client.loop_start()

        # Caricamento modelli AI (CPU o GPU se disponibile)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(margin=16, keep_all=False, device=self.device).eval()
        self.resnet = InceptionResnetV1(pretrained="vggface2").to(self.device).eval()

    # Callback ricezione risultato autenticazione dal serverAccess
    def on_result(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            if data["result"]:
                self.success += 1

            else:
                self.failure += 1

        except Exception as e:
            print(f"[ERRORE RESULT] {e}")

    """
    Pipeline di validazione end-to-end:
    - Scansiona dataset LFW, filtra immagini e mappa nomi cartelle → ID DB
    - Estrae volti (MTCNN) e embedding 512D (InceptionResnetV1)
    - Serializza embedding in Base64 e pubblica su MQTT
    """
    def send_test_images(self):

        for person_name in os.listdir(self.dataset_path):
            person_dir = self.dataset_path / person_name
            if not person_dir.is_dir():
                continue

            images = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if not images:
                continue

            user_id = self.user_dict.get(person_name)
            if user_id is None:
                continue

            sent = 0

            for img_file in images:
                img = cv2.imread(str(person_dir / img_file))
                if img is None:
                    continue

                face = self.mtcnn(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if face is None:
                    continue

                # Estrazione embedding:
                # 1. unsqueeze(0) - aggiunge asse batch (1, C, H, W)
                # 2. resnet()     - forward pass: estrae embedding facciale (1, 512)
                # 3. cpu().numpy()- sposta su CPU e converte in array NumPy standard
                # 4. flatten()    - rimuove la dimensione batch ottenendo un vettore 1D (512,)
                with torch.no_grad():
                    emb = (self.resnet(face.unsqueeze(0).to(self.device)).cpu().numpy().flatten())

                # Controllo dimensione embedding generato dal modello
                if emb.shape[0] != 512:
                    print(f"[WARNING] Dimensione embedding inattesa: {emb.shape}")
                    continue

                # Serializzazione embedding e invio MQTT
                msg = {
                    "id": user_id,
                    "embedding": base64.b64encode(emb.astype(np.float32).tobytes()).decode("utf-8")
                }

                self.client.publish(self.topic, json.dumps(msg), qos=1, retain=False)
                self.total += 1
                sent += 1

                # Piccola attesa per permettere la risposta MQTT
                time.sleep(0.2)
            print(f"[TEST] Inviati {sent} embedding per {person_name} (ID {user_id})")

        # Attesa ricezione ultime risposte dal server
        time.sleep(3)

        if self.total > 0:
            success_percent = (self.success / self.total) * 100
            failure_percent = (self.failure / self.total) * 100

            print("\n========== RISULTATI TEST ==========")
            print(f"Totale richieste: {self.total}")
            print(f"Successi: {self.success} ({success_percent:.2f}%)")
            print(f"Insuccessi: {self.failure} ({failure_percent:.2f}%)")
            print("====================================")

    # Chiusura della connessione MQTT
    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()

if __name__ == "__main__":
    client_test = clientTest()
    client_test.send_test_images()
    client_test.stop()