"""
Modulo di enrollment biometrico.
Scansiona il dataset LFW, estrae embedding facciali, li aggrega, li cifra 
con Fuzzy Extractor + ChaCha20 e li pubblica via MQTT sul server per l'archiviazione.
"""

import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import cv2
import os
import numpy as np
import paho.mqtt.client as mqtt
import json
from pathlib import Path
import sys

from utils.jsonUtils import saveJson
from utils.cryptoUtils import encrypt_embedding_with_fuzzy

class populateDb:

    def __init__(self,dataset_path="/app/lfw",broker="mqtt-broker",port=8883,topic="facenet/embeddings"):

        self.dataset_path = Path(dataset_path)

        # Connessione MQTT
        self.topic = topic

        #MQTT su TLS
        self.client = mqtt.Client(client_id="populateDb")

        self.client.tls_set(
        ca_certs="/app/certs/ca.crt",
        certfile="/app/certs/populateDb.crt",
        keyfile="/app/certs/populateDb.key"
        )

        self.client.connect(broker,port,60)
        self.client.loop_start()

        # Configurazione percorsi e stato infrastruttura
        self.data_dir = Path(__file__).parent.parent / "data"
        self.jsonFile = self.data_dir / "users.json"
        self.done_file = self.data_dir / "populate.done"
        self.db_file = self.data_dir / "facenetIoT.db"

        # Crea la cartella se non esiste
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Controllo dell'esistenza del Db e file json
        if self.db_file.exists() and self.jsonFile.exists() and self.done_file.exists():

            print("[INFO] Database, users.json e populate.done già presenti. ""Skip popolamento.")
            sys.exit(0)

        # Selezione dispositivo (GPU, altrimenti CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Inizializzazione pipeline di visione artificiale in modalità inferenza (.eval())
        # MTCNN: rileva, allinea e ritaglia il volto. margin=16 preserva il contesto strutturale,
        # keep_all=False evita falsi positivi in immagini con più soggetti.
        self.mtcnn = MTCNN(margin=16,keep_all=False,device=self.device).eval()

        # InceptionResnetV1 (FaceNet): estrae embedding 512D pre-addestrato su VGGFace2.
        # .eval() disattiva Dropout e fissa BatchNorm, garantendo embedding deterministici
        # e stabili per la successiva fase di protezione biometrica.
        self.resnet = InceptionResnetV1(pretrained='vggface2',device=self.device).eval()

        """
        .eval() serve a modificare il comportamento di alcuni layer interni
        in modo che l'output sia deterministico e stabile.
        """

        # Registro utenti e contatore ID
        self.userDict = {}
        self.incrementId = 1

    # Metodo di estrazione embedding, cifratura e persistenza
    def process_dataset(self):

        for personName in os.listdir(self.dataset_path):
            personaDir = self.dataset_path / personName

            if not personaDir.is_dir():
                continue
            # Prende solo le immagini supportate
            images = [f for f in os.listdir(personaDir)if f.lower().endswith((".jpg", ".jpeg", ".png"))]

            if not images:
                continue

            images.sort()
            embeddings_list = []

            # Estrazione embedding per ogni immagine dell'utente
            for imgFile in images:
                imgPath = personaDir / imgFile

                # Legge il file dal percorso e lo carica in memoria come array NumPy
                # (H, W, 3) in formato BGR, estrae i pixel dal file.
                img = cv2.imread(str(imgPath))
                if img is None:
                    continue

                # Rilevamento volto e conversione colore
                face = self.mtcnn(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                if face is None:
                    continue

                # Estrazione embedding:
                # 1. unsqueeze(0) - aggiunge asse batch (1, C, H, W) richiesto da PyTorch (numero di immagini elaborate)
                # 2. resnet()     - forward pass: estrae embedding facciale (1, 512)
                # 3. cpu().numpy()- sposta su CPU e converte in array NumPy standard
                # 4. flatten()    - rimuove la dimensione batch ottenendo un vettore 1D (512,) pronto per la protezione
                with torch.no_grad():
                    embedding = (self.resnet(face.unsqueeze(0)).detach().cpu().numpy().flatten())

                # Controllo dimensione embedding generato dal modello
                if embedding.shape[0] != 512:
                    print(f"[WARNING] Dimensione embedding inattesa: {embedding.shape}")
                    continue

                embeddings_list.append(embedding)

            if not embeddings_list:
                continue

            # Creazione template biometrico rappresentativo:
            # - np.stack(): impila gli embedding estratti in una matrice (N, 512)
            # - Gli embedding vengono mediati per ridurre il rumore
            #   dovuto alle variazioni intra-classe del volto.
            # - np.mean(..., axis=0): calcola il centroide matematico
            # Il risultato è un vettore 1D (512,) che rappresenta
            # l'identità media dell'utente.
            embeddings_matrix = np.stack(embeddings_list)

            mean_embedding = np.mean(embeddings_matrix,axis=0)
            # Protezione crittografica con Fuzzy Extractor + ChaCha20
            crypto_payload = encrypt_embedding_with_fuzzy(mean_embedding)

            # Assegnazione ID univoco e aggiornamento registro
            userId = self.incrementId
            self.incrementId += 1
            self.userDict[personName] = userId

            # Preparazione msg MQTT
            msg = {
                "id": userId,
                "name": personName,
                "embedding": crypto_payload["encrypted_embedding"],
                "nonce": crypto_payload["nonce"],
                "helper_data": crypto_payload["helper_data"]
            }

            # Pubblicazione msg:
            # qos=1: garantisce consegna "almeno una volta"
            # retain=False: messaggio effimero
            self.client.publish(self.topic,json.dumps(msg),qos=1,retain=False)
            print(f"[OK] Utente {personName} pubblicato")

        # Persistenza dei dati (personaName: userId)
        saveJson(self.userDict,self.jsonFile)

        # Creazione file di segnalazione
        self.done_file.touch()
        print("[INFO] Populate completato.")

    # Chiusura della connessione MQTT
    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()

if __name__ == "__main__":
    populator = populateDb()
    populator.process_dataset()
    populator.stop()