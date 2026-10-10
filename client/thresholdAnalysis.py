"""
Esperimento - Calibrazione soglia biometrica.

Simula il comportamento reale del sistema:

Calcola:
- Genuine Acceptance / Rejection
- Impostor Acceptance
- FAR
- FRR
- EER
- Threshold consigliata

Il test è indipendente da MQTT e crittografia.
Serve per calibrare FUZZY_THRESHOLD.
"""

import torch
import cv2
import os
import csv
import json
import random

import numpy as np

from pathlib import Path
from facenet_pytorch import MTCNN, InceptionResnetV1

# Configurazione
DATASET_PATH = "/app/lfw"
RESULT_PATH = Path("/app/data/threshold_results")

RESULT_PATH.mkdir(parents=True,exist_ok=True)

# Range soglie analizzate
THRESHOLDS = np.arange(0.30,1.20,0.01)

# Numero massimo confronti impostore
MAX_IMPOSTOR_PAIRS = 100000

class thresholdAnalysis:

    def __init__(self,dataset_path=DATASET_PATH):

        self.dataset_path = Path(dataset_path)
        self.device = torch.device("cuda" if torch.cuda.is_available()else "cpu")

        print(f"[INFO] Device: {self.device}")

        """
        Pipeline identica
        al progetto principale.
        """
        self.mtcnn = MTCNN(margin=16,keep_all=False,device=self.device).eval()
        self.resnet = InceptionResnetV1(pretrained="vggface2",device=self.device).eval()

        """
        Embedding immagini.

        {
            persona:
            [
                embedding1,
                embedding2,
                ...
            ]
        }
        """
        self.embeddings = {}


        """
        Template medio reale
        salvato nel database.

        {
            persona:
            template512D
        }

        """
        self.templates = {}

    # Normalizzazione L2
    def normalize(self,embedding):
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            return embedding / norm
        return embedding


    # Estrazione embedding
    def extract_embedding(self,img_path):

        img = cv2.imread(str(img_path))
        if img is None:
            return None

        face = self.mtcnn(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

        if face is None:
            return None

        with torch.no_grad():

            embedding = (self.resnet(face.unsqueeze(0)).detach().cpu().numpy().flatten())

        if embedding.shape[0] != 512:
            print("[WARNING] "
                    "Embedding dimensione errata:",embedding.shape)
            return None

        """
        Stessa normalizzazione
        usata dal fuzzy extractor.
        """
        embedding = self.normalize(embedding)
        return embedding

    # Caricamento dataset
    def load_dataset(self):

        print("[INFO] Caricamento dataset...")

        users = 0
        images = 0

        for person in os.listdir(self.dataset_path):

            person_dir = (self.dataset_path/person)

            if not person_dir.is_dir():
                continue

            image_files = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]

            """
            Per creare un template
            servono almeno due immagini.
            """
            if len(image_files) < 2:
                continue

            user_embeddings = []

            for img_file in image_files:

                emb = self.extract_embedding(person_dir/img_file)

                if emb is not None:

                    user_embeddings.append(emb)

                    images += 1

            if len(user_embeddings) < 2:
                continue

            self.embeddings[person] = (user_embeddings)

            """
            Creazione template medio.
            Identico a populateDb:

            mean_embedding = media di tutti gli embedding
            """
            template = np.mean(np.stack(user_embeddings),axis=0)

            """
            Dopo la media
            bisogna rinormalizzare.

            """
            template = self.normalize(template)
            self.templates[person] = template

            users += 1

        print(f"[INFO] Utenti caricati: {users}")
        print(f"[INFO] Immagini elaborate: {images}")
        print(f"[INFO] Template creati: {len(self.templates)}")

    # Distanza L2
    @staticmethod
    def l2_distance(e1,e2):

        return np.linalg.norm( e1-e2)

    # Genuine test
    # Simula:
    # Utente corretto
    # nuova immagine

    # confronto:
    # embedding nuova immagine     
    # template medio stesso utente

    def genuine_template_test(self):

        print("[INFO] Genuine template test...")
        results = []

        for person, embeddings in self.embeddings.items():
            template = self.templates[person]

            for emb in embeddings:
                distance = self.l2_distance(emb,template)

                results.append((person,person,distance))

        self.save_csv("genuine_template_distances.csv",results)
        return results

    # Impostor test

    # Simula Attaccante:
    # immagine propria
    
    # contro:
    # template vittima

    def impostor_template_test(self):
        print("[INFO] Impostor template test...")
        results = []

        users = list(self.templates.keys())

        random.shuffle(users)
        counter = 0

        while counter < MAX_IMPOSTOR_PAIRS:

            attacker = random.choice(users)
            target = random.choice(users)
            if attacker == target:
              continue

            template_target = (self.templates[target])

            emb_attacker = random.choice(self.embeddings[attacker])

            distance = self.l2_distance(emb_attacker,template_target)

            results.append((attacker,target,distance))
            counter += 1

        self.save_csv("impostor_template_distances.csv",results)
        return results

    # Salvataggio CSV
    def save_csv(self,filename,data):
        path = RESULT_PATH / filename

        with open(path,"w",newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["user1","user2","distance"])
            writer.writerows(data)

        print(f"[OK] Salvato {path}")


    # FAR / FRR
    def evaluate_threshold(self,threshold,genuine,impostor):

        # Genuine rifiutati
        # distanza > threshold
        false_negative = sum(
            1
            for _,_,distance in genuine
            if distance > threshold
        )

        frr = (false_negative /len(genuine))

        # Impostor accettati
        # distanza <= threshold

        false_positive = sum(
            1
            for _,_,distance in impostor
            if distance <= threshold
        )

        far = (false_positive /len(impostor))

        return far,frr

    # Ricerca threshold EER
    def threshold_search(self,genuine,impostor):

        print("\n========== ANALISI SOGLIA ==========")
        best = None
        rows = []

        for threshold in THRESHOLDS:
            far,frr = self.evaluate_threshold(threshold,genuine,impostor)

            rows.append((round(threshold,2),far,frr))

            print(
                f"{threshold:.2f} "
                f"FAR={far*100:.4f}% "
                f"FRR={frr*100:.4f}%"
            )

            eer_error = abs(far-frr)

            if best is None or eer_error < best[0]:

                best = (eer_error,threshold,far,frr)

        self.save_csv("threshold_results.csv",rows)

        result = {
            "threshold": float(best[1]),
            "FAR": float(best[2]),
            "FRR": float(best[3])
        }


        with open(RESULT_PATH /"threshold_analysis.json","w") as f:
            json.dump(result,f,indent=4)

        print("\n========== RISULTATO ==========")

        print(f"Threshold EER: {best[1]:.2f}")

        print(f"FAR: {best[2]*100:.4f}%")

        print(f"FRR: {best[3]*100:.4f}%")

        print("[OK] Salvato threshold_analysis.json")


    # Statistiche
    def statistics(self,name,data):
        distances = [d for _,_,d in data]

        print(f"\n{name}")

        print("Numero:",len(distances))

        print("Media:",np.mean(distances))

        print("Deviazione standard:",np.std(distances))
        print("95 percentile:",np.percentile(distances,95))
        print("Massimo:",np.max(distances))

    def run(self):
        self.load_dataset()

        genuine = (self.genuine_template_test())
        impostor = (self.impostor_template_test())

        self.statistics("GENUINE STATISTICS",genuine)

        self.statistics("IMPOSTOR STATISTICS",impostor)
        self.threshold_search(genuine,impostor)


if __name__ == "__main__":

    analysis = thresholdAnalysis()
    analysis.run()