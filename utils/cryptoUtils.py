"""
Implementa un meccanismo di derivazione della chiave biometrica
ispirato ai principi dei Fuzzy Extractor, accoppiato a
ChaCha20-Poly1305 per la protezione dei template biometrici.

ChaCha20-Poly1305 aggiunge rispetto al ChaCha20 originale un
meccanismo di autenticazione tramite TAG MAC, permettendo di
verificare che la chiave rigenerata sia realmente quella corretta
prima di accettare la decifratura.
"""

import base64
import hashlib
import secrets
import os

import numpy as np

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from typing import Optional, Dict, List, Tuple


# Soglia L2 per InceptionResnetV1 512D normalizzati (da calibrare sperimentalmente)
FUZZY_THRESHOLD = 1.09

"""
Chiave segreta lato server.

NON viene salvata nel database.

Viene utilizzata come pepper crittografico per impedire
che un attaccante possa ricostruire la chiave solamente
avendo accesso all'helper_data.

La secret deve essere uguale nei componenti:
- populateDb (generazione database)
- serverDb
- serverAccess
"""
SERVER_SECRET = os.getenv("SERVER_SECRET")

if SERVER_SECRET is not None:
    SERVER_SECRET = SERVER_SECRET.encode()

"""
Calcola la magnitudine.

Ricorda:
Magnitudine (norma euclidea): è la "lunghezza geometrica" del vettore embedding 
nello spazio multidimensionale a 512 dimensioni.

L'operazione emb / norm scala il vettore in modo che la sua 
distanza dall'origine diventi esattamente 1, proiettandolo su un'ipersfera 
unitaria di 512D.
"""
def _normalize(emb: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 1e-8 else emb

"""
Fase di Enrollment del Fuzzy Extractor.

- Normalizza l'embedding per riportarlo sull'ipersfera unitaria.
- Quantizza a 4 decimali per introdurre tolleranza deterministica.
- Deriva chiave crittografica K (32 byte) tramite SHA-256.
- Utilizza SERVER_SECRET come pepper lato server.

Restituisce:
(K, helper_data)

helper_data viene salvato nel DB.

SERVER_SECRET rimane solo lato server.
"""
def fuzzy_gen(embedding: np.ndarray) -> Tuple[bytes, List[float]]:

    if SERVER_SECRET is None:
        raise RuntimeError("SERVER_SECRET necessaria per fuzzy_gen")

    emb_norm = _normalize(embedding.astype(np.float32))

    # Arrotonda ogni componente a 4 decimali
    quantized = np.round(emb_norm, decimals=4)

    # Generazione chiave simmetrica da 32 byte
    # helper_data + SERVER_SECRET
    K = hashlib.sha256(quantized.tobytes() + SERVER_SECRET).digest()[:32]

    return K, quantized.tolist()

"""
Fase di verifica del Fuzzy Extractor.

- Recupera helper_data dal DB.
- Normalizza il nuovo embedding.
- Calcola distanza L2.
- Se la distanza è sotto soglia rigenera la chiave.

La chiave viene ricostruita usando:

helper_data + SERVER_SECRET

Questo impedisce la derivazione della chiave
da parte di un attaccante che possiede solo il DB.
"""
def fuzzy_rep(new_embedding: np.ndarray,helper_data: List[float],threshold: float = FUZZY_THRESHOLD) -> Optional[bytes]:

    if SERVER_SECRET is None:
        raise RuntimeError("SERVER_SECRET necessaria per fuzzy_rep")

    # Recupero template enrollment salvato nel DB
    helper_arr = np.array(helper_data, dtype=np.float32)

    # Normalizzazione embedding ricevuto
    new_emb = _normalize(new_embedding.astype(np.float32))

    # Calcolo distanza L2
    dist = np.linalg.norm(new_emb - helper_arr)


    print(
        f"[FUZZY] Distanza: {dist:.4f} | "
        f"Soglia: {threshold} | "
        f"{' PASS' if dist <= threshold else ' FAIL'}")

    if dist > threshold:
        return None

    """
    Rigenerazione chiave.
    La chiave viene ottenuta da:
    helper_data + SERVER_SECRET
    Solo il server possiede SERVER_SECRET.
    """
    quantized = np.round(helper_arr, decimals=4)
    return hashlib.sha256(quantized.tobytes() + SERVER_SECRET).digest()[:32]


"""
Cifratura autenticata tramite ChaCha20-Poly1305.
"""
def chacha20_encrypt(data: bytes, key: bytes, nonce: bytes) -> bytes:

    if len(nonce) != 12:
        raise ValueError("Nonce deve essere 12 byte per ChaCha20-Poly1305.")

    cipher = ChaCha20Poly1305(key)
    return cipher.encrypt(nonce,data,None)

"""
Decifratura autenticata tramite ChaCha20-Poly1305.
"""
def chacha20_decrypt(ciphertext: bytes,key: bytes,nonce: bytes) -> bytes:

    cipher = ChaCha20Poly1305(key)
    return cipher.decrypt(nonce,ciphertext,None)

"""
Prepara payload crittografico per MQTT/DB.
"""
def encrypt_embedding_with_fuzzy(embedding: np.ndarray) -> Dict[str, object]:

    K, helper_data = fuzzy_gen(embedding)

    nonce = secrets.token_bytes(12)

    embedding_bytes = embedding.astype(np.float32).tobytes()
    encrypted = chacha20_encrypt(embedding_bytes,K,nonce)
    
    return {
        "encrypted_embedding":base64.b64encode(encrypted).decode('utf-8'),
        "nonce":base64.b64encode(nonce).decode('utf-8'),
        "helper_data":helper_data
    }

"""
Verifica biometrica.

- Rigenera la chiave tramite helper_data + SERVER_SECRET.
- Prova la decifratura.
- ChaCha20-Poly1305 verifica il TAG.

Se il TAG è valido:
    -> embedding originale

Se il TAG fallisce:
    -> chiave errata
"""
def decrypt_embedding_with_fuzzy(encrypted_embedding: str,nonce: str,helper_data: List[float],new_embedding: np.ndarray) -> Optional[bytes]:

    try:

        K = fuzzy_rep(new_embedding,helper_data)

        if K is None:
            return None

        nonce_bytes = base64.b64decode(nonce + '=' * (-len(nonce) % 4))
        ciphertext = base64.b64decode(encrypted_embedding +'=' * (-len(encrypted_embedding) % 4))

        return chacha20_decrypt(ciphertext,K,nonce_bytes)

    except Exception:
        return None