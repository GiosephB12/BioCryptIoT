FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Dipendenze di sistema solo per OpenCV (senza build-essential per risparmiare ~300MB)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Copia solo requirements per sfruttare la cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copia SOLO il codice sorgente (evita di portare dentro data/, .git, __pycache__, ecc.)
COPY client/ ./client/
COPY server/ ./server/
COPY utils/ ./utils/

# 3. Crea cartella dati vuota (verrà montata come volume da docker-compose)
RUN mkdir -p /app/data

# CMD di fallback (sovrascritto automaticamente da docker-compose.yml)
CMD ["python", "-c", "print('Avvia un servizio specifico tramite docker-compose')"]