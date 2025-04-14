FROM python:3.11-slim

# Installazione delle dipendenze di sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Creazione della directory di lavoro
WORKDIR /app

# Copia dei file di requisiti
COPY requirements.txt .

# Modifica del requirements.txt (poiché 'random' è integrato in Python)
RUN sed -i '/^random$/d' requirements.txt

# Installazione delle dipendenze Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia del codice del progetto
COPY . .

# Comando predefinito
CMD ["python", "main.py"]