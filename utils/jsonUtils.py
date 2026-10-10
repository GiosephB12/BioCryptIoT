import json
import logging
from pathlib import Path
from typing import Dict, Union

# Configurazione logger
logger = logging.getLogger(__name__)

def saveJson(user_dict: Dict[str, int], json_path: Union[str, Path]) -> None:
    """
    Salva il dizionario di mappatura utenti in un file JSON.
    :param user_dict: Dict {nome: id}
    :param json_path: Percorso del file di output
    """
    try:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(user_dict, f, indent=4, ensure_ascii=False)
            
        logger.info(f"[JSON] File salvato correttamente in {json_path}")
        
    except Exception as e:
        logger.error(f"[JSON] Errore durante il salvataggio: {e}")
        raise

def loadJson(json_path: Union[str, Path]) -> Dict[str, int]:
    """
    Carica il dizionario di mappatura utenti da un file JSON.
    :param json_path: Percorso del file di input
    :return: Dict {nome: id}
    """
    try:
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"File non trovato: {json_path}")
            
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        logger.info(f"[JSON] File caricato da {json_path} ({len(data)} record)")
        return data
        
    except Exception as e:
        logger.error(f"[JSON] Errore durante il caricamento: {e}")
        raise