from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch

from app.config import Settings

API_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = API_DIR / "model"

CONFIG_PATH = MODEL_DIR / "model_config.json"
WEIGHTS_PATH = MODEL_DIR / "meilleurs_poids.pt"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fonctions.model_architectures import (
    construire_modele,
    extraire_logits,
)

logger = logging.getLogger(__name__)


@dataclass
class LoadedModelInfo:
    mode: str
    model_name: str | None = None
    architecture: str | None = None
    run_id: str | None = None
    miou: float | None = None
    dice: float | None = None
    nombre_classes: int | None = None
    largeur: int | None = None
    hauteur: int | None = None


class SegmentationModelService:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.device = torch.device("cpu")
        self.largeur = 256
        self.hauteur = 128
        self.nombre_classes = 8
        self.model_config: dict = {}
        self.info = LoadedModelInfo(mode=settings.model_mode)

    @property
    def loaded(self) -> bool:
        if self.settings.model_mode == "mock":
            return True
        return self.model is not None

    def load(self) -> None:
        mode = self.settings.model_mode.lower()

        if mode == "mock":
            logger.warning("Mode mock : aucun modèle réel chargé.")
            self.info = LoadedModelInfo(mode="mock")
            return

        if mode != "local":
            raise ValueError("MODEL_MODE doit être 'mock' ou 'local'.")

        if not CONFIG_PATH.is_file():
            raise FileNotFoundError(f"Configuration introuvable : {CONFIG_PATH}")

        if not WEIGHTS_PATH.is_file():
            raise FileNotFoundError(f"Poids introuvables : {WEIGHTS_PATH}")

        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            self.model_config = json.load(f)

        nom_modele = self.model_config["nom_modele"]
        self.nombre_classes = int(self.model_config["nombre_classes"])
        self.largeur = int(self.model_config["largeur_image"])
        self.hauteur = int(self.model_config["hauteur_image"])

        self.model = construire_modele(
            nom_modele=nom_modele,
            nombre_classes=self.nombre_classes,
            poids_preentraines=False,
        )

        loaded = torch.load(
            WEIGHTS_PATH,
            map_location=self.device,
            weights_only=True,
        )

        state_dict = loaded["state_dict"] if isinstance(loaded, dict) and "state_dict" in loaded else loaded

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        metrics = self.model_config.get("metriques_validation", {})

        self.info = LoadedModelInfo(
            mode="local",
            model_name=nom_modele,
            architecture=self.model_config.get("architecture"),
            run_id=self.model_config.get("run_id"),
            miou=metrics.get("miou"),
            dice=metrics.get("dice"),
            nombre_classes=self.nombre_classes,
            largeur=self.largeur,
            hauteur=self.hauteur,
        )

        logger.info("Modèle chargé avec succès.")

    def predict(self, tensor: torch.Tensor) -> np.ndarray:
        if self.settings.model_mode == "mock":
            return self._mock_predict(tensor)

        if self.model is None:
            raise RuntimeError("Le modèle n'est pas chargé.")

        tensor = tensor.to(self.device)

        with torch.inference_mode():
            output = self.model(tensor)
            logits = extraire_logits(
                output,
                (self.hauteur, self.largeur),
            )
            prediction = logits.argmax(dim=1)

        return prediction.cpu().numpy()

    @staticmethod
    def _mock_predict(tensor: torch.Tensor) -> np.ndarray:
        image = tensor[0]
        gray = image.mean(dim=0)
        return (gray > gray.mean()).long().unsqueeze(0).numpy()