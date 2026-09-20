from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from app.config import Settings

API_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = API_DIR / "model"

CONFIG_PATH = MODEL_DIR / "model_config.json"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fonctions.model_architectures import (
    charger_modele_keras,
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
        self.format_modele = "keras"
        self.format_tenseur = "CHW"
        self.normalisation = "division_par_255"
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

        config_path = CONFIG_PATH
        if self.settings.model_name == "segformer_sans_augmentation":
            config_path = MODEL_DIR / "segformer_config.json"

        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration introuvable : {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            self.model_config = json.load(f)

        self.format_modele = self.model_config["format_modele"]
        if self.format_modele != "keras":
            raise ValueError("Le runtime de production prend uniquement en charge le format Keras.")

        weights_path = MODEL_DIR / self.model_config["fichier_modele"]
        if mode == "huggingface":
            if not self.settings.hf_repo_id:
                raise ValueError("HF_REPO_ID est obligatoire avec MODEL_MODE=huggingface.")
            weights_path = Path(hf_hub_download(
                repo_id=self.settings.hf_repo_id,
                filename=self.model_config["fichier_modele"],
                revision=self.settings.hf_revision,
                cache_dir=self.settings.hf_cache_dir,
            ))
        if not weights_path.is_file():
            raise FileNotFoundError(f"Modèle introuvable : {weights_path}")

        nom_modele = self.model_config["nom_modele"]
        self.nombre_classes = int(self.model_config["nombre_classes"])
        self.largeur = int(self.model_config["largeur_image"])
        self.hauteur = int(self.model_config["hauteur_image"])
        preprocessing = self.model_config.get("preprocessing", {})
        self.format_tenseur = preprocessing.get("format_tenseur", "CHW")
        self.normalisation = preprocessing.get("normalisation", "division_par_255")

        if self.format_modele == "keras":
            if self.format_tenseur != "HWC" or self.normalisation != "integree_au_modele":
                raise ValueError("Le modèle Keras attend une entrée HWC RGB 0..255.")
            self.model = charger_modele_keras(
                weights_path, segformer=nom_modele == "segformer_sans_augmentation"
            )
            if tuple(self.model.input_shape[1:]) != (self.hauteur, self.largeur, 3):
                raise ValueError("Les dimensions configurées ne correspondent pas au modèle Keras.")
            if tuple(self.model.output_shape[1:]) != (self.hauteur, self.largeur, self.nombre_classes):
                raise ValueError("La sortie Keras ne correspond pas aux dimensions/classes configurées.")
        self.model.to(self.device)
        self.model.eval()

        metrics = self.model_config.get("metriques_validation", {})

        self.info = LoadedModelInfo(
            mode=mode,
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
            output = self.model(tensor, training=False)
            return output.argmax(dim=-1).cpu().numpy()

    @staticmethod
    def _mock_predict(tensor: torch.Tensor) -> np.ndarray:
        image = tensor[0]
        gray = image.mean(dim=0)
        return (gray > gray.mean()).long().unsqueeze(0).numpy()
