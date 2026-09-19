import os


def charger_modele_keras(chemin):
    # Le modèle a été entraîné avec Keras sur PyTorch.
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras
    import keras_hub  # Enregistre les couches SegFormer pour la désérialisation.

    if keras.backend.backend() != "torch":
        raise ValueError("Ce modèle nécessite KERAS_BACKEND=torch.")

    # Les pertes et métriques d'entraînement ne sont pas nécessaires à l'API.
    return keras.saving.load_model(chemin, compile=False, safe_mode=True)

