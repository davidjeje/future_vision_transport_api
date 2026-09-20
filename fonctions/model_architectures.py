import os


def charger_modele_keras(chemin, *, segformer=False):
    # Le modèle a été entraîné avec Keras sur PyTorch.
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras
    import keras_hub  # Enregistre les couches SegFormer pour la désérialisation.

    if keras.backend.backend() != "torch":
        raise ValueError("Ce modèle nécessite KERAS_BACKEND=torch.")

    if segformer:
        # Enregistre les classes MiT/SegFormer présentes dans l'archive Keras.
        # Aucun from_preset : tous les poids sont déjà dans l'archive.
        import keras_hub  # noqa: F401

    # Les pertes et métriques d'entraînement ne sont pas nécessaires à l'API.
    return keras.saving.load_model(chemin, compile=False, safe_mode=True)
