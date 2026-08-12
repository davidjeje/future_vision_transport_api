import torch.nn.functional as F
import segmentation_models_pytorch as smp


def construire_modele(
    nom_modele: str,
    nombre_classes: int,
    poids_preentraines: bool = False,
):
    if nom_modele == "unet_mobilenetv2":
        return smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=(
                "imagenet"
                if poids_preentraines
                else None
            ),
            in_channels=3,
            classes=nombre_classes,
        )

    raise ValueError(
        f"Modèle non supporté en production : {nom_modele}"
    )


def extraire_logits(
    sortie,
    taille_cible,
):
    logits = (
        sortie.logits
        if hasattr(sortie, "logits")
        else sortie
    )

    if logits.shape[-2:] != taille_cible:
        logits = F.interpolate(
            logits,
            size=taille_cible,
            mode="bilinear",
            align_corners=False,
        )

    return logits