"""
Script de test pour vérifier la configuration GPU AMD avec DirectML
"""

import sys

def check_pytorch():
    """Vérifie l'installation de PyTorch"""
    try:
        import torch
        print("✓ PyTorch installé")
        print(f"  Version: {torch.__version__}")
        return True
    except ImportError:
        print("✗ PyTorch non installé")
        print("  Installez avec: pip install torch torchvision")
        return False


def check_directml():
    """Vérifie DirectML pour GPU AMD"""
    try:
        import torch_directml
        print("✓ DirectML installé")

        if torch_directml.is_available():
            print("✓ DirectML disponible et fonctionnel")
            device = torch_directml.device()
            print(f"  Device: {device}")
            return True
        else:
            print("✗ DirectML installé mais non disponible")
            print("  Vérifiez vos drivers AMD")
            return False
    except ImportError:
        print("⚠ DirectML non installé (optionnel)")
        print("  Pour utiliser votre GPU AMD, installez avec:")
        print("  pip install torch-directml")
        return False


def check_opencv():
    """Vérifie OpenCV"""
    try:
        import cv2
        print("✓ OpenCV installé")
        print(f"  Version: {cv2.__version__}")
        return True
    except ImportError:
        print("✗ OpenCV non installé")
        print("  Installez avec: pip install opencv-python")
        return False


def check_ultralytics():
    """Vérifie Ultralytics (YOLO)"""
    try:
        import ultralytics
        from ultralytics import YOLO
        print("✓ Ultralytics (YOLO) installé")
        print(f"  Version: {ultralytics.__version__}")
        return True
    except ImportError:
        print("✗ Ultralytics non installé")
        print("  Installez avec: pip install ultralytics")
        return False


def main():
    print("\n" + "="*60)
    print("VÉRIFICATION DE LA CONFIGURATION GPU AMD")
    print("="*60 + "\n")

    all_ok = True

    print("Vérification des packages requis:")
    print("-" * 60)

    all_ok &= check_pytorch()
    all_ok &= check_opencv()
    all_ok &= check_ultralytics()

    print("\nVérification GPU AMD:")
    print("-" * 60)

    gpu_ok = check_directml()

    print("\n" + "="*60)

    if all_ok:
        print("STATUS: Tous les packages requis sont installés ✓")

        if gpu_ok:
            print("GPU AMD: Configuré et prêt ✓")
            print("\nVotre système est prêt pour le comptage de vélos!")
        else:
            print("GPU AMD: Non configuré (le comptage fonctionnera sur CPU)")
            print("\nPour activer le GPU AMD:")
            print("  pip install torch-directml")
    else:
        print("STATUS: Certains packages sont manquants ✗")
        print("\nInstallez les dépendances manquantes avec:")
        print("  pip install -r requirements.txt")

    print("="*60 + "\n")

    # Test basique si tout est installé
    if all_ok:
        print("Test de chargement YOLO...")
        try:
            from ultralytics import YOLO
            print("✓ YOLO peut être importé correctement")
            print("\nLe premier lancement téléchargera le modèle (~6 MB)")
        except Exception as e:
            print(f"✗ Erreur lors de l'import YOLO: {e}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
