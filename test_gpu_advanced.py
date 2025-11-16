"""
Test complet d'utilisation du GPU AMD avec DirectML
Vérifie la configuration et teste les performances GPU vs CPU
"""

import sys
import time
import torch
import cv2
import numpy as np


def print_section(title):
    """Affiche un titre de section"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_pytorch_installation():
    """Test 1: Vérifier l'installation de PyTorch"""
    print_section("TEST 1: Installation PyTorch")

    try:
        print(f"✓ PyTorch version: {torch.__version__}")
        return True
    except Exception as e:
        print(f"✗ Erreur PyTorch: {e}")
        return False


def test_directml_availability():
    """Test 2: Vérifier DirectML"""
    print_section("TEST 2: Disponibilité DirectML")

    try:
        import torch_directml
        print(f"✓ DirectML installé")

        if torch_directml.is_available():
            print(f"✓ DirectML disponible et fonctionnel")
            device = torch_directml.device()
            print(f"✓ Device DirectML créé: {device}")
            return True, device
        else:
            print(f"✗ DirectML installé mais non disponible")
            print(f"  Vérifiez vos drivers AMD")
            return False, None

    except ImportError:
        print(f"✗ DirectML non installé")
        print(f"  Installez avec: pip install torch-directml")
        return False, None
    except Exception as e:
        print(f"✗ Erreur DirectML: {e}")
        return False, None


def test_tensor_operations(device):
    """Test 3: Opérations sur tenseurs GPU"""
    print_section("TEST 3: Opérations Tenseurs sur GPU")

    try:
        # Créer un tensor sur GPU
        print("Creating tensor on GPU...")
        tensor_gpu = torch.randn(1000, 1000, device=device)
        print(f"✓ Tensor créé sur GPU: {tensor_gpu.shape}")

        # Opération matricielle
        print("Performing matrix multiplication on GPU...")
        start = time.time()
        result = torch.matmul(tensor_gpu, tensor_gpu)
        elapsed = time.time() - start
        print(f"✓ Multiplication matricielle sur GPU: {elapsed:.4f}s")

        # Vérifier que le résultat est sur GPU
        print(f"✓ Résultat sur device: {result.device}")

        return True

    except Exception as e:
        print(f"✗ Erreur lors des opérations: {e}")
        return False


def test_yolo_gpu():
    """Test 4: YOLO avec GPU"""
    print_section("TEST 4: YOLO sur GPU")

    try:
        from ultralytics import YOLO
        import torch_directml

        # Créer le device DirectML
        device = torch_directml.device()
        print(f"✓ Device DirectML: {device}")

        # Charger le modèle YOLO
        print("Chargement du modèle YOLOv8n...")
        model = YOLO("yolov8n.pt")

        # Note: Pas besoin de .to(device) avec DirectML
        # On passe le device directement dans l'inférence
        print(f"✓ Modèle YOLO chargé (utilisera {device})")

        # Créer une image de test
        print("Création d'une image de test...")
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Inférence sur GPU
        print("Inférence sur GPU...")
        start = time.time()
        results = model(test_image, device=device, verbose=False)
        elapsed = time.time() - start
        print(f"✓ Inférence YOLO sur GPU: {elapsed:.4f}s")

        return True

    except Exception as e:
        print(f"✗ Erreur YOLO GPU: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_cpu_vs_gpu():
    """Test 5: Benchmark CPU vs GPU"""
    print_section("TEST 5: Benchmark CPU vs GPU")

    try:
        from ultralytics import YOLO
        import torch_directml

        # Créer une image de test
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Charger le modèle
        model = YOLO("yolov8n.pt")

        # Test CPU
        print("\n🖥️  TEST CPU:")
        print("Warm-up...")
        model(test_image, device='cpu', verbose=False)

        print("Benchmark 10 inférences...")
        start = time.time()
        for i in range(10):
            results = model(test_image, device='cpu', verbose=False)
        cpu_time = time.time() - start
        cpu_avg = cpu_time / 10
        print(f"✓ Temps CPU total: {cpu_time:.4f}s")
        print(f"✓ Temps CPU moyen: {cpu_avg:.4f}s par image")

        # Test GPU
        print("\n🎮 TEST GPU (DirectML):")
        device = torch_directml.device()
        # Pas de .to(device) pour DirectML - on passe le device dans l'inférence

        print("Warm-up...")
        model(test_image, device=device, verbose=False)

        print("Benchmark 10 inférences...")
        start = time.time()
        for i in range(10):
            results = model(test_image, device=device, verbose=False)
        gpu_time = time.time() - start
        gpu_avg = gpu_time / 10
        print(f"✓ Temps GPU total: {gpu_time:.4f}s")
        print(f"✓ Temps GPU moyen: {gpu_avg:.4f}s par image")

        # Comparaison
        print(f"\n📊 COMPARAISON:")
        speedup = cpu_time / gpu_time
        print(f"  CPU: {cpu_avg:.4f}s par image")
        print(f"  GPU: {gpu_avg:.4f}s par image")
        if speedup > 1:
            print(f"  🚀 Accélération GPU: {speedup:.2f}x plus rapide")
        else:
            print(f"  ⚠️  CPU plus rapide: {1/speedup:.2f}x")
            print(f"  Cela peut arriver avec de petites images ou des GPUs anciens")

        return True

    except Exception as e:
        print(f"✗ Erreur benchmark: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_opencv_gpu():
    """Test 6: OpenCV avec GPU (optionnel)"""
    print_section("TEST 6: OpenCV GPU Support (Optionnel)")

    try:
        # Vérifier si OpenCV a le support CUDA/GPU
        print(f"OpenCV version: {cv2.__version__}")

        # Note: OpenCV avec DirectML n'est pas directement supporté
        # mais on peut vérifier la version
        print(f"ℹ️  OpenCV utilise le CPU pour le décodage vidéo")
        print(f"ℹ️  L'accélération GPU se fait au niveau de YOLO")

        return True

    except Exception as e:
        print(f"✗ Erreur OpenCV: {e}")
        return False


def get_system_info():
    """Afficher les informations système"""
    print_section("INFORMATIONS SYSTÈME")

    import platform
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")

    try:
        import torch_directml
        print(f"DirectML: Installé")
    except:
        print(f"DirectML: Non installé")

    try:
        from ultralytics import YOLO, __version__
        print(f"Ultralytics: {__version__}")
    except:
        print(f"Ultralytics: Non installé")


def main():
    """Fonction principale"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  TEST COMPLET GPU AMD - DirectML pour YOLO".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)

    get_system_info()

    # Test 1: PyTorch
    if not test_pytorch_installation():
        print("\n❌ ÉCHEC: PyTorch non installé correctement")
        return 1

    # Test 2: DirectML
    has_directml, device = test_directml_availability()
    if not has_directml:
        print("\n❌ ÉCHEC: DirectML non disponible")
        print("\n💡 SOLUTION:")
        print("   pip install torch-directml")
        return 1

    # Test 3: Opérations tenseurs
    if not test_tensor_operations(device):
        print("\n❌ ÉCHEC: Opérations GPU échouées")
        return 1

    # Test 4: YOLO GPU
    if not test_yolo_gpu():
        print("\n❌ ÉCHEC: YOLO sur GPU échoué")
        return 1

    # Test 5: Benchmark
    benchmark_cpu_vs_gpu()

    # Test 6: OpenCV
    test_opencv_gpu()

    # Résumé final
    print_section("RÉSUMÉ")
    print("✅ Tous les tests GPU ont réussi!")
    print("\n💡 POUR UTILISER LE GPU:")
    print("   python bike_counter_multiline.py video.mp4 --device gpu")
    print("   python bike_counter.py video.mp4 --device gpu")
    print("\n📊 Le GPU sera automatiquement utilisé si disponible")
    print("="*70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
