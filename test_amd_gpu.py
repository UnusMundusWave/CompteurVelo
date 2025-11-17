"""
Script de test et configuration pour GPU AMD GFX 1151 (RDNA 3)
Teste DirectML (Windows) et ROCm (Linux) pour accélération GPU
"""

import sys
import platform

def test_pytorch():
    """Teste l'installation de PyTorch"""
    print("="*80)
    print("TEST 1: PyTorch Installation")
    print("="*80)

    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU détecté: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError as e:
        print(f"✗ PyTorch non installé: {e}")
        return False

def test_directml():
    """Teste DirectML (Windows)"""
    print("\n" + "="*80)
    print("TEST 2: DirectML (Windows AMD GPU)")
    print("="*80)

    if platform.system() != "Windows":
        print("⚠️  DirectML est disponible uniquement sur Windows")
        print(f"  Système détecté: {platform.system()}")
        return False

    try:
        import torch
        import torch_directml

        print(f"✓ torch-directml version: {torch_directml.__version__}")

        # Vérifier les devices DirectML disponibles
        device_count = torch_directml.device_count()
        print(f"✓ Nombre de devices DirectML: {device_count}")

        if device_count > 0:
            for i in range(device_count):
                device = torch_directml.device(i)
                print(f"  Device {i}: {device}")

            # Tester une opération simple
            print("\n  Test d'opération GPU...")
            dml = torch_directml.device()
            tensor = torch.randn(1000, 1000).to(dml)
            result = torch.matmul(tensor, tensor)
            print(f"✓ Opération GPU réussie: {result.shape}")

            return True
        else:
            print("✗ Aucun device DirectML détecté")
            return False

    except ImportError as e:
        print(f"✗ torch-directml non installé: {e}")
        print("\n  Pour installer:")
        print("  pip install torch-directml")
        return False
    except Exception as e:
        print(f"✗ Erreur DirectML: {e}")
        return False

def test_rocm():
    """Teste ROCm (Linux)"""
    print("\n" + "="*80)
    print("TEST 3: ROCm (Linux AMD GPU)")
    print("="*80)

    if platform.system() != "Linux":
        print("⚠️  ROCm est disponible uniquement sur Linux")
        print(f"  Système détecté: {platform.system()}")
        return False

    try:
        import torch

        # Vérifier si PyTorch a été compilé avec ROCm
        if hasattr(torch.version, 'hip'):
            print(f"✓ PyTorch compilé avec ROCm/HIP: {torch.version.hip}")
        else:
            print("✗ PyTorch n'est pas compilé avec ROCm")
            print("\n  Pour installer PyTorch avec ROCm:")
            print("  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7")
            return False

        if torch.cuda.is_available():
            print(f"✓ GPU AMD détecté via ROCm")
            print(f"  Nom: {torch.cuda.get_device_name(0)}")
            print(f"  Nombre de GPUs: {torch.cuda.device_count()}")

            # Tester une opération simple
            print("\n  Test d'opération GPU...")
            tensor = torch.randn(1000, 1000).cuda()
            result = torch.matmul(tensor, tensor)
            print(f"✓ Opération GPU réussie: {result.shape}")

            return True
        else:
            print("✗ Aucun GPU détecté par ROCm")
            return False

    except Exception as e:
        print(f"✗ Erreur ROCm: {e}")
        return False

def test_yolo_with_gpu():
    """Teste YOLO avec GPU"""
    print("\n" + "="*80)
    print("TEST 4: YOLO avec GPU")
    print("="*80)

    try:
        from ultralytics import YOLO
        import torch

        print("✓ Ultralytics YOLO installé")

        # Détecter le device approprié
        device = None
        device_name = "CPU"

        # Essayer DirectML (Windows)
        if platform.system() == "Windows":
            try:
                import torch_directml
                if torch_directml.device_count() > 0:
                    device = torch_directml.device()
                    device_name = "DirectML (AMD GPU)"
                    print(f"✓ Device sélectionné: {device_name}")
            except:
                pass

        # Essayer ROCm (Linux)
        if device is None and platform.system() == "Linux":
            if torch.cuda.is_available():
                device = "cuda"
                device_name = f"ROCm (AMD {torch.cuda.get_device_name(0)})"
                print(f"✓ Device sélectionné: {device_name}")

        # Fallback sur CPU
        if device is None:
            device = "cpu"
            device_name = "CPU"
            print(f"⚠️  Utilisation du CPU (GPU non disponible)")

        # Charger un modèle YOLO léger
        print("\n  Chargement du modèle YOLO...")
        model = YOLO("yolov8n.pt")

        # Tester une prédiction
        print(f"  Test de prédiction sur {device_name}...")

        import numpy as np
        # Créer une image de test
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Prédiction avec mesure du temps
        import time
        start_time = time.time()

        try:
            results = model.predict(
                test_image,
                device=device,
                verbose=False,
                conf=0.5
            )
            elapsed = time.time() - start_time

            print(f"✓ Prédiction réussie en {elapsed:.3f}s sur {device_name}")
            print(f"  Détections: {len(results[0].boxes)} objets")

            return True

        except Exception as e:
            print(f"✗ Erreur lors de la prédiction: {e}")

            # Si DirectML échoue, essayer avec CPU
            if device_name != "CPU":
                print(f"\n  Essai avec CPU comme fallback...")
                try:
                    results = model.predict(
                        test_image,
                        device="cpu",
                        verbose=False,
                        conf=0.5
                    )
                    elapsed = time.time() - start_time
                    print(f"✓ Prédiction CPU réussie en {elapsed:.3f}s")
                    print(f"  Note: Le GPU a échoué, CPU fonctionne")
                    return False
                except Exception as e2:
                    print(f"✗ Erreur également avec CPU: {e2}")
                    return False
            return False

    except ImportError as e:
        print(f"✗ Ultralytics YOLO non installé: {e}")
        print("\n  Pour installer:")
        print("  pip install ultralytics")
        return False
    except Exception as e:
        print(f"✗ Erreur YOLO: {e}")
        return False

def get_gpu_info():
    """Affiche les informations système"""
    print("="*80)
    print("INFORMATIONS SYSTÈME")
    print("="*80)

    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")

    # Informations GPU (si disponible)
    try:
        if platform.system() == "Linux":
            import subprocess
            result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True)
            gpu_lines = [line for line in result.stdout.split('\n') if 'VGA' in line or 'Display' in line]
            if gpu_lines:
                print(f"\nGPU détecté:")
                for line in gpu_lines:
                    print(f"  {line}")
    except:
        pass

    print()

def print_recommendations():
    """Affiche les recommandations"""
    print("\n" + "="*80)
    print("RECOMMANDATIONS POUR AMD GFX 1151 (RDNA 3)")
    print("="*80)

    os_type = platform.system()

    if os_type == "Windows":
        print("\n✓ WINDOWS - Utilisez DirectML:")
        print("  1. Installer torch-directml:")
        print("     pip install torch-directml")
        print("\n  2. Mettre à jour les pilotes AMD Adrenalin (dernière version)")
        print("     https://www.amd.com/en/support")
        print("\n  3. Utiliser le device DirectML dans le code:")
        print("     import torch_directml")
        print("     device = torch_directml.device()")
        print("     model.predict(frame, device=device)")

    elif os_type == "Linux":
        print("\n✓ LINUX - Utilisez ROCm:")
        print("  1. Installer ROCm 5.7+ (compatible RDNA 3):")
        print("     https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html")
        print("\n  2. Installer PyTorch avec support ROCm:")
        print("     pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7")
        print("\n  3. Utiliser CUDA device (ROCm émule CUDA):")
        print("     device = 'cuda'")
        print("     model.predict(frame, device=device)")
        print("\n  4. Vérifier que votre GPU est supporté:")
        print("     rocminfo | grep gfx")
        print("     # Doit afficher: gfx1100 ou gfx1101 pour RDNA 3")

    print("\n" + "="*80)
    print("NOTE IMPORTANTE:")
    print("="*80)
    print("Pour YOLO avec DirectML, certaines versions peuvent avoir des problèmes.")
    print("Si vous rencontrez l'erreur 'Cannot set version_counter for inference tensor':")
    print("  - Essayez une version plus récente d'Ultralytics: pip install -U ultralytics")
    print("  - Ou utilisez le CPU qui offre de bonnes performances (32 threads)")
    print()

def main():
    """Fonction principale"""
    get_gpu_info()

    pytorch_ok = test_pytorch()
    directml_ok = test_directml() if platform.system() == "Windows" else False
    rocm_ok = test_rocm() if platform.system() == "Linux" else False
    yolo_ok = test_yolo_with_gpu()

    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ DES TESTS")
    print("="*80)
    print(f"PyTorch:    {'✓ OK' if pytorch_ok else '✗ ÉCHEC'}")
    print(f"DirectML:   {'✓ OK' if directml_ok else '✗ Non disponible/échoué'}")
    print(f"ROCm:       {'✓ OK' if rocm_ok else '✗ Non disponible/échoué'}")
    print(f"YOLO GPU:   {'✓ OK' if yolo_ok else '✗ ÉCHEC (utiliser CPU)'}")

    print_recommendations()

    # Conclusion
    print("="*80)
    print("CONCLUSION")
    print("="*80)

    if directml_ok or rocm_ok:
        if yolo_ok:
            print("✓ GPU AMD GFX 1151 configuré et fonctionnel avec YOLO!")
            print("  Vous pouvez utiliser l'accélération GPU pour le comptage.")
        else:
            print("⚠️  GPU détecté mais YOLO a des problèmes.")
            print("  Recommandation: Utiliser le CPU (performances acceptables avec 32 threads)")
    else:
        print("⚠️  GPU AMD non configuré correctement.")
        print("  Suivez les recommandations ci-dessus pour activer l'accélération GPU.")
        print("  Ou utilisez le CPU qui offre de bonnes performances.")

    print("="*80 + "\n")

if __name__ == "__main__":
    main()
