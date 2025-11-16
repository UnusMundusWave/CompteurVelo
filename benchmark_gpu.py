"""
Benchmark GPU vs CPU pour le comptage de vélos
Compare les performances avec et sans DirectML
"""

import time
import sys
import argparse
import cv2
import numpy as np
from ultralytics import YOLO


def create_test_video(output_path, width=1920, height=1080, num_frames=100):
    """Crée une vidéo de test synthétique"""
    print(f"Création vidéo de test: {num_frames} frames {width}x{height}...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    for i in range(num_frames):
        # Créer une frame aléatoire
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        # Ajouter quelques formes pour simuler des objets
        for _ in range(5):
            x = np.random.randint(0, width - 100)
            y = np.random.randint(0, height - 100)
            cv2.rectangle(frame, (x, y), (x + 100, y + 100), (255, 255, 255), -1)

        out.write(frame)

    out.release()
    print(f"✓ Vidéo de test créée: {output_path}")


def benchmark_device(video_path, device, model_path="yolov8n.pt", num_runs=3):
    """Benchmark sur un device spécifique"""
    device_name = "CPU" if device == 'cpu' else f"GPU ({device})"
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {device_name}")
    print(f"{'='*60}")

    # Charger le modèle
    print("Chargement du modèle...")
    model = YOLO(model_path)

    # Note: Pas de .to(device) pour DirectML
    # On passe le device directement dans les appels d'inférence

    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    times = []

    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}...")

        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        # Warm-up
        if run == 0:
            print("  Warm-up...")
            ret, frame = cap.read()
            if ret:
                model.track(frame, device=device, persist=True, verbose=False)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Inférence
            results = model.track(
                frame,
                device=device,
                persist=True,
                conf=0.5,
                classes=[1],  # bicycles
                verbose=False
            )

            frame_count += 1

            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"  Frame {frame_count}/{total_frames} - FPS: {fps:.2f}", end='\r')

        elapsed = time.time() - start_time
        times.append(elapsed)

        cap.release()

        fps = total_frames / elapsed if elapsed > 0 else 0
        print(f"\n  ✓ Run {run + 1}: {elapsed:.2f}s ({fps:.2f} FPS)")

    # Statistiques
    avg_time = sum(times) / len(times)
    avg_fps = total_frames / avg_time

    print(f"\n📊 RÉSULTATS {device_name}:")
    print(f"  Temps moyen: {avg_time:.2f}s")
    print(f"  FPS moyen: {avg_fps:.2f}")
    print(f"  Temps par frame: {avg_time / total_frames * 1000:.2f}ms")

    return {
        'device': device_name,
        'avg_time': avg_time,
        'avg_fps': avg_fps,
        'times': times
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GPU vs CPU pour le comptage de vélos"
    )
    parser.add_argument("-v", "--video", help="Vidéo à utiliser (sinon crée une vidéo de test)")
    parser.add_argument("-m", "--model", default="yolov8n.pt", help="Modèle YOLO")
    parser.add_argument("-f", "--frames", type=int, default=100,
                       help="Nombre de frames pour vidéo de test (défaut: 100)")
    parser.add_argument("-r", "--runs", type=int, default=3,
                       help="Nombre de runs par device (défaut: 3)")

    args = parser.parse_args()

    # Vidéo de test
    video_path = args.video
    if not video_path:
        video_path = "test_benchmark.mp4"
        create_test_video(video_path, num_frames=args.frames)

    print(f"\n{'█'*60}")
    print("█" + " "*58 + "█")
    print("█" + "  BENCHMARK GPU AMD vs CPU - YOLO".center(58) + "█")
    print("█" + " "*58 + "█")
    print(f"{'█'*60}")
    print(f"\nVidéo: {video_path}")
    print(f"Modèle: {args.model}")
    print(f"Runs: {args.runs}")

    # Benchmark CPU
    cpu_results = benchmark_device(video_path, 'cpu', args.model, args.runs)

    # Benchmark GPU (si disponible)
    gpu_results = None
    try:
        import torch_directml
        if torch_directml.is_available():
            device = torch_directml.device()
            gpu_results = benchmark_device(video_path, device, args.model, args.runs)
        else:
            print("\n⚠️  DirectML installé mais non disponible")
    except ImportError:
        print("\n⚠️  DirectML non installé, GPU benchmark ignoré")
        print("   Installez avec: pip install torch-directml")

    # Comparaison
    print(f"\n{'='*60}")
    print("COMPARAISON FINALE")
    print(f"{'='*60}")

    print(f"\n🖥️  CPU:")
    print(f"  Temps moyen: {cpu_results['avg_time']:.2f}s")
    print(f"  FPS moyen: {cpu_results['avg_fps']:.2f}")

    if gpu_results:
        print(f"\n🎮 GPU (DirectML):")
        print(f"  Temps moyen: {gpu_results['avg_time']:.2f}s")
        print(f"  FPS moyen: {gpu_results['avg_fps']:.2f}")

        speedup = cpu_results['avg_time'] / gpu_results['avg_time']
        fps_improvement = gpu_results['avg_fps'] / cpu_results['avg_fps']

        print(f"\n📈 AMÉLIORATION:")
        if speedup > 1:
            print(f"  🚀 GPU {speedup:.2f}x plus rapide")
            print(f"  🚀 FPS {fps_improvement:.2f}x supérieur")
        else:
            print(f"  ⚠️  CPU {1/speedup:.2f}x plus rapide")
            print(f"     (Normal pour petites vidéos ou GPU ancien)")

        # Estimation pour vidéo longue
        print(f"\n⏱️  ESTIMATION pour vidéo 1 heure (30 FPS = 108000 frames):")
        cpu_hour = (cpu_results['avg_time'] / args.frames) * 108000 / 60
        gpu_hour = (gpu_results['avg_time'] / args.frames) * 108000 / 60

        print(f"  CPU: {cpu_hour:.1f} minutes")
        print(f"  GPU: {gpu_hour:.1f} minutes")
        print(f"  Gain de temps: {cpu_hour - gpu_hour:.1f} minutes")

    print(f"\n{'='*60}")
    print("💡 RECOMMANDATION:")

    if gpu_results and speedup > 1.2:
        print("  ✓ Utilisez le GPU pour de meilleures performances")
        print("  python bike_counter.py video.mp4")
        print("  (Le GPU est activé par défaut)")
    elif gpu_results:
        print("  ~ Le GPU offre peu d'amélioration")
        print("  Utilisez --no-gpu si vous rencontrez des problèmes")
    else:
        print("  ! GPU non disponible")
        print("  Installez DirectML: pip install torch-directml")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
