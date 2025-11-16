"""
Compteur de vélos pour parade
Détecte et compte les vélos traversant une ligne verticale configurable
Utilise YOLO pour la détection et le suivi d'objets
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import argparse
import json
import os


class BikeCounter:
    def __init__(self, video_path, model_path="yolov8n.pt", counting_line_x=None,
                 min_confidence=0.5, show_video=True, output_path=None, use_gpu=True):
        """
        Initialise le compteur de vélos

        Args:
            video_path: Chemin vers la vidéo MP4
            model_path: Chemin vers le modèle YOLO (par défaut yolov8n.pt)
            counting_line_x: Position X de la ligne de comptage (None = milieu de l'image)
            min_confidence: Confiance minimale pour la détection (0.0 à 1.0)
            show_video: Afficher la vidéo pendant le traitement
            output_path: Chemin pour sauvegarder la vidéo annotée (None = pas de sauvegarde)
            use_gpu: Utiliser le GPU AMD via DirectML si disponible
        """
        self.video_path = video_path
        self.model_path = model_path
        self.counting_line_x = counting_line_x
        self.min_confidence = min_confidence
        self.show_video = show_video
        self.output_path = output_path
        self.use_gpu = use_gpu

        # Configurer le device (GPU ou CPU)
        self.device = self._setup_device()

        # Compteurs
        self.count_left_to_right = 0  # Vélos allant de gauche à droite
        self.count_right_to_left = 0  # Vélos allant de droite à gauche
        self.total_count = 0

        # Suivi des positions précédentes pour détecter le sens de passage
        self.track_history = defaultdict(list)
        self.counted_ids = set()  # IDs des vélos déjà comptés

        # Charger le modèle YOLO
        print(f"Chargement du modèle YOLO: {model_path}")
        self.model = YOLO(model_path)

        # Déplacer le modèle sur le device approprié
        if self.device != 'cpu':
            print(f"Déplacement du modèle sur {self.device}...")
            self.model.to(self.device)

        # Classe 1 = bicycle dans COCO dataset
        self.bike_class_id = 1

    def _setup_device(self):
        """Configure le device GPU ou CPU"""
        if not self.use_gpu:
            print("GPU désactivé, utilisation du CPU")
            return 'cpu'

        try:
            import torch_directml
            if torch_directml.is_available():
                device = torch_directml.device()
                print(f"✓ GPU AMD DirectML détecté et activé: {device}")
                return device
            else:
                print("⚠ DirectML installé mais non disponible, utilisation du CPU")
                return 'cpu'
        except ImportError:
            print("⚠ DirectML non installé, utilisation du CPU")
            print("  Pour activer le GPU AMD: pip install torch-directml")
            return 'cpu'
        except Exception as e:
            print(f"⚠ Erreur lors de la configuration GPU: {e}")
            print("  Utilisation du CPU par défaut")
            return 'cpu'

    def process_video(self):
        """Traite la vidéo et compte les vélos"""

        # Ouvrir la vidéo
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo: {self.video_path}")

        # Récupérer les propriétés de la vidéo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nPropriétés vidéo:")
        print(f"  Résolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Frames totales: {total_frames}")

        # Définir la ligne de comptage (milieu par défaut)
        if self.counting_line_x is None:
            self.counting_line_x = width // 2

        print(f"  Ligne de comptage: x = {self.counting_line_x}")

        # Configurer l'enregistrement vidéo si demandé
        out = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            print(f"  Sauvegarde: {self.output_path}")

        print("\nTraitement en cours...")
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Afficher la progression
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"  Progression: {progress:.1f}% ({frame_count}/{total_frames}) - Comptage: {self.total_count}")

                # Détection et suivi avec YOLO
                results = self.model.track(
                    frame,
                    persist=True,  # Activer le suivi entre frames
                    conf=self.min_confidence,
                    classes=[self.bike_class_id],  # Filtrer uniquement les vélos
                    device=self.device,
                    verbose=False
                )

                # Traiter les détections
                annotated_frame = self.process_detections(frame, results)

                # Sauvegarder la frame si demandé
                if out is not None:
                    out.write(annotated_frame)

                # Afficher la vidéo
                if self.show_video:
                    cv2.imshow('Compteur de Vélos', annotated_frame)

                    # Vérifier les touches seulement toutes les 10 frames pour plus de vitesse
                    if frame_count % 10 == 0:
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\nArrêt demandé par l'utilisateur")
                            break
                    else:
                        cv2.waitKey(1)  # Nécessaire pour rafraîchir l'affichage

        finally:
            # Nettoyer
            cap.release()
            if out is not None:
                out.release()
            if self.show_video:
                cv2.destroyAllWindows()

        # Afficher les résultats
        self.print_results()

        # Sauvegarder les résultats dans un fichier JSON
        self.save_results()

    def process_detections(self, frame, results):
        """
        Traite les détections YOLO et met à jour le comptage

        Args:
            frame: Image de la frame
            results: Résultats YOLO

        Returns:
            Frame annotée
        """
        annotated_frame = frame.copy()

        # Dessiner la ligne de comptage
        height = frame.shape[0]
        cv2.line(annotated_frame,
                (self.counting_line_x, 0),
                (self.counting_line_x, height),
                (0, 255, 0), 3)

        # Zone de tolérance autour de la ligne (en pixels)
        tolerance = 20

        # Traiter chaque détection
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes, track_ids, confidences):
                x_center, y_center, w, h = box

                # Enregistrer la position du centre
                self.track_history[track_id].append((x_center, y_center))

                # Garder seulement les 30 dernières positions
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id] = self.track_history[track_id][-30:]

                # Vérifier si le vélo traverse la ligne
                if track_id not in self.counted_ids and len(self.track_history[track_id]) >= 2:
                    prev_x, _ = self.track_history[track_id][-2]
                    curr_x, _ = self.track_history[track_id][-1]

                    # Détection du passage de gauche à droite
                    if prev_x < self.counting_line_x - tolerance and curr_x >= self.counting_line_x - tolerance:
                        self.count_left_to_right += 1
                        self.total_count += 1
                        self.counted_ids.add(track_id)
                        print(f"  ✓ Vélo #{track_id} compté (G→D) - Total: {self.total_count}")

                    # Détection du passage de droite à gauche
                    elif prev_x > self.counting_line_x + tolerance and curr_x <= self.counting_line_x + tolerance:
                        self.count_right_to_left += 1
                        self.total_count += 1
                        self.counted_ids.add(track_id)
                        print(f"  ✓ Vélo #{track_id} compté (D→G) - Total: {self.total_count}")

                # Dessiner le bounding box
                x1 = int(x_center - w/2)
                y1 = int(y_center - h/2)
                x2 = int(x_center + w/2)
                y2 = int(y_center + h/2)

                # Couleur selon si déjà compté
                color = (0, 255, 0) if track_id in self.counted_ids else (255, 0, 0)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # Afficher l'ID et la confiance
                label = f"ID:{track_id} ({conf:.2f})"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Dessiner la trajectoire
                if len(self.track_history[track_id]) > 1:
                    points = np.array(self.track_history[track_id], dtype=np.int32)
                    cv2.polylines(annotated_frame, [points], False, color, 2)

        # Afficher les compteurs
        self.draw_counters(annotated_frame)

        return annotated_frame

    def draw_counters(self, frame):
        """Dessine les compteurs sur la frame"""
        # Fond semi-transparent pour le texte
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Texte
        cv2.putText(frame, f"Total: {self.total_count}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"G->D: {self.count_left_to_right}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"D->G: {self.count_right_to_left}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def print_results(self):
        """Affiche les résultats finaux"""
        print("\n" + "="*50)
        print("RÉSULTATS DU COMPTAGE")
        print("="*50)
        print(f"Total de vélos comptés: {self.total_count}")
        print(f"  - De gauche à droite: {self.count_left_to_right}")
        print(f"  - De droite à gauche: {self.count_right_to_left}")
        print("="*50)

    def save_results(self):
        """Sauvegarde les résultats dans un fichier JSON"""
        results = {
            "video_path": self.video_path,
            "total_count": self.total_count,
            "left_to_right": self.count_left_to_right,
            "right_to_left": self.count_right_to_left,
            "counting_line_x": self.counting_line_x,
            "min_confidence": self.min_confidence,
            "model_used": self.model_path
        }

        output_file = "bike_count_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nRésultats sauvegardés dans: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compteur de vélos pour parade - Détection et comptage avec YOLO"
    )
    parser.add_argument("video", help="Chemin vers la vidéo MP4")
    parser.add_argument("-m", "--model", default="yolov8n.pt",
                       help="Modèle YOLO à utiliser (défaut: yolov8n.pt)")
    parser.add_argument("-l", "--line", type=int, default=None,
                       help="Position X de la ligne de comptage (défaut: milieu)")
    parser.add_argument("-c", "--confidence", type=float, default=0.5,
                       help="Confiance minimale (0.0-1.0, défaut: 0.5)")
    parser.add_argument("-o", "--output", default=None,
                       help="Sauvegarder la vidéo annotée")
    parser.add_argument("--no-display", action="store_true",
                       help="Ne pas afficher la vidéo pendant le traitement")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Forcer l'utilisation du CPU (désactiver GPU)")

    args = parser.parse_args()

    # Vérifier que la vidéo existe
    if not os.path.exists(args.video):
        print(f"Erreur: La vidéo '{args.video}' n'existe pas")
        return

    # Créer et lancer le compteur
    counter = BikeCounter(
        video_path=args.video,
        model_path=args.model,
        counting_line_x=args.line,
        min_confidence=args.confidence,
        show_video=not args.no_display,
        output_path=args.output,
        use_gpu=not args.no_gpu
    )

    counter.process_video()


if __name__ == "__main__":
    main()
