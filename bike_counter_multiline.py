"""
Compteur de vélos pour parade - VERSION MULTI-LIGNES
Détecte et compte les vélos traversant 3 lignes verticales (A, B, C)
Analyse les trajectoires complètes et les combinaisons de passages
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import argparse
import json
import os
from itertools import combinations


class MultiLineBikeCounter:
    def __init__(self, video_path, model_path="yolov8n.pt",
                 line_a=None, line_b=None, line_c=None,
                 min_confidence=0.5, show_video=True, output_path=None):
        """
        Initialise le compteur de vélos multi-lignes

        Args:
            video_path: Chemin vers la vidéo MP4
            model_path: Chemin vers le modèle YOLO (par défaut yolov8n.pt)
            line_a: Position X de la ligne A (None = 1/4 de la largeur)
            line_b: Position X de la ligne B (None = 1/2 de la largeur)
            line_c: Position X de la ligne C (None = 3/4 de la largeur)
            min_confidence: Confiance minimale pour la détection (0.0 à 1.0)
            show_video: Afficher la vidéo pendant le traitement
            output_path: Chemin pour sauvegarder la vidéo annotée
        """
        self.video_path = video_path
        self.model_path = model_path
        self.line_a = line_a
        self.line_b = line_b
        self.line_c = line_c
        self.min_confidence = min_confidence
        self.show_video = show_video
        self.output_path = output_path

        # Tracking des vélos et leurs passages
        self.track_history = defaultdict(list)
        self.bike_crossings = defaultdict(set)  # {bike_id: {A, B, C}}

        # Compteurs par ligne
        self.count_a = 0
        self.count_b = 0
        self.count_c = 0

        # Compteurs par combinaison
        self.combinations_count = {
            'A_only': 0,
            'B_only': 0,
            'C_only': 0,
            'A_and_B': 0,
            'A_and_C': 0,
            'B_and_C': 0,
            'A_B_and_C': 0
        }

        # Vélos terminés (sortis de l'image)
        self.finished_bikes = set()

        # Charger le modèle YOLO
        print(f"Chargement du modèle YOLO: {model_path}")
        self.model = YOLO(model_path)

        # Classe 1 = bicycle dans COCO dataset
        self.bike_class_id = 1

        # Zone de tolérance autour des lignes (pixels)
        self.tolerance = 20

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

        # Définir les lignes de comptage (répartition par défaut)
        if self.line_a is None:
            self.line_a = width // 4
        if self.line_b is None:
            self.line_b = width // 2
        if self.line_c is None:
            self.line_c = (3 * width) // 4

        print(f"\nLignes de comptage:")
        print(f"  Ligne A: x = {self.line_a}")
        print(f"  Ligne B: x = {self.line_b}")
        print(f"  Ligne C: x = {self.line_c}")

        # Configurer l'enregistrement vidéo si demandé
        out = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            print(f"  Sauvegarde: {self.output_path}")

        print("\nTraitement en cours...")
        frame_count = 0

        # Garder trace des IDs vus dans la frame précédente
        previous_ids = set()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Afficher la progression
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    total_bikes = len(self.bike_crossings)
                    print(f"  Progression: {progress:.1f}% ({frame_count}/{total_frames}) - Vélos détectés: {total_bikes}")

                # Détection et suivi avec YOLO
                results = self.model.track(
                    frame,
                    persist=True,
                    conf=self.min_confidence,
                    classes=[self.bike_class_id],
                    verbose=False
                )

                # Traiter les détections
                current_ids = self.process_detections(frame, results)
                annotated_frame = self.draw_annotations(frame, results, current_ids)

                # Détecter les vélos qui ont quitté l'image
                disappeared_ids = previous_ids - current_ids
                for bike_id in disappeared_ids:
                    if bike_id not in self.finished_bikes:
                        self.finalize_bike(bike_id)

                previous_ids = current_ids.copy()

                # Sauvegarder la frame si demandé
                if out is not None:
                    out.write(annotated_frame)

                # Afficher la vidéo
                if self.show_video:
                    cv2.imshow('Compteur Multi-Lignes', annotated_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nArrêt demandé par l'utilisateur")
                        break

        finally:
            # Finaliser tous les vélos restants
            for bike_id in self.bike_crossings.keys():
                if bike_id not in self.finished_bikes:
                    self.finalize_bike(bike_id)

            # Nettoyer
            cap.release()
            if out is not None:
                out.release()
            if self.show_video:
                cv2.destroyAllWindows()

        # Afficher les résultats
        self.print_results()

        # Sauvegarder les résultats
        self.save_results()

    def process_detections(self, frame, results):
        """
        Traite les détections YOLO et enregistre les passages de lignes

        Returns:
            set: IDs des vélos détectés dans cette frame
        """
        current_ids = set()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x_center, y_center, w, h = box
                current_ids.add(track_id)

                # Enregistrer la position
                self.track_history[track_id].append((x_center, y_center))

                # Garder seulement les 50 dernières positions
                if len(self.track_history[track_id]) > 50:
                    self.track_history[track_id] = self.track_history[track_id][-50:]

                # Vérifier le passage des lignes
                if len(self.track_history[track_id]) >= 2:
                    prev_x, _ = self.track_history[track_id][-2]
                    curr_x, _ = self.track_history[track_id][-1]

                    # Vérifier chaque ligne
                    self.check_line_crossing(track_id, prev_x, curr_x, self.line_a, 'A')
                    self.check_line_crossing(track_id, prev_x, curr_x, self.line_b, 'B')
                    self.check_line_crossing(track_id, prev_x, curr_x, self.line_c, 'C')

        return current_ids

    def check_line_crossing(self, bike_id, prev_x, curr_x, line_x, line_name):
        """Vérifie si un vélo a traversé une ligne"""
        # Passage de gauche à droite
        if prev_x < line_x - self.tolerance and curr_x >= line_x - self.tolerance:
            if line_name not in self.bike_crossings[bike_id]:
                self.bike_crossings[bike_id].add(line_name)
                print(f"  ✓ Vélo #{bike_id} a traversé la ligne {line_name} (→)")

        # Passage de droite à gauche
        elif prev_x > line_x + self.tolerance and curr_x <= line_x + self.tolerance:
            if line_name not in self.bike_crossings[bike_id]:
                self.bike_crossings[bike_id].add(line_name)
                print(f"  ✓ Vélo #{bike_id} a traversé la ligne {line_name} (←)")

    def finalize_bike(self, bike_id):
        """Finalise le comptage pour un vélo qui a quitté l'image"""
        if bike_id in self.finished_bikes:
            return

        self.finished_bikes.add(bike_id)
        crossings = self.bike_crossings[bike_id]

        if not crossings:
            return

        # Incrémenter les compteurs individuels
        if 'A' in crossings:
            self.count_a += 1
        if 'B' in crossings:
            self.count_b += 1
        if 'C' in crossings:
            self.count_c += 1

        # Déterminer la combinaison
        crossing_set = frozenset(crossings)

        if crossing_set == {'A'}:
            self.combinations_count['A_only'] += 1
            print(f"  📊 Vélo #{bike_id} - Trajet: A seulement")
        elif crossing_set == {'B'}:
            self.combinations_count['B_only'] += 1
            print(f"  📊 Vélo #{bike_id} - Trajet: B seulement")
        elif crossing_set == {'C'}:
            self.combinations_count['C_only'] += 1
            print(f"  📊 Vélo #{bike_id} - Trajet: C seulement")
        elif crossing_set == {'A', 'B'}:
            self.combinations_count['A_and_B'] += 1
            print(f"  📊 Vélo #{bike_id} - Trajet: A et B")
        elif crossing_set == {'A', 'C'}:
            self.combinations_count['A_and_C'] += 1
            print(f"  📊 Vélo #{bike_id} - Trajet: A et C")
        elif crossing_set == {'B', 'C'}:
            self.combinations_count['B_and_C'] += 1
            print(f"  📊 Vélo #{bike_id} - Trajet: B et C")
        elif crossing_set == {'A', 'B', 'C'}:
            self.combinations_count['A_B_and_C'] += 1
            print(f"  📊 Vélo #{bike_id} - Trajet: A, B et C (COMPLET)")

    def draw_annotations(self, frame, results, current_ids):
        """Dessine les annotations sur la frame"""
        annotated_frame = frame.copy()
        height = frame.shape[0]

        # Dessiner les lignes de comptage avec couleurs différentes
        cv2.line(annotated_frame, (self.line_a, 0), (self.line_a, height), (255, 0, 0), 3)  # Bleu - A
        cv2.line(annotated_frame, (self.line_b, 0), (self.line_b, height), (0, 255, 0), 3)  # Vert - B
        cv2.line(annotated_frame, (self.line_c, 0), (self.line_c, height), (0, 0, 255), 3)  # Rouge - C

        # Labels des lignes
        cv2.putText(annotated_frame, "A", (self.line_a - 15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.putText(annotated_frame, "B", (self.line_b - 15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(annotated_frame, "C", (self.line_c - 15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Dessiner les détections
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes, track_ids, confidences):
                x_center, y_center, w, h = box

                # Bounding box
                x1 = int(x_center - w/2)
                y1 = int(y_center - h/2)
                x2 = int(x_center + w/2)
                y2 = int(y_center + h/2)

                # Couleur selon les lignes traversées
                crossings = self.bike_crossings[track_id]
                if len(crossings) == 3:
                    color = (255, 0, 255)  # Magenta - Toutes les lignes
                elif len(crossings) == 2:
                    color = (0, 255, 255)  # Cyan - Deux lignes
                elif len(crossings) == 1:
                    color = (255, 255, 0)  # Jaune - Une ligne
                else:
                    color = (128, 128, 128)  # Gris - Aucune ligne

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # Label avec ID et lignes traversées
                lines_str = ''.join(sorted(crossings)) if crossings else '-'
                label = f"#{track_id} [{lines_str}]"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Dessiner la trajectoire
                if len(self.track_history[track_id]) > 1:
                    points = np.array(self.track_history[track_id], dtype=np.int32)
                    cv2.polylines(annotated_frame, [points], False, color, 2)

        # Afficher les statistiques
        self.draw_stats(annotated_frame)

        return annotated_frame

    def draw_stats(self, frame):
        """Dessine les statistiques sur la frame"""
        # Fond semi-transparent
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 50), (450, 350), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        y_offset = 80
        line_height = 30

        # Titre
        cv2.putText(frame, "COMPTAGE MULTI-LIGNES", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += line_height + 10

        # Compteurs par ligne
        cv2.putText(frame, f"Ligne A: {self.count_a}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        y_offset += line_height

        cv2.putText(frame, f"Ligne B: {self.count_b}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height

        cv2.putText(frame, f"Ligne C: {self.count_c}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += line_height + 10

        # Combinaisons
        cv2.putText(frame, "Trajets:", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        y_offset += line_height

        cv2.putText(frame, f"  A seul: {self.combinations_count['A_only']}", (30, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        y_offset += 25

        cv2.putText(frame, f"  B seul: {self.combinations_count['B_only']}", (30, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        y_offset += 25

        cv2.putText(frame, f"  C seul: {self.combinations_count['C_only']}", (30, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        y_offset += 25

        cv2.putText(frame, f"  A+B: {self.combinations_count['A_and_B']}", (30, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        y_offset += 25

        cv2.putText(frame, f"  A+C: {self.combinations_count['A_and_C']}", (30, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        y_offset += 25

        cv2.putText(frame, f"  B+C: {self.combinations_count['B_and_C']}", (30, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        y_offset += 25

        cv2.putText(frame, f"  A+B+C: {self.combinations_count['A_B_and_C']}", (30, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    def print_results(self):
        """Affiche les résultats finaux"""
        print("\n" + "="*70)
        print("RÉSULTATS DU COMPTAGE MULTI-LIGNES")
        print("="*70)

        print(f"\n📊 PASSAGES PAR LIGNE:")
        print(f"  Ligne A: {self.count_a} vélos")
        print(f"  Ligne B: {self.count_b} vélos")
        print(f"  Ligne C: {self.count_c} vélos")

        print(f"\n🚴 ANALYSE DES TRAJETS:")
        print(f"  A seulement:        {self.combinations_count['A_only']} vélos")
        print(f"  B seulement:        {self.combinations_count['B_only']} vélos")
        print(f"  C seulement:        {self.combinations_count['C_only']} vélos")
        print(f"  A et B:             {self.combinations_count['A_and_B']} vélos")
        print(f"  A et C:             {self.combinations_count['A_and_C']} vélos")
        print(f"  B et C:             {self.combinations_count['B_and_C']} vélos")
        print(f"  A, B et C (complet): {self.combinations_count['A_B_and_C']} vélos")

        total_unique = sum(self.combinations_count.values())
        print(f"\n✅ TOTAL DE VÉLOS UNIQUES: {total_unique}")
        print("="*70 + "\n")

    def save_results(self):
        """Sauvegarde les résultats dans un fichier JSON"""
        results = {
            "video_path": self.video_path,
            "model_used": self.model_path,
            "lines": {
                "A": self.line_a,
                "B": self.line_b,
                "C": self.line_c
            },
            "counts_per_line": {
                "A": self.count_a,
                "B": self.count_b,
                "C": self.count_c
            },
            "trajectory_analysis": {
                "A_only": self.combinations_count['A_only'],
                "B_only": self.combinations_count['B_only'],
                "C_only": self.combinations_count['C_only'],
                "A_and_B": self.combinations_count['A_and_B'],
                "A_and_C": self.combinations_count['A_and_C'],
                "B_and_C": self.combinations_count['B_and_C'],
                "A_B_and_C": self.combinations_count['A_B_and_C']
            },
            "total_unique_bikes": sum(self.combinations_count.values()),
            "min_confidence": self.min_confidence
        }

        output_file = "bike_count_multiline_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Résultats sauvegardés dans: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compteur multi-lignes de vélos - Analyse des trajectoires A, B, C"
    )
    parser.add_argument("video", help="Chemin vers la vidéo MP4")
    parser.add_argument("-m", "--model", default="yolov8n.pt",
                       help="Modèle YOLO (défaut: yolov8n.pt)")
    parser.add_argument("-a", "--line-a", type=int, default=None,
                       help="Position X de la ligne A (défaut: 1/4 largeur)")
    parser.add_argument("-b", "--line-b", type=int, default=None,
                       help="Position X de la ligne B (défaut: 1/2 largeur)")
    parser.add_argument("-c", "--line-c", type=int, default=None,
                       help="Position X de la ligne C (défaut: 3/4 largeur)")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Confiance minimale (0.0-1.0, défaut: 0.5)")
    parser.add_argument("-o", "--output", default=None,
                       help="Sauvegarder la vidéo annotée")
    parser.add_argument("--no-display", action="store_true",
                       help="Ne pas afficher la vidéo")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Erreur: La vidéo '{args.video}' n'existe pas")
        return

    counter = MultiLineBikeCounter(
        video_path=args.video,
        model_path=args.model,
        line_a=args.line_a,
        line_b=args.line_b,
        line_c=args.line_c,
        min_confidence=args.confidence,
        show_video=not args.no_display,
        output_path=args.output
    )

    counter.process_video()


if __name__ == "__main__":
    main()
