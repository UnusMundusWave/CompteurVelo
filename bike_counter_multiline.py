"""
Compteur de vélos pour parade - VERSION MULTI-LIGNES
Détecte et compte les vélos traversant 5 lignes verticales (A, B, C, D, E)
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
                 line_a=None, line_b=None, line_c=None, line_d=None, line_e=None,
                 min_confidence=0.5, show_video=True, output_path=None,
                 use_gpu=True):
        """
        Initialise le compteur de vélos multi-lignes

        Args:
            video_path: Chemin vers la vidéo MP4
            model_path: Chemin vers le modèle YOLO (par défaut yolov8n.pt)
            line_a: Position X de la ligne A (None = 1/6 de la largeur)
            line_b: Position X de la ligne B (None = 2/6 de la largeur)
            line_c: Position X de la ligne C (None = 3/6 de la largeur)
            line_d: Position X de la ligne D (None = 4/6 de la largeur)
            line_e: Position X de la ligne E (None = 5/6 de la largeur)
            min_confidence: Confiance minimale pour la détection (0.0 à 1.0)
            show_video: Afficher la vidéo pendant le traitement
            output_path: Chemin pour sauvegarder la vidéo annotée
            use_gpu: Utiliser le GPU AMD via DirectML si disponible
        """
        self.video_path = video_path
        self.model_path = model_path
        self.line_a = line_a
        self.line_b = line_b
        self.line_c = line_c
        self.line_d = line_d
        self.line_e = line_e
        self.min_confidence = min_confidence
        self.show_video = show_video
        self.output_path = output_path
        self.use_gpu = use_gpu

        # Configurer le device (GPU ou CPU)
        self.device = self._setup_device()

        # Tracking des objets et leurs passages
        self.track_history = defaultdict(list)
        self.object_crossings = defaultdict(set)  # {object_id: {A, B, C, D, E}}
        self.object_types = {}  # {object_id: 'bike' or 'person'}

        # Compteurs par ligne - VÉLOS
        self.count_bikes_a = 0
        self.count_bikes_b = 0
        self.count_bikes_c = 0
        self.count_bikes_d = 0
        self.count_bikes_e = 0

        # Compteurs par ligne - PERSONNES
        self.count_people_a = 0
        self.count_people_b = 0
        self.count_people_c = 0
        self.count_people_d = 0
        self.count_people_e = 0

        # Compteurs par combinaison - VÉLOS (31 combinaisons possibles pour 5 lignes)
        self.bikes_combinations_count = {
            # Singles (5)
            'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0,
            # Pairs (10)
            'AB': 0, 'AC': 0, 'AD': 0, 'AE': 0, 'BC': 0, 'BD': 0, 'BE': 0, 'CD': 0, 'CE': 0, 'DE': 0,
            # Triples (10)
            'ABC': 0, 'ABD': 0, 'ABE': 0, 'ACD': 0, 'ACE': 0, 'ADE': 0, 'BCD': 0, 'BCE': 0, 'BDE': 0, 'CDE': 0,
            # Quadruples (5)
            'ABCD': 0, 'ABCE': 0, 'ABDE': 0, 'ACDE': 0, 'BCDE': 0,
            # Quintuple (1)
            'ABCDE': 0
        }

        # Compteurs par combinaison - PERSONNES (31 combinaisons possibles pour 5 lignes)
        self.people_combinations_count = {
            # Singles (5)
            'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0,
            # Pairs (10)
            'AB': 0, 'AC': 0, 'AD': 0, 'AE': 0, 'BC': 0, 'BD': 0, 'BE': 0, 'CD': 0, 'CE': 0, 'DE': 0,
            # Triples (10)
            'ABC': 0, 'ABD': 0, 'ABE': 0, 'ACD': 0, 'ACE': 0, 'ADE': 0, 'BCD': 0, 'BCE': 0, 'BDE': 0, 'CDE': 0,
            # Quadruples (5)
            'ABCD': 0, 'ABCE': 0, 'ABDE': 0, 'ACDE': 0, 'BCDE': 0,
            # Quintuple (1)
            'ABCDE': 0
        }

        # Objets terminés (sortis de l'image)
        self.finished_objects = set()

        # Charger le modèle YOLO
        print(f"Chargement du modèle YOLO: {model_path}")
        self.model = YOLO(model_path)

        # Note: Pour DirectML, on ne déplace PAS le modèle avec .to()
        # On passe simplement le device dans les appels .track()
        # Cela évite l'erreur "Cannot set version_counter for inference tensor"

        # Classes COCO: 0 = person, 1 = bicycle
        self.person_class_id = 0
        self.bike_class_id = 1

        # Zone de tolérance autour des lignes (pixels)
        self.tolerance = 20

    def _setup_device(self):
        """Configure le device GPU ou CPU - Support AMD GFX 1151 (RDNA 3)"""
        import torch
        import platform

        # Configurer PyTorch pour utiliser tous les threads CPU disponibles
        torch.set_num_threads(32)  # Optimisation pour CPU 32 threads
        torch.set_num_interop_threads(4)  # Threads pour opérations parallèles

        num_threads = torch.get_num_threads()
        print(f"✓ CPU optimisé: {num_threads} threads configurés")

        if not self.use_gpu:
            print("Mode CPU sélectionné")
            return 'cpu'

        # Tenter d'utiliser le GPU AMD
        print("\n🔍 Détection GPU AMD...")

        # Option 1: DirectML (Windows uniquement)
        if platform.system() == "Windows":
            try:
                import torch_directml
                device_count = torch_directml.device_count()

                if device_count > 0:
                    print(f"✓ DirectML détecté: {device_count} device(s)")
                    device = torch_directml.device()
                    print(f"✓ Utilisation du GPU AMD via DirectML")
                    print(f"  Device: {device}")
                    print(f"  Note: Compatible avec AMD GFX 1151 (RDNA 3)")
                    return device
                else:
                    print("⚠️  DirectML installé mais aucun GPU détecté")

            except ImportError:
                print("⚠️  torch-directml non installé")
                print("   Pour activer le GPU AMD sur Windows:")
                print("   pip install torch-directml")
            except Exception as e:
                print(f"⚠️  Erreur DirectML: {e}")

        # Option 2: ROCm (Linux uniquement)
        elif platform.system() == "Linux":
            try:
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"✓ ROCm détecté")
                    print(f"  GPU: {gpu_name}")
                    print(f"  Nombre de GPUs: {torch.cuda.device_count()}")

                    # Vérifier si c'est bien un GPU AMD RDNA 3
                    if "AMD" in gpu_name.upper() or "Radeon" in gpu_name:
                        print(f"✓ Utilisation du GPU AMD via ROCm")
                        print(f"  Note: Compatible avec AMD GFX 1151 (RDNA 3)")
                        return 'cuda'  # ROCm utilise l'interface CUDA
                    else:
                        print(f"⚠️  GPU détecté mais n'est pas AMD: {gpu_name}")
                else:
                    print("⚠️  ROCm installé mais aucun GPU détecté")
                    print("   Vérifiez que ROCm est correctement installé:")
                    print("   rocminfo | grep gfx")

            except Exception as e:
                print(f"⚠️  ROCm non disponible: {e}")
                print("   Pour activer le GPU AMD sur Linux:")
                print("   1. Installer ROCm 5.7+")
                print("   2. pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7")

        # Fallback sur CPU
        print("\n⚠️  GPU AMD non disponible, utilisation du CPU")
        print("   Le CPU 32 threads offre d'excellentes performances")
        print("   Pour activer le GPU, exécutez: python test_amd_gpu.py")
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

        # Définir les lignes de comptage (répartition par défaut: 1/6, 2/6, 3/6, 4/6, 5/6)
        if self.line_a is None:
            self.line_a = width // 6
        if self.line_b is None:
            self.line_b = (2 * width) // 6
        if self.line_c is None:
            self.line_c = (3 * width) // 6
        if self.line_d is None:
            self.line_d = (4 * width) // 6
        if self.line_e is None:
            self.line_e = (5 * width) // 6

        print(f"\nLignes de comptage:")
        print(f"  Ligne A: x = {self.line_a}")
        print(f"  Ligne B: x = {self.line_b}")
        print(f"  Ligne C: x = {self.line_c}")
        print(f"  Ligne D: x = {self.line_d}")
        print(f"  Ligne E: x = {self.line_e}")

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

        # Flag pour indiquer si on a testé le GPU
        gpu_tested = False

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Afficher la progression
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    total_objects = len(self.object_crossings)
                    print(f"  Progression: {progress:.1f}% ({frame_count}/{total_frames}) - Objets détectés: {total_objects}")

                # Détection et suivi avec YOLO
                # Test de compatibilité GPU sur la première frame
                if not gpu_tested and self.device != 'cpu':
                    try:
                        results = self.model.track(
                            frame,
                            persist=True,
                            conf=self.min_confidence,
                            classes=[self.person_class_id, self.bike_class_id],  # Personnes ET vélos
                            device=self.device,
                            verbose=False
                        )
                        gpu_tested = True
                        print(f"✓ GPU fonctionnel pour YOLO")
                    except RuntimeError as e:
                        if "version_counter" in str(e) or "inference tensor" in str(e):
                            print(f"\n⚠️  Erreur DirectML détectée: {str(e)[:100]}")
                            print("⚠️  Bascule automatique sur CPU...")
                            print("   (Le modèle doit être rechargé)")

                            # IMPORTANT: Recharger le modèle pour éviter l'état corrompu
                            self.device = 'cpu'
                            del self.model  # Libérer l'ancien modèle
                            import gc
                            gc.collect()  # Forcer le garbage collection

                            print(f"   Rechargement du modèle YOLO pour CPU...")
                            self.model = YOLO(self.model_path)

                            results = self.model.track(
                                frame,
                                persist=True,
                                conf=self.min_confidence,
                                classes=[self.person_class_id, self.bike_class_id],  # Personnes ET vélos
                                device='cpu',
                                verbose=False
                            )
                            gpu_tested = True
                            print(f"✓ Bascule sur CPU réussie\n")
                        else:
                            raise
                else:
                    results = self.model.track(
                        frame,
                        persist=True,
                        conf=self.min_confidence,
                        classes=[self.person_class_id, self.bike_class_id],  # Personnes ET vélos
                        device=self.device,
                        verbose=False
                    )

                # Traiter les détections
                current_ids = self.process_detections(frame, results)
                annotated_frame = self.draw_annotations(frame, results, current_ids)

                # Détecter les objets qui ont quitté l'image
                disappeared_ids = previous_ids - current_ids
                for object_id in disappeared_ids:
                    if object_id not in self.finished_objects:
                        self.finalize_object(object_id)

                previous_ids = current_ids.copy()

                # Sauvegarder la frame si demandé
                if out is not None:
                    out.write(annotated_frame)

                # Afficher la vidéo
                if self.show_video:
                    cv2.imshow('Compteur Multi-Lignes', annotated_frame)

                    # Vérifier les touches seulement toutes les 10 frames pour plus de vitesse
                    if frame_count % 10 == 0:
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\nArrêt demandé par l'utilisateur")
                            break
                    else:
                        cv2.waitKey(1)  # Nécessaire pour rafraîchir l'affichage

        finally:
            # Finaliser tous les objets restants
            for object_id in self.object_crossings.keys():
                if object_id not in self.finished_objects:
                    self.finalize_object(object_id)

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
            set: IDs des objets détectés dans cette frame
        """
        current_ids = set()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)  # Récupérer les classes

            for box, track_id, cls in zip(boxes, track_ids, classes):
                x_center, y_center, w, h = box
                current_ids.add(track_id)

                # Enregistrer le type d'objet (person ou bike)
                if track_id not in self.object_types:
                    if cls == self.person_class_id:
                        self.object_types[track_id] = 'person'
                    elif cls == self.bike_class_id:
                        self.object_types[track_id] = 'bike'

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
                    self.check_line_crossing(track_id, prev_x, curr_x, self.line_d, 'D')
                    self.check_line_crossing(track_id, prev_x, curr_x, self.line_e, 'E')

        return current_ids

    def check_line_crossing(self, object_id, prev_x, curr_x, line_x, line_name):
        """Vérifie si un objet (vélo ou personne) a traversé une ligne"""
        # Passage de gauche à droite
        if prev_x < line_x - self.tolerance and curr_x >= line_x - self.tolerance:
            if line_name not in self.object_crossings[object_id]:
                self.object_crossings[object_id].add(line_name)
                obj_type = self.object_types.get(object_id, 'objet')
                obj_label = "Vélo" if obj_type == 'bike' else "Personne"
                print(f"  ✓ {obj_label} #{object_id} a traversé la ligne {line_name} (→)")

        # Passage de droite à gauche
        elif prev_x > line_x + self.tolerance and curr_x <= line_x + self.tolerance:
            if line_name not in self.object_crossings[object_id]:
                self.object_crossings[object_id].add(line_name)
                obj_type = self.object_types.get(object_id, 'objet')
                obj_label = "Vélo" if obj_type == 'bike' else "Personne"
                print(f"  ✓ {obj_label} #{object_id} a traversé la ligne {line_name} (←)")

    def finalize_object(self, object_id):
        """Finalise le comptage pour un objet (vélo ou personne) qui a quitté l'image"""
        if object_id in self.finished_objects:
            return

        self.finished_objects.add(object_id)
        crossings = self.object_crossings[object_id]

        if not crossings:
            return

        # Déterminer le type d'objet
        obj_type = self.object_types.get(object_id, None)
        if obj_type is None:
            return

        obj_label = "Vélo" if obj_type == 'bike' else "Personne"

        # Sélectionner les bons compteurs selon le type
        if obj_type == 'bike':
            combinations_count = self.bikes_combinations_count
        else:  # person
            combinations_count = self.people_combinations_count

        # Incrémenter les compteurs individuels par ligne
        if 'A' in crossings:
            if obj_type == 'bike':
                self.count_bikes_a += 1
            else:
                self.count_people_a += 1

        if 'B' in crossings:
            if obj_type == 'bike':
                self.count_bikes_b += 1
            else:
                self.count_people_b += 1

        if 'C' in crossings:
            if obj_type == 'bike':
                self.count_bikes_c += 1
            else:
                self.count_people_c += 1

        if 'D' in crossings:
            if obj_type == 'bike':
                self.count_bikes_d += 1
            else:
                self.count_people_d += 1

        if 'E' in crossings:
            if obj_type == 'bike':
                self.count_bikes_e += 1
            else:
                self.count_people_e += 1

        # Déterminer la combinaison (clé du dictionnaire = lettres triées)
        combination_key = ''.join(sorted(crossings))

        if combination_key in combinations_count:
            combinations_count[combination_key] += 1

            # Message selon le nombre de lignes traversées
            if len(crossings) == 5:
                print(f"  📊 {obj_label} #{object_id} - Trajet: {combination_key} (COMPLET)")
            elif len(crossings) >= 3:
                print(f"  📊 {obj_label} #{object_id} - Trajet: {combination_key}")
            else:
                print(f"  📊 {obj_label} #{object_id} - Trajet: {combination_key}")

    def draw_annotations(self, frame, results, current_ids):
        """Dessine les annotations sur la frame"""
        annotated_frame = frame.copy()
        height = frame.shape[0]

        # Dessiner les lignes de comptage avec couleurs différentes
        cv2.line(annotated_frame, (self.line_a, 0), (self.line_a, height), (255, 0, 0), 3)     # Bleu - A
        cv2.line(annotated_frame, (self.line_b, 0), (self.line_b, height), (0, 255, 0), 3)     # Vert - B
        cv2.line(annotated_frame, (self.line_c, 0), (self.line_c, height), (0, 0, 255), 3)     # Rouge - C
        cv2.line(annotated_frame, (self.line_d, 0), (self.line_d, height), (255, 255, 0), 3)   # Jaune - D
        cv2.line(annotated_frame, (self.line_e, 0), (self.line_e, height), (255, 0, 255), 3)   # Magenta - E

        # Labels des lignes
        cv2.putText(annotated_frame, "A", (self.line_a - 15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.putText(annotated_frame, "B", (self.line_b - 15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(annotated_frame, "C", (self.line_c - 15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(annotated_frame, "D", (self.line_d - 15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        cv2.putText(annotated_frame, "E", (self.line_e - 15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

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

                # Déterminer le type d'objet
                obj_type = self.object_types.get(track_id, 'unknown')
                obj_prefix = "V" if obj_type == 'bike' else "P"  # V=Vélo, P=Personne

                # Couleur selon les lignes traversées
                crossings = self.object_crossings[track_id]
                if len(crossings) == 5:
                    color = (255, 0, 255)    # Magenta - Toutes les lignes (5)
                elif len(crossings) == 4:
                    color = (255, 128, 255)  # Rose - Quatre lignes
                elif len(crossings) == 3:
                    color = (0, 255, 255)    # Cyan - Trois lignes
                elif len(crossings) == 2:
                    color = (255, 255, 0)    # Jaune - Deux lignes
                elif len(crossings) == 1:
                    color = (255, 165, 0)    # Orange - Une ligne
                else:
                    color = (128, 128, 128)  # Gris - Aucune ligne

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # Label avec type, ID et lignes traversées
                lines_str = ''.join(sorted(crossings)) if crossings else '-'
                label = f"{obj_prefix}#{track_id} [{lines_str}]"
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
        # Calculer les comptages en temps réel (incluant les objets actifs)
        realtime_bikes = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
        realtime_people = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}

        for obj_id, crossings in self.object_crossings.items():
            obj_type = self.object_types.get(obj_id)
            if obj_type == 'bike':
                for line in crossings:
                    realtime_bikes[line] += 1
            elif obj_type == 'person':
                for line in crossings:
                    realtime_people[line] += 1

        # Compter les objets actifs (non finalisés)
        active_bikes = sum(1 for obj_id in self.object_crossings.keys()
                          if obj_id not in self.finished_objects
                          and self.object_types.get(obj_id) == 'bike')
        active_people = sum(1 for obj_id in self.object_crossings.keys()
                           if obj_id not in self.finished_objects
                           and self.object_types.get(obj_id) == 'person')

        # Fond semi-transparent plus grand pour les deux catégories
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 50), (600, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        y_offset = 70
        line_height = 25

        # Titre
        cv2.putText(frame, "COMPTAGE TEMPS REEL - 5 LIGNES", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height + 5

        # VÉLOS
        cv2.putText(frame, "VELOS:", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y_offset += line_height

        cv2.putText(frame, f"A:{realtime_bikes['A']} B:{realtime_bikes['B']} C:{realtime_bikes['C']} D:{realtime_bikes['D']} E:{realtime_bikes['E']}",
                   (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y_offset += line_height - 5

        total_bikes_finished = sum(self.bikes_combinations_count.values())
        cv2.putText(frame, f"Finis: {total_bikes_finished} | Actifs: {active_bikes} | Total: {total_bikes_finished + active_bikes}", (30, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        y_offset += line_height + 5

        # PERSONNES
        cv2.putText(frame, "PERSONNES:", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)
        y_offset += line_height

        cv2.putText(frame, f"A:{realtime_people['A']} B:{realtime_people['B']} C:{realtime_people['C']} D:{realtime_people['D']} E:{realtime_people['E']}",
                   (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y_offset += line_height - 5

        total_people_finished = sum(self.people_combinations_count.values())
        cv2.putText(frame, f"Finis: {total_people_finished} | Actifs: {active_people} | Total: {total_people_finished + active_people}", (30, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    def print_results(self):
        """Affiche les résultats finaux"""
        print("\n" + "="*80)
        print("RÉSULTATS DU COMPTAGE - PASSAGES PAR LIGNE")
        print("="*80)

        # VÉLOS - Simple et clair
        print(f"\n🚴 VÉLOS:")
        print(f"  Ligne A: {self.count_bikes_a:4d} vélos")
        print(f"  Ligne B: {self.count_bikes_b:4d} vélos")
        print(f"  Ligne C: {self.count_bikes_c:4d} vélos")
        print(f"  Ligne D: {self.count_bikes_d:4d} vélos")
        print(f"  Ligne E: {self.count_bikes_e:4d} vélos")

        # Calculer les combinaisons spécifiques pour vélos
        bikes_cd = self._count_combination_with_lines(['C', 'D'], self.bikes_combinations_count)

        print(f"\n  Combinaisons:")
        print(f"    C et D (les deux): {bikes_cd:4d} vélos")

        total_bikes = sum(self.bikes_combinations_count.values())
        print(f"\n  TOTAL VÉLOS UNIQUES: {total_bikes}")

        # PERSONNES - Simple et clair
        print(f"\n👤 PERSONNES:")
        print(f"  Ligne A: {self.count_people_a:4d} personnes")
        print(f"  Ligne B: {self.count_people_b:4d} personnes")
        print(f"  Ligne C: {self.count_people_c:4d} personnes")
        print(f"  Ligne D: {self.count_people_d:4d} personnes")
        print(f"  Ligne E: {self.count_people_e:4d} personnes")

        # Calculer les combinaisons spécifiques pour personnes
        people_cd = self._count_combination_with_lines(['C', 'D'], self.people_combinations_count)

        print(f"\n  Combinaisons:")
        print(f"    C et D (les deux): {people_cd:4d} personnes")

        total_people = sum(self.people_combinations_count.values())
        print(f"\n  TOTAL PERSONNES UNIQUES: {total_people}")

        # Résumé global
        print(f"\n📊 RÉSUMÉ:")
        print(f"  Total vélos:     {total_bikes}")
        print(f"  Total personnes: {total_people}")
        print(f"  TOTAL GÉNÉRAL:   {total_bikes + total_people}")

        print("="*80 + "\n")

        # Note explicative
        print("NOTE: Les compteurs par ligne comptent chaque passage.")
        print("      Un vélo qui traverse A, C et E sera compté 1 fois sur A, 1 fois sur C et 1 fois sur E.")
        print("      'C et D (les deux)' compte les objets qui ont traversé C ET D (peu importe les autres lignes).")
        print("      Le TOTAL UNIQUE compte chaque vélo une seule fois (pas de doublons).\n")

    def _count_combination_with_lines(self, required_lines, combinations_dict):
        """
        Compte combien d'objets ont traversé TOUTES les lignes spécifiées
        (peu importe s'ils ont aussi traversé d'autres lignes)

        Args:
            required_lines: Liste des lignes requises (ex: ['C', 'D'])
            combinations_dict: Dictionnaire des combinaisons à analyser

        Returns:
            Nombre d'objets ayant traversé toutes les lignes requises
        """
        required_set = set(required_lines)
        count = 0

        for combination_key, combination_count in combinations_dict.items():
            # Convertir la clé en set (ex: 'CD' -> {'C', 'D'}, 'ABCDE' -> {'A','B','C','D','E'})
            combination_set = set(combination_key)

            # Si toutes les lignes requises sont présentes dans cette combinaison
            if required_set.issubset(combination_set):
                count += combination_count

        return count

    def save_results(self):
        """Sauvegarde les résultats dans un fichier JSON"""
        results = {
            "video_path": self.video_path,
            "model_used": self.model_path,
            "lines": {
                "A": self.line_a,
                "B": self.line_b,
                "C": self.line_c,
                "D": self.line_d,
                "E": self.line_e
            },
            "bikes": {
                "counts_per_line": {
                    "A": self.count_bikes_a,
                    "B": self.count_bikes_b,
                    "C": self.count_bikes_c,
                    "D": self.count_bikes_d,
                    "E": self.count_bikes_e
                },
                "trajectory_analysis": self.bikes_combinations_count,
                "total_unique": sum(self.bikes_combinations_count.values())
            },
            "people": {
                "counts_per_line": {
                    "A": self.count_people_a,
                    "B": self.count_people_b,
                    "C": self.count_people_c,
                    "D": self.count_people_d,
                    "E": self.count_people_e
                },
                "trajectory_analysis": self.people_combinations_count,
                "total_unique": sum(self.people_combinations_count.values())
            },
            "total_general": sum(self.bikes_combinations_count.values()) + sum(self.people_combinations_count.values()),
            "min_confidence": self.min_confidence
        }

        output_file = "bike_count_multiline_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Résultats sauvegardés dans: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compteur multi-lignes de vélos - Analyse des trajectoires A, B, C, D, E"
    )
    parser.add_argument("video", help="Chemin vers la vidéo MP4")
    parser.add_argument("-m", "--model", default="yolov8n.pt",
                       help="Modèle YOLO (défaut: yolov8n.pt)")
    parser.add_argument("-a", "--line-a", type=int, default=None,
                       help="Position X de la ligne A (défaut: 1/6 largeur)")
    parser.add_argument("-b", "--line-b", type=int, default=None,
                       help="Position X de la ligne B (défaut: 2/6 largeur)")
    parser.add_argument("-c", "--line-c", type=int, default=None,
                       help="Position X de la ligne C (défaut: 3/6 largeur)")
    parser.add_argument("-d", "--line-d", type=int, default=None,
                       help="Position X de la ligne D (défaut: 4/6 largeur)")
    parser.add_argument("-e", "--line-e", type=int, default=None,
                       help="Position X de la ligne E (défaut: 5/6 largeur)")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Confiance minimale (0.0-1.0, défaut: 0.5)")
    parser.add_argument("-o", "--output", default=None,
                       help="Sauvegarder la vidéo annotée")
    parser.add_argument("--no-display", action="store_true",
                       help="Ne pas afficher la vidéo")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Forcer l'utilisation du CPU (désactiver GPU)")

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
        line_d=args.line_d,
        line_e=args.line_e,
        min_confidence=args.confidence,
        show_video=not args.no_display,
        output_path=args.output,
        use_gpu=not args.no_gpu
    )

    counter.process_video()


if __name__ == "__main__":
    main()
