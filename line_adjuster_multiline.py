"""
Outil interactif pour ajuster les 5 lignes de comptage (A, B, C, D, E)
Permet de visualiser et choisir les positions optimales avant le traitement
"""

import cv2
import argparse


class MultiLineAdjuster:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None
        self.frame = None
        self.width = 0
        self.height = 0
        self.frame_number = 0

        # Lignes de comptage
        self.line_a = 0
        self.line_b = 0
        self.line_c = 0
        self.line_d = 0
        self.line_e = 0

        # Ligne actuellement sélectionnée
        self.selected_line = 'C'  # Par défaut, ligne C (milieu)

    def run(self):
        """Lance l'interface d'ajustement"""
        # Ouvrir la vidéo
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            print(f"Erreur: Impossible d'ouvrir la vidéo '{self.video_path}'")
            return

        # Lire la première frame
        ret, self.frame = self.cap.read()
        if not ret:
            print("Erreur: Impossible de lire la vidéo")
            self.cap.release()
            return

        self.height, self.width = self.frame.shape[:2]

        # Initialiser les positions par défaut (1/6, 2/6, 3/6, 4/6, 5/6)
        self.line_a = self.width // 6
        self.line_b = (2 * self.width) // 6
        self.line_c = (3 * self.width) // 6
        self.line_d = (4 * self.width) // 6
        self.line_e = (5 * self.width) // 6

        self.print_instructions()
        self.main_loop()

        # Nettoyer
        self.cap.release()
        cv2.destroyAllWindows()

        # Afficher la commande finale
        self.print_final_command()

    def print_instructions(self):
        """Affiche les instructions"""
        print("\n" + "="*80)
        print("AJUSTEUR MULTI-LIGNES (A, B, C, D, E)")
        print("="*80)
        print(f"Résolution vidéo: {self.width}x{self.height}")
        print(f"\nPositions initiales:")
        print(f"  Ligne A (bleue):    x = {self.line_a}")
        print(f"  Ligne B (verte):    x = {self.line_b}")
        print(f"  Ligne C (rouge):    x = {self.line_c}")
        print(f"  Ligne D (jaune):    x = {self.line_d}")
        print(f"  Ligne E (magenta):  x = {self.line_e}")
        print(f"\n🎮 CONTRÔLES:")
        print(f"  [A] [B] [C] [D] [E] : Sélectionner la ligne à ajuster")
        print(f"  Clic gauche         : Déplacer la ligne sélectionnée")
        print(f"  ← →                 : Déplacer de 10 pixels")
        print(f"  Shift + ← →         : Déplacer de 1 pixel")
        print(f"  ESPACE              : Frame suivante")
        print(f"  R                   : Revenir au début")
        print(f"  X                   : Positions par défaut")
        print(f"  ENTER ou Q          : Confirmer et quitter")
        print("="*80 + "\n")

    def main_loop(self):
        """Boucle principale d'affichage"""
        window_name = 'Ajusteur Multi-Lignes - [A] [B] [C] [D] [E] pour sélectionner'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        while True:
            # Créer la frame d'affichage
            display_frame = self.draw_frame()

            # Afficher
            cv2.imshow(window_name, display_frame)

            # Gérer les touches
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 13:  # Q ou ENTER
                break
            elif key == ord('a') or key == ord('A'):
                self.selected_line = 'A'
                print(f"→ Ligne A sélectionnée (x = {self.line_a})")
            elif key == ord('b') or key == ord('B'):
                self.selected_line = 'B'
                print(f"→ Ligne B sélectionnée (x = {self.line_b})")
            elif key == ord('c') or key == ord('C'):
                self.selected_line = 'C'
                print(f"→ Ligne C sélectionnée (x = {self.line_c})")
            elif key == ord('d') or key == ord('D'):
                self.selected_line = 'D'
                print(f"→ Ligne D sélectionnée (x = {self.line_d})")
            elif key == ord('e') or key == ord('E'):
                self.selected_line = 'E'
                print(f"→ Ligne E sélectionnée (x = {self.line_e})")
            elif key == ord(' '):  # ESPACE
                self.next_frame()
            elif key == ord('r') or key == ord('R'):
                self.reset_to_beginning()
            elif key == ord('x') or key == ord('X'):
                self.reset_to_defaults()
            elif key == 81 or key == 2:  # Flèche gauche
                self.move_selected_line(-10)
            elif key == 83 or key == 3:  # Flèche droite
                self.move_selected_line(10)

    def mouse_callback(self, event, x, y, flags, param):
        """Callback pour les événements souris"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Déplacer la ligne sélectionnée
            x = max(0, min(self.width, x))

            if self.selected_line == 'A':
                self.line_a = x
            elif self.selected_line == 'B':
                self.line_b = x
            elif self.selected_line == 'C':
                self.line_c = x
            elif self.selected_line == 'D':
                self.line_d = x
            elif self.selected_line == 'E':
                self.line_e = x

            print(f"Ligne {self.selected_line} déplacée à x = {x}")

    def move_selected_line(self, delta):
        """Déplace la ligne sélectionnée"""
        if self.selected_line == 'A':
            self.line_a = max(0, min(self.width, self.line_a + delta))
            print(f"Ligne A: x = {self.line_a}")
        elif self.selected_line == 'B':
            self.line_b = max(0, min(self.width, self.line_b + delta))
            print(f"Ligne B: x = {self.line_b}")
        elif self.selected_line == 'C':
            self.line_c = max(0, min(self.width, self.line_c + delta))
            print(f"Ligne C: x = {self.line_c}")
        elif self.selected_line == 'D':
            self.line_d = max(0, min(self.width, self.line_d + delta))
            print(f"Ligne D: x = {self.line_d}")
        elif self.selected_line == 'E':
            self.line_e = max(0, min(self.width, self.line_e + delta))
            print(f"Ligne E: x = {self.line_e}")

    def next_frame(self):
        """Passe à la frame suivante"""
        ret, self.frame = self.cap.read()
        if not ret:
            print("Fin de la vidéo, retour au début")
            self.reset_to_beginning()
        else:
            self.frame_number += 1

    def reset_to_beginning(self):
        """Retourne au début de la vidéo"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, self.frame = self.cap.read()
        self.frame_number = 0
        print("Retour au début de la vidéo")

    def reset_to_defaults(self):
        """Réinitialise aux positions par défaut"""
        self.line_a = self.width // 6
        self.line_b = (2 * self.width) // 6
        self.line_c = (3 * self.width) // 6
        self.line_d = (4 * self.width) // 6
        self.line_e = (5 * self.width) // 6
        print(f"Positions par défaut: A={self.line_a}, B={self.line_b}, C={self.line_c}, D={self.line_d}, E={self.line_e}")

    def draw_frame(self):
        """Dessine la frame avec les lignes et annotations"""
        display_frame = self.frame.copy()

        # Dessiner les lignes avec indication de sélection
        # Ligne A (bleue)
        thickness_a = 5 if self.selected_line == 'A' else 3
        cv2.line(display_frame, (self.line_a, 0), (self.line_a, self.height),
                (255, 0, 0), thickness_a)
        cv2.putText(display_frame, "A", (self.line_a - 15, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        # Ligne B (verte)
        thickness_b = 5 if self.selected_line == 'B' else 3
        cv2.line(display_frame, (self.line_b, 0), (self.line_b, self.height),
                (0, 255, 0), thickness_b)
        cv2.putText(display_frame, "B", (self.line_b - 15, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Ligne C (rouge)
        thickness_c = 5 if self.selected_line == 'C' else 3
        cv2.line(display_frame, (self.line_c, 0), (self.line_c, self.height),
                (0, 0, 255), thickness_c)
        cv2.putText(display_frame, "C", (self.line_c - 15, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Ligne D (jaune)
        thickness_d = 5 if self.selected_line == 'D' else 3
        cv2.line(display_frame, (self.line_d, 0), (self.line_d, self.height),
                (255, 255, 0), thickness_d)
        cv2.putText(display_frame, "D", (self.line_d - 15, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

        # Ligne E (magenta)
        thickness_e = 5 if self.selected_line == 'E' else 3
        cv2.line(display_frame, (self.line_e, 0), (self.line_e, self.height),
                (255, 0, 255), thickness_e)
        cv2.putText(display_frame, "E", (self.line_e - 15, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)

        # Zone d'information (plus grande pour 5 lignes)
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, self.height - 240), (500, self.height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

        y = self.height - 210
        line_height = 30

        # Positions des lignes
        cv2.putText(display_frame, f"Ligne A: {self.line_a} px", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        y += line_height

        cv2.putText(display_frame, f"Ligne B: {self.line_b} px", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += line_height

        cv2.putText(display_frame, f"Ligne C: {self.line_c} px", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y += line_height

        cv2.putText(display_frame, f"Ligne D: {self.line_d} px", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y += line_height

        cv2.putText(display_frame, f"Ligne E: {self.line_e} px", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        y += line_height

        # Ligne sélectionnée
        cv2.putText(display_frame, f"Selectionnee: {self.selected_line}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += line_height

        # Frame
        cv2.putText(display_frame, f"Frame: {self.frame_number}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return display_frame

    def print_final_command(self):
        """Affiche la commande finale à utiliser"""
        print("\n" + "="*80)
        print("POSITIONS FINALES DES LIGNES:")
        print(f"  Ligne A (bleue):    x = {self.line_a}")
        print(f"  Ligne B (verte):    x = {self.line_b}")
        print(f"  Ligne C (rouge):    x = {self.line_c}")
        print(f"  Ligne D (jaune):    x = {self.line_d}")
        print(f"  Ligne E (magenta):  x = {self.line_e}")
        print("\n📋 COMMANDE À UTILISER:")
        print(f"python bike_counter_multiline.py {self.video_path} -a {self.line_a} -b {self.line_b} -c {self.line_c} -d {self.line_d} -e {self.line_e}")
        print("\n💡 Avec sauvegarde vidéo:")
        print(f"python bike_counter_multiline.py {self.video_path} -a {self.line_a} -b {self.line_b} -c {self.line_c} -d {self.line_d} -e {self.line_e} -o output.mp4")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Outil interactif pour ajuster les 5 lignes de comptage (A, B, C, D, E)"
    )
    parser.add_argument("video", help="Chemin vers la vidéo MP4")

    args = parser.parse_args()

    adjuster = MultiLineAdjuster(args.video)
    adjuster.run()


if __name__ == "__main__":
    main()
