"""
Outil interactif pour ajuster la position de la ligne de comptage
Permet de visualiser et choisir la position optimale avant le traitement
"""

import cv2
import argparse


def adjust_line(video_path):
    """
    Ouvre la vidéo et permet d'ajuster la ligne de comptage avec la souris

    Args:
        video_path: Chemin vers la vidéo
    """
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la vidéo '{video_path}'")
        return

    # Lire la première frame
    ret, frame = cap.read()
    if not ret:
        print("Erreur: Impossible de lire la vidéo")
        cap.release()
        return

    height, width = frame.shape[:2]
    line_x = width // 2  # Position initiale au milieu

    print("\n" + "="*60)
    print("AJUSTEUR DE LIGNE DE COMPTAGE")
    print("="*60)
    print(f"Résolution vidéo: {width}x{height}")
    print(f"\nContrôles:")
    print("  - Clic gauche sur l'image: déplacer la ligne")
    print("  - Flèche gauche/droite: déplacer la ligne de 10 pixels")
    print("  - ESPACE: passer à la frame suivante")
    print("  - 'r': revenir au début de la vidéo")
    print("  - 'q' ou ENTER: confirmer et quitter")
    print("="*60)

    def mouse_callback(event, x, y, flags, param):
        """Callback pour les événements souris"""
        nonlocal line_x
        if event == cv2.EVENT_LBUTTONDOWN:
            line_x = x
            print(f"Ligne déplacée à x = {line_x}")

    # Créer la fenêtre et attacher le callback souris
    window_name = 'Ajusteur de ligne - Cliquez pour positionner'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    frame_number = 0

    while True:
        # Créer une copie de la frame pour l'annotation
        display_frame = frame.copy()

        # Dessiner la ligne de comptage
        cv2.line(display_frame, (line_x, 0), (line_x, height), (0, 255, 0), 3)

        # Dessiner la zone de tolérance
        tolerance = 20
        cv2.line(display_frame, (line_x - tolerance, 0),
                (line_x - tolerance, height), (0, 255, 0), 1)
        cv2.line(display_frame, (line_x + tolerance, 0),
                (line_x + tolerance, height), (0, 255, 0), 1)

        # Afficher les informations
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

        cv2.putText(display_frame, f"Position ligne: x = {line_x}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Frame: {frame_number}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Afficher
        cv2.imshow(window_name, display_frame)

        # Gérer les touches
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 13:  # 'q' ou ENTER
            break
        elif key == ord(' '):  # ESPACE - frame suivante
            ret, frame = cap.read()
            if not ret:
                print("Fin de la vidéo, retour au début")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                frame_number = 0
            else:
                frame_number += 1
        elif key == ord('r'):  # 'r' - retour au début
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            frame_number = 0
            print("Retour au début de la vidéo")
        elif key == 81 or key == 2:  # Flèche gauche
            line_x = max(0, line_x - 10)
            print(f"Ligne déplacée à x = {line_x}")
        elif key == 83 or key == 3:  # Flèche droite
            line_x = min(width, line_x + 10)
            print(f"Ligne déplacée à x = {line_x}")

    # Nettoyer
    cap.release()
    cv2.destroyAllWindows()

    # Afficher la commande à utiliser
    print("\n" + "="*60)
    print(f"Position de ligne sélectionnée: x = {line_x}")
    print("\nUtilisez cette commande pour le comptage:")
    print(f"python bike_counter.py {video_path} -l {line_x}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Outil interactif pour ajuster la position de la ligne de comptage"
    )
    parser.add_argument("video", help="Chemin vers la vidéo MP4")

    args = parser.parse_args()

    adjust_line(args.video)


if __name__ == "__main__":
    main()
