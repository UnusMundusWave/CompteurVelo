# Compteur de Vélos pour Parade 🚴

Système de détection et comptage automatique de vélos dans une vidéo de parade, utilisant YOLO pour la détection et le suivi d'objets.

## Fonctionnalités

- ✅ Détection automatique des vélos avec YOLO
- ✅ Suivi des vélos individuels entre les frames
- ✅ Comptage précis lors du passage d'une ligne verticale configurable
- ✅ Distinction des directions (gauche→droite et droite→gauche)
- ✅ Visualisation en temps réel avec annotations
- ✅ Sauvegarde de la vidéo annotée
- ✅ Export des résultats en JSON
- ✅ Support GPU AMD via DirectML (Windows 11)

## Prérequis

- Windows 11
- Python 3.8 ou supérieur
- GPU AMD (optionnel mais recommandé)
- Vidéo MP4 de la parade

## Installation

### 1. Créer un environnement virtuel Python

```bash
# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement virtuel
# Sur Windows:
venv\Scripts\activate

# Vous devriez voir (venv) au début de votre ligne de commande
```

### 2. Installer les dépendances

```bash
# Installer les packages de base
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configuration GPU AMD (optionnel mais recommandé)

Pour utiliser votre GPU AMD sur Windows 11:

```bash
# Installer le support DirectML pour PyTorch
pip install torch-directml
```

### 4. Télécharger le modèle YOLO

Le modèle YOLOv8 nano sera téléchargé automatiquement lors de la première exécution.
Pour utiliser un modèle plus précis (mais plus lent):

```bash
# Télécharger manuellement un modèle plus grand (optionnel)
# yolov8s.pt (small) - plus précis
# yolov8m.pt (medium) - encore plus précis
# yolov8l.pt (large) - très précis mais lent
```

## Utilisation

### Utilisation basique

```bash
python bike_counter.py votre_video.mp4
```

### Options avancées

```bash
# Utiliser un modèle YOLO différent
python bike_counter.py votre_video.mp4 -m yolov8s.pt

# Définir la position de la ligne de comptage (en pixels depuis la gauche)
python bike_counter.py votre_video.mp4 -l 640

# Ajuster le seuil de confiance (0.0 à 1.0)
python bike_counter.py votre_video.mp4 -c 0.6

# Sauvegarder la vidéo annotée
python bike_counter.py votre_video.mp4 -o resultat.mp4

# Traiter sans afficher la vidéo (plus rapide)
python bike_counter.py votre_video.mp4 --no-display -o resultat.mp4

# Exemple complet
python bike_counter.py parade.mp4 -m yolov8s.pt -l 800 -c 0.6 -o parade_annote.mp4
```

### Paramètres

| Paramètre | Description | Défaut |
|-----------|-------------|--------|
| `video` | Chemin vers la vidéo MP4 (requis) | - |
| `-m, --model` | Modèle YOLO (yolov8n/s/m/l/x.pt) | yolov8n.pt |
| `-l, --line` | Position X de la ligne de comptage | Milieu de l'image |
| `-c, --confidence` | Confiance minimale (0.0-1.0) | 0.5 |
| `-o, --output` | Sauvegarder la vidéo annotée | Aucun |
| `--no-display` | Ne pas afficher la vidéo | False |

## Résultats

### Pendant l'exécution

- La vidéo s'affiche avec:
  - **Ligne verte verticale**: ligne de comptage
  - **Boîtes bleues**: vélos détectés
  - **Boîtes vertes**: vélos déjà comptés
  - **Trajectoires**: chemin suivi par chaque vélo
  - **Compteurs**: affichage en temps réel

### Après l'exécution

1. **Console**: Résumé du comptage
2. **bike_count_results.json**: Résultats détaillés
3. **Vidéo annotée**: Si `-o` spécifié

Exemple de `bike_count_results.json`:
```json
{
  "video_path": "parade.mp4",
  "total_count": 42,
  "left_to_right": 38,
  "right_to_left": 4,
  "counting_line_x": 640,
  "min_confidence": 0.5,
  "model_used": "yolov8n.pt"
}
```

## Conseils pour de meilleurs résultats

### Position de la ligne de comptage
- Placez la ligne où les vélos sont bien visibles
- Évitez les zones avec beaucoup d'occlusions
- Pour trouver la bonne position, essayez d'abord avec la valeur par défaut

### Modèle YOLO
- **yolov8n.pt**: Rapide, bon pour tests (recommandé pour débuter)
- **yolov8s.pt**: Bon compromis vitesse/précision
- **yolov8m.pt**: Plus précis, plus lent
- **yolov8l.pt**: Très précis, requiert plus de ressources

### Confiance
- **0.3-0.4**: Détecte plus de vélos mais plus de faux positifs
- **0.5**: Valeur équilibrée (recommandé)
- **0.6-0.7**: Plus strict, moins de faux positifs

### Performance GPU AMD
- Assurez-vous d'avoir installé `torch-directml`
- Les derniers drivers AMD Adrenalin sont recommandés
- Le modèle nano (yolov8n) fonctionne bien sur la plupart des GPU

## Résolution de problèmes

### Le GPU AMD n'est pas détecté
```bash
# Vérifier l'installation DirectML
pip install torch-directml --upgrade

# Vérifier que PyTorch voit DirectML
python -c "import torch_directml; print(torch_directml.is_available())"
```

### Erreur "Impossible d'ouvrir la vidéo"
- Vérifiez le chemin du fichier
- Assurez-vous que le format est MP4
- Essayez de réencoder la vidéo avec VLC ou FFmpeg

### Détections manquées
- Réduisez le seuil de confiance: `-c 0.4`
- Utilisez un modèle plus grand: `-m yolov8s.pt`
- Vérifiez l'éclairage et la qualité de la vidéo

### Comptage en double
- Vérifiez que la ligne n'est pas trop près du bord
- La tolérance de 20 pixels autour de la ligne évite normalement ce problème

## Structure du projet

```
CompteurVelo/
├── bike_counter.py          # Script principal
├── requirements.txt         # Dépendances Python
├── README.md               # Cette documentation
├── config_example.json     # Exemple de configuration
└── bike_count_results.json # Résultats (généré)
```

## Exemples d'utilisation

### Scénario 1: Test rapide
```bash
python bike_counter.py ma_video.mp4
# Appuyez sur 'q' pour quitter
```

### Scénario 2: Production avec sauvegarde
```bash
python bike_counter.py parade.mp4 -m yolov8s.pt -c 0.6 -o resultat.mp4
```

### Scénario 3: Traitement batch sans affichage
```bash
python bike_counter.py video1.mp4 --no-display -o output1.mp4
python bike_counter.py video2.mp4 --no-display -o output2.mp4
```

## Licence

MIT License

## Support

Pour toute question ou problème, créez une issue sur le dépôt GitHub.

## Crédits

- YOLO: [Ultralytics](https://github.com/ultralytics/ultralytics)
- PyTorch DirectML: [Microsoft](https://github.com/microsoft/DirectML)
