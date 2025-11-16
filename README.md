# Compteur de Vélos pour Parade 🚴

Système de détection et comptage automatique de vélos dans une vidéo de parade, utilisant YOLO pour la détection et le suivi d'objets.

## Fonctionnalités

### Mode Simple (1 ligne)
- ✅ Détection automatique des vélos avec YOLO
- ✅ Suivi des vélos individuels entre les frames
- ✅ Comptage précis lors du passage d'une ligne verticale configurable
- ✅ Distinction des directions (gauche→droite et droite→gauche)
- ✅ Visualisation en temps réel avec annotations
- ✅ Sauvegarde de la vidéo annotée
- ✅ Export des résultats en JSON
- ✅ Support GPU AMD via DirectML (Windows 11)

### Mode Multi-Lignes (3 lignes A, B, C) ⭐ NOUVEAU
- ✅ Comptage sur 3 lignes verticales simultanément
- ✅ Analyse des trajectoires complètes
- ✅ Comptage par combinaisons (A seul, B seul, C seul, A+B, A+C, B+C, A+B+C)
- ✅ Identification des vélos qui font le parcours complet
- ✅ Statistiques détaillées des trajets

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
├── bike_counter.py                    # Compteur simple (1 ligne)
├── bike_counter_multiline.py          # Compteur multi-lignes (A, B, C) ⭐
├── line_adjuster.py                   # Ajusteur ligne simple
├── line_adjuster_multiline.py         # Ajusteur multi-lignes ⭐
├── test_gpu.py                        # Test configuration GPU AMD
├── quick_start.bat                    # Installation automatique Windows
├── requirements.txt                   # Dépendances Python
├── README.md                          # Documentation complète
├── config_example.json                # Exemple de configuration
├── .gitignore                         # Fichiers ignorés par Git
├── bike_count_results.json            # Résultats mode simple (généré)
└── bike_count_multiline_results.json  # Résultats multi-lignes (généré)
```

## Exemples d'utilisation - Mode Simple

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

---

## 🎯 MODE MULTI-LIGNES (A, B, C) - Analyse Avancée des Trajectoires

### Vue d'ensemble

Le mode multi-lignes permet d'analyser les trajectoires complètes des vélos en plaçant 3 lignes de comptage verticales (A, B, C). Le système détecte automatiquement quelles lignes chaque vélo traverse et génère des statistiques détaillées sur les parcours.

### Cas d'usage

- **Analyser les flux**: Identifier quels vélos font le parcours complet vs partiel
- **Optimiser la parade**: Comprendre où les vélos quittent le parcours
- **Statistiques avancées**: Voir les combinaisons de trajets les plus fréquentes
- **Validation**: Vérifier que les vélos passent bien par tous les points de contrôle

### Utilisation du mode multi-lignes

#### Étape 1: Ajuster les 3 lignes interactivement

```bash
python line_adjuster_multiline.py votre_video.mp4
```

**Contrôles de l'ajusteur:**
- `A` `B` `C` : Sélectionner la ligne à ajuster
- Clic gauche : Déplacer la ligne sélectionnée
- Flèches ← → : Déplacer de 10 pixels
- ESPACE : Voir la frame suivante
- R : Retour au début de la vidéo
- D : Réinitialiser aux positions par défaut (1/4, 1/2, 3/4)
- ENTER : Confirmer et obtenir la commande

**Visualisation:**
- Ligne **bleue** = Ligne A
- Ligne **verte** = Ligne B
- Ligne **rouge** = Ligne C
- La ligne actuellement sélectionnée est plus épaisse

#### Étape 2: Lancer le comptage multi-lignes

```bash
# Utilisation basique (positions par défaut)
python bike_counter_multiline.py votre_video.mp4

# Avec positions personnalisées
python bike_counter_multiline.py votre_video.mp4 -a 400 -b 800 -c 1200

# Avec sauvegarde vidéo annotée
python bike_counter_multiline.py votre_video.mp4 -a 400 -b 800 -c 1200 -o resultat.mp4

# Production (modèle précis, sans affichage)
python bike_counter_multiline.py parade.mp4 -m yolov8s.pt --no-display -o output.mp4
```

### Options disponibles

| Option | Description | Défaut |
|--------|-------------|--------|
| `video` | Vidéo MP4 à analyser (requis) | - |
| `-m, --model` | Modèle YOLO | yolov8n.pt |
| `-a, --line-a` | Position X de la ligne A | 1/4 largeur |
| `-b, --line-b` | Position X de la ligne B | 1/2 largeur |
| `-c, --line-c` | Position X de la ligne C | 3/4 largeur |
| `--confidence` | Seuil de confiance (0.0-1.0) | 0.5 |
| `-o, --output` | Vidéo annotée de sortie | Aucun |
| `--no-display` | Traiter sans affichage | False |

### Interprétation des résultats

Le système génère des statistiques sur 7 types de trajets :

#### Trajets simples (1 ligne)
- **A seulement**: Vélos détectés uniquement sur la ligne A
- **B seulement**: Vélos détectés uniquement sur la ligne B
- **C seulement**: Vélos détectés uniquement sur la ligne C

#### Trajets doubles (2 lignes)
- **A et B**: Vélos passant par A puis B (ou inversement)
- **A et C**: Vélos passant par A puis C (généralement ont "sauté" B)
- **B et C**: Vélos passant par B puis C

#### Trajet complet (3 lignes)
- **A, B et C**: Vélos ayant traversé les 3 lignes ✅ **PARCOURS COMPLET**

### Exemple de résultats

**Console:**
```
==================================================================
RÉSULTATS DU COMPTAGE MULTI-LIGNES
==================================================================

📊 PASSAGES PAR LIGNE:
  Ligne A: 45 vélos
  Ligne B: 42 vélos
  Ligne C: 38 vélos

🚴 ANALYSE DES TRAJETS:
  A seulement:        3 vélos
  B seulement:        2 vélos
  C seulement:        1 vélos
  A et B:             4 vélos
  A et C:             0 vélos
  B et C:             2 vélos
  A, B et C (complet): 36 vélos

✅ TOTAL DE VÉLOS UNIQUES: 48
==================================================================
```

**Fichier JSON** (`bike_count_multiline_results.json`):
```json
{
  "video_path": "parade.mp4",
  "model_used": "yolov8n.pt",
  "lines": {
    "A": 400,
    "B": 800,
    "C": 1200
  },
  "counts_per_line": {
    "A": 45,
    "B": 42,
    "C": 38
  },
  "trajectory_analysis": {
    "A_only": 3,
    "B_only": 2,
    "C_only": 1,
    "A_and_B": 4,
    "A_and_C": 0,
    "B_and_C": 2,
    "A_B_and_C": 36
  },
  "total_unique_bikes": 48,
  "min_confidence": 0.5
}
```

### Visualisation pendant le traitement

**Couleurs des vélos:**
- 🟣 **Magenta**: Vélo ayant traversé les 3 lignes (A+B+C)
- 🔵 **Cyan**: Vélo ayant traversé 2 lignes
- 🟡 **Jaune**: Vélo ayant traversé 1 ligne
- ⚪ **Gris**: Vélo détecté mais n'a encore traversé aucune ligne

**Labels des vélos:**
- `#123 [ABC]` : Vélo numéro 123 ayant traversé A, B et C
- `#45 [AB]` : Vélo numéro 45 ayant traversé A et B
- `#67 [B]` : Vélo numéro 67 ayant traversé B seulement
- `#89 [-]` : Vélo numéro 89 n'ayant encore traversé aucune ligne

### Conseils pour le mode multi-lignes

#### Positionnement optimal des lignes

1. **Ligne A (début)**: Placer après le point de départ, où les vélos sont déjà bien visibles
2. **Ligne B (milieu)**: Au centre du parcours, point de contrôle intermédiaire
3. **Ligne C (fin)**: Avant la sortie, pour détecter qui termine le parcours

#### Espacement recommandé

- **Minimum**: 100-200 pixels entre chaque ligne
- **Idéal**: 1/4, 1/2, 3/4 de la largeur (positions par défaut)
- Évitez de placer les lignes trop proches pour éviter les erreurs de tracking

#### Analyse des résultats

- **Taux de complétion**: `A_B_and_C / total_unique_bikes * 100`
- **Abandon précoce**: Vélos sur A seulement
- **Abandon tardif**: Vélos sur A+B mais pas C
- **Entrée tardive**: Vélos sur B ou C seulement (entrés en cours de route)

### Exemples d'utilisation multi-lignes

#### Scénario 1: Parade complète avec positions par défaut
```bash
# Ajuster les lignes
python line_adjuster_multiline.py parade.mp4

# Lancer le comptage (utilise 1/4, 1/2, 3/4 par défaut)
python bike_counter_multiline.py parade.mp4 -o resultat_annote.mp4
```

#### Scénario 2: Analyse précise avec positions personnalisées
```bash
# Positions personnalisées pour une vidéo 1920x1080
python bike_counter_multiline.py parade.mp4 -a 480 -b 960 -c 1440 -m yolov8s.pt -o output.mp4
```

#### Scénario 3: Traitement batch pour statistiques
```bash
# Traiter sans affichage pour analyse rapide
python bike_counter_multiline.py parade.mp4 --no-display
# Consulter bike_count_multiline_results.json pour les stats
```

## Licence

MIT License

## Support

Pour toute question ou problème, créez une issue sur le dépôt GitHub.

## Crédits

- YOLO: [Ultralytics](https://github.com/ultralytics/ultralytics)
- PyTorch DirectML: [Microsoft](https://github.com/microsoft/DirectML)
