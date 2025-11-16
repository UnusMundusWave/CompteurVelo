# Guide de Démarrage Rapide - Mode Multi-Lignes

## 🎯 Vue d'ensemble

Le mode multi-lignes vous permet de compter les vélos sur **3 lignes verticales (A, B, C)** et d'analyser leurs trajectoires complètes.

## 📋 Installation

```bash
# Si pas encore fait
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install torch-directml  # Pour GPU AMD
```

## 🚀 Utilisation en 2 étapes

### Étape 1: Positionner les lignes

```bash
python line_adjuster_multiline.py votre_video.mp4
```

**Commandes:**
- `A` `B` `C` → Sélectionner la ligne
- Clic souris → Déplacer la ligne
- Flèches ← → → Ajuster finement
- ESPACE → Frame suivante
- ENTER → Valider

### Étape 2: Lancer le comptage

```bash
# Positions par défaut (1/4, 1/2, 3/4)
python bike_counter_multiline.py votre_video.mp4

# Avec positions personnalisées
python bike_counter_multiline.py votre_video.mp4 -a 400 -b 800 -c 1200

# Avec sauvegarde vidéo
python bike_counter_multiline.py votre_video.mp4 -o resultat.mp4
```

## 📊 Résultats obtenus

Le système analyse **7 types de trajets** :

### Trajets simples
- **A seulement** : Vélos vus uniquement sur A
- **B seulement** : Vélos vus uniquement sur B
- **C seulement** : Vélos vus uniquement sur C

### Trajets combinés
- **A et B** : Passage par A et B (pas C)
- **A et C** : Passage par A et C (pas B)
- **B et C** : Passage par B et C (pas A)

### Trajet complet
- **A, B et C** : Vélos ayant traversé les 3 lignes ✅

## 🎨 Codes couleur pendant le traitement

- 🟣 **Magenta** : 3 lignes traversées (A+B+C)
- 🔵 **Cyan** : 2 lignes traversées
- 🟡 **Jaune** : 1 ligne traversée
- ⚪ **Gris** : Aucune ligne encore

## 📈 Exemple de résultats

```
PASSAGES PAR LIGNE:
  Ligne A: 45 vélos
  Ligne B: 42 vélos
  Ligne C: 38 vélos

ANALYSE DES TRAJETS:
  A seulement:         3 vélos
  B seulement:         2 vélos
  C seulement:         1 vélos
  A et B:              4 vélos
  A et C:              0 vélos
  B et C:              2 vélos
  A, B et C (complet): 36 vélos

TOTAL: 48 vélos uniques

Taux de complétion: 36/48 = 75%
```

## 💡 Conseils

### Positionnement des lignes
- **Ligne A** : Début du parcours (après le départ)
- **Ligne B** : Milieu du parcours
- **Ligne C** : Fin du parcours (avant la sortie)

### Espacement
- Minimum 100-200 pixels entre les lignes
- Idéal : positions 1/4, 1/2, 3/4 de la largeur

### Analyse
- **Taux de complétion** : `A_B_and_C / total * 100`
- **Abandon précoce** : Comptage sur A seulement
- **Abandon tardif** : Comptage sur A+B mais pas C

## 🔧 Options disponibles

```bash
-a, --line-a      Position X de la ligne A (pixels)
-b, --line-b      Position X de la ligne B (pixels)
-c, --line-c      Position X de la ligne C (pixels)
-m, --model       Modèle YOLO (yolov8n/s/m/l.pt)
--confidence      Seuil de confiance (0.0-1.0)
-o, --output      Sauvegarder vidéo annotée
--no-display      Traiter sans affichage
```

## 📁 Fichiers générés

- `bike_count_multiline_results.json` : Résultats détaillés
- Vidéo annotée (si `-o` spécifié)

## 🎓 Exemples d'utilisation

### Analyse rapide
```bash
python bike_counter_multiline.py parade.mp4
```

### Production complète
```bash
python line_adjuster_multiline.py parade.mp4
# Ajuster les lignes visuellement
python bike_counter_multiline.py parade.mp4 -a 480 -b 960 -c 1440 -o output.mp4
```

### Traitement batch
```bash
python bike_counter_multiline.py video1.mp4 --no-display
python bike_counter_multiline.py video2.mp4 --no-display
# Consulter les fichiers JSON pour les résultats
```

## ❓ Questions fréquentes

**Q: Quelle différence avec le mode simple ?**
R: Le mode simple compte sur 1 ligne (total). Le mode multi-lignes analyse les trajectoires sur 3 lignes.

**Q: Les positions par défaut sont bonnes ?**
R: Oui pour commencer (1/4, 1/2, 3/4). Utilisez l'ajusteur pour personnaliser.

**Q: Comment savoir si un vélo a fait le parcours complet ?**
R: Il apparaît dans le compteur "A, B et C (complet)" en magenta.

**Q: Puis-je changer les positions des lignes ?**
R: Oui, avec `-a`, `-b`, `-c` ou l'outil `line_adjuster_multiline.py`.

## 🆘 Support

Consultez le README.md complet pour plus de détails ou créez une issue sur GitHub.
