# Guide de Démarrage Rapide - Mode Multi-Lignes (5 Lignes)

## 🎯 Vue d'ensemble

Le mode multi-lignes vous permet de compter les vélos et personnes sur **5 lignes verticales (A, B, C, D, E)** et d'analyser leurs trajectoires complètes avec une granularité fine.

## 📋 Installation

```bash
# Si pas encore fait
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install torch-directml  # Pour GPU AMD (optionnel)
```

## 🚀 Utilisation en 2 étapes

### Étape 1: Positionner les lignes

```bash
python line_adjuster_multiline.py votre_video.mp4
```

**Commandes:**
- `A` `B` `C` `D` `E` → Sélectionner la ligne
- Clic souris → Déplacer la ligne
- Flèches ← → → Ajuster finement (10 pixels)
- ESPACE → Frame suivante
- R → Revenir au début
- X → Positions par défaut
- ENTER → Valider

### Étape 2: Lancer le comptage

```bash
# Positions par défaut (1/6, 2/6, 3/6, 4/6, 5/6)
python bike_counter_multiline.py votre_video.mp4

# Avec positions personnalisées
python bike_counter_multiline.py votre_video.mp4 -a 300 -b 600 -c 900 -d 1200 -e 1500

# Avec sauvegarde vidéo
python bike_counter_multiline.py votre_video.mp4 -o resultat.mp4
```

## 📊 Résultats obtenus

Le système analyse **31 types de trajets** (toutes les combinaisons non-vides de 5 lignes):

### Trajets simples (5 combinaisons)
- **A** : Vélos vus uniquement sur A
- **B** : Vélos vus uniquement sur B
- **C** : Vélos vus uniquement sur C
- **D** : Vélos vus uniquement sur D
- **E** : Vélos vus uniquement sur E

### Trajets doubles (10 combinaisons)
- **AB**, **AC**, **AD**, **AE** : Passages par A et une autre ligne
- **BC**, **BD**, **BE** : Passages par B et une autre ligne
- **CD**, **CE** : Passages par C et D ou E
- **DE** : Passages par D et E

### Trajets triples (10 combinaisons)
- **ABC**, **ABD**, **ABE**, **ACD**, **ACE**, **ADE**
- **BCD**, **BCE**, **BDE**, **CDE**

### Trajets quadruples (5 combinaisons)
- **ABCD**, **ABCE**, **ABDE**, **ACDE**, **BCDE**

### Trajet complet (1 combinaison)
- **ABCDE** : Vélos ayant traversé les 5 lignes ✅

## 🎨 Codes couleur pendant le traitement

Couleur des bounding boxes selon le nombre de lignes traversées:
- 🟣 **Magenta** : 5 lignes traversées (ABCDE - COMPLET)
- 🌸 **Rose** : 4 lignes traversées
- 🔵 **Cyan** : 3 lignes traversées
- 🟡 **Jaune** : 2 lignes traversées
- 🟠 **Orange** : 1 ligne traversée
- ⚪ **Gris** : Aucune ligne encore

Couleur des lignes de comptage:
- 🔵 **Bleu** : Ligne A
- 🟢 **Vert** : Ligne B
- 🔴 **Rouge** : Ligne C
- 🟡 **Jaune** : Ligne D
- 🟣 **Magenta** : Ligne E

## 📈 Exemple de résultats

```
RÉSULTATS DU COMPTAGE 5 LIGNES - VÉLOS ET PERSONNES
================================================================================

🚴 VÉLOS - PASSAGES PAR LIGNE:
  Ligne A:  45 | Ligne B:  42 | Ligne C:  40 | Ligne D:  38 | Ligne E:  35

🚴 VÉLOS - ANALYSE DES TRAJETS (31 combinaisons):

  Ligne unique (5):
    A    :   2 vélo(s)
    B    :   1 vélo(s)
    E    :   1 vélo(s)

  Deux lignes (10):
    AB   :   3 vélo(s)
    BC   :   2 vélo(s)

  Trois lignes (10):
    ABC  :   4 vélo(s)
    BCD  :   3 vélo(s)

  Quatre lignes (5):
    ABCD :   5 vélo(s)
    ABCE :   2 vélo(s)

  Cinq lignes (parcours complet):
    ABCDE:  32 vélo(s) ✅

✅ TOTAL DE VÉLOS UNIQUES: 48

👤 PERSONNES - PASSAGES PAR LIGNE:
  Ligne A:  23 | Ligne B:  21 | Ligne C:  20 | Ligne D:  18 | Ligne E:  15

👤 PERSONNES - ANALYSE DES TRAJETS (31 combinaisons):
  ...

✅ TOTAL DE PERSONNES UNIQUES: 25

📊 TOTAL GÉNÉRAL: 73 objets (48 vélos + 25 personnes)
```

## 💡 Conseils

### Positionnement des lignes

Pour un **parcours de parade** typique:
- **Ligne A** : Début du parcours surveillé (après départ)
- **Ligne B** : Premier point de contrôle (1/3 du parcours)
- **Ligne C** : Point milieu (centre)
- **Ligne D** : Deuxième point de contrôle (2/3 du parcours)
- **Ligne E** : Fin du parcours surveillé (avant sortie)

### Espacement

- **Minimum** : 100-200 pixels entre les lignes
- **Idéal** : Positions 1/6, 2/6, 3/6, 4/6, 5/6 de la largeur
- **Ajustement** : Utiliser `line_adjuster_multiline.py` pour visualiser

### Analyse des résultats

- **Taux de complétion** : `ABCDE / total * 100`
- **Abandon précoce** : Comptage sur A, AB, ABC uniquement
- **Abandon tardif** : Comptage sur ABCD mais pas ABCDE
- **Raccourcis** : Détections manquant certaines lignes (ex: ACE)

## 🔧 Options disponibles

```bash
-a, --line-a      Position X de la ligne A (pixels)
-b, --line-b      Position X de la ligne B (pixels)
-c, --line-c      Position X de la ligne C (pixels)
-d, --line-d      Position X de la ligne D (pixels)
-e, --line-e      Position X de la ligne E (pixels)
-m, --model       Modèle YOLO (yolov8n/s/m/l.pt)
--confidence      Seuil de confiance (0.0-1.0)
-o, --output      Sauvegarder vidéo annotée
--no-display      Traiter sans affichage
--no-gpu          Forcer l'utilisation du CPU
```

## 📁 Fichiers générés

- `bike_count_multiline_results.json` : Résultats détaillés (tous les compteurs)
- Vidéo annotée (si `-o` spécifié)

### Structure du fichier JSON

```json
{
  "video_path": "parade.mp4",
  "model_used": "yolov8n.pt",
  "lines": {
    "A": 320,
    "B": 640,
    "C": 960,
    "D": 1280,
    "E": 1600
  },
  "bikes": {
    "counts_per_line": {
      "A": 45, "B": 42, "C": 40, "D": 38, "E": 35
    },
    "trajectory_analysis": {
      "A": 2, "B": 1, "C": 0, "D": 0, "E": 1,
      "AB": 3, "AC": 0, ...
      "ABCDE": 32
    },
    "total_unique": 48
  },
  "people": { ... },
  "total_general": 73
}
```

## 🎓 Exemples d'utilisation

### Analyse rapide (positions par défaut)

```bash
python bike_counter_multiline.py parade.mp4
```

### Production complète avec ajustement visuel

```bash
# Étape 1: Ajuster visuellement les lignes
python line_adjuster_multiline.py parade.mp4

# Étape 2: Lancer avec positions personnalisées et sauvegarde
python bike_counter_multiline.py parade.mp4 -a 300 -b 600 -c 900 -d 1200 -e 1500 -o output.mp4
```

### Traitement batch (sans affichage)

```bash
python bike_counter_multiline.py video1.mp4 --no-display
python bike_counter_multiline.py video2.mp4 --no-display
# Consulter les fichiers JSON pour les résultats
```

### Traitement haute précision

```bash
# Utiliser modèle plus précis + seuil confiance plus élevé
python bike_counter_multiline.py parade.mp4 -m yolov8s.pt --confidence 0.6 -o output.mp4
```

## ❓ Questions fréquentes

**Q: Quelle différence avec le mode simple ?**
R: Le mode simple compte sur 1 ligne (total). Le mode multi-lignes analyse les trajectoires sur 5 lignes avec 31 combinaisons.

**Q: Les positions par défaut sont bonnes ?**
R: Oui pour commencer (1/6, 2/6, 3/6, 4/6, 5/6). Utilisez l'ajusteur pour personnaliser selon votre vidéo.

**Q: Comment savoir si un vélo a fait le parcours complet ?**
R: Il apparaît dans le compteur "ABCDE" avec une bounding box magenta.

**Q: Puis-je changer les positions des lignes après le traitement ?**
R: Non, il faut relancer avec les nouvelles positions. Utilisez `line_adjuster_multiline.py` d'abord.

**Q: Combien de lignes sont recommandées ?**
R: 5 lignes offrent une granularité fine. Pour des analyses plus simples, vous pouvez ignorer certaines combinaisons dans les résultats JSON.

**Q: Les 31 combinaisons sont toujours utiles ?**
R: Non. Pour une parade linéaire, ABCDE et les combinaisons séquentielles (ABC, ABCD) sont les plus importantes. Les combinaisons non-séquentielles (ACE, BD) indiquent des détections manquées.

**Q: Que signifie une combinaison comme "ACE" ?**
R: Un objet détecté sur A, C et E mais manqué sur B et D. Cela peut indiquer:
  - Détection temporairement perdue
  - Objet partiellement occulté
  - Trajet très rapide

**Q: Comment interpréter les résultats pour une parade ?**
R:
  - **ABCDE** : Parcours complet normal
  - **ABCD, BCDE** : Parcours presque complet
  - **Combinaisons courtes (A, AB)** : Abandons précoces
  - **Combinaisons non-séquentielles** : Problèmes de détection

## 📊 Statistiques avancées

Vous pouvez calculer des métriques additionnelles à partir du JSON:

```python
import json

with open('bike_count_multiline_results.json', 'r') as f:
    data = json.load(f)

bikes = data['bikes']['trajectory_analysis']

# Taux de complétion
completion_rate = bikes['ABCDE'] / data['bikes']['total_unique'] * 100
print(f"Taux de complétion: {completion_rate:.1f}%")

# Taux d'abandon précoce (A, B, C uniquement)
early_abandon = bikes['A'] + bikes['B'] + bikes['C']
early_abandon_rate = early_abandon / data['bikes']['total_unique'] * 100
print(f"Taux d'abandon précoce: {early_abandon_rate:.1f}%")

# Parcours avec au moins 4 lignes
advanced = sum(v for k, v in bikes.items() if len(k) >= 4)
advanced_rate = advanced / data['bikes']['total_unique'] * 100
print(f"Parcours avancés (4+ lignes): {advanced_rate:.1f}%")
```

## 🆘 Support

Si vous rencontrez des problèmes:

1. Vérifier que la vidéo existe et est accessible
2. Tester avec `line_adjuster_multiline.py` pour visualiser
3. Essayer avec `--no-display` pour isoler les problèmes d'affichage
4. Consulter le README.md complet pour plus de détails
5. Créer une issue sur GitHub avec les détails

## 🔗 Ressources

- [README principal](README.md) - Documentation complète
- [GPU_GUIDE.md](GPU_GUIDE.md) - Configuration GPU AMD DirectML
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Documentation YOLO
