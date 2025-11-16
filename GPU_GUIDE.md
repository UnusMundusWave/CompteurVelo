# Guide d'Utilisation du GPU AMD (DirectML)

## 🎮 Pourquoi utiliser le GPU ?

Le GPU AMD accélère significativement le traitement YOLO, permettant:
- **2-5x plus rapide** que le CPU seul
- Traiter des vidéos longues plus rapidement
- Meilleure utilisation des ressources système

## 📋 Installation DirectML

### Prérequis
- Windows 11
- GPU AMD (toute génération récente)
- Drivers AMD Adrenalin à jour

### Installation

```bash
# Activer l'environnement virtuel
venv\Scripts\activate

# Installer DirectML
pip install torch-directml
```

## ✅ Vérifier la Configuration GPU

### Test Rapide
```bash
python test_gpu.py
```

**Résultat attendu:**
```
✓ PyTorch installé
✓ OpenCV installé
✓ Ultralytics (YOLO) installé
✓ DirectML disponible et fonctionnel
  Device: privateuseone:0
```

### Test Complet
```bash
python test_gpu_advanced.py
```

Ce script teste:
1. Installation PyTorch
2. Disponibilité DirectML
3. Opérations tenseurs sur GPU
4. Inférence YOLO sur GPU
5. Benchmark CPU vs GPU
6. Support OpenCV

## 🚀 Utilisation

### Par Défaut (GPU Activé)

Le GPU est **automatiquement activé** si DirectML est installé:

```bash
# Mode simple - GPU auto
python bike_counter.py video.mp4

# Mode multi-lignes - GPU auto
python bike_counter_multiline.py video.mp4
```

Au démarrage, vous verrez:
```
✓ GPU AMD DirectML détecté et activé: privateuseone:0
Chargement du modèle YOLO: yolov8n.pt
Déplacement du modèle sur privateuseone:0...
```

### Forcer l'Utilisation du CPU

Si vous rencontrez des problèmes avec le GPU:

```bash
# Désactiver le GPU
python bike_counter.py video.mp4 --no-gpu
python bike_counter_multiline.py video.mp4 --no-gpu
```

## 📊 Benchmark GPU vs CPU

### Lancer un Benchmark

```bash
# Benchmark avec vidéo de test
python benchmark_gpu.py

# Benchmark avec votre vidéo
python benchmark_gpu.py -v votre_video.mp4

# Benchmark détaillé (5 runs)
python benchmark_gpu.py -v video.mp4 -r 5
```

### Interprétation des Résultats

**Exemple de sortie:**
```
COMPARAISON FINALE
==============================================================

🖥️  CPU:
  Temps moyen: 45.23s
  FPS moyen: 22.10

🎮 GPU (DirectML):
  Temps moyen: 15.78s
  FPS moyen: 63.37

📈 AMÉLIORATION:
  🚀 GPU 2.87x plus rapide
  🚀 FPS 2.87x supérieur

⏱️  ESTIMATION pour vidéo 1 heure (30 FPS = 108000 frames):
  CPU: 163.7 minutes
  GPU: 57.1 minutes
  Gain de temps: 106.6 minutes
```

## 🔧 Résolution de Problèmes

### Problème: GPU non détecté

**Symptômes:**
```
⚠ DirectML non installé, utilisation du CPU
```

**Solution:**
```bash
pip install torch-directml
```

### Problème: DirectML installé mais non disponible

**Symptômes:**
```
⚠ DirectML installé mais non disponible, utilisation du CPU
```

**Solutions:**
1. Mettre à jour les drivers AMD:
   - Télécharger depuis [AMD Support](https://www.amd.com/en/support)
   - Installer la dernière version AMD Adrenalin

2. Vérifier la compatibilité GPU:
   ```bash
   python -c "import torch_directml; print(torch_directml.is_available())"
   ```

3. Réinstaller DirectML:
   ```bash
   pip uninstall torch-directml
   pip install torch-directml
   ```

### Problème: Erreur pendant l'inférence GPU

**Symptômes:**
- Crash pendant le traitement
- Erreur CUDA/DirectML

**Solutions:**
1. Essayer avec CPU pour vérifier que le problème vient du GPU:
   ```bash
   python bike_counter.py video.mp4 --no-gpu
   ```

2. Réduire la taille du modèle:
   ```bash
   # Utiliser yolov8n au lieu de yolov8s/m/l
   python bike_counter.py video.mp4 -m yolov8n.pt
   ```

3. Vérifier la mémoire GPU disponible

### Problème: CPU à 23% seulement

**Cause:** Le GPU fait le travail ! C'est normal.

**Vérification:**
1. Vérifier le message au démarrage:
   ```
   ✓ GPU AMD DirectML détecté et activé
   ```

2. Le CPU reste à ~20-30% car:
   - Décodage vidéo (CPU)
   - Prétraitement images (CPU)
   - Inférence YOLO (GPU) ← Le gros du travail
   - Affichage OpenCV (CPU)

3. Pour confirmer que le GPU travaille:
   ```bash
   # Lancer le benchmark
   python benchmark_gpu.py -v votre_video.mp4

   # Vous devriez voir une différence significative
   # entre CPU et GPU
   ```

## 💡 Optimisations GPU

### 1. Choisir le Bon Modèle

| Modèle | Taille | Vitesse GPU | Précision | Recommandation |
|--------|--------|-------------|-----------|----------------|
| yolov8n | 6 MB | Très rapide | Bonne | ✅ Défaut, idéal |
| yolov8s | 22 MB | Rapide | Meilleure | Pour vidéos HD |
| yolov8m | 52 MB | Moyen | Très bonne | GPU puissant |
| yolov8l | 87 MB | Lent | Excellente | Précision max |

```bash
# Utiliser un modèle plus petit pour plus de vitesse
python bike_counter.py video.mp4 -m yolov8n.pt

# Ou plus gros pour plus de précision
python bike_counter.py video.mp4 -m yolov8s.pt
```

### 2. Désactiver l'Affichage Vidéo

L'affichage en temps réel ralentit le traitement:

```bash
# Sans affichage = plus rapide
python bike_counter.py video.mp4 --no-display -o output.mp4
```

**Gains:** +20-30% de vitesse

### 3. Ajuster la Confiance

Un seuil plus élevé = moins de détections à traiter:

```bash
# Confiance plus élevée = plus rapide
python bike_counter.py video.mp4 --confidence 0.6
```

## 📈 Performance Attendue

### Configuration Typique
- **GPU:** AMD Radeon RX 5000/6000/7000 series
- **Modèle:** YOLOv8n
- **Vidéo:** 1920x1080, 30 FPS

**Résultats:**
- CPU: ~20-25 FPS
- GPU: ~50-70 FPS
- **Accélération: 2.5-3x**

### Vidéos Longues

Pour une vidéo de **1 heure (30 FPS):**
- CPU: ~2.5 heures de traitement
- GPU: ~50-60 minutes de traitement
- **Gain: 1.5 heures**

## 🎯 Commandes Utiles

```bash
# Test configuration GPU
python test_gpu_advanced.py

# Benchmark complet
python benchmark_gpu.py -v video.mp4 -r 5

# Traitement avec GPU (défaut)
python bike_counter_multiline.py video.mp4 -o output.mp4

# Traitement sans GPU (fallback)
python bike_counter_multiline.py video.mp4 --no-gpu

# Traitement optimisé (GPU, pas d'affichage, modèle léger)
python bike_counter.py video.mp4 -m yolov8n.pt --no-display -o out.mp4
```

## ❓ FAQ

**Q: Mon GPU AMD est-il supporté ?**
R: Tous les GPU AMD récents (RX 5000+) sont supportés via DirectML.

**Q: Pourquoi le CPU est à 23% seulement ?**
R: C'est normal ! Le GPU fait le travail d'inférence YOLO. Le CPU gère uniquement le décodage vidéo et l'affichage.

**Q: Comment vérifier que le GPU travaille vraiment ?**
R: Lancez `python test_gpu_advanced.py` ou `python benchmark_gpu.py`. Vous verrez la différence de vitesse.

**Q: Le GPU est plus lent que le CPU ?**
R: Cela peut arriver avec:
  - De très petites vidéos (overhead GPU > gain)
  - Des GPU anciens
  - Drivers obsolètes

Mettez à jour les drivers AMD.

**Q: Puis-je utiliser plusieurs GPU ?**
R: DirectML utilise automatiquement le GPU principal. Le multi-GPU n'est pas supporté actuellement.

**Q: Dois-je toujours garder le GPU activé ?**
R: Oui, sauf si vous rencontrez des erreurs. Le GPU est plus rapide dans 95% des cas.

## 🆘 Support

Si vous rencontrez des problèmes:

1. Vérifier les drivers AMD
2. Tester avec `python test_gpu_advanced.py`
3. Essayer avec `--no-gpu` pour isoler le problème
4. Créer une issue sur GitHub avec les détails

## 📚 Ressources

- [PyTorch DirectML](https://github.com/microsoft/DirectML)
- [AMD Drivers](https://www.amd.com/en/support)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
