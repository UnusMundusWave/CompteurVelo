@echo off
REM Script de démarrage rapide pour Windows
REM Lance l'installation et la configuration

echo ================================================
echo COMPTEUR DE VELOS - INSTALLATION RAPIDE
echo ================================================
echo.

REM Vérifier si Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python n'est pas installé ou n'est pas dans le PATH
    echo Veuillez installer Python 3.8+ depuis https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Python détecté
python --version

REM Créer l'environnement virtuel s'il n'existe pas
if not exist "venv\" (
    echo.
    echo [2/4] Création de l'environnement virtuel...
    python -m venv venv
    if errorlevel 1 (
        echo ERREUR: Impossible de créer l'environnement virtuel
        pause
        exit /b 1
    )
) else (
    echo.
    echo [2/4] Environnement virtuel existant détecté
)

REM Activer l'environnement virtuel
echo.
echo [3/4] Activation de l'environnement virtuel...
call venv\Scripts\activate.bat

REM Installer les dépendances
echo.
echo [4/4] Installation des dépendances...
echo Cela peut prendre quelques minutes...
python -m pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERREUR lors de l'installation des dépendances
    pause
    exit /b 1
)

echo.
echo ================================================
echo INSTALLATION TERMINEE !
echo ================================================
echo.
echo Pour utiliser le compteur de velos:
echo   1. Activez l'environnement virtuel: venv\Scripts\activate
echo   2. Lancez le script: python bike_counter.py votre_video.mp4
echo.
echo Pour ajuster la ligne de comptage:
echo   python line_adjuster.py votre_video.mp4
echo.
echo Pour plus d'informations, consultez README.md
echo ================================================
echo.

pause
