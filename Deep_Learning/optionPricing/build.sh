# 1. Vérifie si le dossier build existe, sinon le crée
if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Path "build"
}

# 2. Entre dans le dossier build
cd build

# 3. Configuration avec CMake
Write-Host "--- Configuration du projet ---" -ForegroundColor Cyan
cmake -G "MinGW Makefiles" ..

# 4. Compilation
Write-Host "--- Compilation en cours ---" -ForegroundColor Cyan
cmake --build .

# 5. Lancement si la compilation a réussi
if ($LASTEXITCODE -eq 0) {
    Write-Host "--- Lancement du programme ---" -ForegroundColor Green
    
    # On copie le dataset dans build s'il n'y est pas déjà
    if (Test-Path "../options_dataset.csv") {
        Copy-Item "../options_dataset.csv" . -Force
    }

    ./optionPricingDl.exe
} else {
    Write-Host "ERREUR : La compilation a echoue." -ForegroundColor Red
}

# 6. Retour à la racine
cd ..