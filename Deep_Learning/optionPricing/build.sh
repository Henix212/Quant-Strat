# 1. Vérifie si le dossier build existe, sinon le crée
if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Path "build"
}

cd build

cmake -G "MinGW Makefiles" ..

cmake --build .

if ($LASTEXITCODE -eq 0) {
    
    if (Test-Path "../options_dataset.csv") {
        Copy-Item "../options_dataset.csv" . -Force
    }

    ./optionPricingDl.exe
} else {
    Write-Host "ERREUR : La compilation a echoue." -ForegroundColor Red
}

cd ..