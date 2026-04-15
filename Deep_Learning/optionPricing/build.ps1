if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Path "build"
}

cd build
Write-Host "--- Configuration et Compilation ---" -ForegroundColor Cyan
cmake -G "MinGW Makefiles" ..
cmake --build .

if ($LASTEXITCODE -eq 0) {
    Write-Host "--- Lancement du programme ---" -ForegroundColor Green
    if (Test-Path "../options_dataset.csv") {
        Copy-Item "../options_dataset.csv" . -Force
    }
    ./optionPricingDl.exe
} else {
    Write-Host "ERREUR : La compilation a échoué." -ForegroundColor Red
}
cd ..