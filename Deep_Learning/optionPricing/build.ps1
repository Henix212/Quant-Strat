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
    Write-Host "ERROR." -ForegroundColor Red
}
cd ..