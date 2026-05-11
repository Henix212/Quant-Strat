#!/bin/bash

if [ ! -d "build" ]; then
    mkdir build
fi

cd build

cmake ..

cmake --build .

if [ $? -eq 0 ]; then
    if [ -f "../options_dataset.csv" ]; then
        cp "../options_dataset.csv" .
    fi

    ./optionPricingDl
else
    echo "ERREUR : La compilation a échoué."
fi

cd ..