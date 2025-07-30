#!/bin/bash

cd /home/johnma/bcr-scattering-inv
source /home/johnma/miniconda3/bin/activate
conda activate tf113
python subprocess_get_preds.py \
    --initialvalue logs-darwin-1/S2ddataNc6Al128Ns0large-model.h5 \
    --input-prefix temp_inference_data \
    --data-path data \