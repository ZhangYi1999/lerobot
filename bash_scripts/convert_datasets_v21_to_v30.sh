#!/bin/bash

DATASETS=(
    "real_0_put_bowl"
    "real_0_put_bowl_filtered"
    "real_1_stack_bowls"
    "real_1_stack_bowls_filtered"
    "real_2_put_moka_pot"
    "real_2_put_moka_pot_filtered"
    "real_3_close_drawer"
    "real_3_close_drawer_filtered"
    "real_4_put_lego_into_drawer"
    "real_4_put_lego_into_drawer_filtered"
    "real_5_stack_lego"
    "real_5_stack_lego_filtered"
)

for DATASET in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Converting: continuallearning/${DATASET}"
    echo "=========================================="
    python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
        --repo-id="continuallearning/${DATASET}" \
        --push-to-hub=true || {
        echo "SKIPPED: continuallearning/${DATASET} (not found or invalid)"
        continue
    }
done

echo "All conversions completed!"
