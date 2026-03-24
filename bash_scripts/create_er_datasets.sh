#!/bin/bash
set -e

# ==============================================================================
# Create Experience Replay datasets iteratively from real_0 to real_4.
# Each step samples 10 episodes from the old data as replay.
#
# Merge sequence:
#   real_0 + real_1 → er_real_0_1
#   er_real_0_1 + real_2 → er_real_0_2
#   er_real_0_2 + real_3 → er_real_0_3
#   er_real_0_3 + real_4 → er_real_0_4
# ==============================================================================

SAMPLE_N=10
SEED=42
NS="continuallearning"

DATASETS=(
    "real_0_put_bowl_filtered"
    "real_1_stack_bowls_filtered"
    "real_2_put_moka_pot_filtered"
    "real_3_close_drawer_filtered"
    "real_4_put_lego_into_drawer_filtered"
)

# Step 1: real_0 + real_1 → er_real_0_1
echo "=========================================="
echo "Step 1: ${DATASETS[0]} + ${DATASETS[1]} → er_real_0_1"
echo "=========================================="
python -m lerobot.scripts.clare.create_er_dataset \
    --existing_repo_id "${NS}/${DATASETS[0]}" \
    --new_repo_id "${NS}/${DATASETS[1]}" \
    --output_repo_id "${NS}/er_real_0_1" \
    --sample_n_episodes ${SAMPLE_N} \
    --seed ${SEED}

# Steps 2-4: er_real_0_{i-1} + real_{i} → er_real_0_{i}
for i in 2 3 4; do
    PREV=$((i - 1))
    EXISTING="${NS}/er_real_0_${PREV}"
    NEW="${NS}/${DATASETS[$i]}"
    OUTPUT="${NS}/er_real_0_${i}"

    echo ""
    echo "=========================================="
    echo "Step ${i}: ${EXISTING} + ${DATASETS[$i]} → er_real_0_${i}"
    echo "=========================================="
    python -m lerobot.scripts.clare.create_er_dataset \
        --existing_repo_id "${EXISTING}" \
        --new_repo_id "${NEW}" \
        --output_repo_id "${OUTPUT}" \
        --sample_n_episodes ${SAMPLE_N} \
        --seed ${SEED}
done

echo ""
echo "=========================================="
echo "All ER datasets created!"
echo "=========================================="
