#!/bin/bash
# =============================================================================
# SLURM GPU Queue Status — shows per-partition GPU info, job counts, and
# estimated wait times by combining sinfo + squeue.
# Usage:  bash slurm_queue_status.sh [--gpu-only]
#   --gpu-only   hide CPU-only partitions
# =============================================================================

set -euo pipefail

GPU_ONLY=false
for arg in "$@"; do
    [[ "$arg" == "--gpu-only" ]] && GPU_ONLY=true
done

# ── 1. Partition summary from sinfo ──────────────────────────────────────────
# Columns: Partition | GPUs-per-node | Idle/Mix/Alloc/Down nodes | Total nodes
declare -A PART_GRES PART_IDLE PART_MIX PART_ALLOC PART_DOWN PART_TOTAL
declare -a PART_ORDER=()

while IFS='|' read -r part gres total idle mix alloc down; do
    part="${part%\*}"  # strip default marker
    part="$(echo "$part" | xargs)"
    gres="$(echo "$gres" | xargs)"
    [[ -z "$part" || "$part" == "PARTITION" ]] && continue
    if $GPU_ONLY && [[ "$gres" == "(null)" || -z "$gres" ]]; then
        continue
    fi

    # Track partition order (first seen)
    if [[ -z "${PART_GRES[$part]+x}" ]]; then
        PART_ORDER+=("$part")
    fi

    PART_GRES[$part]="$gres"
    PART_IDLE[$part]="${idle:-0}"
    PART_MIX[$part]="${mix:-0}"
    PART_ALLOC[$part]="${alloc:-0}"
    PART_DOWN[$part]="${down:-0}"
    PART_TOTAL[$part]="${total:-0}"
done < <(sinfo -h -o "%P|%G|%D|%Ci" --format="%P|%G|%D|%D|%D|%D" 2>/dev/null || true)

# Fallback: simpler sinfo parse if the above produced nothing
if [[ ${#PART_ORDER[@]} -eq 0 ]]; then
    while IFS=' ' read -r part avail timelimit nodes state nodelist; do
        [[ "$part" == "PARTITION" ]] && continue
        part="${part%\*}"
        if [[ -z "${PART_GRES[$part]+x}" ]]; then
            PART_ORDER+=("$part")
            PART_GRES[$part]=""
            PART_IDLE[$part]=0
            PART_MIX[$part]=0
            PART_ALLOC[$part]=0
            PART_DOWN[$part]=0
            PART_TOTAL[$part]=0
        fi
        case "$state" in
            idle)  PART_IDLE[$part]=$((${PART_IDLE[$part]} + nodes)) ;;
            mix)   PART_MIX[$part]=$((${PART_MIX[$part]} + nodes)) ;;
            alloc*) PART_ALLOC[$part]=$((${PART_ALLOC[$part]} + nodes)) ;;
            down*|drain*) PART_DOWN[$part]=$((${PART_DOWN[$part]} + nodes)) ;;
        esac
        PART_TOTAL[$part]=$((${PART_TOTAL[$part]} + nodes))
    done < <(sinfo -h -o "%P %a %l %D %T %N" 2>/dev/null)

    # Get GRES info separately
    while IFS='|' read -r part gres; do
        part="${part%\*}"
        part="$(echo "$part" | xargs)"
        gres="$(echo "$gres" | xargs)"
        [[ -n "${PART_GRES[$part]+x}" ]] && PART_GRES[$part]="$gres"
    done < <(sinfo -h -o "%P|%G" 2>/dev/null)

    # Apply GPU filter
    if $GPU_ONLY; then
        filtered=()
        for p in "${PART_ORDER[@]}"; do
            [[ "${PART_GRES[$p]}" != "(null)" && -n "${PART_GRES[$p]}" ]] && filtered+=("$p")
        done
        PART_ORDER=("${filtered[@]}")
    fi
fi

# ── 2. Job counts from squeue ────────────────────────────────────────────────
declare -A RUNNING_JOBS PENDING_JOBS

# Running jobs per partition
while IFS=' ' read -r count part; do
    part="$(echo "$part" | xargs)"
    RUNNING_JOBS[$part]="${count}"
done < <(squeue -t R -h -o "%P" 2>/dev/null | sort | uniq -c)

# Pending jobs per partition
while IFS=' ' read -r count part; do
    part="$(echo "$part" | xargs)"
    PENDING_JOBS[$part]="${count}"
done < <(squeue -t PD -h -o "%P" 2>/dev/null | sort | uniq -c)

# ── 3. Estimated wait from squeue --start ────────────────────────────────────
# For pending jobs, SLURM can estimate start times. We find the latest
# estimated start per partition to gauge worst-case wait.
declare -A WAIT_MIN WAIT_MAX WAIT_EARLIEST

now=$(date +%s)

while IFS='|' read -r jobid part name user state start nodes; do
    part="$(echo "$part" | xargs)"
    state="$(echo "$state" | xargs)"
    start="$(echo "$start" | xargs)"
    [[ "$state" != "PENDING" && "$state" != "PD" ]] && continue
    [[ -z "$start" || "$start" == "N/A" || "$start" == "(null)" ]] && continue

    start_epoch=$(date -d "$start" +%s 2>/dev/null || echo 0)
    [[ "$start_epoch" -eq 0 ]] && continue
    wait_secs=$((start_epoch - now))
    [[ "$wait_secs" -lt 0 ]] && wait_secs=0

    if [[ -z "${WAIT_MIN[$part]+x}" ]]; then
        WAIT_MIN[$part]=$wait_secs
        WAIT_MAX[$part]=$wait_secs
        WAIT_EARLIEST[$part]="$start"
    else
        [[ $wait_secs -lt ${WAIT_MIN[$part]} ]] && WAIT_MIN[$part]=$wait_secs
        [[ $wait_secs -gt ${WAIT_MAX[$part]} ]] && WAIT_MAX[$part]=$wait_secs
    fi
done < <(squeue --start -t PD -h -o "%i|%P|%j|%u|%T|%S|%D" 2>/dev/null)

# ── Helper: format seconds to human-readable ─────────────────────────────────
fmt_time() {
    local s=$1
    if [[ $s -lt 60 ]]; then
        echo "<1m"
    elif [[ $s -lt 3600 ]]; then
        echo "$((s / 60))m"
    elif [[ $s -lt 86400 ]]; then
        echo "$((s / 3600))h$((s % 3600 / 60))m"
    else
        echo "$((s / 86400))d$((s % 86400 / 3600))h"
    fi
}

# ── 4. Print table ───────────────────────────────────────────────────────────
printf "\n"
printf "%-30s  %-22s  %5s  %5s  %5s  %5s  %6s  %6s  %-14s  %-14s\n" \
    "PARTITION" "GPU_RES" "TOTAL" "IDLE" "MIX" "DOWN" "RUN_J" "PD_J" "WAIT_MIN" "WAIT_MAX"
printf "%s\n" "$(printf '=%.0s' {1..140})"

for part in "${PART_ORDER[@]}"; do
    gres="${PART_GRES[$part]:-—}"
    total="${PART_TOTAL[$part]:-0}"
    idle="${PART_IDLE[$part]:-0}"
    mix="${PART_MIX[$part]:-0}"
    down="${PART_DOWN[$part]:-0}"
    run="${RUNNING_JOBS[$part]:-0}"
    pend="${PENDING_JOBS[$part]:-0}"

    if [[ -n "${WAIT_MIN[$part]+x}" ]]; then
        wmin="$(fmt_time ${WAIT_MIN[$part]})"
        wmax="$(fmt_time ${WAIT_MAX[$part]})"
    elif [[ "$pend" -gt 0 ]]; then
        wmin="unknown"
        wmax="unknown"
    else
        wmin="—"
        wmax="—"
    fi

    printf "%-30s  %-22s  %5s  %5s  %5s  %5s  %6s  %6s  %-14s  %-14s\n" \
        "$part" "$gres" "$total" "$idle" "$mix" "$down" "$run" "$pend" "$wmin" "$wmax"
done

printf "\n"
printf "Timestamp: %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
printf "WAIT_MIN/MAX = estimated wait for the earliest/latest pending job in each partition (from squeue --start)\n"
printf "Tip: use --gpu-only to hide CPU partitions\n\n"
