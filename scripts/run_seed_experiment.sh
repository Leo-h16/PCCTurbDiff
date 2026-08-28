#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_SEEDS="2883413570083077179,7318462950184729365,4927816359048271538,8652391746083519274,3175948260391756482"

EXPERIMENT_NAME=""
CHECKPOINT_ROOT=""
TRAIN_GPUS="3,5,6,7"
EVAL_GPU="3"
SEEDS="$DEFAULT_SEEDS"
MODE="all"
SAMPLES_ROOT=""
RESULTS_DIR=""
EXPENSIVE=0
TRAIN_OVERRIDES=()

usage() {
    echo "Usage: $0 --name NAME --checkpoint-root DIR [options] [-- TRAIN_OVERRIDE ...]"
    echo
    echo "Options:"
    echo "  --name NAME             Experiment/W&B name (required)"
    echo "  --checkpoint-root DIR   Root directory for this experiment's checkpoints"
    echo "  --train-gpus IDS        CUDA devices for training (default: 3,5,6,7)"
    echo "  --eval-gpu ID           CUDA device for evaluation (default: 3)"
    echo "  --seeds S1,S2,...       Comma-separated seeds"
    echo "  --samples-root DIR      Evaluation sample directory"
    echo "  --results-dir DIR       Evaluation metrics directory"
    echo "  --mode all|train|eval   Run both stages or only one (default: all)"
    echo "  --expensive             Enable expensive evaluation metrics"
    echo "  -h, --help              Show this help"
    echo
    echo "Arguments after -- are passed to train.py as Hydra overrides."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --checkpoint-root)
            CHECKPOINT_ROOT="$2"
            shift 2
            ;;
        --train-gpus)
            TRAIN_GPUS="$2"
            shift 2
            ;;
        --eval-gpu)
            EVAL_GPU="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --samples-root)
            SAMPLES_ROOT="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --expensive)
            EXPENSIVE=1
            shift
            ;;
        --)
            shift
            TRAIN_OVERRIDES=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$EXPERIMENT_NAME" || -z "$CHECKPOINT_ROOT" ]]; then
    echo "--name and --checkpoint-root are required." >&2
    usage >&2
    exit 2
fi

if [[ "$MODE" != "all" && "$MODE" != "train" && "$MODE" != "eval" ]]; then
    echo "--mode must be one of: all, train, eval" >&2
    exit 2
fi

SAMPLES_ROOT="${SAMPLES_ROOT:-$PROJECT_ROOT/results/$EXPERIMENT_NAME/samples}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results/$EXPERIMENT_NAME/metrics}"
IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
IFS=',' read -r -a TRAIN_GPU_ARRAY <<< "$TRAIN_GPUS"

cd "$PROJECT_ROOT"

if [[ "$MODE" == "all" || "$MODE" == "train" ]]; then
    echo "Training ${#SEED_ARRAY[@]} seeds; checkpoints: $CHECKPOINT_ROOT"
    for seed in "${SEED_ARRAY[@]}"; do
        echo "Training seed $seed"
        # Use a fresh Python process for every seed. A Hydra multirun would reuse
        # the process after DDP training and leave torch.distributed initialized.
        CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" python ./train.py \
            "trainer.devices=${#TRAIN_GPU_ARRAY[@]}" \
            "seed=$seed" \
            eval_testset=false \
            "checkpoint_root=$CHECKPOINT_ROOT" \
            "wandb.name=$EXPERIMENT_NAME-$seed" \
            "${TRAIN_OVERRIDES[@]}"
    done
fi

if [[ "$MODE" == "all" || "$MODE" == "eval" ]]; then
    mkdir -p "$SAMPLES_ROOT" "$RESULTS_DIR"
    eval_options=()
    if [[ "$EXPENSIVE" -eq 1 ]]; then
        eval_options+=(--expensive)
    fi

    for seed in "${SEED_ARRAY[@]}"; do
        checkpoint="$CHECKPOINT_ROOT/seed_$seed/best.ckpt"
        samples="$SAMPLES_ROOT/seed_$seed.h5"

        if [[ ! -f "$checkpoint" ]]; then
            echo "Checkpoint not found: $checkpoint" >&2
            exit 1
        fi
        if [[ -e "$samples" ]]; then
            echo "Samples already exist: $samples" >&2
            echo "Choose another --samples-root or move the existing file." >&2
            exit 1
        fi

        echo "Evaluating seed $seed"
        CUDA_VISIBLE_DEVICES="$EVAL_GPU" python ./scripts/eval_ckpt.py \
            --device cuda \
            --seed "$seed" \
            --results-dir "$RESULTS_DIR" \
            "${eval_options[@]}" \
            "$checkpoint" \
            "$samples"
    done

    echo "Combined metrics: $RESULTS_DIR/summary.csv"
fi
