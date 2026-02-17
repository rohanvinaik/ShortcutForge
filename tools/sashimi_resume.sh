#!/bin/bash
# Resume Sashimi Mode from where we left off
# Batches 1-2 complete (examples 0-999), resuming from 1000
# Uses evaluate_model.py directly in a loop with skip/max

set -e
cd /Users/rohanvinaik/apple-shortcuts

MODEL="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
ADAPTER="models/baseline-v1-mlx"
EVAL_FILE="training_data/shortcutdsl_train_expanded.jsonl"
OUTPUT="training_data/distillation_log_full.jsonl"
BATCH_SIZE=500
TIMEOUT=60
TOTAL=6679
START=1000  # Already completed 0-999

echo "=== Sashimi Mode Resume — $(date) ==="
echo "  Resuming from example $START of $TOTAL"
echo "  Existing entries: $(wc -l < $OUTPUT)"
echo ""

processed=$START
batch_num=2  # We finished batch 2

while [ $processed -lt $TOTAL ]; do
    batch_num=$((batch_num + 1))
    remaining=$((TOTAL - processed))
    current_batch=$((BATCH_SIZE < remaining ? BATCH_SIZE : remaining))

    echo ""
    echo "  Batch $batch_num: examples $((processed+1))-$((processed+current_batch)) of $TOTAL — $(date)"

    python training/evaluate_model.py \
        --model-path "$MODEL" \
        --adapter-path "$ADAPTER" \
        --eval-file "$EVAL_FILE" \
        --skip-examples $processed \
        --max-examples $current_batch \
        --log-distillation \
        --distillation-output "$OUTPUT" \
        --append-distillation \
        --chat-template llama3 \
        --timeout $TIMEOUT 2>&1 | tail -5

    processed=$((processed + current_batch))
    entries=$(wc -l < "$OUTPUT")
    echo "  Batch $batch_num complete — $(date) — $entries total entries"
done

echo ""
echo "=== Generation complete — $(date) ==="
echo "  Total entries: $(wc -l < $OUTPUT)"

# Now run curation and merge (steps 2-3)
echo ""
echo "  [Step 2] Curating distillation data..."
python training/distillation_curator.py "$OUTPUT" \
    --output training_data/distilled_curated.jsonl 2>&1 | tail -5

echo ""
echo "  [Step 2b] Converting to chat format..."
python -c "
import json, sys
sys.path.insert(0, 'training')
from build_distill_data import convert_to_chat_format
n = convert_to_chat_format(
    'training_data/distilled_curated.jsonl',
    'training_data/distilled_chat.jsonl',
    verbose=True
)
print(f'  Converted {n} entries')
"

echo ""
echo "  [Step 3] Merging gold + distilled..."
python -c "
import json, sys
sys.path.insert(0, 'training')
from build_distill_data import merge_gold_and_distilled
stats = merge_gold_and_distilled(
    'training_data/shortcutdsl_train_expanded.jsonl',
    'training_data/distilled_chat.jsonl',
    'training_data/merged_train.jsonl',
    verbose=True
)
print(f'  Merged {stats[\"merged_count\"]} examples')
"

echo ""
echo "=== SASHIMI MODE COMPLETE — $(date) ==="
echo "  Output: training_data/merged_train.jsonl"
