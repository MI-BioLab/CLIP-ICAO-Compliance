#!/bin/bash

# The first argument is the path to the TONO dataset root
if [ $# -ne 1 ]; then
    echo "Usage: $0 <path_to_tono_dataset_root>"
    exit 1
fi

TONO_ROOT="$1"

# Common parameters
common="--split-file $TONO_ROOT/train.txt --n-generations 50 --n-prompts 100 --keep-top-k-prompts 10 --exploitation-k-prompts 20 --exploration-ratio 0.5 --spare-top-k --llm-model-path models/phi-4-q4.gguf --device cuda:0"

model_variants=("clip_iqa" "oai_large14_336")
model_names=("clip-iqa" "openai/clip-vit-large-patch14-336")

mkdir -p history

# Iterate over arrays using indices
for i in "${!model_variants[@]}"; do
    variant="${model_variants[$i]}"
    model="${model_names[$i]}"
    # Iterate over each group
    for group in cap la ceg mkup sm sun tq expos oof sat bkg light pixel poster; do
        compliant="$TONO_ROOT/icao"
        seed="tono_prompts/seed_prompts/${group}.txt"
        requirement="tono_prompts/icao_requirements/${group}.txt"
        if [[ "$group" == "la" ]]; then
            non_compliant="$TONO_ROOT/la_1 $TONO_ROOT/la_2"
        else
            non_compliant="$TONO_ROOT/${group}"
        fi

        seed_export_history="history/${variant}_${group}_seed_history.csv"
        req_export_history="history/${variant}_${group}_req_history.csv"
        seed_stdout_file="history/${variant}_${group}_seed.out"
        req_stdout_file="history/${variant}_${group}_req.out"
        seed_stderr_file="history/${variant}_${group}_seed.err"
        req_stderr_file="history/${variant}_${group}_req.err"

        if [[ ! -f "$seed_export_history" ]]; then
            echo "Running prompt learning with model $model for group $group, using seed prompts"
            python prompt_learning.py --seed-prompts-file "$seed" --compliant-roots "$compliant" --non-compliant-roots $non_compliant $common --export-history "$seed_export_history" --model "$model" > "$seed_stdout_file" 2> "$seed_stderr_file"
        else
            echo "Skipping prompt learning with model $model for group $group, using seed prompts (already run)"
        fi
        if [[ ! -f "$req_export_history" ]]; then
            echo "Running prompt learning with model $model for group $group, using ICAO requirement"
            python prompt_learning.py --requirement-file "$requirement" --compliant-roots "$compliant" --non-compliant-roots $non_compliant $common --export-history "$req_export_history" --model "$model" > "$req_stdout_file" 2> "$req_stderr_file"
        else
            echo "Skipping prompt learning with model $model for group $group, using ICAO requirement (already run)"
        fi
    done
done
