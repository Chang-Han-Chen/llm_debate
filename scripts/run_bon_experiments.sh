#!/bin/bash
set -eoux pipefail

exp_dir=./exp/bon_comparison_claude3.5new
mkdir -p $exp_dir

# Dataset and model configuration
limit=47 # limit number of questions to speed up if needed
dataset_args="++max_num_from_same_story=5 ++split=both ++human_experiments=8 ++limit=$limit"

# Define models and BoN values to test
judge_models=("gpt-4-1106-preview" "claude-3-sonnet-20240229-v1")
debater_models=("gpt-4-1106-preview" "claude-3-sonnet-20240229-v1")

bon_values=(1 2 4 8)

# Create results file header
results_file="$exp_dir/results.csv"
echo "judge_model,debater_model,bon_value,accuracy" > $results_file

# Main experiment loop
for judge_model in "${judge_models[@]}"; do
    for debater_model in "${debater_models[@]}"; do
        for bon_value in "${bon_values[@]}"; do
            echo "Running experiment with:"
            echo "Judge: $judge_model"
            echo "Debater: $debater_model"
            echo "BoN: $bon_value"
            
            # Configure debate arguments
            debate_args="++correct_debater.language_model.model=$debater_model \
                        ++incorrect_debater.language_model.model=$debater_model \
                        ++correct_preference.language_model.model=$debater_model \
                        ++incorrect_preference.language_model.model=$debater_model \
                        ++correct_debater.BoN=$bon_value \
                        ++incorrect_debater.BoN=$bon_value \
                        ++correct_debater.language_model.temperature=0.8 \
                        ++incorrect_debater.language_model.temperature=0.8"

            # Create experiment-specific directory
            curr_exp_dir="$exp_dir/judge.${judge_model}/debater.${debater_model}/bon.${bon_value}"
            mkdir -p "$curr_exp_dir"

            # Run debate
            python3 -m core.debate exp_dir="$curr_exp_dir" +experiment=debate $debate_args $dataset_args

            # Run judge
            judge_args="++judge.language_model.model=$judge_model ++judge_name=$judge_model"
            python3 -m core.judge exp_dir="$curr_exp_dir" +experiment=debate $judge_args

            # Score accuracy
            score_args="++judge_name=$judge_model"
            python3 -m core.scoring.accuracy exp_dir="$curr_exp_dir" +experiment=debate $score_args

            # Extract accuracy and append to results file
            accuracy=$(tail -n 1 "$curr_exp_dir/debate_sim/accuracy.csv" | cut -d',' -f2)
            echo "$judge_model,$debater_model,$bon_value,$accuracy" >> $results_file
        done
    done
done