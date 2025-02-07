#!/bin/bash
set -eoux pipefail

exp_dir=./exp/tree_search_comparison
mkdir -p $exp_dir

# Dataset and model configuration
limit=20 # limit number of questions to speed up if needed
dataset_args="++max_num_from_same_story=5 ++split=both ++human_experiments=8 ++limit=$limit"

# Define models
judge_model=""gpt-4-1106-preview""
debater_model=""gpt-4-1106-preview""

# Define tree search configurations to test
# Format: "root_branching:child_branching:depth"
tree_configs=(
    "2:2:2"   
)

# Main experiment loop
for tree_config in "${tree_configs[@]}"; do
    # Parse tree configuration
    IFS=':' read -r root_branching child_branching depth <<< "$tree_config"
    
    echo "Running experiment with:"
    echo "Judge: $judge_model"
    echo "Debater: $debater_model"
    echo "Tree config: root_branching=$root_branching, child_branching=$child_branching, depth=$depth"
    
    # Configure debate arguments
    debate_args="++correct_debater.language_model.model=$debater_model \
                ++incorrect_debater.language_model.model=$debater_model \
                ++correct_preference.language_model.model=$debater_model \
                ++incorrect_preference.language_model.model=$debater_model \
                ++correct_debater.tree_search.root_branching=$root_branching \
                ++incorrect_debater.tree_search.root_branching=$root_branching \
                ++correct_debater.tree_search.child_branching=$child_branching \
                ++incorrect_debater.tree_search.child_branching=$child_branching \
                ++correct_debater.tree_search.depth=$depth \
                ++incorrect_debater.tree_search.depth=$depth \
                ++correct_debater.tree_search.exploration_temp=0.8 \
                ++incorrect_debater.tree_search.exploration_temp=0.8"

    # Run debate
    curr_exp_dir="$exp_dir/tree_${root_branching}_${child_branching}_${depth}_model_${debater_model}"
    mkdir -p $curr_exp_dir
    python3 -m core.debate exp_dir="$curr_exp_dir" +experiment=debate $debate_args $dataset_args

    # Run judge
    judge_args="++judge.language_model.model=$judge_model ++judge_name=$judge_model"
    python3 -m core.judge exp_dir="$curr_exp_dir" +experiment=debate $judge_args

    # Score accuracy
    score_args="++judge_name=$judge_model"
    python3 -m core.scoring.accuracy exp_dir="$curr_exp_dir" +experiment=debate $score_args
done