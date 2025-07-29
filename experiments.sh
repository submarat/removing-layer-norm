#!/bin/bash
# sequential_pythia_experiments.sh
# Run experiments ONE AT A TIME to avoid memory issues

set -e

echo "🔬 SEQUENTIAL PYTHIA EXPERIMENTS"
echo "================================"
echo "⏰ Start: $(date '+%H:%M:%S')"
echo "🎯 Running ONE experiment at a time"
echo "⏱️  Each takes ~30 minutes"
echo "📊 Total time: ~3 hours for 6 experiments"
echo "================================"

mkdir -p logs

# Function to run experiment and wait for completion
run_experiment() {
    local config=$1
    local env_vars=$2
    local description=$3
    local timestamp=$(date '+%m%d_%H%M')
    local log_file="logs/${config}_${timestamp}.log"
    
    echo ""
    echo "🔄 STARTING: $description"
    echo "📋 Config: $config"
    echo "🌍 Env: $env_vars"
    echo "📝 Log: $log_file"
    echo "⏰ Start: $(date '+%H:%M:%S')"
    echo "================================"
    
    # Run experiment and wait for completion
    if [ -n "$env_vars" ]; then
        echo "🚀 Running: $env_vars python train.py --config $config --mode without_ln"
        eval "$env_vars python train.py --config $config --mode without_ln" 2>&1 | tee "$log_file"
    else
        echo "🚀 Running: python train.py --config $config --mode without_ln"
        python train.py --config "$config" --mode without_ln 2>&1 | tee "$log_file"
    fi
    
    echo "✅ COMPLETED: $description at $(date '+%H:%M:%S')"
    echo ""
}

# Core experiments - most important first
echo "🚀 STARTING CORE EXPERIMENTS"

# 1. Test early vs late start (most important)
run_experiment "pythia-70m_start10" "" "Early start (step 10)"
run_experiment "pythia-70m_start100" "" "Late start (step 100)"

# 2. Test auxiliary loss effectiveness
run_experiment "pythia-70m_aux" "" "With auxiliary loss"

# 3. Test std recomputation
run_experiment "pythia-70m_start10" "EXP_RECOMPUTE_STD_ON_REAL=1" "Early start + std recomp"

# 4. Test medium start
run_experiment "pythia-70m_start50" "" "Medium start (step 50)"

# 5. Best combination candidate
run_experiment "pythia-70m_start10_aux" "" "Early start + aux loss"

echo ""
echo "🎯 ALL EXPERIMENTS COMPLETED!"
echo "=============================="
echo "⏰ Finished: $(date '+%H:%M:%S')"
echo "📊 Results in: logs/"
echo "🔍 Next: Analyze loss curves to find best parameters"
echo ""
echo "🌅 Good morning! Time to analyze results."