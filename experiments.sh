#!/bin/bash
# pythia_70m_parameter_scaling.sh
# Run pruned experimental configurations testing each parameter separately

set -e

echo "🔬 PYTHIA-70M PARAMETER SCALING EXPERIMENTS"
echo "============================================"
echo "⏰ Start: $(date '+%H:%M:%S')"
echo "🎯 Running 10 experiments (3+3+3+1 baseline)"
echo "📊 Testing each parameter separately for efficiency"
echo "⏱️  Each takes ~15-20 minutes"
echo "📊 Total time: ~2.5-3.5 hours for 10 experiments"
echo "============================================"

mkdir -p logs/parameter_scaling

# Function to run experiment and wait for completion
run_experiment() {
    local config=$1
    local description=$2
    local timestamp=$(date '+%m%d_%H%M')
    local log_file="logs/parameter_scaling/${config}_${timestamp}.log"
    
    echo ""
    echo "🔄 STARTING: $description"
    echo "📋 Config: $config"
    echo "📝 Log: $log_file"
    echo "⏰ Start: $(date '+%H:%M:%S')"
    echo "================================"
    
    # Run experiment and wait for completion
    echo "🚀 Running: python train.py --config $config"
    EXP_RECOMPUTE_STD_ON_REAL=1 python train.py --config "$config" 2>&1 | tee "$log_file"
    
    echo "✅ COMPLETED: $description at $(date '+%H:%M:%S')"
    echo ""
}

# Parameter scaling experiments
echo "🚀 STARTING PARAMETER SCALING EXPERIMENTS"

echo ""
echo "📊 BASELINE EXPERIMENT"
echo "======================"
run_experiment "pythia-70m_exp_baseline" "Baseline: aux_loss=0.001, max_steps=200, gaps=4"

echo ""
echo "📊 AUX_LOSS_WEIGHT SCALING (keeping max_steps=200, gaps=4)"
echo "=========================================================="
run_experiment "pythia-70m_exp_aux_000005" "aux_loss_weight=0.00005 (vs baseline 0.001)"
run_experiment "pythia-70m_exp_aux_00001" "aux_loss_weight=0.0001 (vs baseline 0.001)"
run_experiment "pythia-70m_exp_aux_00005" "aux_loss_weight=0.0005 (vs baseline 0.001)"


echo ""
echo "📊 MAX_STEPS SCALING (keeping aux_loss=0.001, gaps=4)"
echo "====================================================="
run_experiment "pythia-70m_exp_steps_120" "max_steps=120 (vs baseline 200)"
run_experiment "pythia-70m_exp_steps_150" "max_steps=150 (vs baseline 200)"
run_experiment "pythia-70m_exp_steps_170" "max_steps=170 (vs baseline 200)"

echo ""
echo "📊 GAP SCALING (keeping aux_loss=0.001, max_steps=200)"
echo "======================================================"
run_experiment "pythia-70m_exp_gap_8" "gap_ln1qk/gap_ln1v=8 (vs baseline 4)"
run_experiment "pythia-70m_exp_gap_12" "gap_ln1qk/gap_ln1v=12 (vs baseline 4)"
run_experiment "pythia-70m_exp_gap_18" "gap_ln1qk/gap_ln1v=18 (vs baseline 4)"

echo ""
echo "🎯 ALL PARAMETER SCALING EXPERIMENTS COMPLETED!"
echo "==============================================="
echo "⏰ Finished: $(date '+%H:%M:%S')"
echo "📊 Results in: logs/parameter_scaling/"
echo ""
echo "🔍 ANALYSIS WORKFLOW:"
echo "===================="
echo "1. 📊 Compare aux_loss_weight effects:"
echo "   - Plot final loss vs aux_loss_weight [0.001, 0.2, 0.4, 0.5]"
echo "   - Analyze convergence stability"
echo ""
echo "2. ⏱️ Compare max_steps effects:"
echo "   - Plot final loss vs max_steps [120, 150, 170, 200]"
echo "   - Identify diminishing returns point"
echo ""
echo "3. 🔄 Compare gap effects:"
echo "   - Plot final loss vs gap_size [4, 8, 12, 18]"
echo "   - Analyze LayerNorm removal stability"
echo ""
echo "4. 🎯 Determine optimal parameters:"
echo "   - Identify best single parameter changes"
echo "   - Consider combining best-performing values"
echo ""
echo "📈 Next steps:"
echo "- If clear winners emerge, test optimal combination"
echo "- If parameters interact, run targeted 2-factor experiments"
echo ""
echo "🌅 Parameter scaling complete! Time to analyze results."