#!/bin/bash

# 🎯 Agentic AI Video Object Tracking - Demo Script
# This script demonstrates the key capabilities of our system

echo "🚀 Agentic AI Video Object Tracking Demo"
echo "========================================"

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "capstone310" ]]; then
    echo "⚠️  Please activate conda environment first:"
    echo "   conda activate capstone310"
    exit 1
fi

# Demo 1: Successful tracking with MLLM system
echo ""
echo "📹 Demo 1: MLLM-Enhanced Intelligent Tracking"
echo "Target: Woman in white clothes dancing"
echo "Expected: High-quality tracking with optimal parameters"
echo ""
python agent_main_mllm.py \
    --video demo2.mp4 \
    --text "track the woman in white clothes dancing" \
    --output_dir ./demo1_mllm_success

# Demo 2: Traditional system comparison
echo ""
echo "📹 Demo 2: Traditional System (For Comparison)"
echo "Target: Same woman in white clothes dancing"
echo "Expected: Basic tracking with default parameters"
echo ""
python agent_main.py \
    --video demo2.mp4 \
    --text "track the woman in white clothes dancing" \
    --output_dir ./demo2_traditional

# Demo 3: Target not found handling
echo ""
echo "📹 Demo 3: Intelligent Target Detection"
echo "Target: Red car (doesn't exist in dance video)"
echo "Expected: System detects target absence and handles gracefully"
echo ""
python agent_main_mllm.py \
    --video demo2.mp4 \
    --text "track the red car" \
    --output_dir ./demo3_target_not_found

# Demo 4: Complex scene tracking
echo ""
echo "📹 Demo 4: Complex Scene with Multiple Similar Objects"
echo "Target: Man in white jacket"
echo "Expected: Precise target selection among similar objects"
echo ""
python agent_main_mllm.py \
    --video demo3.mp4 \
    --text "track the man wearing white jacket" \
    --output_dir ./demo4_complex_scene

echo ""
echo "✅ Demo completed! Check the output directories for results:"
echo "   - demo1_mllm_success/     : MLLM intelligent tracking"
echo "   - demo2_traditional/      : Traditional system comparison"
echo "   - demo3_target_not_found/ : Target detection handling"
echo "   - demo4_complex_scene/    : Complex scene tracking"
echo ""
echo "🎓 Key innovations demonstrated:"
echo "   ✅ Autonomous parameter optimization"
echo "   ✅ Multi-modal video analysis"
echo "   ✅ Intelligent target existence detection"
echo "   ✅ Natural language target description"
echo "   ✅ Adaptive tracking strategies"
