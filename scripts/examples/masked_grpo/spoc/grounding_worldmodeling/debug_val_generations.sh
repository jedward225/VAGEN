#!/bin/bash

# Debug script for validation generation logging
echo "=== SPOC Validation Generation Debug ==="

# Check current training log for key patterns
EXPERIMENT_NAME="grounding_worldmodeling-spoc-masked_grpo"
LOG_FILE="$EXPERIMENT_NAME.log"

if [ -f "$LOG_FILE" ]; then
    echo "1. Checking for validation generation logging patterns in $LOG_FILE..."
    
    echo "--- WandB logger initialization ---"
    grep -n "wandb.*init\|logger.*wandb" "$LOG_FILE" | tail -5
    
    echo "--- Val generations to log setting ---"
    grep -n "val_generations_to_log_to_wandb" "$LOG_FILE" | tail -3
    
    echo "--- Validation debug messages ---"
    grep -n "validation at global step.*begins\|validation at global step.*ends" "$LOG_FILE" | tail -10
    
    echo "--- Response length stats ---"
    grep -n "response_length" "$LOG_FILE" | tail -5
    
    echo "--- Action validity stats ---"
    grep -n "action_is_valid\|action_is_effective" "$LOG_FILE" | tail -5
    
    echo "--- Any WandB validation table logs ---"
    grep -n "val/generations\|validation.*table\|wandb.*log" "$LOG_FILE" | tail -5
    
    echo "--- Recent validation metrics ---"
    grep -n "val/score\|val/success" "$LOG_FILE" | tail -5
    
else
    echo "Log file $LOG_FILE not found!"
fi

echo ""
echo "2. Checking WandB connection and project..."
python3 -c "
import wandb
try:
    # Check if we can access wandb
    wandb_run = wandb.Api().runs(path='vagen_spoc')
    print(f'Found {len(wandb_run)} runs in vagen_spoc project')
    
    # Get the latest run
    if wandb_run:
        latest_run = wandb_run[0]
        print(f'Latest run: {latest_run.name} - {latest_run.state}')
        
        # Check if val/generations exists
        summary = latest_run.summary
        if 'val/generations' in summary:
            print('✓ val/generations found in summary')
        else:
            print('✗ val/generations NOT found in summary')
            print('Available validation keys:', [k for k in summary.keys() if k.startswith('val/')])
    
except Exception as e:
    print(f'WandB connection error: {e}')
"

echo ""
echo "3. Testing SPOC environment config_id generation..."
python3 -c "
import sys
sys.path.append('/home/jed/VAGEN')
from vagen.env.spoc.env_config import SpocEnvConfig

config = SpocEnvConfig(
    task_type='FetchType',
    chores_split='train',
    render_mode='vision',
    max_actions_per_step=3
)

print(f'Config ID: {config.config_id()}')
print('✓ Config ID generation works')
"

echo ""
echo "4. Checking current tmux sessions..."
tmux list-sessions 2>/dev/null | grep spoc || echo "No SPOC tmux sessions found"

echo ""
echo "5. Quick validation generation test (if server is running)..."
python3 -c "
import requests
import sys

try:
    response = requests.get('http://localhost:5000/health', timeout=5)
    if response.status_code == 200:
        print('✓ SPOC server is responding on port 5000')
    else:
        print(f'✗ Server responded with status {response.status_code}')
except Exception as e:
    print(f'✗ Cannot connect to server: {e}')
"

echo ""
echo "=== Debug Summary ==="
echo "The most likely issues for missing val/generations:"
echo "1. Model generating only 1-token responses (check response_length in logs)"
echo "2. No valid actions being parsed (check action_is_valid in logs)"  
echo "3. WandB logger not properly initialized (check wandb init messages)"
echo "4. Recording data structure missing required fields"
echo ""
echo "Next steps:"
echo "- Restart training with the fixed SpocEnvConfig (added render_mode field)"
echo "- Check for increased response lengths and valid actions"
echo "- Monitor for val/generations appearing in WandB after next validation"