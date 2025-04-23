#!/bin/bash
#SBATCH --job-name=adaptiveevict
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60G
#SBATCH --time=15:00:00

module purge
module load WebProxy
export http_proxy=http://10.73.132.63:8080
export https_proxy=http://10.73.132.63:8080

PROJECT_DIR="/scratch/user/ajayjagan2511/AdaptiveEvictLLMfScratch/"
cd $PROJECT_DIR
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

source ~/.bashrc
conda activate ISR

echo "Job started at $(date)"
echo "Running on $(hostname -s).grace.hprc.tamu.edu"
nvidia-smi


echo "Training RL agent..."
python train_rl_agent.py \
    --training_data_path data/rl_training_tiny.json \
    --vocab_size 22369 \
    --embed_dim 256 \
    --max_seq_len 256 \
    --n_layers 8 \
    --n_heads 8 \
    --forward_mul 4 \
    --max_primary_size 1024 \
    --max_secondary_size 2048 \
    --lambda_cost 0.1 \
    --semantic_weight 0.3 \
    --cache_miss_penalty 0.5 \
    --perplexity_weight 0.2 \
    --attention_weight 0.3 \
    --gradient_weight 0.2 \
    --hidden_dim 256 \
    --lr 3e-4 \
    --gamma 0.99 \
    --tau 0.005 \
    --alpha 0.2 \
    --num_episodes 1000 \
    --max_steps 1000 \
    --batch_size 64 \
    --save_interval 100

echo "Job finished at $(date)"
