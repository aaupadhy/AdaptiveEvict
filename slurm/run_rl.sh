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

PROJECT_DIR="/scratch/user/sarveshgs22/AdaptiveEvictLLMfScratch/"
cd $PROJECT_DIR
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

source ~/.bashrc
conda activate ISR

echo "Job started at $(date)"
echo "Running on $(hostname -s).grace.hprc.tamu.edu"
nvidia-smi

echo "Generating RL training data..."
python generate_rl_data.py \
    --num_prompts 1000 \
    --min_length 50 \
    --max_length 200 \
    --output_file data/rl_training_data.json \
    --model_path ./saved_models \
    --data_path data \
    --data_file train.txt \
    --network_type llama \
    --load_model \
    --load_tokenizer \
    --embed_dim 256 \
    --n_layers 8 \
    --n_heads 8 \
    --forward_mul 4 \
    --train_tokens_len 64 \
    --batch_size 64 \
    --dropout 0.1 \
    --n_workers 2 \
    --max_merged_tokens 5000

echo "Training RL agent..."
python train_rl_agent.py \
    --training_data_path data/rl_training_data.json \
    --vocab_size 6969 \
    --embed_dim 256 \
    --max_seq_len 64 \
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
