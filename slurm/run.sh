#!/bin/bash
#SBATCH --job-name=adaptiveevict
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=15:00:00

module purge
module load WebProxy
export http_proxy=http://10.73.132.63:8080
export https_proxy=http://10.73.132.63:8080

PROJECT_DIR="/scratch/user/sarveshgs22/AdaptiveEvictLLMfScratch/"
cd $PROJECT_DIR
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

COMPUTE_NODE=$(hostname -s)

source ~/.bashrc
conda activate ISR

echo "Job started at $(date)"
echo "Running on ${COMPUTE_NODE}.grace.hprc.tamu.edu"
nvidia-smi

python main.py --network_type llama --max_merged_tokens 5000 --embed_dim 256 --n_heads 8 --forward_mul 4 --n_layers 8 --batch_size 64 --train_tokens_len 64 --epochs 48 --warmup_epochs 5

echo "Job finished at $(date)"