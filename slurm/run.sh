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

PROJECT_DIR="/scratch/user/aaupadhy/college/projects/AdaptiveEvictLLMfScratch/"
cd $PROJECT_DIR
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

COMPUTE_NODE=$(hostname -s)
# echo "ssh -N -L 8787:${COMPUTE_NODE}:8787 aaupadhy@grace.hprc.tamu.edu"

source ~/.bashrc
conda activate ML

echo "Job started at $(date)"
echo "Running on ${COMPUTE_NODE}.grace.hprc.tamu.edu"
nvidia-smi

# torchrun \
#     --nnodes=1 \
#     --nproc_per_node=1 \
#     --master_port=$MASTER_PORT \
#     --master_addr="127.0.0.1" \
#     --node_rank=0 \
#     --max_restarts=0 \
#     --start_method=spawn \
#     main.py --mode all


python main.py --network_type llama
echo "Job finished at $(date)"