#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1  
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=waltermayor@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --array=1

echo "loding modules"
module load python/3.10
nvidia-smi

#alg=${1:-'dqn'}

cd $HOME/scratch/johan-s/purejaxql
echo "loding env"
source .venv-pqn/bin/activate

cd $HOME/scratch/johan-s/drl
#python ppo_rnn.py
python ppo.py

#if [ "$alg" = "pqn-rnn" ]; then
#    else
#    fi



