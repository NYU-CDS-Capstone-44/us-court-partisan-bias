#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu
#SBATCH --job-name=TestRNNLSTM
#SBATCH --mail-type=END
#SBATCH --mail-user=amr10211@nyu.edu
#SBATCH --output=RNNLSTM_test_topic_results_%j.out

module purge

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

singularity exec $nv \
  --overlay /scratch/amr10211/LLM_env/overlay-50G-10M.ext3:ro \
  /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; python /home/amr10211/us-court-partisan-bias/RNN_LSTM_test_topic.py"

