#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu
#SBATCH --job-name=ParReg
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=amh9750@nyu.edu
#SBATCH --output=partisan_bert_32batch_3epoch_2e5lr_1wd_%j.out

module purge

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

singularity exec $nv \
  --overlay /scratch/amh9750/LLM_env/overlay-50G-10M.ext3:ro \
  /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; python /scratch/amh9750/capstone/capstone_bert_ft/partisan_bert_32batch_3epoch_2e5lr_1wd.py"
