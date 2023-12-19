#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu
#SBATCH --job-name=SCTest
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=amh9750@nyu.edu
#SBATCH --output=sc_test_set_%j.out

module purge

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

singularity exec $nv \
  --overlay /scratch/amh9750/LLM_env/overlay-50G-10M.ext3:ro \
  /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; python /scratch/amh9750/capstone/bert_inference/sc_test_set_legal_bert.py"
