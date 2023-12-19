#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=2Chunk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=amh9750@nyu.edu
#SBATCH --output=Chunk_2_chunks_%j.out

module purge

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

singularity exec $nv \
  --overlay /scratch/amh9750/LLM_env/overlay-50G-10M.ext3:ro \
  /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; python /scratch/amh9750/capstone/final_lc_csv/chunk_2_chunks.py"
