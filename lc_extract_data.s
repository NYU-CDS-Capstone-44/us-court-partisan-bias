#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=ExtractLC
#SBATCH --mail-type=END
#SBATCH --mail-user=amh9750@nyu.edu
#SBATCH --output=slurm_%j.out
SRCDIR=$SCRATCH/capstone
VENVDIR=$SRCDIR/venv
source $VENVDIR/bin/activate
RUNDIR=$SCRATCH/amh9750/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR
DATADIR=$VAST/amh9750
export DATADIR
cp $SRCDIR/lc_extract_data.py $RUNDIR
cd $RUNDIR
python3 lc_extract_data.py
deactivate
