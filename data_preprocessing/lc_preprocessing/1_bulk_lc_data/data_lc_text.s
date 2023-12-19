#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=ExtractLC
#SBATCH --mail-type=END
#SBATCH --mail-user=amh9750@nyu.edu
#SBATCH --output=lowerdata_%j.out
SRCDIR=$SCRATCH/capstone
VENVDIR=$SRCDIR/venv
source $VENVDIR/bin/activate
RUNDIR=$SCRATCH/amh9750/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR
DATADIR=$VAST/amh9750
export DATADIR
cp $SRCDIR/data_lc_text.py $RUNDIR
cd $RUNDIR
python3 data_lc_text.py
deactivate

