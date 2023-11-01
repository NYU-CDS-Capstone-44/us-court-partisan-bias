#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=DecompressFile
#SBATCH --mail-type=END
#SBATCH --mail-user=amr10211@nyu.edu
#SBATCH --output=slurm_%j.out
SRCDIR=$VAST/
RUNDIR=$VAST/amr10211/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR
DATADIR=$VAST/amr10211
export DATADIR
cp $SRCDIR/decompress.py $RUNDIR
cd $RUNDIR
python3 decompress.py

