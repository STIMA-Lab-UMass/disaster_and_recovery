#!/bin/bash
#
#SBATCH --job-name=yemen_files
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zshah@umass.edu
#SBATCH --ntasks=2
#SBATCH -p longq
#SBATCH --time=06-01:00:00
#SBATCH --mem-per-cpu=25000
srun python timeseries_clustering.py