#!/bin/bash

#SBATCH --job-name=MCMC_BOSS_NERCOME_40
#SBATCH --mail-type=ALL             # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=M.J.Looijmans@sms.ed.ac.uk
#SBATCH --time=5-00:00:00
#SBATCH --output=%x.%j.out

source /usr/local/anaconda/3.9/etc/profile.d/conda.sh
conda activate nbodykit

python3 ../scripts/power-spectrum-likelihood-analysis-nercome.py 40
