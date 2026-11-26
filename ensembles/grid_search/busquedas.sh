#!/bin/bash
#SBATCH --job-name=grid-searchs-trees
#SBATCH -p short              
#SBATCH --array=0-11%4                  
#SBATCH --cpus-per-task=4              
#SBATCH --mem=4G                      
#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1                
#SBATCH --output=logs/%x_%A_%a.out      # %A = id del array, %a = índice de tarea
#SBATCH --error=logs/%x_%A_%a.err
##SBATCH --mail-type=END,FAIL           # descomenta si tu cluster envía correo
##SBATCH --mail-user=tu@correo.com

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "Start:  $(date)"

source /hpcfs/apps/conda4.12.0/etc/profile.d/conda.sh    
conda activate gs


export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd $HOME/ML_Tarea
echo "Tarea $SLURM_ARRAY_TASK_ID en $(hostname), $SLURM_CPUS_PER_TASK cores"
python -u run_one.py $SLURM_ARRAY_TASK_ID

echo "End:    $(date)"

