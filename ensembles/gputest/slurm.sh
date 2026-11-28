#!/bin/bash
#SBATCH --job-name=tabular-test

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=./log/%j.out
#SBATCH --error=./log/%j.err


echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"

module load cuda
nvidia-smi


cat /proc/self/cgroup

export TABPFN_TOKEN="tabpfn_sk_61TBAkdW3FNl93F_tAcRfL_qWrclpo8v12aRH1okghI"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4


micromamba run -n torch_gpu python -u fit.py

echo "End:    $(date)"


