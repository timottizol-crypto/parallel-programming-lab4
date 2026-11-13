#!/bin/bash
#SBATCH --job-name=cuda_lab
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=batch

module load cuda/8.0
cd $SLURM_SUBMIT_DIR

echo "N,Block,Mode,Time_ms" > cuda_results.csv

for N in 256 512 1024 2048; do
    if [ ! -f A_${N}.txt ]; then
        python3 generate_matrices.py $N
        mv A.txt A_${N}.txt
        mv B.txt B_${N}.txt
    fi
    
    for B in 8 16 32; do
        ./main_cuda $N A_${N}.txt B_${N}.txt $B tiled | awk -F'[=, ]+' '{print $2","$4","$6","$8}' >> cuda_results.csv
        if [ "$B" -eq 16 ]; then
            ./main_cuda $N A_${N}.txt B_${N}.txt $B naive | awk -F'[=, ]+' '{print $2","$4","$6","$8}' >> cuda_results.csv
        fi
    done
done