#!/bin/bash
#SBATCH --account=hai_ricci
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH -J PECAN_TEST
#SBATCH --output=/p/project/hai_ricci/wayland1/aidos/PECAN/pecan/scripts/logs/pecan%j.out
#SBATCH --error=/p/project/hai_ricci/wayland1/aidos/PECAN/pecan/scripts/logs/pecan%j.out
#SBATCH --partition=booster
#SBATCH --time=13:00:00
#SBATCH --gres=gpu:4

source ~/.bashrc
module load git

cd $PECAN/pecan
echo $pwd
conda activate pecan3.9
python condensation.py -n 256 -d double_annulus -o double_annulus.npz