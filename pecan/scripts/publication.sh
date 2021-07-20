#!/usr/bin/env bash
#
# Create data sets for upcoming diffusion condensation publication. Feel
# free to add additional scenarios here, but check that each output name
# is unique---else, the script will overwrite everything.

N_POINTS=128
DATASETS=(blobs hyperuniform_circle petals simplex)

KERNELS=(box gaussian laplacian)

for DATASET in "${DATASETS[@]}"; do
  for KERNEL in "${KERNELS[@]}"; do
    echo "Running condensation for $DATASET with '$KERNEL' kernel..."
    poetry run python condensation.py --kernel $KERNEL              \
                                      --data $DATASET               \
                                      -n ${N_POINTS}                \
                                      -c CalculateDiffusionHomology \
                                      -o data/publication/${DATASET}_${KERNEL}_n${N_POINTS}.npz 
  done
done
