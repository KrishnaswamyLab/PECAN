#!/usr/bin/env bash
#
# Create data sets for upcoming diffusion condensation publication. Feel
# free to add additional scenarios here, but check that each output name
# is unique---else, the script will overwrite everything.

SEED=2021

N_POINTS=128
DATASETS=(barbell blobs double_annulus hyperuniform_circle petals simplex)

KERNELS=(alpha box gaussian laplacian)

for DATASET in "${DATASETS[@]}"; do
  for KERNEL in "${KERNELS[@]}"; do
    echo "Running condensation for $DATASET with '$KERNEL' kernel..."
    # if [[ $DATASET = "double_annulus" ]]; then
      poetry run python condensation.py --kernel ${KERNEL}                                         \
                                        --data ${DATASET}                                          \
                                        -s ${SEED}                                                 \
                                        -n ${N_POINTS}                                             \
                                        -c CalculateDiffusionHomology CalculatePersistentHomology CalculateLawvereHomology  \
                                        -o data/publication/${DATASET}_${KERNEL}_n${N_POINTS}.npz 

    # else
    #   poetry run python condensation.py --kernel ${KERNEL}            \
    #                                     --data ${DATASET}             \
    #                                     -s ${SEED}                    \
    #                                     -n ${N_POINTS}                \
    #                                     -c CalculateDiffusionHomology \
    #                                     -o data/publication/${DATASET}_${KERNEL}_n${N_POINTS}.npz 
    # fi
  done
done
