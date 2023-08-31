#!/usr/bin/env bash
#
# Create data sets for upcoming diffusion condensation publication. Feel
# free to add additional scenarios here, but check that each output name
# is unique---else, the script will overwrite everything.

# module load miniconda
# conda activate py39
# poetry shell

SEED=2021

# N_POINTS=488
N_POINTS=979
DATASETS=(swiss_roll_hole.txt)

KERNELS=(alpha box gaussian laplacian)

for DATASET in "${DATASETS[@]}"; do
  for KERNEL in "${KERNELS[@]}"; do
    echo "Running condensation for $DATASET with '$KERNEL' kernel..."
    # if [[ $DATASET = "double_annulus" ]]; then
      poetry run python ../pecan/condensation.py --kernel ${KERNEL}                                         \
                                        --data ${DATASET}                                          \
                                        -s ${SEED}                                                 \
                                        -n ${N_POINTS}                                             \
                                        -c CalculateDiffusionHomology CalculatePersistentHomology CalculateLawvereHomology  \
                                        -o ${DATASET}_${KERNEL}_n${N_POINTS}.npz #\
                                        # -e 1

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
