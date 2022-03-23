DATASET=$1
N_POINTS=$2
KERNEL=gaussian
SEED=23

poetry run python condensation.py --kernel ${KERNEL}            \
                                  --data ${DATASET}             \
                                  -s ${SEED}                    \
                                  -n ${N_POINTS}                \
                                  -c CalculateDiffusionHomology \
                                  -o /tmp/test.npz --force

poetry run python analyse_intrinsic_distance.py /tmp/test.npz
poetry run python calculate_intrinsic_diffusion_homology.py /tmp/test.npz
