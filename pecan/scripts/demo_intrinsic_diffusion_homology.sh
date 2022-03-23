DATASET=$1
N_POINTS=$2
KERNEL=gaussian
SEED=23
SCRIPT=analyse_merge_thresholds.py

poetry run python ${SCRIPT}       --kernel ${KERNEL}            \
                                  --data ${DATASET}             \
                                  -s ${SEED}                    \
                                  -n ${N_POINTS}                \
                                  -o /tmp/test.npz --force

poetry run python analyse_intrinsic_distance.py /tmp/test.npz
poetry run python calculate_intrinsic_diffusion_homology.py /tmp/test.npz
