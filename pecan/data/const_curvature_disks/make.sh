for n in 64 128 256; do
  for k in `seq -2 0.1 2.1`; do
    poetry run python ../../condensation.py -n $n -K $k -d 'const_curvature_disk' -o '.';
  done
done