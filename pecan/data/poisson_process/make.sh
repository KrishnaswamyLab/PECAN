for n in 16 32 64 128 256 512; do
  for i in `seq 5`; do 
    poetry run python ../../condensation.py -d 'poisson_process' -n $n -o .
  done
done 
