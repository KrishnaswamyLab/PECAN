for n in 16 32 64 128 256 512; do
  for r in 0.01 0.25 0.50 0.75 0.99; do
    for i in `seq 5`; do 
      poetry run python ../../condensation.py -r $r -n $n -d 'annulus' -o '.'
    done
  done
done 
