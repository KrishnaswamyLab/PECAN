rm /tmp/mag.txt
echo > /tmp/mag.txt

for i in `seq 20`; do
  echo $i
  python condensation.py -n 128 -c CalculateMagnitude -d annulus -o test.npz --force
  python extract_data.py test.npz >> /tmp/mag.txt
done
