rm /tmp/mag.txt
echo > /tmp/mag.txt

for i in `seq 20`; do
  echo $i
  python condensation.py -n $(expr $i + 128) -c CalculateMagnitude -d petals -o test.npz --force
  python extract_data.py test.npz >> /tmp/mag.txt
done
