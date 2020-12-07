for f in $@; do
zip -m ${f%%.*}.zip $f
zip -m ${f%%.*}.zip --out ${f%%.*}_split.zip -s 20m
echo "Zipping $f file..."
done
