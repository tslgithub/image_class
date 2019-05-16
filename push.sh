echo "push origin master"
echo "input file name"
read FILE
git add $FILE
git commit -m "update $FILE"
git push origin master
