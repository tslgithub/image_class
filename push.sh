#!/usr/bin/env bash
#echo "push origin master"
#echo "input file name"
#read FILE
#git add $FILE
#git commit -m "update $FILE"
#git push origin master

echo "push update file: $1"
git add $1
git commit -m "update $1"
git push origin master

