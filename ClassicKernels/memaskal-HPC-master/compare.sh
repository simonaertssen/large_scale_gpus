#!/bin/bash
print=0

now=$(date +"%d_%m_%Y_%H:%M:%S")
filename=./tests/$now.txt

echo "Running comparissons"

echo "Rows = Cols" >> $filename
for i in {500..18500..1000}
do 
	rows=$i
	cols=$i
	echo "Calculating for $rows x $cols";
	for j in {1..3}
	do
		echo " * Erwthma $j";
		./erwt$j.out $rows $cols $print | grep "Time elapsed:" | awk '{printf "%f;",$3}' >> $filename
	done
	echo "" >> $filename
done

sleep 5

echo "Rows = Cols/2" >> $filename
for i in {500..20500..1000}
do 
	rows=$(($i/2))
	cols=$i
	echo "Calculating for $rows x $cols";
	for j in {1..3}
	do
		echo " * Erwthma $j";
		./erwt$j.out $rows $cols $print | grep "Time elapsed:" | awk '{printf "%f;",$3}' >> $filename
	done
	echo "" >> $filename
done

sleep 5

echo "Rows/2 = Cols" >> $filename
for i in {500..20500..1000}
do 
	rows=$i
	cols=$(($i/2))
	echo "Calculating for $rows x $cols";
	for j in {1..3}
	do
		echo " * Erwthma $j";
		./erwt$j.out $rows $cols $print | grep "Time elapsed:" | awk '{printf "%f;",$3}' >> $filename
	done
	echo "" >> $filename
done

