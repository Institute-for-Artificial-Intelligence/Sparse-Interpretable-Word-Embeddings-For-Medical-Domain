#!/bin/bash



mkdir ./spow_result_vectors

lambda=("0.5" "1" "0.1")


for i in "${lambda[@]}" ; do
echo 'Coef lambda:' $i
./sparse-coding/sparse.o /home/server01/workspace/pmc/Skip_Gram/sg_pmc_300_min20_win16_20k.txt 10 $i 1e-5 4 ./spow_result_vectors/spow_sg_h1000_$i.txt
done

