#!/bin/bash



mkdir ./result_vector

RL=("0.5" "1" "2")
PSL=("0.5" "1" "0.2")

for i in "${RL[@]}" ; do
echo 'Coef RL:' $i
python3 ./SPINE/code/model/main.py --input /home/server01/workspace/pmc/Skip_Gram/sg_pmc_300_min20_win16_20k.txt --num_epochs 5000 --denoising --noise 0.2 --sparsity 0.92 --output ./result_vector/pmc_sg_spine_h1000_rl_$i --hdim 1000 --rl $i --simloss 0
done

for i in "${PSL[@]}" ; do
echo 'Coef PSL:' $i
python3 ./SPINE/code/model/main.py --input /home/server01/workspace/pmc/Skip_Gram/sg_pmc_300_min20_win16_20k.txt --num_epochs 5000 --denoising --noise 0.2 --sparsity 0.92 --output ./result_vector/pmc_sg_spine_h1000_psl_$i --hdim 1000 --psl $i --simloss 0
done
