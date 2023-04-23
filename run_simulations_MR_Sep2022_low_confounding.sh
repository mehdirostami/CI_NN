#!/bin/bash



for sim_id in {1..100} 
do  
	# python sim_compare_consistency_Sep2022.py $sim_id $1 7 750 32 0.25 0.25
	# python sim_compare_consistency_Sep2022.py $sim_id $1 5 2000 32 0.25 0.25
	# python sim_compare_consistency_Sep2022.py $sim_id $1 3 5000 32 0.25 0.25
	python3 sim_compare_consistency_Sep2022.py $sim_id $1 3 750 8 0.01 0.01 
	# python sim_compare_consistency_Sep2022.py $sim_id $1 3 7500 300 0.01 0.01
	# python sim_compare_consistency_Sep2022.py $sim_id $1 5 10000 32 0.25 0.25

done
