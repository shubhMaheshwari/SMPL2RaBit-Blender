#!/bin/bash
### Script Usage: run.sh 
### Should run:
### 	- installation
### 	- genereate_smpl_motion_retargetting_experiments.py
###		- Run pinnochio 
###		- Run spider match 
###		- Evaluate 


## Generate experiments 

# Changing variables:
distortions=("" "decimate" "gaussian-noise")
shape_distortions=("False", "True")
motion_paths=("", "")

# Loop through each distortion
for distortion in "${distortions[@]}"
do
	echo "Retargetting for $distortion distortion:"
	
	echo "Running Command: python genereate_smpl_motion_retargetting_experiments.py --source SMPL --target SMPL --distortions $distortion --debug --gpu -f"

	python genereate_smpl_motion_retargetting_experiments.py --source SMPL --target SMPL --distortions $distortion --debug --gpu -f 
	
	# python genereate_smpl_motion_retargetting_experiments.py --source SMPL --target SMPL --distortions $distortion --debug --gpu -f --shape True  
	# python genereate_smpl_motion_retargetting_experiments.py --source SMPL --target SMPL --distortions $distortion --debug --gpu -f --pose "Example amass data"  
	# python genereate_smpl_motion_retargetting_experiments.py --source SMPL --target SMPL --distortions $distortion --debug --gpu -f --shape True --pose "Example amass data" 
	break
done
