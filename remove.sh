#!/bin/bash
# remove.sh

######################################################
##########        REMOVE BASH      ###################
######################################################


# This bash is executable only for deleting all dataset 
# .mat and .npy files, to make possible to push the Github 
# project 

# Firstly we delete the datasets .mat files

hDatasetFile = C:\Users\ferna\Desktop\computerGeneratedHolography\output\dataset\output\dataset\hDataset.mat

echo "Start deleting .mat files..."
if [ -f "$hDatasetFile" ];then
	rm -r $hDatasetFile
	echo "Deleted hologram database!"
else
	echo "Hologram database does not exist!"
fi

echo "All files removed!"