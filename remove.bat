@ECHO OFF

ECHO ###########################################################
ECHO ################## REMOVE DATASETS ########################
ECHO ###########################################################

:: This bat is executable only for deleting all dataset .mat and
:: .npy files, to make possible to push the Github project

SET hDataset_file=C:\Users\ferna\Desktop\computerGeneratedHolography\output\dataset\hDataset.mat
SET rDataset_file=C:\Users\ferna\Desktop\computerGeneratedHolography\output\dataset\rDataset.mat
SET wd_results_file=C:\Users\ferna\Desktop\computerGeneratedHolography\wd_results.npy

ECHO This batch fill will delete the datasets (.mat and .npy)

SET /p delDatasets=Delete datasets [y/n]?: 

IF "%delDatasets%" == "n" (
	ECHO Closing batch file...
	PAUSE
	EXIT
)

IF "%delDatasets%" == "y" (
	ECHO Start deleting .mat files...
)

IF EXIST %hDataset_file% (
	DEL %hDataset_file%
	ECHO Deleted hologram database!
) ELSE (
	ECHO Hologram database does not exist!
)

IF EXIST %rDataset_file% (
	DEL %rDataset_file%
	ECHO Deleted reconstructed images database!
) ELSE (
	ECHO Reconstructed images database does not exist!
)

ECHO Start deleting .npy files

IF EXIST %wd_results_file% (
	DEL %wd_results_file%
	ECHO Deleted wigner distributions database!
) ELSE (
	ECHO Wigner distributions database does not exist!
)

ECHO All files removed!

ECHO ###########################################################

PAUSE