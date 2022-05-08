#!/bin/bash -l
#SBATCH --job-name=rhee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=ALL
#SBATCH --partition=day
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=10GB

start_time=$( date )

# To configure GNU Environment for Mothur
module load Anaconda3/2021.05 

# Activate my conda env
conda activate mapdn

# list the environment loaded by the modules.
# Can remove the two lines below if you want.
# module list
# conda list

# Note: SLURM_JOBID is a unique number for every job.
# These are generic variables.

# Below is the Python file that would be run. Replace
# lab05-sample.py by your own file name.
SCRIPT=process_input_data.py
SCRIPT_PATH=$MYGROUP/MAPDN/data_preparation/$SCRIPT 

SCRATCH=$MYSCRATCH/run_conda/$SLURM_JOBID
RESULTS=$MYGROUP/conda_results

###############################################
# Creates a unique directory in the SCRATCH directory for this job to run in.
if [ ! -d $SCRATCH ]; then 
    mkdir -p $SCRATCH 
fi 
echo Working SCRATCH directory is $SCRATCH

###############################################
# Creates a unique directory in your GROUP directory for the results of this job
if [ ! -d $RESULTS ]; then 
     mkdir -p $RESULTS
fi 
echo Results will be stored in $RESULTS/$SLURM_JOBID

#############################################
# Rather than copying the large CIFAR-10 batch files to the $SCRATCH directory
# (where the Python file will be run), we create symbolic links to the data files
# in that directory.

cd ${SLURM_SUBMIT_DIR}
echo "SLURM_SUBMIT_DIR is"
echo ${SLURM_SUBMIT_DIR}

# copy the mothur analysis script to SCRATCH
cp ${SCRIPT_PATH} ${SCRATCH}

# go to the /scratch... directory and create symbolic links to the
# files for the input data
cd ${SCRATCH}

cp -r $MYGROUP/MAPDN/data_preparation .
cd data_preparation

# we can delete the line below. It just shows the contents of
# the /scratch... directory before running Python.
ls -al

# now run our Python script file
# we can't just call main at the bottom of the file as that causes recursion
# instead we just import main to kick it off
python3 rhee.py


# remove all the symbolic link files (we don't need them)
# (the actual files that they point to are not affected by this
# removal)
/bin/rm *load-pv.csv

#############################################
# Now move the output produced by our Python script from
# the /scratch... directory to my home directory.
cd $HOME
mv ${SCRATCH} ${RESULTS}

echo "mv ${SCRATCH} ${RESULTS}"
echo "Please see the ${RESULTS} directory for any output"

echo
echo "Mothur MPI job started  at $start_time"
echo "Mothur MPI job finished at `date`"


