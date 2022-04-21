#!/bin/bash -l
#SBATCH --job-name=bootstrap-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=ALL
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50GB

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
SCRIPT=bootstrap.py
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

cp $MYGROUP/MAPDN/data_preparation/bootstrap.py . 
cp $MYGROUP/input_data/1642-load-pv-reduced.csv .
cp $MYGROUP/input_data/2335-load-pv-reduced.csv .
cp $MYGROUP/input_data/2361-load-pv-reduced.csv .
cp $MYGROUP/input_data/2818-load-pv-reduced.csv .
cp $MYGROUP/input_data/3039-load-pv-reduced.csv .
cp $MYGROUP/input_data/3456-load-pv-reduced.csv .
cp $MYGROUP/input_data/3538-load-pv-reduced.csv .
cp $MYGROUP/input_data/4031-load-pv-reduced.csv .
cp $MYGROUP/input_data/4373-load-pv-reduced.csv .
cp $MYGROUP/input_data/4767-load-pv-reduced.csv .
cp $MYGROUP/input_data/5746-load-pv-reduced.csv .
cp $MYGROUP/input_data/6139-load-pv-reduced.csv .
cp $MYGROUP/input_data/661-load-pv-reduced.csv .
cp $MYGROUP/input_data/7536-load-pv-reduced.csv .
cp $MYGROUP/input_data/7719-load-pv-reduced.csv .
cp $MYGROUP/input_data/7800-load-pv-reduced.csv .
cp $MYGROUP/input_data/7901-load-pv-reduced.csv .
cp $MYGROUP/input_data/7951-load-pv-reduced.csv .
cp $MYGROUP/input_data/8156-load-pv-reduced.csv .
cp $MYGROUP/input_data/8386-load-pv-reduced.csv .
cp $MYGROUP/input_data/8565-load-pv-reduced.csv .
cp $MYGROUP/input_data/9019-load-pv-reduced.csv .
cp $MYGROUP/input_data/9160-load-pv-reduced.csv .
cp $MYGROUP/input_data/9278-load-pv-reduced.csv .
cp $MYGROUP/input_data/9922-load-pv-reduced.csv .


# we can delete the line below. It just shows the contents of
# the /scratch... directory before running Python.
ls -al

# now run our Python script file
# we can't just call main at the bottom of the file as that causes recursion
# instead we just import main to kick it off
python ${SCRIPT}


# remove all the symbolic link files (we don't need them)
# (the actual files that they point to are not affected by this
# removal)
cd ${SCRATCH}
/bin/rm *load-pv-reduced.csv

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


