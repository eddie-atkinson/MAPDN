#!/bin/bash -l
#SBATCH --job-name=process-input-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=ALL
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100GB

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
SCRIPT=test.py


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


# go to the /scratch... directory and create symbolic links to the
# files for the input data
cd ${SCRATCH}
ln -s $MYGROUP/MAPDN/test.py
ln -s $MYGROUP/MAPDN/environments
ln -s $MYGROUP/MAPDN/args
ln -s $MYGROUP/MAPDN/trial
# we can delete the line below. It just shows the contents of
# the /scratch... directory before running Python.
ls -al

# now run our Python script file
python ./${SCRIPT} --save-path trial/model_save --alg maddpg --alias 0 --scenario J_50percent --voltage-barrier-type l1 --test-mode batch

# remove all the symbolic link files (we don't need them)
# (the actual files that they point to are not affected by this
# removal)

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


