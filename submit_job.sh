#!/bin/bash 
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#
# TODO:
#   - Set name of the job below changing "Test" value.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter.
#   - Select the partition (queue) you want to run the job in:
#     - short : For jobs that have maximum run time of 120 mins. Has higher priority.
#     - long  : For jobs that have maximum run time of 7 days. Lower priority than short.
#     - longer: For testing purposes, queue has 31 days limit but only 3 nodes.
#   - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input and output file names below.
#   - If you do not want mail please remove the line that has --mail-type
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch examle_submit.sh

# -= Resources =-
#
#SBATCH --job-name=Vectornet-BGUNDUZ21
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=mid
#SBATCH --qos=users
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:00:00
#SBATCH --output=vectornet.out
#SBATCH --gres=gpu:tesla_v100:1

################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################



# ## Load Python 3.6.3
# echo "Activating Python 3.6.3..."
# module load python/3.6.3

# echo "Activating Cuda 11.4..."
# moddule load cuda/11.4

## Load GCC-7.2.1
# echo "Activating GCC-7.2.1..."
# module load gcc/7.2.1

echo ""
echo "======================================================================================"
env
echo "======================================================================================"
echo ""

# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

echo "Running Job...!"
echo "==============================================================================="
#Activate conda
echo "Activating Conda Env"
module load anaconda/3.21.05
conda init bash
conda activate vectornet-reimpl
source activate vectornet-reimpl
echo "Python version"
python -c 'import sys; print(sys.version_info[:])'

echo "Running Python script..."
which python3
python3 train.py
