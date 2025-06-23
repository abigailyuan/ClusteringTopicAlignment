#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Tue Jun 17 2025 17:52:27 GMT+1000 (Australian Eastern Standard Time)

# Partition for the job:
#SBATCH --partition=gpu-h100

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="bertopic_wsj"

# The project ID which this job should run under:
#SBATCH --account="punim2412"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Number of GPUs requested per node:
#SBATCH --gres=gpu:2
# The amount of memory in megabytes per node:
#SBATCH --mem=65536

# Use this email address:
#SBATCH --mail-user=meng.yuan@unimelb.edu.au

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=3-0:0:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load CUDA/12.2.0
module load foo/3.2-gcc-4.9.2

# The job command(s):
source venvs/liverag/bin/activate
python run_bertopic_force_sweep.py --collections wsj --start 50 --stop 100 --step 10 --train-script bertopic_train.py --output-dir Results/BERTOPIC



##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
