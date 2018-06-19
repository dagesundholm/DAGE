#!/bin/bash

# Wrapper for sbatch
# Names the output files in a systematic way

input=${1:-$HOME/chlorophyll.xyz}
step=${2:-0.2}
lmax=${3:-8}
maxlevel=${4:-2}
for i in {1..4}; do shift; done
SLURMargs="$@"
output=$PWD/gbfmm_$(basename $input)_${step}_${lmax}_${maxlevel}.%j

sbatch ${SLURMargs} -o${output} -e${output} <<EOL
#!/bin/bash
#SBATCH -N 1
#SBATCH -J gbfmm
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH

module purge
module load gcc/4.8.2 cuda/6.0 mkl/11.1.1 intelmpi
module list

set -xe

cd ${SLURM_SUBMIT_DIR:-.}
pwd

# Infiniband communication
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export I_MPI_FABRICS=shm:ofa
srun --mpi=mvapich $USERAPPL/gbfmm.x $input $step $lmax $maxlevel
EOL

exit

