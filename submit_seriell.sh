#!/bin/bash -l
#
# allocate 1 nodes (4 CPUs) for 3 hours
#PBS -l nodes=1:ppn=4:gtx1080ti,walltime=24:00:00
#
#Mail bei abbruch
#PBS -m a
# job name
#PBS -N CNN_Training
# stdout and stderr files
#PBS -o /home/vault/capm/sn0515/PhD/Th_U-Wire/logs/${PBS_JOBID}.out
#PBS -e /home/vault/capm/sn0515/PhD/Th_U-Wire/logs/${PBS_JOBID}.err
#
# first nosn-empty non-comment line ends PBS options

# jobs always start in $HOME -
cd /home/vault/capm/sn0515/PhD/Th_U-Wire/


# run
echo -e -n "Start:\t" && date

module load python/2.7-anaconda
module load root/6.08.02

echo "/home/vault/capm/sn0515/PhD/Th_U-Wire/wrapper.sh $config"
/home/vault/capm/sn0515/PhD/Th_U-Wire/wrapper.sh $config

wait

echo -e -n "Ende:\t" && date 

qdel ${PBS_JOBID}
