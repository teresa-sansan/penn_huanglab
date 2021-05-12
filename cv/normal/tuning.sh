#!/bin/bash
#PBS -l nodes=1:ppn=2
#PBS -l walltime=12:00:00
#PBS -l pmem=20gb
#PBS -A yuh371_a_g_gc_default
#PBS -N tuning_overall

#

cd ~/work/project/not_conflict/cv
python tuning.py $1 $2 $3 $4
#echo python tuning.py $1 $2 $3 $4
