#!/bin/sh
#$ -S /bin/sh
#$ -cwd
#$ -V
#$ -m e
#$ -M ericmjl@mit.edu
#$ -pe whole_nodes 1
#$ -l mem_free=60G
#############################################

python train_all_graphs.py 20000
