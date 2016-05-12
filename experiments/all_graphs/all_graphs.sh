#!/bin/sh
#$ -S /bin/sh
#$ -cwd
#$ -V
#$ -m e
#$ -M ericmjl@mit.edu
#$ -pe whole_nodes 1
#$ -l mem_free=70G
#############################################

python train_all_graphs.py 1000
