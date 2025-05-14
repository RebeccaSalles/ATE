#!/bin/bash

#INFLUDB CONFIG
influxdb_host="localhost"
influxdb_port=8086
influxdb_login=""
influxdb_password=""
influxdb_db="test"


#DATA FOLDER
source_pwd="/home/user/data/"
#CURRENT EXECUTION FOLDER
exec_folder="/home/user/runnerTest"
#RESULT FOLDER
res_pwd="$exec_folder/user/results/"

mkdir -p $res_pwd


declare -a models=("cblof" "copod" "deepsvdd" "ecod" "hbos" "iforest" "knn" "loda" "mcd" "pca" "knn" "loda" "mcd" "pca")


cmd=main-xp1.py
for entry in `ls $source_pwd`; do
    echo $entry 
    for model in ${models[@]}; do
       echo $model
       NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1  CUDA_VISIBLE_DEVICES= python3.9 $exec_folder/$cmd --source $source_pwd$entry --hostinflux $influxdb_host --portinflux $influxdb_port --logininflux $influxdb_login --passwordinflux $influxdb_password --dbinflux $influxdb_db --result_folder $res_pwd --result $model-$entry --result2 G-$model-$entry  --n 1  --models $model
    done
done

