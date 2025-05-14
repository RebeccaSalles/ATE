import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import atelib
import argparse
import tensorflow as tf
import time
import socket
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

parser = argparse.ArgumentParser()

parser.add_argument("--nodes",
                    help="Size number of nodes : 1,2,5,10,20,30,40,50,60,70,80,160,320",
                    nargs='+',
                    type=int,
                    default=[1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 160, 320])

parser.add_argument("--nb",
                    help="Size number of values : 300000, 500000, 1000000, 3000000,5000000, 10000000",
                    nargs='+',
                    type=int,
                    default=[300000, 500000, 1000000, 3000000, 5000000, 10000000])

parser.add_argument("--headers",
                    help="Columns of csv : global_time, precision, recall, f1, aucroc, aucpr, diff, tn, fp, fn, tp",
                    nargs='+',
                    default=["global_time", "split", "windows_time", "fit_time", "aggr_time", "acc_fit_time",
                             "acc_pred_time",
                             "acc_df_time", "precision", "recall", "f1", "aucroc", "aucpr",
                             "diff", "tn", "fp", "fn", "tp"])

parser.add_argument("--models",
                    help="Models to run: hbos, iforest, ocsvm, cblof, copod, ecod, featuringbagging, knn, loda, lof, mcd, pca, cof, sod, sogaal, mogaal, deepsvdd,deepsvdd_ae,vae,ae",
                    nargs='+',
                    default=["copod", "deepsvdd", "ecod", "pca", "hbos", "iforest", "knn", "loda", "mcd", "cblof"])

parser.add_argument("--n", help="Number of run", type=int, default=1)
parser.add_argument("--limit", help="limit", type=int, default=-1)
parser.add_argument("--result_folder", help="Result folder name", type=str, default="./results/")
parser.add_argument("--result", help="Result file name", type=str, default="results.csv")
parser.add_argument("--result2", help="Global Result file name", type=str, default="global-results.csv")
parser.add_argument("--source", help="Source file name", type=str,
                    default="../data/GutenTAG/cbf-channels-single-of-2/test.csv")
                    
parser.add_argument("--hostinflux", help="Host of influxDB server", type=str,
                    default="localhost")
parser.add_argument("--portinflux", help="Port of influxDB server", type=int,
                    default=8086)
parser.add_argument("--logininflux", help="Login of influxDB server", type=str,
                    default="")
parser.add_argument("--passwordinflux", help="Password of influxDB server", type=str,
                    default="")                                                                                
parser.add_argument("--dbinflux", help="DB of influxDB server", type=str,
                    default="test")
                    
args = parser.parse_args()
source_filename = args.source

res_filename = args.result
res_filename2 = args.result2
result_folder = args.result_folder

limit = args.limit

headers = args.headers 
listOfModels = args.models
n = args.n

nbElements = args.nb
nodes = args.nodes

GutenTAG_1 = pd.read_csv(source_filename)
data_x = GutenTAG_1.drop(["timestamp", "is_anomaly"], axis=1)
data_y = GutenTAG_1.is_anomaly
X_VALUES = data_x.to_numpy()
Y_ANOMALY = data_y.to_numpy()

atelib.writeCsvHeader2(result_folder + "/" + res_filename, headers)
atelib.writeCsvHeader2(result_folder + "/" + res_filename2, headers)

header = False

xp=1
for model in listOfModels:
    for i in range(n):
        print("run XP : ", xp)
        print("run: ", i + 1, " of ", n, " xp: ", xp, " nodes :","NA", " windows: ",nbElements," split: ","NA")
        tags = {
            "host":socket.gethostname(),
            "run": str(i + 1),
            "n": str(n),
            "xp": str(xp),
            "model": model,
            "windows": nbElements,
            "nodes": "NA",
            "split":"NA",
            "file": res_filename
        }
        atelib.writeInit(tags,args.hostinflux,args.portinflux,args.logininflux,args.passwordinflux,args.dbinflux)
        atelib.writeStart(tags,args.hostinflux,args.portinflux,args.logininflux,args.passwordinflux,args.dbinflux)
        res = atelib.executeXP1(model, X_VALUES, Y_ANOMALY)
        atelib.writeCsvLine(result_folder + "/" + res_filename, i, xp, model, "", "NA",nbElements, res,
                          headers)
        atelib.writeStop(tags,args.hostinflux,args.portinflux,args.logininflux,args.passwordinflux,args.dbinflux)
        atelib.writeInit(tags,args.hostinflux,args.portinflux,args.logininflux,args.passwordinflux,args.dbinflux)
        time.sleep(15)
