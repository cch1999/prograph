import argparse
import torch as torch
import pandas as pd
from tqdm import tqdm
from graphein.construct_graphs import ProteinGraph
import csv
import sys

csv.field_size_limit(sys.maxsize)

def read_dict(path):
    'Reads Python dictionary stored in a csv file'
    dictionary = {}
    for key, val in csv.reader(open(path)):
        dictionary[key] = val
    return dictionary

partition = read_dict("partition_single.csv")

partition = partition["train"]
partition = partition[1:-1].split("', '")
print(partition)

pg = ProteinGraph(granularity='CA', insertions=False, keep_hets=True,
                  node_featuriser='meiler',
                  get_contacts_path='~/projects/repos/getcontacts',
                  pdb_dir='files/pdbs/',
                  contacts_dir='files/contacts/',
                  exclude_waters=True, covalent_bonds=False, include_ss=True)

exceptions = 0

df = pd.read_csv("labels_reduced.csv")

pdbs = df["pdb_id"].to_list()



for i in tqdm(range(df.shape[0])):

    pdb_id = df.loc[i, "pdb_id"]

    try:
        g = pg.dgl_graph_from_pdb_code(pdb_code=pdb_id)
    except:
        df = df.drop(i)

    print(g)
    print(df.shape)

print(exceptions)

df.to_csv("labels_reduced_dropped.csv")
