import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
from session_dataset import SessionDataset
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx



if __name__=='__main__':

    iscx_root = r'D:\SH\TrafficClassification\vpn-gcn\datasets\iscx'
    vnat_root = r'D:\SH\TrafficClassification\vpn-gcn\datasets\vnat-vpn'
    

    roots = [iscx_root, vnat_root]

    for root in roots:
        if not os.path.isdir(root + '\\raw'):
            os.mkdir(root + '\\raw')
        if not os.path.isdir(root + '\\processed'):
            os.mkdir(root + '\\processed')
        
        cmd = "move /Y " + root + "\\*.json " + root + '\\raw'
        os.system(cmd)

    iscx_dataset = SessionDataset(root=iscx_root)
    vnat_dataset = SessionDataset(root=vnat_root)

    print(iscx_dataset[1])
    print(vnat_dataset[1])

    print('ALL DONE!!!!!')
