import torch
from torch_geometric.data import InMemoryDataset, Dataset, Data

from tqdm import tqdm
import torch_geometric.transforms as T
from pathlib import Path
import json
import numpy as np
from utils import *


class SessionDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(SessionDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        raw_files = list(Path(self.root + '\\raw').rglob('*.json'))
        raw_files = [item.name for item in raw_files]
        return raw_files

    @property
    def processed_file_names(self):
        return 'data.pt'
    
    def download(self):
        pass

    def process(self):

        pbar = tqdm(total=len(self.raw_paths), desc='Files Done: ')

        data_list = []
        for file_number, raw_file in enumerate(self.raw_paths):

            with open(raw_file, 'r') as file_handle:
                json_data = json.load(file_handle)
                
                id = json_data["id"]
                features = json_data["features"]
                edge_indices = json_data["edge_indices"]
                class_label = json_data["class"]
                class_vector = json_data["class_vector"]

                edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long)
                x = torch.tensor(features, dtype=torch.float)
                y = torch.tensor(np.array([class_vector], dtype=np.float32), dtype=torch.float)
                graph = Data(x=x, edge_index=edge_index, y=y)

                data_list.append(graph)

            pbar.update(1)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
