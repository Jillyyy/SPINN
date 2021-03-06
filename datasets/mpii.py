"""
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np

from .base_JointsDataset import BaseJointsDataset

class MPIIDataset(torch.utils.data.Dataset):

    def __init__(self, options, cfg, **kwargs):
        self.dataset_list = ['mpii']
        self.dataset_dict = {'mpii': 0}
        # self.datasets = [BaseJointsDataset(options, ds, **kwargs) for ds in self.dataset_list]
        self.dataset = BaseJointsDataset(options, cfg, 'mpii', **kwargs)
        # total_length = sum([len(ds) for ds in self.datasets])
        # length_itw = sum([len(ds) for ds in self.datasets])
        self.length =len(self.dataset)
        # self.num_joints = 16
        """
        Data distribution inside each batch:
        30% H36M - 60% ITW - 10% MPI-INF
        """
        # self.partition = [.3, .6*len(self.datasets[1])/length_itw,
        #                   .6*len(self.datasets[2])/length_itw,
        #                   .6*len(self.datasets[3])/length_itw, 
        #                   .6*len(self.datasets[4])/length_itw,
        #                   0.1]
        # self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        # p = np.random.rand()
        # for i in range(6):
        #     if p <= self.partition[i]:
        return self.dataset[index]

    def __len__(self):
        return self.length
