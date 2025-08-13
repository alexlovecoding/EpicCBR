#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import scipy.sparse as sp 

import torch
from torch.utils.data import Dataset, DataLoader

# Print matrix statistics such as average interactions, ratio of nonzero rows/columns, matrix density, etc.
def print_statistics(X, string):
    print('>'*10 + string + '>'*10 )
    # Calculate the average number of interactions per row
    print('Average interactions', X.sum(1).mean(0).item())
    # Get row and column indices of nonzero elements
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    # Get unique indices of nonzero rows
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    # Get unique indices of nonzero columns
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    # Calculate the ratio of nonzero rows
    print('Non-zero rows', len(unique_nonzero_row_indice)/X.shape[0])
    # Calculate the ratio of nonzero columns
    print('Non-zero columns', len(unique_nonzero_col_indice)/X.shape[1])
    # Calculate matrix density
    print('Matrix density', len(nonzero_row_indice)/(X.shape[0]*X.shape[1]))

# Custom dataset class for bundle data in training phase
class BundleTrainDataset(Dataset):
    def __init__(self, conf, u_b_pairs, u_b_graph, num_bundles, u_b_for_neg_sample, b_b_for_neg_sample, neg_sample=1):
        # Save config
        self.conf = conf
        # User-bundle positive sample pairs
        self.u_b_pairs = u_b_pairs
        # User-bundle interaction graph (sparse matrix)
        self.u_b_graph = u_b_graph
        # Total number of bundles
        self.num_bundles = num_bundles
        # Number of negative samples
        self.neg_sample = neg_sample

        # User-bundle info for negative sampling
        self.u_b_for_neg_sample = u_b_for_neg_sample
        # Bundle-bundle info for negative sampling
        self.b_b_for_neg_sample = b_b_for_neg_sample

        # Calculate interaction count for each bundle
        self.bundle_interaction_counts = self.calculate_bundle_interaction_counts()
        # Filter high-interaction bundles for BPR calculation
        self.high_interaction_bundles = [bundle for bundle, count in self.bundle_interaction_counts.items() if count > self.conf["interaction_threshold"]]

    def calculate_bundle_interaction_counts(self):
        bundle_interaction_counts = {}
        for user, bundle in self.u_b_pairs:
            if bundle in bundle_interaction_counts:
                bundle_interaction_counts[bundle] += 1
            else:
                bundle_interaction_counts[bundle] = 1
        return bundle_interaction_counts

    # Get a sample by index
    def __getitem__(self, index):
        conf = self.conf
        # Get user and positive bundle for current index
        user_b, pos_bundle = self.u_b_pairs[index]
        # Initialize list for positive and negative bundles
        all_bundles = [pos_bundle]

        # Negative sampling
        while True:
            i = np.random.randint(self.num_bundles)
            if self.u_b_graph[user_b, i] == 0 and not i in all_bundles:                                                          
                all_bundles.append(i)                                                                                                   
                if len(all_bundles) == self.neg_sample+1:                                                                               
                    break      

        # Convert user and bundle indices to PyTorch LongTensor and return
        return torch.LongTensor([user_b]), torch.LongTensor(all_bundles)

    # Return dataset length
    def __len__(self):
        return len(self.u_b_pairs)

# Custom dataset class for bundle data in test phase
class BundleTestDataset(Dataset):
    def __init__(self, u_b_pairs, u_b_graph, u_b_graph_train, num_users, num_bundles):
        # User-bundle pairs
        self.u_b_pairs = u_b_pairs
        # User-bundle interaction graph (test set)
        self.u_b_graph = u_b_graph
        # User-bundle interaction graph (train set), used to mask interactions already in train set
        self.train_mask_u_b = u_b_graph_train

        # Total number of users
        self.num_users = num_users
        # Total number of bundles
        self.num_bundles = num_bundles

        # Generate index tensor for all users
        self.users = torch.arange(num_users, dtype=torch.long).unsqueeze(dim=1)
        # Generate index tensor for all bundles
        self.bundles = torch.arange(num_bundles, dtype=torch.long)

    # Get a sample by index
    def __getitem__(self, index):
        # Convert the interaction graph of the specified user in test set to PyTorch tensor
        u_b_grd = torch.from_numpy(self.u_b_graph[index].toarray()).squeeze()
        # Convert the interaction graph of the specified user in train set to PyTorch tensor
        u_b_mask = torch.from_numpy(self.train_mask_u_b[index].toarray()).squeeze()

        return index, u_b_grd, u_b_mask

    # Return dataset length
    def __len__(self):
        return self.u_b_graph.shape[0]

# Dataset management class for loading and processing all data
class Datasets():
    def __init__(self, conf):
        # Path to data files
        self.path = conf['data_path']
        # Dataset name
        self.name = conf['dataset']
        # Batch size for training set
        batch_size_train = conf['batch_size_train']
        # Batch size for test and validation set
        batch_size_test = conf['batch_size_test']

        # Get basic data info: number of users, bundles, items
        self.num_users, self.num_bundles, self.num_items = self.get_data_size()

        self.item_relation_types = conf.get('item_relation_types', [])

        # Read item popularity file
        self.item_popularity = self.get_item_popularity()

        # Get bundle-item interaction pairs and graph
        b_i_pairs, b_i_graph = self.get_bi()
        # Get bundle-item interaction pairs and graph with popularity
        w_b_i_pairs, w_b_i_graph = self.get_bi_weighted()
        # Get user-item interaction pairs and graph
        u_i_pairs, u_i_graph = self.get_ui()
        # Get item-item interaction graph
        i_i_graph = self.get_ii()

        # Get user-bundle interaction pairs and graph for train/val/test
        u_b_pairs_train, u_b_graph_train = self.get_ub("train")
        u_b_pairs_val, u_b_graph_val = self.get_ub("tune")
        u_b_pairs_test, u_b_graph_test = self.get_ub("test")

        # Info for negative sampling, currently set to None
        u_b_for_neg_sample, b_b_for_neg_sample = None, None

        # Create dataset objects for train/val/test
        self.bundle_train_data = BundleTrainDataset(conf, u_b_pairs_train, u_b_graph_train, self.num_bundles, u_b_for_neg_sample, b_b_for_neg_sample, conf["neg_num"])
        self.bundle_val_data = BundleTestDataset(u_b_pairs_val, u_b_graph_val, u_b_graph_train, self.num_users, self.num_bundles)
        self.bundle_test_data = BundleTestDataset(u_b_pairs_test, u_b_graph_test, u_b_graph_train, self.num_users, self.num_bundles)

        # Store user-bundle, user-item, bundle-item interaction graphs
        self.graphs = [u_b_graph_train, u_i_graph, b_i_graph, i_i_graph, w_b_i_graph]

        # Create dataloaders for train/val/test
        self.train_loader = DataLoader(self.bundle_train_data, batch_size=batch_size_train, shuffle=True, num_workers=10, drop_last=True)
        self.val_loader = DataLoader(self.bundle_val_data, batch_size=batch_size_test, shuffle=False, num_workers=20)
        self.test_loader = DataLoader(self.bundle_test_data, batch_size=batch_size_test, shuffle=False, num_workers=20)

    # Get basic data info: number of users, bundles, items
    def get_data_size(self):
        name = self.name
        # If dataset name contains '_', take the part before '_'
        if "_" in name:
            name = name.split("_")[0]
        # Open data size info file
        with open(os.path.join(self.path, self.name, '{}_data_size.txt'.format(name)), 'r') as f:
            # Read and convert to int list, take first three values
            return [int(s) for s in f.readline().split('\t')][:3]
    

    def get_item_popularity(self):
        # Open item popularity info file
        with open(os.path.join(self.path, self.name, 'item_popularity.txt'), 'r') as f:
            # Read and convert to dict
            item_popularity = {int(line.split()[0]): float(line.split()[1]) for line in f.readlines()}
        return item_popularity

    # Get bundle-item interaction pairs and graph with popularity
    def get_bi_weighted(self):
        # Open bundle-item interaction file
        with open(os.path.join(self.path, self.name, 'bundle_item.txt'), 'r') as f:
            # Read and convert to tuple list
            b_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        # Convert to numpy array
        indice = np.array(b_i_pairs, dtype=np.int32)
        # Use item popularity as edge weight, default to 2 if missing
        values = np.array([self.item_popularity.get(item_id, 2.0) for _, item_id in b_i_pairs], dtype=np.float32)
        # Create sparse matrix for bundle-item interaction graph
        b_i_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_bundles, self.num_items)).tocsr()

        # Print statistics
        print_statistics(b_i_graph, 'Weighted-B-I statistics')

        return b_i_pairs, b_i_graph

    # Get bundle-item interaction pairs and graph
    def get_bi(self):
        # Open bundle-item interaction file
        with open(os.path.join(self.path, self.name, 'bundle_item.txt'), 'r') as f:
            # Read and convert to tuple list
            b_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        # Convert to numpy array
        indice = np.array(b_i_pairs, dtype=np.int32)
        # All ones for existing interactions
        values = np.ones(len(b_i_pairs), dtype=np.float32)
        # Create sparse matrix for bundle-item interaction graph
        b_i_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_bundles, self.num_items)).tocsr()

        # Print statistics
        print_statistics(b_i_graph, 'B-I statistics')

        return b_i_pairs, b_i_graph

    # Get user-item interaction pairs and graph
    def get_ui(self):
        # Open user-item interaction file
        with open(os.path.join(self.path, self.name, 'user_item.txt'), 'r') as f:
            # Read and convert to tuple list
            u_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        # Convert to numpy array
        indice = np.array(u_i_pairs, dtype=np.int32)
        # All ones for existing interactions
        values = np.ones(len(u_i_pairs), dtype=np.float32)
        # Create sparse matrix for user-item interaction graph
        u_i_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_items)).tocsr()

        # Print statistics
        print_statistics(u_i_graph, 'U-I statistics')

        return u_i_pairs, u_i_graph

    # Get user-bundle interaction pairs and graph for different tasks (train/val/test)
    def get_ub(self, task):
        # Open user-bundle interaction file for the specified task
        with open(os.path.join(self.path, self.name, 'user_bundle_{}.txt'.format(task)), 'r') as f:
            # Read and convert to tuple list
            u_b_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        # Convert to numpy array
        indice = np.array(u_b_pairs, dtype=np.int32)
        # All ones for existing interactions
        values = np.ones(len(u_b_pairs), dtype=np.float32)
        # Create sparse matrix for user-bundle interaction graph
        u_b_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_bundles)).tocsr()

        # Print statistics
        print_statistics(u_b_graph, "U-B statistics in %s" %(task))

        return u_b_pairs, u_b_graph


    def get_ii(self):
        # Open item-item interaction file
        with open(os.path.join(self.path, self.name, 'item_relations.txt'), 'r') as f:
            lines = f.readlines()
        
        # Process all lines at once
        data = [
            (int(parts[0]), int(parts[1]), float(parts[2]))
            for line in lines
            for parts in [line[:-1].split(' ')]
            if float(parts[2]) in self.item_relation_types # Only use specified edge types, discard others
        ]

        # Separate data
        i_i_pairs = [(item1, item2) for item1, item2, _ in data]
        values = np.array([relation_type for _, _, relation_type in data])

        # Use numpy conditional assignment to modify values
        para = [0.2,0.5,1]
        values = np.where(values == 1, para[0], values)
        values = np.where(values == 2, para[1], values)
        values = np.where(values == 3, para[2], values)

        # Convert to numpy array
        indice = np.array(i_i_pairs, dtype=np.int32)
        # Create sparse matrix for item-item interaction graph
        i_i_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_items, self.num_items)).tocsr()
        # Print statistics
        print_statistics(i_i_graph, 'I-I statistics')

        has_zero_edges = np.any(i_i_graph.data == 0)
        if has_zero_edges:
            raise ValueError(f"Invalid zero-value edge found!")

        return i_i_graph
