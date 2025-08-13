#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 
from scipy.sparse import csr_matrix, vstack , diags

# Calculate BPR (Bayesian Personalized Ranking) loss
def cal_bpr_loss(pred):
    # pred: [bs, 1+neg_num], bs is batch size, neg_num is number of negative samples
    if pred.shape[1] > 2:
        # Extract prediction scores of negative samples
        negs = pred[:, 1:]
        # Extract prediction scores of positive samples and expand dimensions to match negative samples
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        # Handling when the number of negative samples is 1
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    # Calculate BPR loss using sigmoid and log functions
    loss = - torch.log(torch.sigmoid(pos - negs))  # [bs]
    # Calculate average loss
    loss = torch.mean(loss)

    return loss

# Perform Laplacian transform on the graph
def laplace_transform(graph):
    # Calculate the inverse of the square root of the sum of each row's elements to build a diagonal matrix
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    # Calculate the inverse of the square root of the sum of each column's elements to build a diagonal matrix
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    # Perform Laplacian transform
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph

# Convert a sparse matrix to a PyTorch sparse tensor
def to_tensor(graph):
    # Convert the graph to a COO format sparse matrix
    graph = graph.tocoo()
    # Extract non-zero element values of the graph
    values = graph.data
    # Extract row and column indices of non-zero elements
    indices = np.vstack((graph.row, graph.col))
    # Create a PyTorch sparse float tensor
    graph = torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph

# Randomly drop edges of the graph
def np_edge_dropout(values, dropout_ratio):
    # Generate a mask of 0s or 1s with dropout_ratio probability
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    # Drop edges based on the mask
    values = mask * values
    return values

# Define EpicCBR model class
class EpicCBR(nn.Module):
    def __init__(self, conf, raw_graph):
        # Call the constructor of the parent class
        super().__init__()
        # Save configuration information
        self.conf = conf
        # Get device information, e.g., CPU or GPU
        device = self.conf["device"]
        self.device = device

        # Dimension of embedding vectors
        self.embedding_size = conf["embedding_size"]
        # L2 regularization coefficient
        self.embed_L2_norm = conf["l2_reg"]
        # Number of users
        self.num_users = conf["num_users"]
        # Number of bundles
        self.num_bundles = conf["num_bundles"]
        # Number of items
        self.num_items = conf["num_items"]
        # Number of propagation layers
        self.num_layers = self.conf["num_layers"]
        # Temperature parameter for contrastive loss
        self.c_temp = self.conf["c_temp"]

        self.scenario_weight = self.conf["scenario_weight"]

        # Fusion weight configuration
        self.fusion_weights = conf['fusion_weights']

        # Initialize embedding vectors
        self.init_emb()
        # Initialize fusion weights
        self.init_fusion_weights()

        # Ensure the raw graph is a list
        assert isinstance(raw_graph, list)
        # Extract user-bundle, user-item, bundle-item, item-item, and weighted bundle-item graphs
        self.ub_graph, self.ui_graph, self.bi_graph, self.ii_graph, self.w_bi_graph = raw_graph

        self.II_propagation_graph = to_tensor(laplace_transform(self.ii_graph)).to(device)

        # Generate non-dropout propagation graphs for testing
        self.UB_propagation_graph_ori = self.get_propagation_graph(self.ub_graph)

        # Note: When modifying any graph, be sure to modify both the training and testing graphs simultaneously!!!
        # Note: When modifying any graph, be sure to modify both the training and testing graphs simultaneously!!!
        # Note: When modifying any graph, be sure to modify both the training and testing graphs simultaneously!!!
        
        self.UI_propagation_graph_ori = self.get_propagation_graph(self.ui_graph)
        self.UI_aggregation_graph_ori = self.get_aggregation_graph(self.ui_graph)

        self.BI_propagation_graph_ori = self.get_propagation_graph(self.w_bi_graph)
        self.BI_aggregation_graph_ori = self.get_aggregation_graph(self.w_bi_graph)

        self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf["UB_ratio"])

        self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf["UI_ratio"])
        self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph, self.conf["UI_ratio"])

        self.BI_propagation_graph = self.get_propagation_graph(self.w_bi_graph, self.conf["BI_ratio"])
        self.BI_aggregation_graph = self.get_aggregation_graph(self.w_bi_graph, self.conf["BI_ratio"])

        # cold scenario
        self.UI_propagation_graph_ori_cold = self.get_propagation_graph(self.multiply_and_normalize(self.ui_graph,self.ii_graph,self.conf['edges_enhanced_strength']))
        self.UI_aggregation_graph_ori_cold = self.get_aggregation_graph(self.multiply_and_normalize(self.ui_graph,self.ii_graph,self.conf['edges_enhanced_strength']))

        self.BI_propagation_graph_ori_cold = self.get_propagation_graph(self.w_bi_graph)
        self.BI_aggregation_graph_ori_cold = self.get_aggregation_graph(self.w_bi_graph)

        self.UI_propagation_graph_cold = self.get_propagation_graph(self.multiply_and_normalize(self.ui_graph,self.ii_graph,self.conf['edges_enhanced_strength']), self.conf["UI_ratio"])
        self.UI_aggregation_graph_cold = self.get_aggregation_graph(self.multiply_and_normalize(self.ui_graph,self.ii_graph,self.conf['edges_enhanced_strength']), self.conf["UI_ratio"])

        self.BI_propagation_graph_cold = self.get_propagation_graph(self.w_bi_graph, self.conf["BI_ratio"])
        self.BI_aggregation_graph_cold = self.get_aggregation_graph(self.w_bi_graph, self.conf["BI_ratio"])
        
        # If augmentation type is MD, initialize modal dropout layers
        if self.conf['aug_type'] == 'MD':
            self.init_md_dropouts()
        # If augmentation type is Noise, initialize noise parameters
        elif self.conf['aug_type'] == "Noise":
            self.init_noise_eps()

        self.item_relations = self.load_item_relations()
        
    # Initialize modal dropout layers
    def init_md_dropouts(self):
        # Initialize dropout layer for user-bundle graph
        self.UB_dropout = nn.Dropout(self.conf["UB_ratio"], True)
        # Initialize dropout layer for user-item graph
        self.UI_dropout = nn.Dropout(self.conf["UI_ratio"], True)
        # Initialize dropout layer for bundle-item graph
        self.BI_dropout = nn.Dropout(self.conf["BI_ratio"], True)
        # Dictionary to store dropout layers
        self.mess_dropout_dict = {
            "UB": self.UB_dropout,
            "UI": self.UI_dropout,
            "BI": self.BI_dropout
        }

    # Initialize noise parameters
    def init_noise_eps(self):
        # Noise parameter for user-bundle graph
        self.UB_eps = self.conf["UB_ratio"]
        # Noise parameter for user-item graph
        self.UI_eps = self.conf["UI_ratio"]
        # Noise parameter for bundle-item graph
        self.BI_eps = self.conf["BI_ratio"]
        # Dictionary to store noise parameters
        self.eps_dict = {
            "UB": self.UB_eps,
            "UI": self.UI_eps,
            "BI": self.BI_eps
        }

    # Initialize embedding vectors
    def init_emb(self):
        # Initialize user embedding vectors
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        # Initialize user embedding vectors with Xavier normal distribution
        nn.init.xavier_normal_(self.users_feature)
        # Initialize bundle embedding vectors
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        # Initialize bundle embedding vectors with Xavier normal distribution
        nn.init.xavier_normal_(self.bundles_feature)
        # Initialize item embedding vectors
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        # Initialize item embedding vectors with Xavier normal distribution
        nn.init.xavier_normal_(self.items_feature)
        # cold
        # Initialize user embedding vectors
        self.users_feature_cold = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        # Initialize user embedding vectors with Xavier normal distribution
        nn.init.xavier_normal_(self.users_feature_cold)
        # Initialize item embedding vectors
        self.items_feature_cold = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        # Initialize item embedding vectors with Xavier normal distribution
        nn.init.xavier_normal_(self.items_feature_cold)

    # Initialize fusion weights
    def init_fusion_weights(self):
        # Ensure the number of modal fusion weights matches the number of graphs
        assert (len(self.fusion_weights['modal_weight']) == 3), \
            "The number of modal fusion weights does not correspond to the number of graphs"

        # Ensure the number of layer fusion weights matches the number of layers
        assert (len(self.fusion_weights['UB_layer']) == self.num_layers + 1) and\
               (len(self.fusion_weights['UI_layer']) == self.num_layers + 1) and \
               (len(self.fusion_weights['BI_layer']) == self.num_layers + 1) and \
            "The number of layer fusion weights does not correspond to number of layers"

        # Convert modal fusion weights to a PyTorch tensor
        modal_coefs = torch.FloatTensor(self.fusion_weights['modal_weight'])
        
        modal_coefs_cold = torch.FloatTensor(self.fusion_weights['modal_weight_cold'])
        # Convert user-bundle graph layer fusion weights to a PyTorch tensor
        UB_layer_coefs = torch.FloatTensor(self.fusion_weights['UB_layer'])
        # Convert user-item graph layer fusion weights to a PyTorch tensor
        UI_layer_coefs = torch.FloatTensor(self.fusion_weights['UI_layer'])
        # Convert bundle-item graph layer fusion weights to a PyTorch tensor
        BI_layer_coefs = torch.FloatTensor(self.fusion_weights['BI_layer'])

        # Expand dimensions of modal fusion weights and move to the specified device
        self.modal_coefs = modal_coefs.unsqueeze(-1).unsqueeze(-1).to(self.device)

        self.modal_coefs_cold = modal_coefs_cold.unsqueeze(-1).unsqueeze(-1).to(self.device)

        # Expand dimensions of user-bundle graph layer fusion weights and move to the specified device
        self.UB_layer_coefs = UB_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)
        # Expand dimensions of user-item graph layer fusion weights and move to the specified device
        self.UI_layer_coefs = UI_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)
        # Expand dimensions of bundle-item graph layer fusion weights and move to the specified device
        self.BI_layer_coefs = BI_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)

    def multiply_and_normalize(self, x_i_graph, ii_graph, default_value=0.5):
        # 1. Ensure x_i_graph edge weights are 1 (if original edge weights are not 1, binarize first)
        x_i_binary = x_i_graph.copy()
        x_i_binary.data = np.ones_like(x_i_binary.data)  # Ensure original edge weights are 1
        
        # 2. Calculate the product matrix (U×I) @ (I×I) = U×I, set non-zero values to default_value
        multiplied_csr = x_i_graph @ ii_graph
        multiplied_csr.data = np.full_like(multiplied_csr.data, default_value)  # Key: directly assign to data
        
        # 3. Matrix addition (utilizing efficient merging of CSR addition)
        combined_csr = multiplied_csr + x_i_binary  # Automatically merges elements at the same position
        
        # 4. Laplace normalization (optimized CSR path)
        return laplace_transform(combined_csr)

    def get_propagation_graph_with_ii(self, bipartite_graph, ii_graph, modification_ratio=0):
        # Get device information
        device = self.device
        num_part_1 = bipartite_graph.shape[0]
        num_items = ii_graph.shape[0]
        if ii_graph.shape[0] != bipartite_graph.shape[1] or ii_graph.shape[1] != num_items:
            raise ValueError(f"The size of ii_graph ({ii_graph.shape}) does not match the number of items in bipartite_graph, expected size is ({bipartite_graph.shape[1]}, {bipartite_graph.shape[1]}).")

        # Build propagation graph, filling in the II graph at the same time
        upper_left = sp.csr_matrix((num_part_1, num_part_1))
        upper_right = bipartite_graph
        lower_left = bipartite_graph.T
        lower_right = ii_graph

        propagation_graph = sp.bmat([[upper_left, upper_right], [lower_left, lower_right]])

        # If modification ratio is not 0
        if modification_ratio != 0:
            # If augmentation type is ED (Edge Dropping)
            if self.conf["aug_type"] == "ED":
                # Convert propagation graph to COO format
                graph = propagation_graph.tocoo()
                # Randomly drop edges
                values = np_edge_dropout(graph.data, modification_ratio)
                # Reconstruct the propagation graph
                propagation_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        # Perform Laplacian transform on the propagation graph, convert to tensor, and move to the specified device
        return to_tensor(laplace_transform(propagation_graph)).to(device)


    # Get propagation graph
    def get_propagation_graph(self, bipartite_graph, modification_ratio=0):
        # Get device information
        device = self.device
        # Build propagation graph by combining the bipartite graph with its transpose
        propagation_graph = sp.bmat([[sp.csr_matrix((bipartite_graph.shape[0], bipartite_graph.shape[0])), bipartite_graph], 
                                    [bipartite_graph.T, sp.csr_matrix((bipartite_graph.shape[1], bipartite_graph.shape[1]))]])

        # If modification ratio is not 0
        if modification_ratio != 0:
            # If augmentation type is ED (Edge Dropping)
            if self.conf["aug_type"] == "ED":
                # Convert propagation graph to COO format
                graph = propagation_graph.tocoo()
                # Randomly drop edges
                values = np_edge_dropout(graph.data, modification_ratio)
                # Reconstruct the propagation graph
                propagation_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        # Perform Laplacian transform on the propagation graph, convert to tensor, and move to the specified device
        return to_tensor(laplace_transform(propagation_graph)).to(device)

    # Get aggregation graph
    def get_aggregation_graph(self, bipartite_graph, modification_ratio=0):
        # Get device information
        device = self.device

        # If modification ratio is not 0
        if modification_ratio != 0:
            # If augmentation type is ED (Edge Dropping)
            if self.conf["aug_type"] == "ED":
                # Convert bipartite graph to COO format
                graph = bipartite_graph.tocoo()
                # Randomly drop edges
                values = np_edge_dropout(graph.data, modification_ratio)
                # Reconstruct the bipartite graph
                bipartite_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        # Calculate the degree of each node, adding a small constant to prevent division by zero
        bundle_size = bipartite_graph.sum(axis=1) + 1e-8
        # Normalize the bipartite graph
        bipartite_graph = sp.diags(1/bundle_size.A.ravel()) @ bipartite_graph
        # Convert the processed bipartite graph to a tensor and move to the specified device
        return to_tensor(bipartite_graph).to(device)

    def propagate_ii(self, graph, item_feature, layer_coef, test):
        # Store features of each layer
        all_features = [item_feature]

        # Perform multi-layer propagation
        for i in range(self.num_layers):
            # Update features through graph convolution
            item_feature = torch.spmm(graph, item_feature)
            # L2 normalize the features
            all_features.append(F.normalize(item_feature, p=2, dim=1))

        # Stack features of each layer and multiply by layer fusion weights
        all_features = torch.stack(all_features, 1) * layer_coef
        # Sum the features of each layer
        all_features = torch.sum(all_features, dim=1)

        return all_features

    # Perform graph propagation operation (second-level embedding)
    def propagate(self, graph, A_feature, B_feature, graph_type, layer_coef, test):
        # Concatenate A features and B features
        features = torch.cat((A_feature, B_feature), 0)
        # Store features of each layer
        all_features = [features]

        # Perform multi-layer propagation
        for i in range(self.num_layers):
            # Update features through graph convolution
            features = torch.spmm(graph, features) ## sparse matrix and dense matrix multiplication, interaction sparse matrix and first-level embedding (torch), the following features is the embedding matrix not the result after embedding, the former is EU, EB
            # If augmentation type is MD and not in testing phase
            if self.conf["aug_type"] == "MD" and not test:
                # Get the corresponding dropout layer
                mess_dropout = self.mess_dropout_dict[graph_type]
                # Apply dropout operation
                features = mess_dropout(features)
            # If augmentation type is Noise and not in testing phase
            elif self.conf["aug_type"] == "Noise" and not test:
                # Generate random noise
                random_noise = torch.rand_like(features).to(self.device)
                # Get the corresponding noise parameter
                eps = self.eps_dict[graph_type]
                # Add noise
                features += torch.sign(features) * F.normalize(random_noise, dim=-1) * eps

            # L2 normalize the features
            all_features.append(F.normalize(features, p=2, dim=1))

        # Stack features of each layer and multiply by layer fusion weights
        all_features = torch.stack(all_features, 1) * layer_coef
        # Sum the features of each layer
        all_features = torch.sum(all_features, dim=1)
        # Separate A features and B features
        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature

    # Perform graph aggregation operation (indirect)
    def aggregate(self, agg_graph, node_feature, graph_type, test):
        # Perform aggregation operation through matrix multiplication
        aggregated_feature = torch.matmul(agg_graph, node_feature)

        # If augmentation type is MD and not in testing phase
        if self.conf["aug_type"] == "MD" and not test:
            # Get the corresponding dropout layer
            mess_dropout = self.mess_dropout_dict[graph_type]
            # Apply dropout operation
            aggregated_feature = mess_dropout(aggregated_feature)
        # If augmentation type is Noise and not in testing phase
        elif self.conf["aug_type"] == "Noise" and not test:
            # Generate random noise
            random_noise = torch.rand_like(aggregated_feature).to(self.device)
            # Get the corresponding noise parameter
            eps = self.eps_dict[graph_type]
            # Add noise
            aggregated_feature += torch.sign(aggregated_feature) * F.normalize(random_noise, dim=-1) * eps

        return aggregated_feature

    # Fuse features of users and bundles
    def fuse_users_bundles_feature(self, users_feature, bundles_feature):
        # Stack user features
        users_feature = torch.stack(users_feature, dim=0)
        # Stack bundle features
        bundles_feature = torch.stack(bundles_feature, dim=0)

        # Modal aggregation, weighted sum of user features according to modal fusion weights
        users_rep = torch.sum(users_feature * self.modal_coefs, dim=0)
        # Modal aggregation, weighted sum of bundle features according to modal fusion weights
        bundles_rep = torch.sum(bundles_feature * self.modal_coefs, dim=0)

        return users_rep, bundles_rep

    # Fuse features of users and bundles (cold)
    def fuse_users_bundles_feature_cold(self, users_feature, bundles_feature):
        # Stack user features
        users_feature = torch.stack(users_feature, dim=0)
        # Stack bundle features
        bundles_feature = torch.stack(bundles_feature, dim=0)

        # Modal aggregation, weighted sum of user features according to modal fusion weights
        users_rep = torch.sum(users_feature * self.modal_coefs_cold, dim=0)
        # Modal aggregation, weighted sum of bundle features according to modal fusion weights
        bundles_rep = torch.sum(bundles_feature * self.modal_coefs_cold, dim=0)

        return users_rep, bundles_rep

    # Get multi-modal representations
    def get_multi_modal_representations(self, test=False):
        #  =============================  UB graph propagation  =============================
        if test:
            # In the testing phase, use non-dropout propagation graph for propagation
            UB_users_feature, UB_bundles_feature = self.propagate(self.UB_propagation_graph_ori, self.users_feature, self.bundles_feature, "UB", self.UB_layer_coefs, test)
        else:
            # In the training phase, use propagation graph with dropout for propagation
            UB_users_feature, UB_bundles_feature = self.propagate(self.UB_propagation_graph, self.users_feature, self.bundles_feature, "UB", self.UB_layer_coefs, test)

        #  =============================  UI graph propagation  =============================
        if test:
            # In the testing phase, use non-dropout propagation graph for propagation
            UI_users_feature, UI_items_feature = self.propagate(self.UI_propagation_graph_ori, self.users_feature, self.items_feature, "UI", self.UI_layer_coefs, test)
            # In the testing phase, use non-dropout aggregation graph for aggregation
            UI_bundles_feature = self.aggregate(self.BI_aggregation_graph_ori, UI_items_feature, "BI", test)
        else:
            # In the training phase, use propagation graph with dropout for propagation
            UI_users_feature, UI_items_feature = self.propagate(self.UI_propagation_graph, self.users_feature, self.items_feature, "UI", self.UI_layer_coefs, test)
            # In the training phase, use aggregation graph with dropout for aggregation
            UI_bundles_feature = self.aggregate(self.BI_aggregation_graph, UI_items_feature, "BI", test)
        #  =============================  BI graph propagation  =============================
        if test:
            # In the testing phase, use non-dropout propagation graph for bundle-item graph propagation
            BI_bundles_feature, BI_items_feature = self.propagate(self.BI_propagation_graph_ori, self.bundles_feature, self.items_feature, "BI", self.BI_layer_coefs, test)
            # In the testing phase, use non-dropout aggregation graph to aggregate user features from item features
            BI_users_feature = self.aggregate(self.UI_aggregation_graph_ori, BI_items_feature, "UI", test)
        else:
            # In the training phase, use propagation graph with dropout for bundle-item graph propagation
            BI_bundles_feature, BI_items_feature = self.propagate(self.BI_propagation_graph, self.bundles_feature, self.items_feature, "BI", self.BI_layer_coefs, test)
            # In the training phase, use aggregation graph with dropout to aggregate user features from item features
            BI_users_feature = self.aggregate(self.UI_aggregation_graph, BI_items_feature, "UI", test)
            
        
        # Collect user features obtained from three graph propagations
        users_feature = [UB_users_feature, UI_users_feature, BI_users_feature]
        # Collect bundle features obtained from three graph propagations
        bundles_feature = [UB_bundles_feature, UI_bundles_feature, BI_bundles_feature]
        # Fuse user and bundle features from different graphs
        users_rep, bundles_rep = self.fuse_users_bundles_feature(users_feature, bundles_feature)

        # cold:
        #  =============================  UI graph propagation  =============================
        if test:
            # In the testing phase, use non-dropout propagation graph for propagation
            UI_users_feature_cold, UI_items_feature_cold = self.propagate(self.UI_propagation_graph_ori_cold, self.users_feature_cold, self.items_feature_cold, "UI", self.UI_layer_coefs, test)
            # In the testing phase, use non-dropout aggregation graph for aggregation
            UI_bundles_feature_cold = self.aggregate(self.BI_aggregation_graph_ori_cold, UI_items_feature_cold, "BI", test)
        else:
            # In the training phase, use propagation graph with dropout for propagation
            UI_users_feature_cold, UI_items_feature_cold = self.propagate(self.UI_propagation_graph_cold, self.users_feature_cold, self.items_feature_cold, "UI", self.UI_layer_coefs, test)
            # In the training phase, use aggregation graph with dropout for aggregation
            UI_bundles_feature_cold = self.aggregate(self.BI_aggregation_graph_cold, UI_items_feature_cold, "BI", test)
        #  =============================  BI graph propagation  =============================
        if test:
            # In the testing phase, use non-dropout propagation graph for bundle-item graph propagation
            BI_bundles_feature_cold, BI_items_feature_cold = self.propagate(self.BI_propagation_graph_ori_cold, UI_bundles_feature_cold, self.items_feature_cold, "BI", self.BI_layer_coefs, test)
            # In the testing phase, use non-dropout aggregation graph to aggregate user features from item features
            BI_users_feature_cold = self.aggregate(self.UI_aggregation_graph_ori_cold, BI_items_feature_cold, "UI", test)
        else:
            # In the training phase, use propagation graph with dropout for bundle-item graph propagation
            BI_bundles_feature_cold, BI_items_feature_cold = self.propagate(self.BI_propagation_graph_cold, UI_bundles_feature_cold, self.items_feature_cold, "BI", self.BI_layer_coefs, test)
            # In the training phase, use aggregation graph with dropout to aggregate user features from item features
            BI_users_feature_cold = self.aggregate(self.UI_aggregation_graph_cold, BI_items_feature_cold, "UI", test)
            
        # Collect user features obtained from three graph propagations
        users_feature_cold = [UI_users_feature_cold, BI_users_feature_cold]
        # Collect bundle features obtained from three graph propagations
        bundles_feature_cold = [UI_bundles_feature_cold, BI_bundles_feature_cold]
        # Fuse user and bundle features from different graphs
        users_rep_cold, bundles_rep_cold = self.fuse_users_bundles_feature_cold(users_feature_cold, bundles_feature_cold)
        
        # Perform modal fusion
        k = self.scenario_weight #modify this to control whether the model is more cold or hot
        users_rep_all = torch.cat((users_rep*k, users_rep_cold*(1-k)), dim=1)
        bundles_rep_all = torch.cat((bundles_rep*k, bundles_rep_cold*(1-k)), dim=1)

        # scen_c_loss = self.cal_c_loss(users_rep,users_rep_cold) + self.cal_c_loss(bundles_rep,bundles_rep_cold)

        return users_rep_all, bundles_rep_all, users_rep, bundles_rep, users_rep_cold, bundles_rep_cold

    # Calculate contrastive loss
    def cal_c_loss(self, pos, aug):
        # pos: [batch_size, :, emb_size], positive sample features
        # aug: [batch_size, :, emb_size], augmented sample features
        # Extract the first feature of positive samples
        pos = pos[:, 0, :]
        # Extract the first feature of augmented samples
        aug = aug[:, 0, :]

        # L2 normalize positive sample features
        pos = F.normalize(pos, p=2, dim=1)
        # L2 normalize augmented sample features
        aug = F.normalize(aug, p=2, dim=1)
        # Calculate similarity score between positive and augmented samples
        pos_score = torch.sum(pos * aug, dim=1)  # [batch_size]
        # Calculate similarity score matrix between positive samples and all augmented samples
        ttl_score = torch.matmul(pos, aug.permute(1, 0))  # [batch_size, batch_size]

        # Apply exponential function to positive similarity scores and divide by temperature parameter
        pos_score = torch.exp(pos_score / self.c_temp)  # [batch_size]
        # Apply exponential function to all similarity score matrix and sum along rows
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1)  # [batch_size]

        # Calculate contrastive loss
        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss

    def load_item_relations(self):# item_relations set independently of config
        item_relations = {}
        with open(os.path.join(self.conf['data_path'], self.conf['dataset'], 'item_relations.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line[:-1].split(' ')
                item1 = int(parts[0])
                item2 = int(parts[1])
                relation_type = float(parts[2])
                if relation_type == 3:## Have tried various combinations of 1 and 2 with no effect, next try 1 and 3
                    if item1 not in item_relations:
                        item_relations[item1] = {'positive': [], 'negative': []}
                    item_relations[item1]['positive'].append(item2)
                elif relation_type == 4:
                    if item1 not in item_relations:
                        item_relations[item1] = {'positive': [], 'negative': []}
                    item_relations[item1]['negative'].append(item2)
        '''
        all_items = set(range(self.num_items))
        for item in item_relations:
            positive_items = set(item_relations[item]['positive'])
            negative_items = set(item_relations[item]['negative'])  
            # If you need to complete negative relations, you can use the following code
            # negative_items = negative_items.union(all_items - positive_items - {item})
            item_relations[item]['negative'] = list(negative_items)
        '''
        return item_relations

    def cal_ii_single_item_loss(self, k):
        # Randomly sample k items from all items
        random_items = np.random.choice(self.num_items, k, replace=False)

        total_loss = 0
        valid_items_count = 0

        for item in random_items:
            if item in self.item_relations:
                # Get positive samples for the current item
                positive_samples = self.item_relations[item]['positive']
                num_positive = len(positive_samples)

                if num_positive > 0:
                    # Get negative candidates for the current item
                    negative_candidates = self.item_relations[item]['negative']

                    if len(negative_candidates) >= num_positive:
                        # Randomly select the same number of negative samples as positive samples from the negative candidates
                        negative_samples = np.random.choice(negative_candidates, num_positive, replace=False)

                        positive_samples = torch.tensor(positive_samples, device=self.device)
                        negative_samples = torch.tensor(negative_samples, device=self.device)

                        item_feature = F.normalize(self.items_feature_cold[item], p=2, dim=0)
                        positive_features = F.normalize(self.items_feature_cold[positive_samples], p=2, dim=1)
                        negative_features = F.normalize(self.items_feature_cold[negative_samples], p=2, dim=1)

                        positive_scores = torch.sum(item_feature * positive_features, dim=1)
                        negative_scores = torch.sum(item_feature * negative_features, dim=1)

                        positive_scores = torch.exp(positive_scores / self.c_temp)
                        negative_scores = torch.exp(negative_scores / self.c_temp)

                        total_scores = torch.cat([positive_scores, negative_scores])
                        ii_loss = - torch.mean(torch.log(positive_scores / total_scores.sum()))
                        total_loss += ii_loss
                        valid_items_count += 1

        if valid_items_count == 0:
            return torch.tensor(0.0, device=self.device)
        else:
            return total_loss / valid_items_count

    # Calculate total loss
    def cal_loss(self, users_feature, bundles_feature, users_feature_warm, bundles_feature_warm, users_feature_cold, bundles_feature_cold):
        # users_feature / bundles_feature: [bs, 1+neg_num, emb_size]
        # Calculate dot product of user features and bundle features as prediction score
        pred = torch.sum(users_feature * bundles_feature, 2)
        # Calculate BPR loss
        bpr_loss = cal_bpr_loss(pred)

        # Calculate user-view contrastive loss
        u_view_cl = self.cal_c_loss(users_feature, users_feature)
        # Calculate bundle-view contrastive loss
        b_view_cl = self.cal_c_loss(bundles_feature, bundles_feature)
        # Calculate scenario contrastive loss
        scen_c_loss = self.cal_c_loss(users_feature_warm, users_feature_cold) + self.cal_c_loss(bundles_feature_warm, bundles_feature_cold)

        # Calculate contrastive loss for IIgraph
        k = self.conf["k_item_loss"]  # Directly get the value of k from the configuration file
        # The value of k can be adjusted as needed
        ii_single_item_loss = self.cal_ii_single_item_loss(k)
        # Calculate contrastive loss
        if k == 0:
            c_losses = [u_view_cl, b_view_cl]
        else:
            c_losses = [u_view_cl, b_view_cl, 0.5*ii_single_item_loss]

        # Calculate average contrastive loss
        c_loss = sum(c_losses) / len(c_losses) + self.conf['scen_lambda']*scen_c_loss

        return bpr_loss, c_loss, ii_single_item_loss, scen_c_loss

    # Forward propagation function
    def forward(self, batch, ED_drop=False):
        # Edge dropping can be done per batch or per epoch, controlled by the training loop
        if ED_drop:
            # Regenerate propagation graph for user-bundle graph
            self.UB_propagation_graph = self.get_propagation_graph(self.H_iub, self.conf["UB_ratio"])
            # Regenerate propagation graph for user-item graph
            self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf["UI_ratio"])
            # Regenerate aggregation graph for user-item graph
            self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph, self.conf["UI_ratio"])
            # Regenerate propagation graph for bundle-item graph
            self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, self.conf["BI_ratio"])
            # Regenerate aggregation graph for bundle-item graph
            self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, self.conf["BI_ratio"])

        # users: [bs, 1], user indices in the batch
        # bundles: [bs, 1+neg_num], bundle indices in the batch
        users, bundles = batch
        # Get multi-modal user and bundle representations
        users_rep, bundles_rep, users_rep_warm, bundles_rep_warm, users_rep_cold, bundles_rep_cold = self.get_multi_modal_representations()

        # Get user representations based on user indices and expand dimensions to match the number of bundles
        users_embedding = users_rep[users].expand(-1, bundles.shape[1], -1)
        users_embedding_warm = users_rep_warm[users].expand(-1, bundles.shape[1], -1)
        users_embedding_cold = users_rep_cold[users].expand(-1, bundles.shape[1], -1)
        # Get bundle representations based on bundle indices
        bundles_embedding = bundles_rep[bundles]
        bundles_embedding_warm = bundles_rep_warm[bundles]
        bundles_embedding_cold = bundles_rep_cold[bundles]

        # Calculate BPR loss and contrastive loss
        bpr_loss, c_loss, ii_c_loss, scen_c_loss = self.cal_loss(users_embedding, bundles_embedding, users_embedding_warm, bundles_embedding_warm, users_embedding_cold, bundles_embedding_cold)

        return bpr_loss, c_loss, ii_c_loss, scen_c_loss

    # Evaluation function
    def evaluate(self, propagate_result, users):
        # Separate user features and bundle features from propagation result
        users_feature, bundles_feature, _, _, _, _ = propagate_result
        # Calculate score matrix between users and bundles
        scores = torch.mm(users_feature[users], bundles_feature.t())
        return scores