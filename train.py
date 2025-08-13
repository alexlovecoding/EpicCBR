#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import json
import argparse
from tqdm import tqdm
from itertools import product
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
from utility import Datasets
from models.EpicCBR import EpicCBR

# Define command line argument parser
def get_cmd():
    # Create argument parser
    parser = argparse.ArgumentParser()
    # Experiment settings
    # Specify GPU id to use
    parser.add_argument("-g", "--gpu", default="0", type=str, help="which gpu to use")
    # Specify dataset to use, options: NetEase, iFashion
    parser.add_argument("-d", "--dataset", default="NetEase", type=str, help="which dataset to use, options: NetEase, iFashion")
    # Specify model to use, now only supports EpicCBR
    parser.add_argument("-m", "--model", default="EpicCBR", type=str, help="which model to use, options: EpicCBR")
    # Auxiliary info, will be appended to log file name
    parser.add_argument("-i", "--info", default="", type=str, help="any auxilary info that will be appended to the log file name")
    # Parse arguments
    args = parser.parse_args()

    return args

# Main function, program entry
def main():
    # Load config from file
    conf = yaml.safe_load(open("./config.yaml"))
    print("load config file done!")

    # Get command line arguments as dict
    paras = get_cmd().__dict__
    # Get dataset name
    dataset_name = paras["dataset"]

    # Ensure selected model is supported
    assert paras["model"] in ["EpicCBR"], "Pls select models from: EpicCBR"

    # Get config for dataset
    conf = conf[dataset_name]
    # Add dataset and model name to config
    conf["dataset"] = dataset_name.split("_")[0]
    conf["model"] = paras["model"]
    # Load dataset
    dataset = Datasets(conf)

    # Add GPU id and info from command line to config
    conf["gpu"] = paras["gpu"]
    conf["info"] = paras["info"]

    # Get user/bundle/item count from dataset
    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items

    # Set visible GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = conf["gpu"]
    # Select device (GPU or CPU)
    device = torch.device('cuda')
    conf["device"] = device
    print(conf)

    # Iterate over hyperparameter combinations
    for lr, l2_reg, UB_ratio, UI_ratio, BI_ratio, embedding_size, num_layers, c_lambda, c_temp, scenario_weight, scen_lambda, edges_enhanced_strength in \
            product(conf['lrs'], conf['l2_regs'], conf['UB_ratios'], conf['UI_ratios'], conf['BI_ratios'], conf["embedding_sizes"], conf["num_layerss"], conf["c_lambdas"], conf["c_temps"], conf["scenario_weights"], conf["scen_lambdas"], conf["edges_enhanced_strengths"]):
        # Define log, tensorboard, and checkpoint paths
        log_path = "./log/%s/%s" % (conf["dataset"], conf["model"])
        run_path = "./runs/%s/%s" % (conf["dataset"], conf["model"])
        checkpoint_model_path = "./checkpoints/%s/%s/model" % (conf["dataset"], conf["model"])
        checkpoint_conf_path = "./checkpoints/%s/%s/conf" % (conf["dataset"], conf["model"])
        # Create directories if needed
        if not os.path.isdir(run_path):
            os.makedirs(run_path)
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        if not os.path.isdir(checkpoint_model_path):
            os.makedirs(checkpoint_model_path)
        if not os.path.isdir(checkpoint_conf_path):
            os.makedirs(checkpoint_conf_path)

        # Update config for l2 regularization and embedding size
        conf["l2_reg"] = l2_reg
        conf["embedding_size"] = embedding_size

        # Experiment settings info list
        settings = []
        if conf["info"] != "":
            settings += [conf["info"]]

        settings += [conf["aug_type"]]
        if conf["aug_type"] == "ED":
            settings += [str(conf["ed_interval"])]
        if conf["aug_type"] == "OP":
            # When augmentation type is OP, ensure UB_ratio, UI_ratio, BI_ratio are 0
            assert UB_ratio == 0 and UI_ratio == 0 and BI_ratio == 0

        settings += ["Neg_%d" % (conf["neg_num"]), str(conf["batch_size_train"]), str(lr), str(l2_reg),
                     str(embedding_size)]

        # Update config for dropout rates and layer count
        conf["UB_ratio"] = UB_ratio
        conf["UI_ratio"] = UI_ratio
        conf["BI_ratio"] = BI_ratio
        conf["num_layers"] = num_layers
        settings += [str(UB_ratio), str(UI_ratio), str(BI_ratio), str(num_layers)]
        settings += ["_".join([str(conf['fusion_weights']["modal_weight"]), str(conf['fusion_weights']["UB_layer"]),
                               str(conf['fusion_weights']["UI_layer"]), str(conf['fusion_weights']["BI_layer"])])]

        # Update config for contrastive loss coefficient and temperature
        conf["c_lambda"] = c_lambda
        conf["c_temp"] = c_temp
        settings += [str(c_lambda), str(c_temp)]
        # Add scenario_weight to settings
        conf["scenario_weight"] = scenario_weight
        settings += [str(scenario_weight)]
        conf["scen_lambda"] = scen_lambda
        settings += [str(scen_lambda)]
        conf["edges_enhanced_strength"] = edges_enhanced_strength
        settings += [str(edges_enhanced_strength)]

        # Join settings into string
        setting = "_".join(settings)
        log_path = log_path + "/" + setting
        run_path = run_path + "/" + setting
        checkpoint_model_path = checkpoint_model_path + "/" + setting
        checkpoint_conf_path = checkpoint_conf_path + "/" + setting

        # Create tensorboard writer
        run = SummaryWriter(run_path)

        # Create model according to config
        if conf['model'] == 'EpicCBR':
            model = EpicCBR(conf, dataset.graphs).to(device)
        else:
            raise ValueError("Unimplemented model %s" % (conf["model"]))

        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=conf["l2_reg"])

        # Calculate batch count per epoch
        batch_cnt = len(dataset.train_loader)
        # Calculate test interval and edge drop interval in batches
        test_interval_bs = int(batch_cnt * conf["test_interval"])
        ed_interval_bs = int(batch_cnt * conf["ed_interval"])

        # Initialize best metrics and best performance
        best_metrics, best_perform = init_best_metrics(conf)
        best_epoch = 0
        # Start training loop
        for epoch in range(conf['epochs']):
            # Calculate anchor batch index for current epoch
            epoch_anchor = epoch * batch_cnt
            # Set model to train mode
            model.train(True)
            # Create progress bar
            pbar = tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader))

            for batch_i, batch in pbar:
                # Set model to train mode
                model.train(True)
                # Zero optimizer gradients
                optimizer.zero_grad()
                # Move batch data to device
                batch = [x.to(device) for x in batch]
                # Calculate global batch index
                batch_anchor = epoch_anchor + batch_i

                # Whether to perform edge drop
                ED_drop = False
                if conf["aug_type"] == "ED" and (batch_anchor + 1) % ed_interval_bs == 0:
                    ED_drop = True
                # Forward pass to compute loss
                bpr_loss, c_loss, ii_c_loss, scen_c_loss = model(batch, ED_drop=ED_drop)
                # Compute total loss
                loss = bpr_loss + conf["c_lambda"] * c_loss
                # Backward pass
                loss.backward()
                # Update model parameters
                optimizer.step()

                # Detach loss tensors to avoid gradient computation
                loss_scalar = loss.detach()
                bpr_loss_scalar = bpr_loss.detach()
                c_loss_scalar = c_loss.detach()
                ii_c_loss_scalar = ii_c_loss.detach()
                scen_c_loss_scalar = scen_c_loss.detach()
                # Write loss info to tensorboard
                run.add_scalar("loss_bpr", bpr_loss_scalar, batch_anchor)
                run.add_scalar("loss_c", c_loss_scalar, batch_anchor)
                run.add_scalar("loss", loss_scalar, batch_anchor)

                # Update progress bar description
                pbar.set_description("epoch: %d, loss: %.4f, bpr_loss: %.4f, c_loss: %.4f, ii_c_loss: %.4f, scen_c_loss: %.4f" %(epoch, loss_scalar, bpr_loss_scalar, c_loss_scalar, ii_c_loss_scalar, scen_c_loss_scalar))

                # Test at test interval
                if (batch_anchor + 1) % test_interval_bs == 0:
                    metrics = {}
                    # Test on validation set
                    metrics["val"] = test(model, dataset.val_loader, conf)
                    # Test on test set
                    metrics["test"] = test(model, dataset.test_loader, conf)
                    # Log metrics, update best metrics and best performance
                    best_metrics, best_perform, best_epoch = log_metrics(conf, model, metrics, run, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform, best_epoch)

# Initialize best metrics
def init_best_metrics(conf):
    best_metrics = {}
    best_metrics["val"] = {}
    best_metrics["test"] = {}
    for key in best_metrics:
        best_metrics[key]["recall"] = {}
        best_metrics[key]["ndcg"] = {}
    # Initialize best metrics to 0 for each topk value
    for topk in conf['topk']:
        for key, res in best_metrics.items():
            for metric in res:
                best_metrics[key][metric][topk] = 0
    best_perform = {}
    best_perform["val"] = {}
    best_perform["test"] = {}

    return best_metrics, best_perform

# Write metrics to log file and tensorboard
def write_log(run, log_path, topk, step, metrics):
    # Get current time
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Get validation and test metrics
    val_scores = metrics["val"]
    test_scores = metrics["test"]

    # Write metrics to tensorboard
    for m, val_score in val_scores.items():
        test_score = test_scores[m]
        run.add_scalar("%s_%d/Val" %(m, topk), val_score[topk], step)
        run.add_scalar("%s_%d/Test" %(m, topk), test_score[topk], step)

    # Build log strings for validation and test metrics
    val_str = "%s, Top_%d, Val:  recall: %f, ndcg: %f" %(curr_time, topk, val_scores["recall"][topk], val_scores["ndcg"][topk])
    test_str = "%s, Top_%d, Test: recall: %f, ndcg: %f" %(curr_time, topk, test_scores["recall"][topk], test_scores["ndcg"][topk])

    # Write log strings to log file
    log = open(log_path, "a")
    log.write("%s\n" %(val_str))
    log.write("%s\n" %(test_str))
    log.close()

    # Print log information
    print(val_str)
    print(test_str)

# Record metrics, update best metrics and best performance
def log_metrics(conf, model, metrics, run, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform, best_epoch):
    # Write log information for each topk value
    for topk in conf["topk"]:
        write_log(run, log_path, topk, batch_anchor, metrics)

    # Open log file
    log = open(log_path, "a")

    topk_ = 20
    print("top%d as the final evaluation standard" %(topk_))
    # Check if current validation metrics are better than best metrics
    if metrics["val"]["recall"][topk_] > best_metrics["val"]["recall"][topk_] and metrics["val"]["ndcg"][topk_] > best_metrics["val"]["ndcg"][topk_]:
        # Save model parameters
        torch.save(model.state_dict(), checkpoint_model_path)
        # Save config
        dump_conf = dict(conf)
        del dump_conf["device"]
        json.dump(dump_conf, open(checkpoint_conf_path, "w"))
        # Update best epoch
        best_epoch = epoch
        # Get current time
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Update best metrics
        for topk in conf['topk']:
            for key, res in best_metrics.items():
                for metric in res:
                    best_metrics[key][metric][topk] = metrics[key][metric][topk]

            # Update best performance information
            best_perform["test"][topk] = "%s, Best in epoch %d, TOP %d: REC_T=%.5f, NDCG_T=%.5f" %(curr_time, best_epoch, topk, best_metrics["test"]["recall"][topk], best_metrics["test"]["ndcg"][topk])
            best_perform["val"][topk] = "%s, Best in epoch %d, TOP %d: REC_V=%.5f, NDCG_V=%.5f" %(curr_time, best_epoch, topk, best_metrics["val"]["recall"][topk], best_metrics["val"]["ndcg"][topk])
            # Print best performance information
            print(best_perform["val"][topk])
            print(best_perform["test"][topk])
            # Write best performance information to log file
            log.write(best_perform["val"][topk] + "\n")
            log.write(best_perform["test"][topk] + "\n")

    # Close log file
    log.close()

    return best_metrics, best_perform, best_epoch

# Test on validation or test set
def test(model, dataloader, conf):
    tmp_metrics = {}
    # Initialize temporary metrics dictionary
    for m in ["recall", "ndcg"]:
        tmp_metrics[m] = {}
        for topk in conf["topk"]:
            tmp_metrics[m][topk] = [0, 0]

    # Get device
    device = conf["device"]
    # Set model to eval mode
    model.eval()
    # Get multi-modal representations
    rs = model.get_multi_modal_representations(test=True) ##user的feature组和bundle的feature组
    # Iterate over data loader
    for users, ground_truth_u_b, train_mask_u_b in dataloader:
        # Evaluate and get predictions
        pred_b = model.evaluate(rs, users.to(device))
        # Exclude interactions already in training set
        pred_b -= 1e8 * train_mask_u_b.to(device)
        # Compute metrics
        tmp_metrics = get_metrics(tmp_metrics, ground_truth_u_b.to(device), pred_b, conf["topk"])

    metrics = {}
    # Compute final metrics
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]

    return metrics


# Compute metrics
def get_metrics(metrics, grd, pred, topks):
    tmp = {"recall": {}, "ndcg": {}}
    # Compute recall and NDCG for each topk value
    for topk in topks:## topk列表的个数
        # Get topk indices of predictions for each user
        _, col_indice = torch.topk(pred, topk)
        # Generate corresponding row indices
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)##？没搞懂，似乎是返回用户对应行号
        
        # Check if predictions hit the ground truth
        is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)
        # Compute recall
        tmp["recall"][topk] = get_recall(pred, grd, is_hit, topk)
        # Compute NDCG
        tmp["ndcg"][topk] = get_ndcg(pred, grd, is_hit, topk)

    for m, topk_res in tmp.items():
        for topk, res in topk_res.items():
            for i, x in enumerate(res):
                metrics[m][topk][i] += x

    return metrics


def get_recall(pred, grd, is_hit, topk):
    epsilon = 1e-8
    # Count number of hits in topk predictions for each sample
    hit_cnt = is_hit.sum(dim=1)
    # Get number of positive samples for each sample
    num_pos = grd.sum(dim=1)
    num_pos = torch.tensor(num_pos, device=torch.device('cuda'))
    # Remove samples without positive samples (avoid division by zero)
    # Denominator is total sample count minus count of samples without positive samples
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    # Numerator is sum of recall for all samples, recall is hit count divided by positive sample count (plus small epsilon to avoid division by zero)
    nomina = (hit_cnt / (num_pos + epsilon)).sum().item()

    return [nomina, denorm]


def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk, device):
        # Compute Discounted Cumulative Gain (DCG)
        # Divide hit (0 or 1) by log2(position) from position 2 onwards (since log of 0 is undefined)
        hit = hit / torch.log2(torch.arange(2, topk + 2, device=device, dtype=torch.float))
        # Sum DCG for each sample
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        # Compute Ideal Discounted Cumulative Gain (IDCG)
        hit = torch.zeros(topk, dtype=torch.float,device=device)## tokp大小的行向量
        # Set positions within range of true positive sample count to 1
        hit[:num_pos] = 1##有多少个行向量就设置为多少，越多IDCG越高
        # Compute IDCG
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1 + topk, dtype=torch.float, device=device)
    IDCGs[0] = 1  # Avoid 0/0 case
    for i in range(1, topk + 1):
        # Pre-compute IDCG for different true positive sample counts
        IDCGs[i] = IDCG(i, topk, device)

    # Clamp true positive sample count between 0 and topk, convert to long
    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    # Compute DCG for current predictions
    dcg = DCG(is_hit, topk, device)

    # Get corresponding IDCG based on true positive sample count
    idcg = IDCGs[num_pos]
    # Compute Normalized Discounted Cumulative Gain (NDCG)
    ndcg = dcg / idcg.to(device)##is_hit()/num_pos(总共实际)
    # Denominator is total sample count minus count of samples without positive samples
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    # Numerator is sum of NDCG for all samples
    nomina = ndcg.sum().item()

    return [nomina, denorm]


if __name__ == "__main__":
    main()
