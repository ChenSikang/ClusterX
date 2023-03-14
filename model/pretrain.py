import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import dgl
import pickle
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler, SGD
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy import stats
from tqdm import tqdm
from dgllife.utils.eval import Meter
from torch.utils.data import DataLoader
from utils import DockingDataset_labeled, collate_pretrain, data_preprocessing, split_datasets, label2onehot, classification_performance_evaluation, draw_confusion_matrix, accuracy, Focal_Loss, rand_hyperparams, set_random_seed
from PotentialNet import PotentialNet
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
# NOTE: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = "0"

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def update_msg_from_scores(msg, scores):
    for metric, score in scores.items():
        msg += ', {} {:.4f}'.format(metric, score)
    return msg


def run_a_train_epoch(args, epoch, model, data_loader,
                      loss_criterion, optimizer, cls_losses, rec_losses, losses):
    model.train()

    epoch_loss = 0
    y_pred, y_target = [], []
    for i, batch in enumerate(data_loader):
        lg, bg1, bg2, idx, labels = batch

        adj_label = lg.adjacency_matrix().to_dense().to(device)
        bigraph_canonical = bg1.to(device)
        knn_graph = bg2.to(device)
        labels = labels.to(device)

        prediction, A_pred, z = model(bigraph_canonical, knn_graph)
        
        cls_loss = loss_criterion(prediction, labels.squeeze().long())

        rec_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        loss = 10 * cls_loss + rec_loss

        cls_losses.append(float(cls_loss))
        rec_losses.append(float(rec_loss))
        losses.append(float(loss))
        y_pred.append(prediction)
        y_target.append(labels.squeeze().long())

        epoch_loss += loss.data.item() * len(idx)
        print(
            f'loss:{loss}, cls_loss{cls_loss}, rec_loss{rec_loss}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    accuracy_score, precision_score, recall_score, f1_macro, f1_micro = classification_performance_evaluation(
        y_pred, y_target)
    avg_loss = epoch_loss / len(data_loader.dataset)
    total_scores = {'accuracy': accuracy_score, 'precision': precision_score,
                    'recall': recall_score, 'f1_macro': f1_macro, 'f1_micro': f1_micro}
    msg = 'epoch {:d}/{:d}, training | loss {:.4f}, cls_loss{}, rec_loss{}'.format(
        epoch + 1, args.epochs, avg_loss, cls_loss, rec_loss)
    msg = update_msg_from_scores(msg, total_scores)
    print(msg)

    if epoch + 1 == args.epochs:
        draw_confusion_matrix(y_pred, y_target, draw=True)
    # else:
    #     draw_confusion_matrix(y_pred, y_target)

    return total_scores


def run_an_eval_epoch(args, model, data_loader):
    model.eval()

    with torch.no_grad():
        y_pred, y_target = [], []
        for i, batch in enumerate(data_loader):
            lg, bg1, bg2, idx, labels = batch
            labels = labels.to(device)
            bigraph_canonical = bg1.to(device)
            knn_graph = bg2.to(device)

            prediction, A_pred, z = model(bigraph_canonical, knn_graph)

            y_pred.append(prediction)
            y_target.append(labels.squeeze().long())

    accuracy_score, precision_score, recall_score, f1_macro, f1_micro = classification_performance_evaluation(
        y_pred, y_target)
    total_scores = {'accuracy': accuracy_score, 'precision': precision_score,
                    'recall': recall_score, 'f1_macro': f1_macro, 'f1_micro': f1_micro}
    return total_scores


def train(args):
    # only for init

    # datasets = pickle.load(open(args.data_dir, 'rb')).item()
    datasets = np.load(args.data_dir, allow_pickle=True).item()
    train_set, val_set, test_set = split_datasets(datasets, train_ratio=0.6)

    iteration = 0
    train_dataloader = DataLoader(DockingDataset_labeled(data_file=train_set),
                                  batch_size=args.batch_size,
                                  collate_fn=collate_pretrain,
                                  shuffle=True,
                                  drop_last=True,
                                  pin_memory=True)
    val_dataloader = DataLoader(DockingDataset_labeled(data_file=val_set),
                                batch_size=args.batch_size,
                                collate_fn=collate_pretrain,
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True)
    test_dataloader = DataLoader(DockingDataset_labeled(data_file=test_set),
                                 batch_size=args.batch_size,
                                 collate_fn=collate_pretrain,
                                 shuffle=True,
                                 drop_last=True,
                                 pin_memory=True)
    print(f'train_dataloader sample: {len(train_dataloader.dataset.ligands)}')
    print(f'val_dataloader sample: {len(val_dataloader.dataset.ligands)}')
    print(f'test_dataloader sample: {len(test_dataloader.dataset.ligands)}')

    model = PotentialNet(n_etypes=len(args.distance_bins)+5,
                         f_in=args.f_in,
                         f_bond=args.f_bond,
                         f_spatial=args.f_spatial,
                         f_gather=args.f_gather,
                         n_rows_fc=args.n_rows_fc,
                         n_bond_conv_steps=args.n_bond_conv_steps,
                         n_spatial_conv_steps=args.n_spatial_conv_steps,
                         out_size=97,
                         dropouts=args.dropouts)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9,
    #                 weight_decay=args.weight_decay, nesterov=False)

    model.train()
    model.to(0)
    tqdm.write(str(model))
    tqdm.write(
        "{} trainable parameters."
        .format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    tqdm.write(
        "{} total parameters."
        .format(sum(p.numel() for p in model.parameters())))

    trial_train_acc, trial_val_acc, trial_test_acc = np.zeros(
        args.trials), np.zeros(args.trials), np.zeros(args.trials)

    for trial in range(args.trials):
        print(f'\n Running trial {trial + 1}/{args.trials}: \n')

        train_acc, val_acc, test_acc = np.zeros(
            args.epochs), np.zeros(args.epochs), np.zeros(args.epochs)
        cls_losses, rec_losses, losses = [], [], []
        for epoch in range(args.epochs):
            train_scores = run_a_train_epoch(
                args, epoch, model, train_dataloader, loss_fn, optimizer, cls_losses, rec_losses, losses)
            train_acc[epoch] = train_scores['accuracy']
            if len(val_set) > 0:
                val_scores = run_an_eval_epoch(
                    args, model, val_dataloader)
                val_msg = update_msg_from_scores(
                    'validation results', val_scores)
                print(val_msg)
                val_acc[epoch] = val_scores['accuracy']
            if len(test_set) > 0:
                test_scores = run_an_eval_epoch(
                    args, model, test_dataloader)
                test_msg = update_msg_from_scores('test results', test_scores)
                print(test_msg)
                test_acc[epoch] = test_scores['accuracy']
            tqdm.write('train_scores {}, val_scores:{}, test_scores:{}'.
                       format(train_scores, val_scores, test_scores))

            checkpoint_dict = {'stage1': model.stage_1_model.state_dict(
            ), 'stage2': model.stage_2_model.state_dict()}
            torch.save(
                checkpoint_dict, f"/home/sikang/cluster-AE/odc-pnet/pretrain/pretrain_trial{trial}_epoch{epoch}.pkl"
            )

        # save results on the epoch with best validation acc
        best_epoch = np.argmax(val_acc)
        trial_train_acc[trial] = train_acc[best_epoch]
        trial_val_acc[trial] = val_acc[best_epoch]
        trial_test_acc[trial] = test_acc[best_epoch]
        print('Best test epoch: ', best_epoch + 1)

def checkpoint_model(model, epoch, step, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    model.train()

    checkpoint_dict = {
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "step": step,
        "epoch": epoch,
    }

    torch.save(checkpoint_dict, output_path)

    # return the computed metrics so it can be used to update the training loop
    return checkpoint_dict


def sum_ligand_features(h, batch_num_nodes):
    """
    Compute the sum of only ligand features `h` according to the batch information `batch_num_nodes`.
    """
    node_nums = torch.cumsum(batch_num_nodes, dim=0)
    B = int(len(batch_num_nodes) / 2)  # actual batch size
    ligand_idx = [list(range(node_nums[0]))]  # first ligand
    for i in range(2, len(node_nums), 2):  # the rest of ligands in the batch
        ligand_idx.append(list(range(node_nums[i-1], node_nums[i])))
    # sum over each ligand
    return ligand_idx


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='train model')

    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--work_dir', type=str, default='./work_dir')
    parser.add_argument('--checkpoint', type=bool, default=True)
    parser.add_argument('--checkpoint_iter', type=int, default=10000)
    parser.add_argument('--checkpoint_dir', type=str, default='./saveMod')

    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--dropouts', type=list, default=[0.25, 0.25, 0.25])
    parser.add_argument('--weight_decay', type=int, default=1e-05)

    parser.add_argument('--max_num_neighbors', type=int, default=4)
    parser.add_argument('--distance_bins', type=list,
                        default=[1.5, 2.5, 3.5, 4.5])
    parser.add_argument('--f_in', type=int, default=44)
    # has to be larger than f_in
    parser.add_argument('--f_bond', type=int, default=48)
    parser.add_argument('--f_gather', type=int, default=48)
    # better to be the same as f_gather
    parser.add_argument('--f_spatial', type=int, default=48)
    parser.add_argument('--n_rows_fc', type=list, default=[48, 24])
    parser.add_argument('--n_bond_conv_steps', type=int, default=2)
    parser.add_argument('--n_spatial_conv_steps', type=int, default=1)

    args = parser.parse_args()
    print(args)
    train(args)
