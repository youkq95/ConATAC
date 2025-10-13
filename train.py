# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from copy import deepcopy
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from Model import ConvAndAttentionModel
import os
import pandas as pd
from FocalLoss import FocalLoss
from sklearn.utils import class_weight
import pickle
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run the model with specified parameters.')
    parser.add_argument('--cell_line', type=str, required=True, help='Name of the cell line')
    parser.add_argument('--mode', type=str, required=True, help='Mode for the data')
    parser.add_argument('--bw_type', type=str, required=True, help='BW type for the data')
    return parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_data(data_file, labels_file):
    data = np.load(data_file)['data']
    labels = np.load(labels_file)['labels']
    return data, labels


def process_features_norm(features):
    n_samples, n_timesteps, n_features = features.shape
    scaled_features = features.copy()
    for feature_idx in range(n_features):
        if feature_idx == 2:
            feature_column = scaled_features[:, :, feature_idx].reshape(-1, 1)
            feature_column = np.log1p(feature_column)
            minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
            normalized_results = minmax_scaler.fit_transform(feature_column)
            scaled_features[:, :, feature_idx] = normalized_results.reshape((n_samples, n_timesteps))
    return scaled_features


def process_features(features, base_dir):
    n_samples, n_timesteps, n_features = features.shape
    scaled_features = features.copy()
    scaler_metrics = np.zeros((n_features, 4))
    for feature_idx in range(n_features):
        feature_column = features[:, :, feature_idx].reshape(-1, 1)
        std_scaler = StandardScaler()
        std_results = std_scaler.fit_transform(feature_column)
        scaler_metrics[feature_idx, 0] = std_scaler.mean_[0]
        scaler_metrics[feature_idx, 1] = std_scaler.scale_[0]
        minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
        normalized_results = minmax_scaler.fit_transform(std_results)
        scaler_metrics[feature_idx, 2] = minmax_scaler.min_[0]
        scaler_metrics[feature_idx, 3] = minmax_scaler.scale_[0]
        scaled_features[:, :, feature_idx] = normalized_results.reshape((n_samples, n_timesteps))
    with open(f'{base_dir}scaler/scaler_metrics_V2.pkl',
              'wb') as f:
        pickle.dump(scaler_metrics, f)
    return scaled_features


def process_features_predict(features, base_dir):
    with open(f'{base_dir}/scaler/scaler_metrics_V2.pkl',
              'rb') as f:
        scaler_metrics = pickle.load(f)
    n_samples, n_timesteps, n_features = features.shape
    scaled_features = features.copy()
    for feature_idx in range(n_features):
        feature_column = scaled_features[:, :, feature_idx].reshape(-1, 1)
        stdmean = scaler_metrics[feature_idx, 0]
        stdscale = scaler_metrics[feature_idx, 1]
        feature_column -= stdmean
        feature_column /= stdscale
        minmaxmin = scaler_metrics[feature_idx, 2]
        minmaxscale = scaler_metrics[feature_idx, 3]
        feature_column *= minmaxscale
        feature_column += minmaxmin
        clipped_data = np.clip(feature_column, -1, 1)
        scaled_features[:, :, feature_idx] = clipped_data.reshape((n_samples, n_timesteps))
    return scaled_features


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_data, batch_labels in train_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()
    accuracy = 100.0 * correct / total
    return running_loss / len(train_loader), accuracy


def evaluate(model, val_loader, criterion, device, threshold=0.5, return_preds=False):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    if return_preds:
        all_labels = []
        all_probs = []

    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            val_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            predicted = (probs[:, 1] >= threshold).long()
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

            if return_preds:
                all_labels.append(batch_labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

    accuracy = 100.0 * correct / total

    if return_preds:
        final_labels = np.concatenate(all_labels, axis=0)
        final_probs = np.concatenate(all_probs, axis=0)
        return val_loss / len(val_loader), accuracy, final_labels, final_probs
    else:
        return val_loss / len(val_loader), accuracy

def main():
    args = parse_arguments()

    cell_line = args.cell_line
    mode = args.mode
    bw_type = args.bw_type

    base_dir = f'{cell_line}/{mode}/traindata_{bw_type}/'
    if not os.path.exists(base_dir + 'scaler'):
        os.makedirs(base_dir + 'scaler', exist_ok=True)
        os.makedirs(base_dir + 'picture', exist_ok=True)
        os.makedirs(base_dir + 'predict_test', exist_ok=True)

    data_dic = f'{base_dir}'

    train_data, train_labels = load_data(data_dic + '/train_data.npz', data_dic + '/train_labels.npz')
    val_data, val_labels = load_data(data_dic + '/val_data.npz', data_dic + '/val_labels.npz')
    test_data, test_labels = load_data(data_dic + '/test_data.npz', data_dic + '/test_labels.npz')

    train_data_Norm = process_features_norm(train_data)
    val_data_Norm = process_features_norm(val_data)
    test_data_Norm = process_features_norm(test_data)

    train_data_scaled = process_features(train_data_Norm, data_dic)
    val_data_scaled = process_features_predict(val_data_Norm, data_dic)
    test_data_scaled = process_features_predict(test_data_Norm, data_dic)

    train_dataset = CustomDataset(train_data_scaled, train_labels)
    val_dataset = CustomDataset(val_data_scaled, val_labels)
    test_dataset = CustomDataset(test_data_scaled, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    first_conv_out_channels = 32
    first_kernel_size = (2, 24)
    second_conv_out_channels = 64
    second_kernel_size = (2, 32)
    attention_output_dim = 16
    maxpool_kernel_size = 16
    model = ConvAndAttentionModel(
        first_conv_out_channels=first_conv_out_channels,
        first_kernel_size=first_kernel_size,
        second_conv_out_channels=second_conv_out_channels,
        second_kernel_size=second_kernel_size,
        attention_output_dim=attention_output_dim,
        maxpool_kernel_size=maxpool_kernel_size)
    ground_truth = train_labels
    unique_classes = np.unique(ground_truth)
    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=unique_classes,
                                                      y=ground_truth)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.FloatTensor(class_weights).to(device)
    criterion = FocalLoss(alpha=0.5, gamma=2, weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    model.to(device)
    num_epochs = 200
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    min_val_loss = float('inf')
    best_epoch = 0
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_epoch = epoch + 1
    print(f"model saved")
    print(f"Minimum test loss {min_val_loss:.4f} at epoch {best_epoch} ")
    model_save_path = f"{data_dic}/Model_epoch{best_epoch}.pth"
    torch.save(model.state_dict(), model_save_path)

    best_model = ConvAndAttentionModel(
        first_conv_out_channels=first_conv_out_channels,
        first_kernel_size=first_kernel_size,
        second_conv_out_channels=second_conv_out_channels,
        second_kernel_size=second_kernel_size,
        attention_output_dim=attention_output_dim,
        maxpool_kernel_size=maxpool_kernel_size
    )
    best_model.load_state_dict(torch.load(model_save_path, weights_only=True))
    best_model.to(device)
    _, _, test_true_labels, test_probs = evaluate(best_model, test_loader, criterion, device, return_preds=True)

    np.save(data_dic + f'/predict_test/test_probs.npy', test_probs)
    np.save(data_dic + f'/predict_test/test_labels.npy', test_true_labels)


if __name__ == '__main__':
    main()