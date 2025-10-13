import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from Model import ConvAndAttentionModel
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Model prediction script parameters")
    parser.add_argument("--predict_cell_line", type=str, required=True, help="The cell line to be predicted, such as GM12878")
    parser.add_argument("--mode", type=str, required=True, help="mode, P or E")
    parser.add_argument("--bw_type", type=str, required=True, help="bigwig type, sc or bulk")
    parser.add_argument("--model_cell_line", type=str, required=True, help="Cell lines used to train the model, such as K562")
    parser.add_argument("--model_path", type=str, required=True, help="model path .pth")

    return parser.parse_args()

# ------------------ 数据处理 ------------------
def process_and_save_data(input_dir, labels_csv, output_dir):
    """读取 features_sc 中的 .npy 数据，并匹配标签，保存为 npz"""
    data_list = []
    labels = []

    # 加载标签 CSV
    labels_df = pd.read_csv(labels_csv)

    # 根据文件 ID 逐个读取 .npy
    for npy_file in labels_df['ID']:
        npy_file_name = npy_file + '.npy'
        file_path = os.path.join(input_dir, npy_file_name)

        matrix = np.load(file_path)
        data_list.append(matrix)

        # 查找对应标签
        label = labels_df.loc[labels_df['ID'] == npy_file, 'Y'].values
        if len(label) == 0:
            raise ValueError(f"The label corresponding to file {npy_file} was not found.  ")
        labels.append(label[0])

    # 转 numpy 数组
    data_array = np.stack(data_list, axis=0)
    labels_array = np.array(labels)

    # 保存到 output_dir
    np.savez(os.path.join(output_dir, "predict_data.npz"), data=data_array)
    np.savez(os.path.join(output_dir, "predict_labels.npz"), labels=labels_array)

    print(f"数据 shape={data_array.shape}, 标签 length={len(labels_array)}\n")


class CustomDataset(Dataset):
    """自定义 PyTorch Dataset，用于模型推理"""
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ------------------ 模型加载 ------------------
def load_model(model_save_path):
    """加载 ConvAndAttentionModel 并切换到 eval 模式"""
    model = ConvAndAttentionModel(
        first_conv_out_channels=32,
        first_kernel_size=(2, 24),
        second_conv_out_channels=64,
        second_kernel_size=(2, 32),
        attention_output_dim=16,
        maxpool_kernel_size=16
    )
    model.load_state_dict(torch.load(model_save_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model


# ------------------ 数据加载 ------------------
def load_data(data_file, labels_file):
    """从 npz 文件中读取 data 和 labels"""
    data = np.load(data_file)['data']
    labels = np.load(labels_file)['labels']
    return data, labels


def process_features_norm(features):
    """对特征做 log1p 和 MinMaxScaler 归一化（只处理 feature_idx == 2）"""
    n_samples, n_timesteps, n_features = features.shape
    scaled_features = features.copy()

    for feature_idx in range(n_features):
        if feature_idx == 2:
            feature_column = scaled_features[:, :, feature_idx].reshape(-1, 1)
            feature_column = np.log1p(feature_column)  # log1p
            minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
            normalized_results = minmax_scaler.fit_transform(feature_column)
            scaled_features[:, :, feature_idx] = normalized_results.reshape((n_samples, n_timesteps))

    return scaled_features


def process_features_predict(features, data_dir):
    """基于保存的 scaler_metrics 标准化 + MinMax 截断到 [-1,1]"""
    with open(f'{data_dir}scaler/scaler_metrics_V2.pkl', 'rb') as f:
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


def load_and_process_data(file_data, file_labels, data_dir):
    """加载并归一化 features"""
    data, labels = load_data(file_data, file_labels)
    data_norm = process_features_norm(data)
    data_scaled = process_features_predict(data_norm, data_dir)
    return data_scaled, labels


# ------------------ 模型推理 ------------------
def test_model_on_data(model, data_loader):
    """用模型在 DataLoader 上推理"""
    all_probs = []
    all_labels = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            batch_data = batch_data.to(device)
            outputs = model(batch_data)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch_labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def model_predict(predict_dir, model_save_path, data_dir):
    """加载模型并推理，保存预测结果"""
    data_to_test_path = os.path.join(predict_dir, 'predict_data.npz')
    labels_to_test_path = os.path.join(predict_dir, 'predict_labels.npz')

    # 加载模型
    model = load_model(model_save_path)

    # 加载并处理测试数据
    test_data_scaled, test_labels = load_and_process_data(data_to_test_path, labels_to_test_path, data_dir)

    # 创建 DataLoader
    test_dataset = CustomDataset(test_data_scaled, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 推理
    predicted_probs, predicted_labels = test_model_on_data(model, test_loader)

    np.save(os.path.join(predict_dir, 'predicted_probs.npy'), predicted_probs)
    np.save(os.path.join(predict_dir, 'predicted_labels.npy'), predicted_labels)

    print("The prediction results have been saved as .npy file.")


# ------------------ 主程序入口 ------------------
def main():
    args = parse_arguments()

    predict_cell_line = args.predict_cell_line
    mode = args.mode
    bw_type = args.bw_type
    model_cell_line = args.model_cell_line
    model_path = args.model_path

    base_dir = f'{predict_cell_line}/{mode}/'
    data_dir = f'{predict_cell_line}/{mode}/traindata_{bw_type}/'

    # 检查 features_sc 是否存在
    if not os.path.exists(f"{predict_cell_line}/{mode}/features_{bw_type}"):
        print('feature not OK')

    # 生成 predict_data.npz 与 predict_labels.npz
    if not os.path.exists(f"{predict_cell_line}/{mode}/predict_based_{model_cell_line}/predict_labels.npz"):
        input_dir = f"{predict_cell_line}/{mode}/features_{bw_type}"
        output_dir = f"{predict_cell_line}/{mode}/predict_based_{model_cell_line}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        process_and_save_data(input_dir, f"{predict_cell_line}/{mode}/traindata_withlabel.csv", output_dir)
    else:
        print("feature for model exist.")

    print("feature for model ready.")

    # 模型推理
    if not os.path.exists(f"{predict_cell_line}/{mode}/predict_based_{model_cell_line}/predicted_probs.npy"):
        model_save_path = f"{model_path}"
        model_predict(f"{predict_cell_line}/{mode}/predict_based_{model_cell_line}", model_save_path, data_dir)
    else:
        print("predicted_labels exist.")

    print("predicted ready.")

    # 合并预测概率到原始 CSV
    alldata = pd.read_csv(f"{predict_cell_line}/{mode}/traindata_withlabel.csv")
    predicted_probs = np.load(f"{predict_cell_line}/{mode}/predict_based_{model_cell_line}/predicted_probs.npy")
    alldata['predicted_probs'] = predicted_probs[:, 1]
    choosedata = alldata[['ID', 'predicted_probs']]
    choosedata.to_csv(f"{predict_cell_line}/{mode}/predict_based_{model_cell_line}/probs.bed", index=False)

if __name__ == "__main__":
    main()