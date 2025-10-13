import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import argparse

def process_group(df, label, train_df):
    df = df.copy()
    df['center'] = (df['start'] + df['end']) / 2
    df['start'] = df['center'].astype(int) - 1000
    df['end'] = df['center'].astype(int) + 1000
    df['newID'] = df['chr'] + '-' + df['start'].astype(str) + '-' + df['end'].astype(str)
    ids = df['newID'].tolist()
    mask = (
        train_df['chromosome'].isin(df['chr']) &
        train_df['start'].isin(df['start']) &
        train_df['end'].isin(df['end']) &
        train_df['ID'].isin(ids)
    )
    train_df.loc[mask, 'Y'] = label
    return train_df

def downsample_dataframe(df, target_column='Y'):
    count_0 = (df[target_column] == 0).sum()
    count_1 = (df[target_column] == 1).sum()
    if count_1 == 0 or count_0 == 0:
        print("Warning: The number of samples in a certain category is 0, making it impossible to balance sampling.")
        return df
    if count_1 > count_0:
        df_majority = df[df[target_column] == 1]
        df_minority = df[df[target_column] == 0]
    else:
        df_majority = df[df[target_column] == 0]
        df_minority = df[df[target_column] == 1]

    df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=42)
    df_balanced = pd.concat([df_majority_downsampled, df_minority]) \
                   .sample(frac=1, random_state=42) \
                   .reset_index(drop=True)
    return df_balanced

def save_split_to_npz(split_name, csv_path, npy_dir, output_dir):
    split_df = pd.read_csv(csv_path)
    data_list = []
    labels = []
    for _, row in split_df.iterrows():
        file_id = row['ID']
        label = row['Y']
        npy_file_path = os.path.join(npy_dir, f"{file_id}.npy")
        if os.path.exists(npy_file_path):
            matrix = np.load(npy_file_path)
            data_list.append(matrix)
            labels.append(label)
        else:
            print(f"no such file：{npy_file_path}")
    if len(data_list) == 0:
        print(f"warning：{split_name} no any data")
        return
    data_array = np.stack(data_list, axis=0)
    labels_array = np.array(labels)
    np.savez(os.path.join(output_dir, f"{split_name}_data.npz"), data=data_array)
    np.savez(os.path.join(output_dir, f"{split_name}_labels.npz"), labels=labels_array)
    print(f"{split_name} data shape={data_array.shape}, label length={len(labels_array)}\n")

def filter_sequences(df):
    filtered_df = df[df['sequence'].apply(lambda seq: seq.count('N') <= 100)]
    return filtered_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and balance DNA data.')
    parser.add_argument('--xlsx_path', type=str, required=True, help='Path to the Excel file')
    parser.add_argument('--cell_line', type=str, required=True, help='Name of the cell line in the Excel file')
    parser.add_argument('--mode', type=str, required=True, help='Mode')
    parser.add_argument('--bw_type', type=str, required=True, help='BW type')

    args = parser.parse_args()

    xlsx_path = args.xlsx_path
    cell_line = args.cell_line
    mode = args.mode
    bw_type = args.bw_type

    train_path = f'{cell_line}/{mode}/traindata.csv'
    out_path_withlabel = f'{cell_line}/{mode}/traindata_withlabel.csv'
    output_dir = f'{cell_line}/{mode}/traindata_{bw_type}'
    input_dir = f'{cell_line}/{mode}/features_{bw_type}/'

    p_dna = pd.read_excel(xlsx_path, sheet_name=cell_line)
    train_data = pd.read_csv(train_path)

    train_data = filter_sequences(train_data)

    train_data = process_group(p_dna[p_dna['condensation_propensity'] == 'HCP'], 1, train_data)
    train_data = process_group(p_dna[p_dna['condensation_propensity'] == 'LCP'], 2, train_data)

    df_selected = train_data[train_data['Y'].isin([1, 2])].copy()
    df_selected.loc[df_selected['Y'] == 2, 'Y'] = 0
    df_selected.to_csv(out_path_withlabel, index=False)

    train_val_df, test_df = train_test_split(df_selected, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.111, random_state=42)
    train_df_balanced = downsample_dataframe(train_df, target_column='Y')

    os.makedirs(output_dir, exist_ok=True)

    train_df_balanced.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    for split_name in ["train", "val", "test"]:
        csv_path = os.path.join(output_dir, f"{split_name}.csv")
        save_split_to_npz(split_name, csv_path, input_dir, output_dir)

    print("Data splitting completed (training set has been balanced and sampled), save directory:", output_dir)