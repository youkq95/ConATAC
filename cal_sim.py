import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
import argparse
import re

def cosine_similarity(matrix):
    similarity_matrix = np.dot(matrix, matrix.T)
    norms = np.linalg.norm(matrix, axis=1)
    outer_norms = np.outer(norms, norms)
    outer_norms[outer_norms == 0] = 1e-10
    cosine_sim = similarity_matrix / outer_norms
    return cosine_sim

def process_file(input_count_file, motif_class_file, output_suffix, chrclsn, posdf):
    countmotif = pd.read_csv(input_count_file).set_index('ID').fillna(0)
    motifcls = pd.read_csv(motif_class_file)
    allresultdict_ALL = {}

    for network in tqdm(motifcls['Cluster'].unique()):
        motifls = motifcls[motifcls['Cluster'] == network]['Motif'].to_list()
        allresultdict = {}
        for i in list(chrclsn.keys()):
            clsnum = chrclsn[i]
            kmdf = posdf[posdf['chromosome'] == i][['ID', 'position']]
            if kmdf.empty:
                continue
            positions = kmdf.iloc[:, 1].values.reshape(-1, 1)
            kmeans = KMeans(n_clusters=clsnum, n_init=10, random_state=42)
            kmeans.fit(positions)
            kmdf['cluster'] = kmeans.labels_
            sim_dict = {}
            for cls in list(kmdf['cluster'].unique()):
                subdf = kmdf[kmdf['cluster'] == cls]
                motifcount = countmotif[countmotif.index.isin(subdf['ID'])][motifls]
                similarity_matrix = cosine_similarity(motifcount)
                sim_dict[cls] = pd.DataFrame(similarity_matrix, index=list(motifcount.index),
                                             columns=list(motifcount.index))
            all_sheets_results = []
            for key, df in sim_dict.items():
                processed_data = []
                indexls = []
                for index, row in df.iterrows():
                    numeric_values = row.apply(pd.to_numeric, errors='coerce')
                    filtered_values = numeric_values.round(3)[numeric_values.round(3) != 1]
                    if len(filtered_values) == 0:
                        feature_value = 0
                    elif len(filtered_values) == 1:
                        feature_value = filtered_values.iloc[0]
                    else:
                        top_two_values = filtered_values.nlargest(2)
                        feature_value = top_two_values.mean()
                    processed_data.append([feature_value])
                    indexls.append(index)
                processed_df = pd.DataFrame(processed_data, columns=['Feature'], index=indexls)
                processed_df['Sheet'] = key
                all_sheets_results.append(processed_df)

            final_result = pd.concat(all_sheets_results)
            allresultdict[i] = final_result
        allresultdict_ALL[network] = allresultdict

    allresultdf_list = []
    for key in tqdm(allresultdict_ALL.keys()):
        allresultdict = allresultdict_ALL[key]
        allresultls = []
        for i in allresultdict.keys():
            allresultls.append(allresultdict[i])
        allresultdf = pd.concat(allresultls)
        allresultdf.columns = [key + '_fea', key + 'cluster']
        allresultdf_list.append(allresultdf[key + '_fea'])

    allresultdf_ALL = pd.concat(allresultdf_list, axis=1)
    allresultdf_ALL_fillna0 = allresultdf_ALL.fillna(0)
    return allresultdf_ALL_fillna0



def read_chrclsn_from_file(coef_path):
    chrclsn = {}
    with open(coef_path, 'r') as f:
        content = f.read()
    matches = re.findall(r'"(chr[0-9XY]+)"\s*:\s*(\d+)', content)
    for chrom, num in matches:
        chrclsn[chrom] = int(num)

    return chrclsn

def combine_results(posdf, input_files, chrclsn):
    result_dfs = []
    for input_count_file, motif_class_file, output_suffix in input_files:
        result_df = process_file(input_count_file, motif_class_file, output_suffix, chrclsn, posdf)
        result_dfs.append(result_df)
    final_result_df = pd.concat(result_dfs, axis=1)
    return final_result_df

def main(posdf_path, input_files, out_path, coef_path):
    posdf = pd.read_csv(posdf_path, sep='\t', header=None)
    posdf.columns = ['chromosome', 'start', 'end', 'ID']
    posdf['position'] = (posdf['start'] + posdf['end']) / 2

    chrclsn = read_chrclsn_from_file(coef_path)

    final_result_df = combine_results(posdf, input_files, chrclsn)
    final_result_df.to_csv(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some input files.')
    parser.add_argument('posdf', type=str, help='Path to the posdf file (CSV or TXT)')
    parser.add_argument('input_files', type=str, nargs='+', help='Paths to input files in the format: count_file, motif_class_file, output_suffix')
    parser.add_argument('out_path', type=str, help='Path to the output file')
    parser.add_argument('--coef', type=str, required=True, help='Path to coefficient txt file')

    args = parser.parse_args()

    input_files = []
    for i in range(0, len(args.input_files), 3):
        if i + 2 < len(args.input_files):
            count_file = args.input_files[i]
            motif_class_file = args.input_files[i + 1]
            output_suffix = args.input_files[i + 2]
            input_files.append((count_file, motif_class_file, output_suffix))

    main(args.posdf, input_files, args.out_path, args.coef)