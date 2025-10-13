import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
import argparse
import re

def cosine_similarity(matrix):
    """计算余弦相似度矩阵
    输入: shape为(n_samples, n_features)的矩阵
    输出: shape为(n_samples, n_samples)的相似度矩阵
    """
    similarity_matrix = np.dot(matrix, matrix.T)
    norms = np.linalg.norm(matrix, axis=1)
    # 防止除零错误
    outer_norms = np.outer(norms, norms)
    outer_norms[outer_norms == 0] = 1e-10
    cosine_sim = similarity_matrix / outer_norms
    return cosine_sim


def process_file(input_count_file, motif_class_file, output_suffix, chrclsn, posdf):
    """处理单个数据文件并输出结果
    Args:
        input_count_file: 输入计数文件路径
        motif_class_file: motif 或 modification 分类文件路径
        output_suffix: 输出文件名后缀
        chrclsn: 染色体聚类数量配置
        posdf: 位置信息 DataFrame
    Returns:
        全部处理后的 DataFrame
    """
    # 读取数据
    countmotif = pd.read_csv(input_count_file).set_index('ID').fillna(0)
    motifcls = pd.read_csv(motif_class_file)
    allresultdict_ALL = {}  # 最外层字典，存储所有网络的结果

    for network in tqdm(motifcls['Cluster'].unique()):  # 获取所有独特的网络类型
        motifls = motifcls[motifcls['Cluster'] == network]['Motif'].to_list()  # 获取该网络对应的所有motif列表
        allresultdict = {}
        for i in list(chrclsn.keys()):
            # 1. 进行位置聚类
            clsnum = chrclsn[i]  # 当前染色体要分成的类别数
            kmdf = posdf[posdf['chromosome'] == i][['ID', 'position']]
            positions = kmdf.iloc[:, 1].values.reshape(-1, 1)
            kmeans = KMeans(n_clusters=clsnum,n_init=10,random_state=42)
            kmeans.fit(positions)
            kmdf['cluster'] = kmeans.labels_

            # 2. 计算每个聚类内的相似度
            sim_dict = {}
            for cls in list(kmdf['cluster'].unique()):
                subdf = kmdf[kmdf['cluster'] == cls]
                motifcount = countmotif[countmotif.index.isin(subdf['ID'])][motifls]
                similarity_matrix = cosine_similarity(motifcount)
                sim_dict[cls] = pd.DataFrame(similarity_matrix, index=list(motifcount.index),
                                             columns=list(motifcount.index))

            # 3. 对每个聚类的相似度矩阵进行处理
            all_sheets_results = []
            for key, df in sim_dict.items():
                processed_data = []
                indexls = []
                for index, row in df.iterrows():
                    # 后续列为数值部分，排除掉 1 并找到最大的两个数值
                    numeric_values = row.apply(pd.to_numeric, errors='coerce')  # 转换为数值类型，忽略无法转换的值
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
                # 创建 DataFrame
                processed_df = pd.DataFrame(processed_data, columns=['Feature'], index=indexls)
                processed_df['Sheet'] = key
                all_sheets_results.append(processed_df)

            # 汇总当前染色体所有 sheet 的结果
            final_result = pd.concat(all_sheets_results)
            allresultdict[i] = final_result
        allresultdict_ALL[network] = allresultdict

    # 聚合所有网络的结果
    allresultdf_list = []
    for key in tqdm(allresultdict_ALL.keys()):
        allresultdict = allresultdict_ALL[key]
        allresultls = []
        for i in allresultdict.keys():
            allresultls.append(allresultdict[i])
        allresultdf = pd.concat(allresultls)
        allresultdf.columns = [key + '_fea', key + 'cluster']  # 防止列重复
        allresultdf_list.append(allresultdf[key + '_fea'])  # 保存 fea 列

    # 合并所有特征列为一个 DataFrame
    allresultdf_ALL = pd.concat(allresultdf_list, axis=1)
    allresultdf_ALL_fillna0 = allresultdf_ALL.fillna(0)
    return allresultdf_ALL_fillna0



def read_chrclsn_from_file(coef_path):
    chrclsn = {}
    with open(coef_path, 'r') as f:
        content = f.read()

    # 用正则匹配 "chrX":数字
    matches = re.findall(r'"(chr[0-9XY]+)"\s*:\s*(\d+)', content)
    for chrom, num in matches:
        chrclsn[chrom] = int(num)

    return chrclsn

def combine_results(posdf, input_files, chrclsn):
    result_dfs = []
    for input_count_file, motif_class_file, output_suffix in input_files:
        # 注意：这里假设你在其他位置定义了 process_file 函数
        result_df = process_file(input_count_file, motif_class_file, output_suffix, chrclsn, posdf)
        result_dfs.append(result_df)
    # 按照索引拼接所有结果
    final_result_df = pd.concat(result_dfs, axis=1)
    return final_result_df

def main(posdf_path, input_files, out_path, coef_path):
    posdf = pd.read_csv(posdf_path, sep='\t', header=None)
    posdf.columns = ['chromosome', 'start', 'end', 'ID']
    posdf['position'] = (posdf['start'] + posdf['end']) / 2

    # 从外部 txt 文件读取 chrclsn
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

    # 解析输入文件为元组列表
    input_files = []
    for i in range(0, len(args.input_files), 3):
        if i + 2 < len(args.input_files):
            count_file = args.input_files[i]
            motif_class_file = args.input_files[i + 1]
            output_suffix = args.input_files[i + 2]
            input_files.append((count_file, motif_class_file, output_suffix))

    # 调用主程序
    main(args.posdf, input_files, args.out_path, args.coef)