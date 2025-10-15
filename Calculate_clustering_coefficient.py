#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import os
import argparse
import ast

chromosomes_order = [f'chr{i}' for i in range(1, 22 + 1)] + ['chrX']

def process_and_plot(all_peaks_df, gold_df, title_prefix, output_subdir, cell_line):
    output_dir = os.path.join(cell_line, f"{output_subdir}_kmeans_ari_curves")
    os.makedirs(output_dir, exist_ok=True)

    best_n_by_chr = {}
    chromosomes = sorted(all_peaks_df['chr'].unique())
    for chrom in chromosomes:
        print(f"--- processing {title_prefix} chr: {chrom} ---")
        chr_peaks_all = all_peaks_df[all_peaks_df['chr'] == chrom].reset_index(drop=True)
        chr_peaks_gold = gold_df[gold_df['chr'] == chrom]

        if chr_peaks_gold.empty or chr_peaks_gold['hub'].nunique() < 2:
            print(f"warning: {chrom} lacks sufficient hub data, skipped.")
            continue

        X = chr_peaks_all[['center']].to_numpy()
        max_ari = -1
        best_n = None
        ari_scores = []
        n_values = []
        n_range = range(2, min(len(chr_peaks_all), 400))

        peakid_to_idx = {pid: idx for idx, pid in enumerate(chr_peaks_all['peak_id'])}
        gold_indices = chr_peaks_gold['peak_id'].map(peakid_to_idx).to_numpy()

        for n in n_range:
            km = KMeans(n_clusters=n, init='k-means++', random_state=42, n_init=10)
            pred_labels = km.fit_predict(X)
            ari = adjusted_rand_score(
                chr_peaks_gold['hub'],
                pred_labels[gold_indices]
            )
            ari_scores.append(ari)
            n_values.append(n)
            if ari > max_ari:
                max_ari = ari
                best_n = n

        best_n_by_chr[chrom] = (best_n, max_ari)

        plt.figure(figsize=(12, 7))
        plt.plot(n_values, ari_scores, marker='.', linestyle='-', markersize=5, label='ARI Score')
        if best_n is not None:
            plt.axvline(x=best_n, color='r', linestyle='--', linewidth=1, label=f'Best n = {best_n}')
            plt.scatter(best_n, max_ari, color='red', s=100, zorder=5, label=f'Max ARI = {max_ari:.4f}')
        plt.title(f'{title_prefix} ARI vs. Number of Clusters (n) for {chrom}', fontsize=16)
        plt.xlabel('Number of Clusters (n)', fontsize=12)
        plt.ylabel('Adjusted Rand Index (ARI)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=10)
        output_filename = os.path.join(output_dir, f'{chrom}.pdf')
        plt.savefig(output_filename, format='pdf', bbox_inches='tight')
        plt.close()

    output_mode_dir = os.path.join(cell_line, output_subdir)
    os.makedirs(output_mode_dir, exist_ok=True)
    coefficient_file = os.path.join(output_mode_dir, f'{output_subdir}_coefficient.txt')
    with open(coefficient_file, 'w') as f:
        f.write(', '.join([f'"{chrom}":{n}' for chrom, (n, _) in best_n_by_chr.items() if n is not None]) + '\n')

    print(f"\n--- {output_subdir}_coefficient.txt down ---")
    return best_n_by_chr

def calculate_and_write_combined_coefficients(p_results, e_results, cell_line):
    ep_coefficients = {}
    for chrom in chromosomes_order:
        p_value = p_results.get(chrom, [0, 0])[0] if isinstance(p_results.get(chrom), tuple) else p_results.get(chrom)
        e_value = e_results.get(chrom, [0, 0])[0] if isinstance(e_results.get(chrom), tuple) else e_results.get(chrom)
        if p_value and e_value:
            ep_coefficients[chrom] = (p_value + e_value) / 2

    output_mode_dir = os.path.join(cell_line, 'EP')
    os.makedirs(output_mode_dir, exist_ok=True)
    coefficient_file = os.path.join(output_mode_dir, 'EP_coefficient.txt')
    with open(coefficient_file, 'w') as f:
        f.write(', '.join([f'"{chrom}":{v}' for chrom, v in ep_coefficients.items()]) + '\n')

    print("\n--- EP_coefficient.txt down ---")

def load_coefficient_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"can't find {path}, please check.")
    with open(path, 'r') as f:
        content = f.read().strip()
    if not content:
        return {}
    return ast.literal_eval("{" + content + "}")

def adjust_mouse_coefficients(file_path, cell_line):
    gm_df = pd.read_excel(file_path, sheet_name="GM12878")
    mouse_df = pd.read_excel(file_path, sheet_name=cell_line)
    mouse_df = mouse_df[mouse_df['chr'].astype(str).str.len() <= 6]
    
    mouse_chromosomes_order = [f'chr{i}' for i in range(1, 19 + 1)] + ['chrX']
    
    gm_df = gm_df[gm_df['chr'].isin(mouse_chromosomes_order)]
    mouse_df = mouse_df[mouse_df['chr'].isin(mouse_chromosomes_order)]

    gm_counts = gm_df['chr'].value_counts().to_dict()
    mouse_counts = mouse_df['chr'].value_counts().to_dict()
    coefficients = {}
    for chrom in mouse_chromosomes_order:
        gm_val = gm_counts.get(chrom, 0)
        mouse_val = mouse_counts.get(chrom, 0)
        if gm_val > 0:
            coefficients[chrom] = mouse_val / gm_val
        else:
            coefficients[chrom] = 0.0
    
    p_coeff_path = os.path.join("GM12878", 'P', 'P_coefficient.txt')
    if not os.path.exists(p_coeff_path):
        raise FileNotFoundError(f"There is no {p_coeff_path}")
    
    p_results = load_coefficient_file(p_coeff_path)
    
    adjusted_results = {}
    for chrom, val in p_results.items():
        coef = coefficients.get(chrom, 1.0)
        adjusted_results[chrom] = val * coef
    
    output_dir = os.path.join(cell_line, "P")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "P_coefficient.txt")
    with open(output_file, 'w') as f:
        f.write(", ".join([f'"{chrom}":{int(round(v))}' for chrom, v in adjusted_results.items()]) + "\n")
    
    print(f"\n--- P_coefficient.txt down ---")

def main(file_path, cell_line, mode, species):
    if species == "mouse":
        adjust_mouse_coefficients(file_path, cell_line)
        return 
    
    df = pd.read_excel(file_path, sheet_name=cell_line)
    df = df[df['chr'].astype(str).str.len() <= 6]
    df['chr'] = pd.Categorical(df['chr'], categories=chromosomes_order, ordered=True)
    df['center'] = (df['start'] + df['end']) / 2

    if mode == "P":
        all_peaks_df = df[df['chromHMM_state_type'] == 'Promoter']
        gold_df = all_peaks_df[(all_peaks_df['condensation_propensity'] == 'HCP') & (all_peaks_df['hub'].notna())]
        process_and_plot(all_peaks_df, gold_df, 'Promoter', 'P', cell_line)

    elif mode == "E":
        all_peaks_df = df[df['chromHMM_state_type'] == 'Enhancer']
        gold_df = all_peaks_df[(all_peaks_df['condensation_propensity'] == 'HCP') & (all_peaks_df['hub'].notna())]
        process_and_plot(all_peaks_df, gold_df, 'Enhancer', 'E', cell_line)

    elif mode == "EP":
        p_path = os.path.join(cell_line, 'P', 'P_coefficient.txt')
        e_path = os.path.join(cell_line, 'E', 'E_coefficient.txt')
        p_results = load_coefficient_file(p_path)
        e_results = load_coefficient_file(e_path)
        calculate_and_write_combined_coefficients(p_results, e_results, cell_line)

    else:
        raise ValueError("mode must be P, E or EP")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process clustering ARI curves for genomic data.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the Excel file (e.g. TableS1.xlsx)")
    parser.add_argument("--cell_line", type=str, required=True, help="Cell line in the Excel file (e.g. GM12878)")
    parser.add_argument("--mode", type=str, required=True, choices=["P", "E", "EP"], help="mode: P=Promoter, E=Enhancer, EP=Combine both")
    parser.add_argument("--species", type=str, required=True, choices=["human", "mouse"], help="Species type: human or mouse")
    args = parser.parse_args()
    main(args.file_path, args.cell_line, args.mode, args.species)