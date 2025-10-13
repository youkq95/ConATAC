#!/usr/bin/env python3
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from copy import deepcopy
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import pyranges as pr
import shutil
import argparse
import subprocess

def processmotiffile(fimoresult):
    hmmmotif=pd.read_csv(fimoresult,sep='\t',comment='#')
    hmmmotif['start']=hmmmotif['start'].astype(int)
    hmmmotif['stop']=hmmmotif['stop'].astype(int)
    hmmmotif['ID']=hmmmotif['sequence_name'].str.split('(',expand=True)[0].str.replace(':','-')
    hmmmotif['s']=hmmmotif['ID'].str.split('-',expand=True)[1].astype(int)
    hmmmotif['e']=hmmmotif['ID'].str.split('-',expand=True)[2].astype(int)
    hmmmotif['chr']=hmmmotif['ID'].str.split('-',expand=True)[0]
    hmmmotif['s1'] = hmmmotif.apply(lambda row: row['s'] + row['start'] , axis=1)
    hmmmotif['e1'] = hmmmotif.apply(lambda row: row['s'] + row['stop'] , axis=1)
    return hmmmotif

def read_fasta(file_path):
    fasta_dict = {}
    with open(file_path, 'r') as file:
        header = None
        sequence = []
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if header is not None:
                    fasta_dict[header.replace(':','-')] = ''.join(sequence)
                header = line[1:]
                sequence = []  
            else:
                sequence.append(line)
        if header is not None:
            fasta_dict[header.replace(':','-')] = ''.join(sequence)
    return fasta_dict

def CPSC_pipeline(cell_line, work_dic, mode, bw_type, file_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pre_dic = os.path.join(work_dic, cell_line, mode)
    os.makedirs(pre_dic, exist_ok=True)

    print(f"all files saved to: {pre_dic}")

    # 读取 Excel
    sheet_name = cell_line
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    filtered_df = df[df['end'] - df['start'] <= 10000]
    ATAC = filtered_df[['chr', 'start', 'end', 'chromHMM_state_type', 'condensation_propensity']]
    ATAC = ATAC.drop_duplicates(['chr','start','end'])

    # 模式选择
    if mode == 'P':
        sel=['Promoter']
    elif mode == 'E':
        sel=['Enhancer']
    elif mode == 'EP':
        sel=['Enhancer','Promoter']

    p_dna = ATAC[ATAC['chromHMM_state_type'].isin(sel)].copy()
    p_dna['center'] = (p_dna['start']+p_dna['end'])/2
    p_dna.to_csv(f"{pre_dic}/{mode}_dna.bed", index=False, sep='\t', header=None)

    pcol4 = pd.DataFrame()
    pcol4['chromosome'] = p_dna['chr']
    pcol4['start'] = p_dna['center'].astype(int) - 1000
    pcol4['end'] = p_dna['center'].astype(int) + 1000
    pcol4['ID'] = pcol4['chromosome']+'-'+pcol4['start'].astype(str)+'-'+pcol4['end'].astype(str)
    pcol4 = pcol4[pcol4['chromosome'].str.len() < 6]
    pcol4.to_csv(f"{pre_dic}/{mode}.bed", index=False, sep='\t', header=None)

    if not os.path.exists(f"{pre_dic}/{mode}.bed.fasta"):
        genome_path = os.path.join(script_dir, "public_data", "GRCh38.p13.genome.fa")
        subprocess.run([
            "bedtools", "getfasta",
            "-fi", genome_path,
            "-bed", f"{pre_dic}/{mode}.bed",
            "-fo", f"{pre_dic}/{mode}.bed.fasta"
        ], check=True)
    else:
        print(f"{mode}.bed.fasta exists.")

    print(cell_line+' :Successfully Get Fasta ')

    if not os.path.exists(f"{pre_dic}/fimo_runs/all_fimo_results.tsv"):
        fimo_path = os.path.join(script_dir, "fimo.sh")
        subprocess.run(["bash", fimo_path, cell_line, mode], check=True, cwd=work_dic)
    else:
        print(f"all_fimo_results.tsv exists.")

    print(cell_line + ': Successfully Get MOTIF')

    if not os.path.exists(f"{pre_dic}/motif_count.csv") or not os.path.exists(f"{pre_dic}/motif_pos.csv"):
        fimoresult = f"{pre_dic}/fimo_runs/all_fimo_results.tsv"
        hmmmotif = processmotiffile(fimoresult)
        hmmmotif[hmmmotif['ID'].isin(pcol4['ID'])][['motif_alt_id','strand','chr','s1','e1']] \
            .to_csv(f"{pre_dic}/motif_pos.csv", index=False)
        allregionmotif = hmmmotif[hmmmotif['ID'].isin(pcol4['ID'])]
        countmotif = allregionmotif.pivot_table(
            index='ID', columns='motif_alt_id', aggfunc='count', values='sequence_name').fillna(0)
        countmotif.to_csv(f"{pre_dic}/motif_count.csv")
    else:
        print(f"motif_count exists.")

    print(cell_line+' :Successfully Get motif_count')

    # 模体相似度
    if not os.path.exists(f"{pre_dic}/motif_sim.csv"):
        cal_sim_path = os.path.join(script_dir, "cal_sim.py")
        motif_path = os.path.join(script_dir, "Data", "motif_family_9.csv")
        subprocess.run([
            "python", cal_sim_path,
            f"{pre_dic}/{mode}.bed",
            f"{pre_dic}/motif_count.csv",
            motif_path,
            f"{mode}_motif",
            f"{pre_dic}/motif_sim.csv",
            "--coef", f"{pre_dic}/{mode}_coefficient.txt"
        ], stdout=open(f"{pre_dic}/cal_sim.log", "w"), stderr=subprocess.STDOUT)
    else:
        print(f"motif_sim.csv exists.")

    print(cell_line + ' :Successfully Get motif_sim')

    # 训练数据
    if not os.path.exists(f"{pre_dic}/traindata.csv"):
        fasta_dict = read_fasta(f"{pre_dic}/{mode}.bed.fasta")
        TRAIN = pcol4.copy()
        TRAIN['Y'] = 0
        TRAIN['sequence'] = TRAIN['ID'].map(fasta_dict)
        TRAIN.to_csv(f"{pre_dic}/traindata.csv", index=False)
    else:
        print(f"traindata exists.")

    print(cell_line+' :Successfully Get traindata')

    # 距离计算
    if not os.path.exists(f"{pre_dic}/distance.csv"):
        df_regions = pd.read_csv(f"{pre_dic}/traindata.csv")
        df_regions['Chromosome'] = df_regions['chromosome']
        df_regions['Start'] = pd.to_numeric(df_regions['start'])
        df_regions['End'] = pd.to_numeric(df_regions['end'])
        pr_regions = pr.PyRanges(df_regions.reset_index(drop=False))
        gtf_path = os.path.join(script_dir, "public_data", "gencode.v47.basic.annotation.gtf")
        pr_gtf = pr.read_gtf(gtf_path)
        pr_transcripts = pr_gtf[pr_gtf.Feature == 'transcript']
        tss_df = pr_transcripts.df.copy()
        tss_df['TSS'] = tss_df.apply(lambda row: row['Start'] if row['Strand'] == '+' else row['End'], axis=1)
        tss_df['End'] = tss_df['TSS'] + 1
        pr_tss = pr.PyRanges(tss_df[['Chromosome', 'TSS', 'End', 'gene_name', 'gene_id', 'Strand']].rename(columns={'TSS': 'Start'}))
        nearest_tss = pr_regions.nearest(pr_tss)
        result_df = nearest_tss.df
        result_df = result_df[['Chromosome', 'Start', 'End', 'gene_name', 'gene_id', 'Strand', 'Distance']]
        result_df['ID'] = result_df['Chromosome'].astype(str) + '-' + result_df['Start'].astype(str) + '-' + result_df['End'].astype(str)
        result_df = result_df.rename(columns={'Distance': 'Distance_to_TSS_bp'})
        result_df.to_csv(f"{pre_dic}/distance.csv", index=False)
    else:
        print(f"distance exists.")

    print(cell_line+' :Successfully Get distance')

    # database.sh
    if not os.path.exists(f"{pre_dic}/features_{bw_type}"):
        database_path = os.path.join(script_dir, "database.sh")
        subprocess.run(["bash", database_path, cell_line, bw_type, mode], check=True, cwd=work_dic)
    else:
        print(f"features exists.")

    print(f"all {mode} features ready")


def main():
    parser = argparse.ArgumentParser(description="Run CPSC pipeline")
    parser.add_argument("--work_dic", type=str, required=True, help="work dictionary")
    parser.add_argument("--cell_line", type=str, required=True, help="cell line")
    parser.add_argument("--mode", type=str, required=True, choices=["P", "E", "EP"], help="mode")
    parser.add_argument("--bw_type", type=str, required=True, help="BigWig type")
    parser.add_argument("--file_path", type=str, required=True, help="Excel path")
    args = parser.parse_args()

    CPSC_pipeline(args.cell_line, args.work_dic, args.mode, args.bw_type, args.file_path)
    print('\n'+args.cell_line+'\nAll file ready\n################\n')

if __name__ == "__main__":
    main()