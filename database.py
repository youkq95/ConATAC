# -*- coding: utf-8 -*-

import os
import argparse
import datetime
import logging
import traceback
import tempfile
import shutil
import sqlite3
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import pyBigWig
from sklearn.preprocessing import OneHotEncoder

def init_worker(motif_db_path, motif_to_family_map, new_features_arr, extra_features_arr, bw_filenames_list):
    global worker_motif_db_path, worker_motif_to_family, worker_new_features, worker_extra_features, worker_bw_files

    logging.info(f"Initializing worker PID: {os.getpid()}")

    worker_motif_db_path = motif_db_path
    worker_motif_to_family = motif_to_family_map
    worker_new_features = new_features_arr
    worker_extra_features = extra_features_arr
    worker_bw_files = [pyBigWig.open(bw_file) for bw_file in bw_filenames_list]

def prepare_motif_database(csv_path, db_path):
    if os.path.exists(db_path):
        logging.info(f"Found existing motif database: {db_path}. Skipping creation.")
        return

    logging.warning(f"Motif database not found. Creating new one from {csv_path}. This may take a while...")

    try:
        conn = sqlite3.connect(db_path)
        chunk_size = 500000  
        for i, chunk in enumerate(pd.read_csv(
                csv_path,
                names=["motif_alt_id", "strand", "chr", "s1", "e1"],
                dtype={"motif_alt_id": "str", "strand": "str", "chr": "str", "s1": "str", "e1": "str"},
                chunksize=chunk_size
        )):
            chunk['s1'] = pd.to_numeric(chunk['s1'], errors='coerce')
            chunk['e1'] = pd.to_numeric(chunk['e1'], errors='coerce')
            chunk.dropna(inplace=True)
            chunk.to_sql('motifs', conn, if_exists='append', index=False)
            logging.info(f"  ... Wrote chunk {i + 1}")

        logging.info("Finished writing data. Now creating database indexes...")
        cursor = conn.cursor()
        cursor.execute("CREATE INDEX idx_chr ON motifs (chr);")
        cursor.execute("CREATE INDEX idx_s1 ON motifs (s1);")
        cursor.execute("CREATE INDEX idx_e1 ON motifs (e1);")
        conn.commit()
        logging.info("Database indexes created successfully.")

    except Exception as e:
        logging.error(f"Failed to create motif database: {e}")
        if os.path.exists(db_path):
            os.remove(db_path)
        raise
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'processing_log_{timestamp}.log')
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - PID:%(process)d - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger()


def load_motif_to_family(filename):
    motif_to_family = {}
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip().rstrip(',')
            if line and ':' in line:
                key, value = line.split(':', 1)
                motif_to_family[key.strip().strip("'\"")] = value.strip().strip("'\"")
    return motif_to_family


def sliding_window(chromosome, start, end, sequence, segment_motifs, all_families_sorted, family_index_map,
                   window_size=200, step_size=100):
    windows_meta, features_list = [], []
    if len(sequence) < window_size:
        return windows_meta, np.array(features_list)

    bigwig_data_segment = [np.nan_to_num(bw.values(chromosome, start - 1, end, numpy=True), nan=0.0) for bw in
                           worker_bw_files]

    for i in range(0, len(sequence) - window_size + 1, step_size):
        sub_seq = sequence[i: i + window_size]
        sub_start, sub_end = start + i, start + i + window_size - 1
        windows_meta.append({'chr': chromosome, 'start': sub_start, 'end': sub_end})

        nt_array = np.fromiter(sub_seq, dtype='<U1')
        nucleotide_counts = np.sum(nt_array[:, None] == np.array(['A', 'T', 'C', 'G']), axis=0)
        at_content = (nucleotide_counts[0] + nucleotide_counts[1]) / window_size
        cg_content = (nucleotide_counts[2] + nucleotide_counts[3]) / window_size

        bw_sums = [bw_arr[i: i + window_size].sum() for bw_arr in bigwig_data_segment]

        motifs_in_window = segment_motifs[(segment_motifs['s1'] <= sub_end) & (segment_motifs['e1'] >= sub_start)]

        if not motifs_in_window.empty:
            motif_seqs = motifs_in_window.apply(lambda row: sub_seq[max(0, int(row['s1'] - sub_start)):min(window_size,
                                                                                                           int(row[
                                                                                                                   'e1'] - sub_start) + 1)],
                                                axis=1)
            motif_at_content = (motif_seqs.str.count('A') + motif_seqs.str.count('T')).sum()
            motif_cg_content = (motif_seqs.str.count('C') + motif_seqs.str.count('G')).sum()
            family_counts_series = motifs_in_window['motif_alt_id'].map(worker_motif_to_family).value_counts()
            family_counts = family_counts_series.reindex(all_families_sorted, fill_value=0).values
        else:
            motif_at_content, motif_cg_content = 0, 0
            family_counts = np.zeros(len(all_families_sorted), dtype=int)

        combined_features = np.concatenate(
            [[at_content, cg_content], bw_sums, [motif_at_content, motif_cg_content], family_counts])
        features_list.append(combined_features)

    return windows_meta, np.array(features_list)

def process_segment_refactored(args):
    i, row, window_size, step_size, all_families_sorted, family_index_map, save_dir = args
    chromosome, start, end, sequence = row['chromosome'], row['start'], row['end'], row['sequence']

    conn = None
    try:
        conn = sqlite3.connect(worker_motif_db_path)
        query = "SELECT * FROM motifs WHERE chr = ? AND s1 <= ? AND e1 >= ?;"
        segment_motifs = pd.read_sql_query(query, conn, params=(chromosome, end, start))

        windows_meta, features = sliding_window(
            chromosome, start, end, sequence, segment_motifs,
            all_families_sorted, family_index_map, window_size, step_size
        )

        if features.shape[0] == 0:
            return (i, True, "No windows generated, skipped file save.")

        new_features_for_windows = np.tile(worker_new_features[i], (features.shape[0], 1))
        extra_features_for_windows = np.tile(worker_extra_features[i], (features.shape[0], 1))
        segment_all_features = np.column_stack((features, new_features_for_windows, extra_features_for_windows))

        filename = f"{chromosome}-{start}-{end}.npy"
        save_path = os.path.join(save_dir, filename)
        np.save(save_path, segment_all_features)

        return (i, True, save_path)

    except Exception:
        return (i, False, traceback.format_exc())
    finally:
        if conn:
            conn.close()

def main(args):
    output_dir = os.path.join(args.work_dir, args.test_dir)
    log_dir = os.path.join(output_dir, "logs")
    save_dir = os.path.join(output_dir, f"features_{args.bw_type}")
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(log_dir)
    logger.info(f"Starting feature extraction process with args: {args}")
    logger.warning(
        "Output mode is set to individual .npy files per segment. This may be slow due to high I/O operations.")

    try:
        motif_csv_path = os.path.join(output_dir, "motif_pos.csv")
        motif_db_path = os.path.join(output_dir, "motif_pos.sqlite")
        prepare_motif_database(motif_csv_path, motif_db_path)

        traindata_path = os.path.join(output_dir, "traindata.csv")
        data = pd.read_csv(traindata_path)

        motif_to_family = load_motif_to_family(args.motif_family_file)
        all_families = sorted(set(motif_to_family.values()))
        family_index = {family: idx for idx, family in enumerate(all_families)}

        logger.info("Loading and aligning feature files...")
        fea_df = pd.read_csv(os.path.join(output_dir, "motif_sim.csv")).rename({'Unnamed: 0': 'ID'}, axis=1)
        dist_df = pd.read_csv(os.path.join(output_dir, "distance.csv"))
        data = pd.merge(data, fea_df, on='ID', how='left').merge(dist_df, on='ID', how='left')

        feature_columns = [col for col in data.columns if col.endswith('_fea')]
        new_features = data[feature_columns].values
        extra_feature_columns = [col for col in data.columns if col.endswith('Distance_to_TSS_bp')]
        extra_features = data[extra_feature_columns].values

        initializer_args = (motif_db_path, motif_to_family, new_features, extra_features, args.bw_filenames)

        tasks = [(i, row, args.window_size, args.step_size, all_families, family_index, save_dir) for i, row in
                 data.iterrows()]

        logger.info(f"Starting processing of {len(data)} segments with {args.num_workers} workers.")

        success_count = 0
        failure_count = 0

        with Pool(processes=args.num_workers, initializer=init_worker, initargs=initializer_args) as pool:
            results = pool.imap_unordered(process_segment_refactored, tasks)

            for i, (idx, status, message) in enumerate(results):
                if status:
                    success_count += 1
                    if (success_count % 500 == 0): 
                        logger.info(f"Successfully processed {success_count} segments...")
                else:
                    failure_count += 1
                    logger.error(f"Failed to process segment index {idx}. Error:\n{message}")

        logger.info("=== Processing Complete ===")
        logger.info(f"Total segments processed: {len(data)}")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Failed: {failure_count}")
        logger.info(f"Output files are located in: {save_dir}")


    except Exception as e:
        logger.error(f"A fatal error occurred in the main process: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="High-performance feature extraction for genomic sequences (V2 - DB enabled).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    run_group = parser.add_argument_group('Standard Execution Arguments')
    run_group.add_argument('--work_dir', type=str, help='Path to the work directory.')
    run_group.add_argument('--test_dir', type=str, help='Path to the test/experiment sub-directory.')
    run_group.add_argument('--bw_filenames', type=str, nargs='+', help='List of bigWig filenames separated by spaces.')
    run_group.add_argument('--motif_family_file', type=str, help='Path to the motif-to-family mapping file.')
    run_group.add_argument('--bw_type', type=str, choices=['sc', 'bulk'], help='Type of bw file: sc or bulk.')
    config_group = parser.add_argument_group('Configuration Arguments')
    config_group.add_argument('--num_workers', type=int, default=8, help='Number of worker processes to use.')
    config_group.add_argument('--window_size', type=int, default=200, help='Size of the sliding window.')
    config_group.add_argument('--step_size', type=int, default=100, help='Step size for the sliding window.')
    args = parser.parse_args()

    required_args = ['work_dir', 'test_dir', 'bw_filenames', 'motif_family_file', 'bw_type']
    if not all(getattr(args, arg) for arg in required_args):
        parser.error(
            "For a standard run, --work_dir, --test_dir, --bw_filenames, --bw_type and --motif_family_file are required.")
    main(args)