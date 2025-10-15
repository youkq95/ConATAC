#!/bin/bash
set -euo pipefail
export LANG=zh_CN.UTF-8
CHOOSE_DIR=$1
MODE=$2
MOTIF_FILE="./public_data/JASPAR2022_CORE_vertebrates_non-redundant_v2.meme"
INPUT_FASTA="${CHOOSE_DIR}/${MODE}/${MODE}.bed.fasta"
OUTPUT_BASE_DIR="${CHOOSE_DIR}/${MODE}/fimo_runs"
SPLIT_DIR="${CHOOSE_DIR}/${MODE}/fasta_splits"

NUM_PEAKS_PER_FILE=1000 
MAX_JOBS=86              

if [ ! -f "${MOTIF_FILE}" ]; then
  echo "error:MOTIF file not exist - ${MOTIF_FILE}"
  exit 1
fi

if [ ! -f "${INPUT_FASTA}" ]; then
  echo "error:FASTA file not exist - ${INPUT_FASTA}"
  exit 1
fi

echo "making work dic..."
mkdir -p "${SPLIT_DIR}"
mkdir -p "${OUTPUT_BASE_DIR}"

echo "split FASTA file..."
rm -f "${SPLIT_DIR}"/*.fa

awk -v size="${NUM_PEAKS_PER_FILE}" -v outdir="${SPLIT_DIR}" '
  BEGIN { n_seq=0; }
  /^>/ {
    if (n_seq % size == 0) {
      file = sprintf("%s/split_%d.fa", outdir, n_seq/size + 1);
      close(file);
    }
    n_seq++;
    print >> file;
  }
  !/^>/ { print >> file; }
' "${INPUT_FASTA}"

if [ -z "$(ls -A "${SPLIT_DIR}"/*.fa 2>/dev/null)" ]; then
  echo "error:FASTA file split failed"
  exit 1
fi

echo "start FIMO"

pids=()
job_count=0

fasta_chunks=("${SPLIT_DIR}"/*.fa)
for fasta_chunk in "${fasta_chunks[@]}"; do
  chunk_name=$(basename "${fasta_chunk}" .fa)
  output_dir="${OUTPUT_BASE_DIR}/${chunk_name}_fimo_out"
  
  rm -rf "${output_dir}"
  
  echo "Submitting ${chunk_name} task..."
  
  ../../meme/bin/fimo \
    --verbosity 1 \
    --no-pgc \
    --thresh 1.0E-4 \
    --oc "${output_dir}" \
    "${MOTIF_FILE}" \
    "${fasta_chunk}" &
  
  pids+=("$!")
  job_count=$((job_count + 1))
  
  if [ "$job_count" -ge "$MAX_JOBS" ]; then
    wait -n 
    job_count=$((job_count - 1)) 
  fi
done

echo "waiting..."
wait

echo "Merge all result files..."
output_file="${OUTPUT_BASE_DIR}/all_fimo_results.tsv"
fimo_files=()

while IFS= read -r -d '' file; do
	fimo_files+=("$file")
done < <(find "${OUTPUT_BASE_DIR}" -name "fimo.tsv" -print0)

if [ ${#fimo_files[@]} -eq 0 ]; then
	echo "Warning: No fimo.tsv file found for merging!"
	exit 1
fi

head -n 1 "${fimo_files[0]}" > "${output_file}"

for f in "${fimo_files[@]}"; do
	tail -n +2 "$f" >> "${output_file}"
done

echo "All operations completed! The merge results are saved in ${output_file}"

