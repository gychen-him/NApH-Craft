#!/bin/bash
# Batch run AlphaFold3 for all mutant sequences

: "${AF3_DIR:?Set AF3_DIR to the AlphaFold3 source directory}"
: "${AF3_MODEL_DIR:?Set AF3_MODEL_DIR to the AlphaFold3 model directory}"
: "${AF3_DB_DIR:?Set AF3_DB_DIR to the AlphaFold3 database directory}"
INPUT_DIR=$(dirname $0)
OUTPUT_DIR=$INPUT_DIR/af3_outputs
mkdir -p $OUTPUT_DIR

echo 'Running 8PSB_C17-C21...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/8PSB_C17-C21.json \
    --output_dir=$OUTPUT_DIR/8PSB_C17-C21 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 8JFQ_C3-C11...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/8JFQ_C3-C11.json \
    --output_dir=$OUTPUT_DIR/8JFQ_C3-C11 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 7ZEM_C19-C21...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/7ZEM_C19-C21.json \
    --output_dir=$OUTPUT_DIR/7ZEM_C19-C21 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 7OQT_C9-C11...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/7OQT_C9-C11.json \
    --output_dir=$OUTPUT_DIR/7OQT_C9-C11 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 6GH0_C10-C18...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/6GH0_C10-C18.json \
    --output_dir=$OUTPUT_DIR/6GH0_C10-C18 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 8R4W_C3-C14...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/8R4W_C3-C14.json \
    --output_dir=$OUTPUT_DIR/8R4W_C3-C14 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 7X8N_C10-C23...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/7X8N_C10-C23.json \
    --output_dir=$OUTPUT_DIR/7X8N_C10-C23 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 7ATZ_C3-C7...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/7ATZ_C3-C7.json \
    --output_dir=$OUTPUT_DIR/7ATZ_C3-C7 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 7ALU_C1-C15...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/7ALU_C1-C15.json \
    --output_dir=$OUTPUT_DIR/7ALU_C1-C15 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 6ZX6_C6-C21...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/6ZX6_C6-C21.json \
    --output_dir=$OUTPUT_DIR/6ZX6_C6-C21 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 6XT7_C3-C21...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/6XT7_C3-C21.json \
    --output_dir=$OUTPUT_DIR/6XT7_C3-C21 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 6L8M_C3-C8...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/6L8M_C3-C8.json \
    --output_dir=$OUTPUT_DIR/6L8M_C3-C8 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 6JCD_C16-C21...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/6JCD_C16-C21.json \
    --output_dir=$OUTPUT_DIR/6JCD_C16-C21 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 6FTU_C5-C21...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/6FTU_C5-C21.json \
    --output_dir=$OUTPUT_DIR/6FTU_C5-C21 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 5ZEV_C3-C8...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/5ZEV_C3-C8.json \
    --output_dir=$OUTPUT_DIR/5ZEV_C3-C8 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 5YEY_C6-C18...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/5YEY_C6-C18.json \
    --output_dir=$OUTPUT_DIR/5YEY_C6-C18 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 5O4D_C10-C23...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/5O4D_C10-C23.json \
    --output_dir=$OUTPUT_DIR/5O4D_C10-C23 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 2O3M_C16-C20...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/2O3M_C16-C20.json \
    --output_dir=$OUTPUT_DIR/2O3M_C16-C20 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 2O3M_C17-C19...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/2O3M_C17-C19.json \
    --output_dir=$OUTPUT_DIR/2O3M_C17-C19 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 2O3M_C1-C12...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/2O3M_C1-C12.json \
    --output_dir=$OUTPUT_DIR/2O3M_C1-C12 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 2M27_C2-C13...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/2M27_C2-C13.json \
    --output_dir=$OUTPUT_DIR/2M27_C2-C13 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 2LPW_C2-C18...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/2LPW_C2-C18.json \
    --output_dir=$OUTPUT_DIR/2LPW_C2-C18 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 2KYP_C1-C13...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/2KYP_C1-C13.json \
    --output_dir=$OUTPUT_DIR/2KYP_C1-C13 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 2KM3_C11-C14...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/2KM3_C11-C14.json \
    --output_dir=$OUTPUT_DIR/2KM3_C11-C14 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 2HY9_C14-C25...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/2HY9_C14-C25.json \
    --output_dir=$OUTPUT_DIR/2HY9_C14-C25 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 2GKU_C1-C20...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/2GKU_C1-C20.json \
    --output_dir=$OUTPUT_DIR/2GKU_C1-C20 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 2GKU_C13-C24...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/2GKU_C13-C24.json \
    --output_dir=$OUTPUT_DIR/2GKU_C13-C24 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 2F8U_C10-C15...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/2F8U_C10-C15.json \
    --output_dir=$OUTPUT_DIR/2F8U_C10-C15 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

echo 'Running 1XAV_C20-C22...'
python $AF3_DIR/run_alphafold.py \
    --json_path=$INPUT_DIR/1XAV_C20-C22.json \
    --output_dir=$OUTPUT_DIR/1XAV_C20-C22 \
    --model_dir="${AF3_MODEL_DIR}" \
    --db_dir="${AF3_DB_DIR}"

