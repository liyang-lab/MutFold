#!/bin/bash
#SBATCH --job-name=esm2_concat_test
#SBATCH --account=jianzhi1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=esm2_concat_test_%j.log
#SBATCH --error=esm2_concat_test_%j.err

echo "Date: $(date)"

module load singularity

# Update File Paths 
TEST_PROTEINS=(
    "protein-ID"
)

export DMSFOLD_DIR="path"
export CONTAINER="path"
export PYTHON_PACKAGES="path"
export FASTA_DIR="path"
export DMS_DIR="path"
export ALIGNMENTS="path"
export OUTPUT="path"
export ESM2_DIR="path"

# Create a temporary directory to store data 

export TEMP_DIR="path"
mkdir -p ${TEMP_DIR}/{fasta,dms,alignments} ${OUTPUT}

echo "Setting up ${#TEST_PROTEINS[@]} proteins..."
for PROTEIN in "${TEST_PROTEINS[@]}"; do
    echo "  - ${PROTEIN}"
    cp ${FASTA_DIR}/${PROTEIN}.fasta ${TEMP_DIR}/fasta/
    cp ${DMS_DIR}/${PROTEIN}_dms.csv ${TEMP_DIR}/dms/${PROTEIN}_dms.csv
    
    if [ -d "${ALIGNMENTS}/${PROTEIN}" ]; then
        mkdir -p ${TEMP_DIR}/alignments/${PROTEIN}
        cp -r ${ALIGNMENTS}/${PROTEIN}/* ${TEMP_DIR}/alignments/${PROTEIN}/
    fi
done

echo "Running MutFold with ESM-2 integration..."

singularity exec --nv \
    --env PYTHONUSERBASE=${PYTHON_PACKAGES} \
    --bind ${DMSFOLD_DIR}:/dmsfold \
    --bind ${TEMP_DIR}/fasta:/fasta \
    --bind ${TEMP_DIR}/dms:/dms \
    --bind ${TEMP_DIR}/alignments:/alignments \
    --bind ${OUTPUT}:/output \
    --bind ${ESM2_DIR}:/esm2 \
    --bind ${PYTHON_PACKAGES}:${PYTHON_PACKAGES} \
    ${CONTAINER} \
    bash -c "
        export PYTHONPATH=/dmsfold:\$PYTHONPATH
        cd /dmsfold
        
        python3 predict_with_dmsfold.py \
            /fasta/ \
            /dms/ \
            /dmsfold/examples/mmcif_dummy/ \
            --openfold_checkpoint_path /dmsfold/openfold/resources/dmsfold_weights.pt \
            --use_precomputed_alignments /alignments/ \
            --output_dir /output \
            --model_device cuda:0 \
            --config_preset model_5_ptm \
            --skip_relaxation
    "

echo ""
echo "Job completed: $(date)"
echo ""
echo "Results saved to: ${OUTPUT}/predictions/"
ls -lh ${OUTPUT}/predictions/
