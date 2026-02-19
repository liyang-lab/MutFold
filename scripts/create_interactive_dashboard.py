#!/usr/bin/env python3
"""
Create Interactive Dashboard 
"""
import sys
from pathlib import Path

BASE_DIR = Path('set/base/directory/pathh')
sys.path.insert(0, str(BASE_DIR))

from mutfold_modules.visualization.enhanced_heatmap import EnhancedDMSHeatmap

# Input Protein Domain ID 
protein_id = 'protein_ID'

# Set data paths
pdb = BASE_DIR / f'path/to/pdb/file'
dms = BASE_DIR / f'path/to/dms/data'
fasta = BASE_DIR / f'path/to/fasta/data'
per_residue = BASE_DIR / f'path/to/per/residue/plddt/analysis'
tm_score_fie = BASE_DIR / f'path/to/tm/data'
lddt_score_file = BASE_DIR / f'path/to/lddt/data'

# Set structure file paths
structure_dir = BASE_DIR / 'path/to/directory/with/all/structure/info'
exp_structure = BASE_DIR / 'path/to/experimental/structures'
mutfold_structure = structure_dir / 'path'
af2_structure = structure_dir / 'path'
super_structure = structure_dir / 'path'


output = BASE_DIR / 'set/output/directory'

print(f"Creating interactive dashboard for {protein_id}")

heatmap = EnhancedDMSHeatmap(protein_id)
heatmap.load_dms_data(str(dms))
heatmap.load_structure_quality(str(pdb))
heatmap.load_plddt_improvement(str(per_residue))

if fasta.exists():
    heatmap.load_sequence(str(fasta))

if tm_score_file.exists():
    heatmap.load_tm_score(str(tm_score_file))

if lddt_score_file.exists():
    heatmap.load_lddt_score(str(lddt_score_file))

heatmap.set_structure_files(
    mutfold_pdb=str(mutfold_structure) if mutfold_structure.exists() else None,
    af2_pdb=str(af2_structure) if af2_structure.exists() else None
)

heatmap.generate_enhanced_heatmap(str(output))

print(f"✓ Success: {output}")
