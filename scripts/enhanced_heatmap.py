"""
Enhanced MutFold Heatmap with Structure Quality Integration
Shows DMS Fitness + pLDDT improvement + TM-score + lDDT overlay
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Bio.PDB import PDBParser
from Bio import SeqIO
import warnings
warnings.filterwarnings('ignore')


class EnhancedDMSHeatmap:
    """
    Enhanced heatmap showing mutation effects AND structure quality
    """

    def __init__(self, protein_id, sequence_offset=None):
        self.protein_id = protein_id
        self.amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
        self.dms_data = None
        self.structure_data = None
        self.sequence = None
        self.sequence_offset = sequence_offset
        self.plddt_improvement_data = None
        self.tm_score = None
        self.lddt_score = None
        self.structure_files = {}

    def load_dms_data(self, dms_file):
        df = pd.read_csv(dms_file)

        if 'seq_n' in df.columns:
            df = df.rename(columns={
                'seq_n': 'position',
                'mut_res': 'mutation',
                'ddG': 'fitness_score'
            })
            if 'wt_res' in df.columns:
                df['wildtype'] = df['wt_res']

        df['position'] = df['position'].astype(int)
        df['mutation'] = df['mutation'].astype(str).str.strip().str.upper()
        df['fitness_score'] = pd.to_numeric(df['fitness_score'], errors='coerce')

        if 'wildtype' in df.columns:
            before = len(df)
            df['wildtype'] = df['wildtype'].astype(str).str.strip().str.upper()
            df = df[df['mutation'] != df['wildtype']]
            removed = before - len(df)
            if removed > 0:
                print(f"  Removed {removed} WT→WT entries (synonymous mutations)")

        self.dms_data = df.dropna(subset=['fitness_score'])
        
        if self.sequence_offset is None and self.sequence is not None:
            self._auto_detect_offset()
        
        return self

    def load_sequence(self, fasta_file):
        try:
            record = SeqIO.read(fasta_file, 'fasta')
            self.sequence = str(record.seq)
            
            if self.sequence_offset is None and self.dms_data is not None:
                self._auto_detect_offset()
        except Exception:
            self.sequence = None
        return self

    def _auto_detect_offset(self):
        if self.dms_data is None or self.sequence is None:
            return
        
        first_pos = int(self.dms_data['position'].min())
        first_wt = self.dms_data[self.dms_data['position'] == first_pos]['wildtype'].values[0]
        
        for i, aa in enumerate(self.sequence, start=1):
            if aa == first_wt:
                self.sequence_offset = first_pos - i
                print(f"  Auto-detected offset: {self.sequence_offset}")
                return
        
        print(f"  Warning: Could not auto-detect offset. Using DMS wildtype column.")
        self.sequence_offset = 0

    def load_structure_quality(self, pdb_file):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(self.protein_id, pdb_file)

        residue_plddts = {}
        for model in structure:
            for chain in model:
                for residue in chain:
                    pos = residue.get_id()[1]
                    if 'CA' in residue:
                        residue_plddts[pos] = residue['CA'].get_bfactor()

        self.structure_data = pd.DataFrame(
            [{'position': k, 'plddt': v} for k, v in residue_plddts.items()]
        )
        return self

    def load_plddt_improvement(self, per_residue_file):
        df = pd.read_csv(per_residue_file)
        self.plddt_improvement_data = df[df['protein_id'] == self.protein_id].copy()

        if not self.plddt_improvement_data.empty:
            self.plddt_improvement_data['position'] = (
                self.plddt_improvement_data['position'].astype(int)
            )
            print(f"  ✓ Loaded pLDDT improvement: {len(self.plddt_improvement_data)} residues")
        else:
            print(f"  ⚠️  No pLDDT improvement data for {self.protein_id}")
        
        return self

    def load_tm_score(self, tm_score_file):
        df = pd.read_csv(tm_score_file)
        row = df[df['protein'] == self.protein_id]

        if not row.empty:
            self.tm_score = {
                'mutfold': float(row['tm_dmsfold'].values[0]) if 'tm_dmsfold' in row.columns else None,
                'af2': float(row['tm_af2'].values[0]) if 'tm_af2' in row.columns else None,
            }
        else:
            self.tm_score = None
        return self

    def load_lddt_score(self, lddt_score_file):
        df = pd.read_csv(lddt_score_file)
        row = df[df['protein'] == self.protein_id]

        if not row.empty:
            self.lddt_score = {
                'mutfold': float(row['lddt_dmsfold'].values[0]) if 'lddt_dmsfold' in row.columns else None,
                'af2': float(row['lddt_af2'].values[0]) if 'lddt_af2' in row.columns else None,
                'delta': float(row['lddt_delta'].values[0]) if 'lddt_delta' in row.columns else None
            }
        else:
            self.lddt_score = None
        return self

    def set_structure_files(self, mutfold_pdb=None, af2_pdb=None):
        """Set paths to structure files for 3D visualization"""
        self.structure_files = {
            'mutfold': mutfold_pdb,
            'af2': af2_pdb
        }
        return self

    def _read_pdb_file(self, pdb_path):
        if pdb_path is None:
            return None
        try:
            with open(pdb_path, 'r') as f:
                return f.read()
        except:
            return None

    def create_fitness_matrix(self):
        positions = sorted(self.dms_data['position'].unique())
        matrix = np.full((len(self.amino_acids), len(positions)), np.nan)

        for i, aa in enumerate(self.amino_acids):
            for j, pos in enumerate(positions):
                row = self.dms_data[
                    (self.dms_data['position'] == pos) &
                    (self.dms_data['mutation'] == aa)
                ]
                if not row.empty:
                    matrix[i, j] = row['fitness_score'].values[0]

        return matrix, positions

    def create_plddt_improvement_array(self, positions):
        if self.plddt_improvement_data is None or self.plddt_improvement_data.empty:
            return np.zeros(len(positions))

        vals = []
        for pos in positions:
            row = self.plddt_improvement_data[
                self.plddt_improvement_data['position'] == int(pos)
            ]
            vals.append(float(row['plddt_improvement'].values[0]) if not row.empty else 0.0)

        return np.array(vals)

    def _get_wildtype_aa(self, position):
        if 'wildtype' in self.dms_data.columns:
            wt = self.dms_data[self.dms_data['position'] == position]['wildtype']
            if not wt.empty:
                return wt.values[0]
        
        if self.sequence and self.sequence_offset is not None:
            seq_position = position - self.sequence_offset
            if 1 <= seq_position <= len(self.sequence):
                return self.sequence[seq_position - 1]
        
        return 'X'

    def generate_enhanced_heatmap(self, output_file):
        fitness_matrix, positions = self.create_fitness_matrix()
        plddt_improvements = self.create_plddt_improvement_array(positions)
        
        has_improvements = (self.plddt_improvement_data is not None and 
                          not self.plddt_improvement_data.empty and
                          not np.all(plddt_improvements == 0))
        
        has_structure_metrics = (self.tm_score is not None) or (self.lddt_score is not None)

        if has_improvements and has_structure_metrics:
            fig = make_subplots(
                rows=4, cols=1,
                row_heights=[0.60, 0.15, 0.08, 0.17],
                specs=[
                    [{"type": "xy"}],
                    [{"type": "xy"}],
                    [{"type": "xy"}],
                    [{"type": "table"}]
                ],
                subplot_titles=['DMS Fitness Landscape', 'ΔpLDDT (MutFold − AF2)', 'Wild-Type Sequence', 'Structure Quality Metrics'],
                vertical_spacing=0.08
            )
        elif has_improvements and not has_structure_metrics:
            fig = make_subplots(
                rows=3, cols=1,
                row_heights=[0.75, 0.15, 0.1],
                vertical_spacing=0.06,
                subplot_titles=['DMS Fitness Landscape', 'ΔpLDDT (MutFold − AF2)', 'Wild-Type Sequence']
            )
        elif not has_improvements and has_structure_metrics:
            fig = make_subplots(
                rows=3, cols=1,
                row_heights=[0.75, 0.1, 0.15],
                specs=[
                    [{"type": "xy"}],
                    [{"type": "xy"}],
                    [{"type": "table"}]
                ],
                subplot_titles=['DMS Fitness Landscape', 'Wild-Type Sequence', 'Structure Quality Metrics'],
                vertical_spacing=0.08
            )
        else:
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.9, 0.1],
                vertical_spacing=0.06,
                subplot_titles=['DMS Fitness Landscape', 'Wild-Type Sequence']
            )

        hover_text = []
        text_annotations = []

        for i, aa in enumerate(self.amino_acids):
            hrow, trow = [], []
            for j, pos in enumerate(positions):
                val = fitness_matrix[i, j]
                wt_aa = self._get_wildtype_aa(pos)
                
                if not np.isnan(val):
                    msg = f"<b>Position: {pos}</b><br>WT: {wt_aa} → Mut: {aa}<br>Fitness: {val:.3f}"
                    hrow.append(msg)
                    trow.append('')
                else:
                    hrow.append("No data")
                    trow.append('X')
            hover_text.append(hrow)
            text_annotations.append(trow)

        fig.add_trace(
            go.Heatmap(
                z=fitness_matrix,
                x=positions,
                y=self.amino_acids,
                hovertext=hover_text,
                hoverinfo='text',
                text=text_annotations,
                texttemplate='%{text}',
                textfont=dict(color='gray', size=14),
                colorscale=[
                    [0.0, '#d73027'],
                    [0.3, '#fc8d59'],
                    [0.5, '#ede1d1'],
                    [0.7, '#91bfdb'],
                    [1.0, '#4575b4']
                ],
                zmid=0,
                colorbar=dict(title='Fitness Score')
            ),
            row=1, col=1
        )

        if has_improvements:
            fig.add_trace(
                go.Bar(
                    x=positions,
                    y=plddt_improvements,
                    marker_color=['green' if x > 0 else 'red' if x < 0 else 'gray' for x in plddt_improvements],
                    hovertemplate='Pos %{x}<br>ΔpLDDT: %{y:+.2f}<extra></extra>'
                ),
                row=2, col=1
            )
            fig.update_yaxes(title_text='ΔpLDDT', row=2, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

        wt_row = 3 if has_improvements else 2
        if self.sequence:
            fig.add_trace(
                go.Heatmap(
                    z=[list(range(len(positions)))],
                    x=positions,
                    y=['WT'],
                    text=[[self._get_wildtype_aa(p) for p in positions]],
                    texttemplate='%{text}',
                    textfont=dict(family='Courier New', size=10),
                    showscale=False,
                    colorscale=[[0, 'lightgray'], [1, 'lightgray']],
                    hovertemplate='Pos %{x}<br>WT: %{text}<extra></extra>'
                ),
                row=wt_row, col=1
            )

        if has_structure_metrics:
            table_row = 4 if has_improvements else 3
            table_data = self._create_structure_quality_table()
            
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['<b>Metric</b>', '<b>MutFold</b>', '<b>AlphaFold2</b>', '<b>Δ</b>'],
                        fill_color='#f0f0f0',
                        align='left',
                        font=dict(size=12, color='black')
                    ),
                    cells=dict(
                        values=[
                            table_data['metrics'],
                            table_data['mutfold'],
                            table_data['af2'],
                            table_data['delta']
                        ],
                        fill_color=[
                            ['white'] * len(table_data['metrics']),
                            table_data['mutfold_colors'],
                            table_data['af2_colors'],
                            table_data['delta_colors']
                        ],
                        align='left',
                        font=dict(size=11),
                        height=30
                    )
                ),
                row=table_row, col=1
            )

        title = f"<b>{self.protein_id}</b>"
        
        if has_improvements and has_structure_metrics:
            height = 1000
        elif has_improvements or has_structure_metrics:
            height = 900
        else:
            height = 700
        
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=16)),
            height=height,
            width=1400,
            hovermode='closest',
            showlegend=False
        )

        fig.update_xaxes(title_text="Protein Position", row=wt_row, col=1)
        fig.update_yaxes(title_text="Amino Acid", row=1, col=1)

        has_structures = (self.structure_files.get('mutfold') is not None or 
                         self.structure_files.get('af2') is not None)
        
        if has_structures:
            self._generate_html_with_structure_viewer(output_file, fig)
        else:
            fig.write_html(output_file)
        
        return fig

    def _generate_html_with_structure_viewer(self, output_file, fig):
        # Save regular plotly first
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp:
            fig.write_html(tmp.name, include_plotlyjs='cdn')
            tmp_path = tmp.name
        
        with open(tmp_path, 'r') as f:
            plotly_html = f.read()
        
        os.unlink(tmp_path)
        
        # Read PDB files
        mutfold_pdb = self._read_pdb_file(self.structure_files.get('mutfold'))
        af2_pdb = self._read_pdb_file(self.structure_files.get('af2'))
        
        def escape_js(pdb_str):
            if pdb_str is None:
                return 'null'
            # Simple escaping
            escaped = pdb_str.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n').replace('\r', '')
            return "'" + escaped + "'"
        
        mutfold_js = escape_js(mutfold_pdb)
        af2_js = escape_js(af2_pdb)

        
        # Build buttons
        buttons = []
        if mutfold_pdb:
            buttons.append('<button class="struct-btn" onclick="showStructure(\'mutfold\')">MutFold</button>')
        if af2_pdb:
            buttons.append('<button class="struct-btn" onclick="showStructure(\'af2\')">AlphaFold2</button>')
        buttons.append('<button class="struct-btn reset-btn" onclick="resetViewer()">Reset View</button>')
        buttons_html = '\n            '.join(buttons)
        
        # Inject viewer BEFORE closing body tag with explicit positioning
        viewer_section = f'''
<div id="structure-section" style="display: block; position: relative; width: 100%; max-width: 1400px; margin: 80px auto 40px; padding: 0 20px; clear: both;">
    <h3 style="font-size: 14px; font-weight: 600; margin-bottom: 12px; color: #333;">3D Structure Comparison</h3>
    <div style="margin-bottom: 12px;">
        {buttons_html}
    </div>
    <div id="structure-viewer" style="position: relative; width: 100%; height: 600px; border: 1px solid #ddd; background: #000;"></div>
</div>

<script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>

<script>
var pdbData = {{
    mutfold: {mutfold_js},
    af2: {af2_js}
}};

var viewer = null;
var currentStructure = null;

function initViewer() {{
    var element = document.getElementById('structure-viewer');
    if (element && typeof $3Dmol !== 'undefined') {{
        viewer = $3Dmol.createViewer(element, {{backgroundColor: 'white'}});
    }}
}}

function showStructure(type) {{
    if (!viewer) initViewer();
    if (!viewer) return;
    
    viewer.clear();
    currentStructure = type;
    
    var buttons = document.querySelectorAll('.struct-btn');
    buttons.forEach(function(btn) {{ btn.classList.remove('active'); }});
    
    if (type === 'mutfold' && pdbData.mutfold) {{
        viewer.addModel(pdbData.mutfold, 'pdb');
        viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
        buttons[0].classList.add('active');
    }} else if (type === 'af2' && pdbData.af2) {{
        viewer.addModel(pdbData.af2, 'pdb');
        viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
        var btn = pdbData.mutfold ? buttons[1] : buttons[0];
        btn.classList.add('active');
    }}
    
    viewer.zoomTo();
    viewer.render();
}}

function resetViewer() {{
    if (!viewer || !currentStructure) return;
    showStructure(currentStructure);
}}

if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', function() {{
        setTimeout(function() {{
            initViewer();
            if (pdbData.mutfold) showStructure('mutfold');
            else if (pdbData.af2) showStructure('af2');
        }}, 100);
    }});
}} else {{
    setTimeout(function() {{
        initViewer();
        if (pdbData.mutfold) showStructure('mutfold');
        else if (pdbData.af2) showStructure('af2');
    }}, 100);
}}
</script>

<style>
.struct-btn {{
    padding: 8px 16px;
    margin-right: 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
    background: white;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s;
}}
.struct-btn:hover {{ background: #f5f5f5; }}
.struct-btn.active {{ background: #4CAF50; color: white; border-color: #4CAF50; }}
</style>
'''
        
        # Insert before </body>
        final_html = plotly_html.replace('</body>', viewer_section + '\n</body>')
        
        with open(output_file, 'w') as f:
            f.write(final_html)

    def _create_structure_quality_table(self):
        metrics = []
        mutfold_vals = []
        af2_vals = []
        delta_vals = []
        mutfold_colors = []
        af2_colors = []
        delta_colors = []

        if self.tm_score:
            metrics.append('TM-score')
            tm_mutfold = self.tm_score.get('mutfold')
            tm_af2 = self.tm_score.get('af2')
            
            mutfold_vals.append(f"{tm_mutfold:.3f}" if tm_mutfold else "—")
            af2_vals.append(f"{tm_af2:.3f}" if tm_af2 else "—")
            
            if tm_mutfold and tm_af2:
                delta = tm_mutfold - tm_af2
                delta_vals.append(f"{delta:+.3f}")
                delta_colors.append('#d4edda' if delta > 0 else '#f8d7da' if delta < 0 else 'white')
            else:
                delta_vals.append("—")
                delta_colors.append('white')
            
            mutfold_colors.append('#d4edda' if tm_mutfold and tm_mutfold > 0.5 else 'white')
            af2_colors.append('#d4edda' if tm_af2 and tm_af2 > 0.5 else 'white')

        if self.lddt_score:
            metrics.append('lDDT-score')
            lddt_mutfold = self.lddt_score.get('mutfold')
            lddt_af2 = self.lddt_score.get('af2')
            lddt_delta = self.lddt_score.get('delta')
            
            mutfold_vals.append(f"{lddt_mutfold:.2f}" if lddt_mutfold else "—")
            af2_vals.append(f"{lddt_af2:.2f}" if lddt_af2 else "—")
            
            if lddt_delta is not None:
                delta_vals.append(f"{lddt_delta:+.2f}")
                delta_colors.append('#d4edda' if lddt_delta > 0 else '#f8d7da' if lddt_delta < 0 else 'white')
            else:
                delta_vals.append("—")
                delta_colors.append('white')
            
            mutfold_colors.append('#d4edda' if lddt_mutfold and lddt_mutfold > 60 else 'white')
            af2_colors.append('#d4edda' if lddt_af2 and lddt_af2 > 60 else 'white')

        if not metrics:
            metrics = ['No data']
            mutfold_vals = ['—']
            af2_vals = ['—']
            delta_vals = ['—']
            mutfold_colors = ['white']
            af2_colors = ['white']
            delta_colors = ['white']

        return {
            'metrics': metrics,
            'mutfold': mutfold_vals,
            'af2': af2_vals,
            'delta': delta_vals,
            'mutfold_colors': mutfold_colors,
            'af2_colors': af2_colors,
            'delta_colors': delta_colors
        }
