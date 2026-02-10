"""
Shared settings and display-name dictionaries.

All configurable look-up tables (palette, metric configs, kernel/perm display
names, etc.) live here so that every script in ``scripts/`` imports from a
single source of truth.
"""

# =============================================================================
# Professional Plot Style - Palette
# =============================================================================

# Colorblind-friendly palette for categorical data (Tol's muted scheme)
PALETTE = [
    '#332288', '#88CCEE', '#44AA99', '#117733',
    '#999933', '#DDCC77', '#CC6677', '#882255',
    '#AA4499', '#DDDDDD',
]

# =============================================================================
# Metric Configuration (used by plot_utils for display names / log scale)
# =============================================================================

METRIC_CONFIG = {
    # Performance metrics
    'gflops': {'display': 'GFLOPS', 'log_scale': True},
    'speedup': {'display': 'Speedup', 'log_scale': False},

    # Bandwidth metrics
    'bandwidth_max': {'display': 'Bandwidth', 'log_scale': True},
    'rel_bandwidth': {'display': 'Relative Bandwidth', 'log_scale': True},
    'bandwidth_improvement': {'display': 'Bandwidth Reduction', 'log_scale': False},

    # Density metrics (block sizes added dynamically below)
    'density': {'display': 'Density', 'log_scale': True},

    # Locality metrics
    'rel_row_spread': {'display': 'Relative Row Spread', 'log_scale': True},
    'locality_vertical_adjacency_ratio': {'display': 'Vertical Adjacency Ratio', 'log_scale': False},
    'row_spread_improvement': {'display': 'Row Spread Reduction', 'log_scale': False},
    'col_spread_improvement': {'display': 'Col Spread Reduction', 'log_scale': False},
    'vertical_adjacency_improvement': {'display': 'Vertical Adjacency Improvement', 'log_scale': False},
}

# Block sizes available for block density metrics
BLOCK_SIZES = [4, 8, 16, 32, 64, 128]

for _bs in BLOCK_SIZES:
    METRIC_CONFIG[f'block_density_{_bs}'] = {'display': f'Block Density (BS {_bs})', 'log_scale': True}
    METRIC_CONFIG[f'density_improvement_{_bs}'] = {'display': f'Density Improvement (BS {_bs})', 'log_scale': False}

# =============================================================================
# Reordering Algorithm Display Names
# =============================================================================

PERM_NAMES = {
    'SB_rcm':          'RCM',
    'SB_amd':          'AMD',
    'SB_metis':        'Metis',
    'SB_patoh':        'PaToH',
    'SB_rabbit':       'Rabbit',
    'SB_gray':         'Gray',
    'SB_degree':       'Degree',
    'GROOT_reorder':   'GROOT',
    'SPARTA_reorder':  'SPARTA',
    'random1D':        'Random',
}

# =============================================================================
# Kernel Display Names
# =============================================================================

KERNEL_NAMES = {
    'ASPT_SPMM': 'ASPT',
    'CUSPARSE_SPMM_BSR_bs32': 'cuSP BSR',
    'CUSPARSE_SPMM_CSR': 'cuSP CSR',
    'DTC_SPMM': 'DTC',
    'FLASHSPARSE_SPMM': 'FlashSP',
    'SMAT_SPMM_bs32': 'SMAT',
}

# =============================================================================
# All Structural Metrics (correlation_table)
# ---------------------------------------------------------------------------
# Set 'enabled' to True/False to include/exclude from correlation tables.
# 'name' is the short display name for column headers.
# 'full_name' is the long display name (used in LaTeX captions / legends).
# 'higher_is_better' indicates the direction of improvement (None = N/A).
# =============================================================================

ALL_METRICS = {
    # --- Bandwidth ---
    'bandwidth_max':                       {'name': 'BW',      'full_name': 'Bandwidth',                          'enabled': False, 'higher_is_better': False},
    'bandwidth_avg':                       {'name': 'ABW',     'full_name': 'Average Bandwidth',                  'enabled': False, 'higher_is_better': False},
    'rel_bandwidth':                       {'name': 'RBW',     'full_name': 'Relative Bandwidth',                 'enabled': True,  'higher_is_better': False},
    # --- Row spread / locality ---
    'locality_avg_row_spread':             {'name': 'ARS',     'full_name': 'Average Row Spread',                 'enabled': False, 'higher_is_better': False},
    'locality_max_row_spread':             {'name': 'MRS',     'full_name': 'Maximum Row Spread',                 'enabled': False, 'higher_is_better': False},
    'rel_row_spread':                      {'name': 'RRS',     'full_name': 'Relative Row Spread',                'enabled': True,  'higher_is_better': False},
    # --- Column spread ---
    'locality_avg_col_spread':             {'name': 'ACS',     'full_name': 'Average Column Spread',              'enabled': False, 'higher_is_better': False},
    'locality_max_col_spread':             {'name': 'MCS',     'full_name': 'Maximum Column Spread',              'enabled': False, 'higher_is_better': False},
    # --- Vertical adjacency ---
    'locality_consecutive_vertical_pairs': {'name': 'CVP',     'full_name': 'Consecutive Vertical Pairs',         'enabled': False, 'higher_is_better': True},
    'locality_vertical_adjacency_ratio':   {'name': 'VAR',     'full_name': 'Vertical Adjacency Ratio',           'enabled': True,  'higher_is_better': True},
    # --- NNZ distribution ---
    'locality_avg_nnz_per_row':            {'name': 'ANR',     'full_name': 'Average NNZ per Row',                'enabled': False, 'higher_is_better': None},
    'locality_max_nnz_per_row':            {'name': 'MNR',     'full_name': 'Maximum NNZ per Row',                'enabled': False, 'higher_is_better': None},
    'locality_num_empty_rows':             {'name': 'NER',     'full_name': 'Number of Empty Rows',               'enabled': False, 'higher_is_better': None},
    'locality_num_empty_cols':             {'name': 'NEC',     'full_name': 'Number of Empty Columns',            'enabled': False, 'higher_is_better': None},
    # --- Profile ---
    'locality_profile':                    {'name': 'Prof',    'full_name': 'Profile',                            'enabled': False, 'higher_is_better': False},
    # --- Overall density ---
    'density':                             {'name': 'Dens',    'full_name': 'Density',                            'enabled': False, 'higher_is_better': None},
    # --- Block density (per block size) ---
    'block_density_4':                     {'name': 'BD4',     'full_name': 'Block Density $4{\\times}4$',         'enabled': False, 'higher_is_better': True},
    'block_density_8':                     {'name': 'BD8',     'full_name': 'Block Density $8{\\times}8$',         'enabled': True,  'higher_is_better': True},
    'block_density_16':                    {'name': 'BD16',    'full_name': 'Block Density $16{\\times}16$',       'enabled': False, 'higher_is_better': True},
    'block_density_32':                    {'name': 'BD32',    'full_name': 'Block Density $32{\\times}32$',       'enabled': True,  'higher_is_better': True},
    'block_density_64':                    {'name': 'BD64',    'full_name': 'Block Density $64{\\times}64$',       'enabled': False, 'higher_is_better': True},
    'block_density_128':                   {'name': 'BD128',   'full_name': 'Block Density $128{\\times}128$',     'enabled': True,  'higher_is_better': True},
    # --- Avg blocks per row (per block size) ---
    'avg_blocks_per_row_4':                {'name': 'ABR4',    'full_name': 'Avg Blocks/Row $4{\\times}4$',        'enabled': False, 'higher_is_better': False},
    'avg_blocks_per_row_8':                {'name': 'ABR8',    'full_name': 'Avg Blocks/Row $8{\\times}8$',        'enabled': False, 'higher_is_better': False},
    'avg_blocks_per_row_16':               {'name': 'ABR16',   'full_name': 'Avg Blocks/Row $16{\\times}16$',      'enabled': False, 'higher_is_better': False},
    'avg_blocks_per_row_32':               {'name': 'ABR32',   'full_name': 'Avg Blocks/Row $32{\\times}32$',      'enabled': False, 'higher_is_better': False},
    'avg_blocks_per_row_64':               {'name': 'ABR64',   'full_name': 'Avg Blocks/Row $64{\\times}64$',      'enabled': False, 'higher_is_better': False},
    'avg_blocks_per_row_128':              {'name': 'ABR128',  'full_name': 'Avg Blocks/Row $128{\\times}128$',    'enabled': False, 'higher_is_better': False},
    # --- Max blocks per row (per block size) ---
    'max_blocks_per_row_4':                {'name': 'MBR4',    'full_name': 'Max Blocks/Row $4{\\times}4$',        'enabled': False, 'higher_is_better': False},
    'max_blocks_per_row_8':                {'name': 'MBR8',    'full_name': 'Max Blocks/Row $8{\\times}8$',        'enabled': False, 'higher_is_better': False},
    'max_blocks_per_row_16':               {'name': 'MBR16',   'full_name': 'Max Blocks/Row $16{\\times}16$',      'enabled': False, 'higher_is_better': False},
    'max_blocks_per_row_32':               {'name': 'MBR32',   'full_name': 'Max Blocks/Row $32{\\times}32$',      'enabled': False, 'higher_is_better': False},
    'max_blocks_per_row_64':               {'name': 'MBR64',   'full_name': 'Max Blocks/Row $64{\\times}64$',      'enabled': False, 'higher_is_better': False},
    'max_blocks_per_row_128':              {'name': 'MBR128',  'full_name': 'Max Blocks/Row $128{\\times}128$',    'enabled': False, 'higher_is_better': False},
}

# Derived lists from ALL_METRICS (do not edit manually)
METRICS = [k for k, v in ALL_METRICS.items() if v['enabled']]
METRIC_NAMES = {k: v['name'] for k, v in ALL_METRICS.items()}
METRIC_FULL_NAMES = {k: v['full_name'] for k, v in ALL_METRICS.items()}

# Block density metric display names
BLOCK_DENSITY_METRIC_NAMES = {f'block_density_{bs}': f'${bs}\\times{bs}$' for bs in BLOCK_SIZES}

# Block density metrics in order
BLOCK_DENSITY_METRICS = [f'block_density_{bs}' for bs in BLOCK_SIZES]

# =============================================================================
# Filtering Settings
# =============================================================================

# Families to keep fully (not reduce to one representative).
# These contain diverse, non-duplicate matrices commonly used in sparse matrix research.
KEEP_FULL_FAMILIES = [
    'DIMACS10',    # Graph challenge benchmarks
    'SNAP',        # Stanford social/web networks
    'LAW',         # Web graphs (Laboratory for Web Algorithmics)
    'Newman',      # Network science graphs
    'Gleich',      # Web and social network graphs
    'Janna',       # Large-scale FEM problems
    'Norris',      # Structural engineering benchmarks
    'vanHeukelum', # Cage graphs, unique structure
]
