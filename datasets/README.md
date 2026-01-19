# datasets/

Storage for sparse matrices from the SuiteSparse Matrix Collection.

## Structure

```
datasets/
├── matrices_list_mtx.txt      # Full paths to all .mtx files (one per line)
├── matrices_metadata.csv      # Matrix properties (rows, cols, nnz, symmetry, etc.)
├── small-matrices-test/       # Subset for quick testing
│   ├── matrices_list_mtx.txt
│   ├── matrices_list.txt
│   └── SuiteSparse_*/         # Downloaded matrices
└── [SuiteSparse_*/]           # Full matrix collections
```

## File Formats

### matrices_list_mtx.txt

Plain text file with one absolute path per line:
```
/path/to/datasets/SuiteSparse_10000_100000_7/matrix1/matrix1.mtx
/path/to/datasets/SuiteSparse_10000_100000_7/matrix2/matrix2.mtx
```

Used by YAML configs to iterate over matrices.

### matrices_metadata.csv

CSV with matrix properties:
- `name`: Matrix name
- `group`: SuiteSparse group/family
- `rows`, `cols`: Dimensions
- `nnz`: Number of non-zeros
- `is_symmetric`: Boolean
- `pattern_symmetry`: Structural symmetry

### Matrix Market (.mtx)

Standard sparse matrix format (COO):
```
%%MatrixMarket matrix coordinate real general
% Comments
rows cols nnz
row1 col1 value1
row2 col2 value2
...
```

## Matrix Selection

Matrices are downloaded based on criteria in [yamls/matrices.yaml](../yamls/matrices.yaml):
- NNZ ranges (e.g., 10K-100K, 100K-1M)
- Structural properties
- Real-valued matrices

Downloaded using `ssgetpy` Python package.

## Testing

Use `small-matrices-test/` for quick validation before full experiments.
