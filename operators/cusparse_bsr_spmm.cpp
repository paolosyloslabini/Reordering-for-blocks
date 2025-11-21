// cuSPARSE BSR SpMM using native cuSPARSE C API
// Computes C = alpha * A * B + beta * C where A is sparse BSR, B and C are dense

#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <algorithm>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while (0)

#define CHECK_CUSPARSE(call) \
    do { \
        cusparseStatus_t err = call; \
        if (err != CUSPARSE_STATUS_SUCCESS) { \
            std::cerr << "cuSPARSE error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

// Simple CSR matrix struct
struct CSRMatrix {
    int m, n, nnz;
    std::vector<float> values;
    std::vector<int> rowPtr;
    std::vector<int> colInd;
};

// Read Matrix Market file as CSR
CSRMatrix readMatrixMarketCSR(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }
    
    std::string line;
    do {
        std::getline(file, line);
    } while (line[0] == '%');
    
    int m, n, nnz;
    std::istringstream iss(line);
    iss >> m >> n >> nnz;
    
    std::vector<int> row_indices(nnz), col_indices(nnz);
    std::vector<float> values(nnz);
    
    for (int i = 0; i < nnz; i++) {
        int row, col;
        float val;
        file >> row >> col >> val;
        row--; col--;  // Convert to 0-based
        row_indices[i] = row;
        col_indices[i] = col;
        values[i] = val;
    }
    file.close();
    
    // Build CSR
    std::vector<int> rowPtr(m + 1, 0);
    for (int i = 0; i < nnz; i++) {
        rowPtr[row_indices[i] + 1]++;
    }
    for (int i = 0; i < m; i++) {
        rowPtr[i + 1] += rowPtr[i];
    }
    
    std::vector<int> colInd(nnz);
    std::vector<float> csrVal(nnz);
    std::vector<int> next(m, 0);
    
    for (int i = 0; i < nnz; i++) {
        int row = row_indices[i];
        int dest = rowPtr[row] + next[row];
        colInd[dest] = col_indices[i];
        csrVal[dest] = values[i];
        next[row]++;
    }
    
    return {m, n, nnz, csrVal, rowPtr, colInd};
}

void printTimer(const char* name, double ms) {
    std::cout << "<Timer name=\"" << name << "\" value=\"" << ms << "\"/>" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix.mtx> [--blocksize N] [--n-cols N] [--n-iterations N]" << std::endl;
        return 1;
    }
    
    // Parse arguments
    const char* matrixFile = argv[1];
    int blockDim = 8;
    int nCols = 32;
    int nIterations = 5;
    float alpha = 1.0f;
    float beta = 0.0f;
    
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--blocksize") == 0 && i + 1 < argc) {
            blockDim = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-cols") == 0 && i + 1 < argc) {
            nCols = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-iterations") == 0 && i + 1 < argc) {
            nIterations = atoi(argv[++i]);
        }
    }
    
    // Load matrix as CSR
    auto start = std::chrono::high_resolution_clock::now();
    CSRMatrix csr = readMatrixMarketCSR(matrixFile);
    auto end = std::chrono::high_resolution_clock::now();
    double loadingMs = std::chrono::duration<double, std::milli>(end - start).count();

    // Calculate padded dimensions for BSR (must be multiple of blockDim)
    int mb = (csr.m + blockDim - 1) / blockDim;
    int nb = (csr.n + blockDim - 1) / blockDim;
    int padded_m = mb * blockDim;
    int padded_n = nb * blockDim;

    // Allocate device memory for CSR
    float *d_csrVal, *d_B, *d_C;
    int *d_csrRowPtr, *d_csrColInd;
    size_t csrValSize = csr.nnz * sizeof(float);
    size_t csrRowPtrSize = (csr.m + 1) * sizeof(int);
    size_t csrColIndSize = csr.nnz * sizeof(int);
    size_t denseBSize = padded_n * nCols * sizeof(float);
    size_t denseCSize = padded_m * nCols * sizeof(float);

    start = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMalloc(&d_csrVal, csrValSize));
    CHECK_CUDA(cudaMalloc(&d_csrRowPtr, csrRowPtrSize));
    CHECK_CUDA(cudaMalloc(&d_csrColInd, csrColIndSize));
    CHECK_CUDA(cudaMalloc(&d_B, denseBSize));
    CHECK_CUDA(cudaMalloc(&d_C, denseCSize));

    CHECK_CUDA(cudaMemcpy(d_csrVal, csr.values.data(), csrValSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csrRowPtr, csr.rowPtr.data(), csrRowPtrSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csrColInd, csr.colInd.data(), csrColIndSize, cudaMemcpyHostToDevice));

    // Initialize B and C with random data (use padded dimensions)
    std::vector<float> B(padded_n * nCols, 0.0f), C(padded_m * nCols, 0.0f);
    for (auto& v : B) v = static_cast<float>(rand()) / RAND_MAX;
    for (auto& v : C) v = static_cast<float>(rand()) / RAND_MAX;
    CHECK_CUDA(cudaMemcpy(d_B, B.data(), denseBSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, C.data(), denseCSize, cudaMemcpyHostToDevice));

    end = std::chrono::high_resolution_clock::now();
    double transferMs = std::chrono::duration<double, std::milli>(end - start).count();

    // Create cuSPARSE handle
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Create matrix descriptors for CSR format
    cusparseMatDescr_t descrA;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    cusparseMatDescr_t descrC;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrC));
    CHECK_CUSPARSE(cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO));

    // Convert CSR to BSR on device
    int nnzb = 0;
    int *d_bsrRowPtr;
    CHECK_CUDA(cudaMalloc(&d_bsrRowPtr, (mb + 1) * sizeof(int)));
    
    // Get nnzb (using legacy API - no modern replacement available)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    CHECK_CUSPARSE(cusparseXcsr2bsrNnz(
        handle,
        CUSPARSE_DIRECTION_ROW,
        csr.m, csr.n,
        descrA,
        d_csrRowPtr, d_csrColInd,
        blockDim,
        descrC,
        d_bsrRowPtr,
        &nnzb));
    
    float *d_bsrVal;
    int *d_bsrColInd;
    CHECK_CUDA(cudaMalloc(&d_bsrVal, nnzb * blockDim * blockDim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bsrColInd, nnzb * sizeof(int)));
    
    CHECK_CUSPARSE(cusparseScsr2bsr(
        handle,
        CUSPARSE_DIRECTION_ROW,
        csr.m, csr.n,
        descrA,
        d_csrVal, d_csrRowPtr, d_csrColInd,
        blockDim,
        descrC,
        d_bsrVal, d_bsrRowPtr, d_bsrColInd));
    #pragma GCC diagnostic pop

    // Create matrix descriptors for generic API
    // BSR matrix: mb x nb blocks, each block is blockDim x blockDim
    // Dense matrices: must match actual padded dimensions
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    CHECK_CUSPARSE(cusparseCreateBsr(
        &matA, mb, nb, nnzb,
        blockDim, blockDim,
        d_bsrRowPtr, d_bsrColInd, d_bsrVal,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F,
        CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, padded_n, nCols, nCols, d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, padded_m, nCols, nCols, d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    
    // Allocate buffer
    size_t bufferSize;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
    
    void* buffer;
    CHECK_CUDA(cudaMalloc(&buffer, bufferSize));
    
    // Warm-up
    CHECK_CUSPARSE(cusparseSpMM(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, buffer));
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));
    
    float totalMs = 0;
    for (int iter = 0; iter < nIterations; iter++) {
        CHECK_CUDA(cudaEventRecord(startEvent));
        
        CHECK_CUSPARSE(cusparseSpMM(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, buffer));
        
        CHECK_CUDA(cudaEventRecord(stopEvent));
        CHECK_CUDA(cudaEventSynchronize(stopEvent));
        
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, startEvent, stopEvent));
        totalMs += ms;
    }
    float avgMs = totalMs / nIterations;
    
    // Cleanup
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrC));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    CHECK_CUDA(cudaFree(d_csrVal));
    CHECK_CUDA(cudaFree(d_csrRowPtr));
    CHECK_CUDA(cudaFree(d_csrColInd));
    CHECK_CUDA(cudaFree(d_bsrVal));
    CHECK_CUDA(cudaFree(d_bsrRowPtr));
    CHECK_CUDA(cudaFree(d_bsrColInd));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(buffer));
    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));
    
    // Print results
    printTimer("loading", loadingMs);
    printTimer("transfer", transferMs);
    printTimer("operation", avgMs);
    
    return 0;
}
