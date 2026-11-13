#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

__global__ void naiveMul(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

template<int T>
__global__ void tiledMul(const float* A, const float* B, float* C, int N) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + T - 1) / T; ++t) {
        if (row < N && (t * T + threadIdx.x) < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + t * T + threadIdx.x];
        else 
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (t * T + threadIdx.y) < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * T + threadIdx.y) * N + col];
        else 
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < T; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = sum;
}

int readNFromFile(const char* path) {
    ifstream in(path);
    if (!in.is_open()) { cerr << "Error opening: " << path << "\n"; exit(1); }
    int n; in >> n; return n;
}

void readMatrixAfterN(const char* path, float* M, int N) {
    ifstream in(path);
    if (!in.is_open()) { cerr << "Error opening: " << path << "\n"; exit(1); }
    int n_file; in >> n_file;
    for (int i = 0; i < N * N; ++i) in >> M[i];
}

void writeMatrix(const char* path, const float* M, int N) {
    ofstream out(path);
    out << N << "\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) out << M[i * N + j] << " ";
        out << "\n";
    }
    out.close();
}

int main(int argc, char** argv) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " N A.txt B.txt BLOCK_SIZE [naive/tiled]\n";
        return 1;
    }
    int N = atoi(argv[1]);
    int block_size = atoi(argv[4]);
    string mode = (argc > 5) ? argv[5] : "tiled";

    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];
    
    readMatrixAfterN(argv[2], h_A, N);
    readMatrixAfterN(argv[3], h_B, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(block_size, block_size);
    dim3 grid((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    if (mode == "naive") {
        naiveMul<<<grid, block>>>(d_A, d_B, d_C, N);
    } else {
        if (block_size == 8) tiledMul<8><<<grid, block>>>(d_A, d_B, d_C, N);
        else if (block_size == 16) tiledMul<16><<<grid, block>>>(d_A, d_B, d_C, N);
        else if (block_size == 32) tiledMul<32><<<grid, block>>>(d_A, d_B, d_C, N);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA_ERROR: %s\n", cudaGetErrorString(err));
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("N=%d, Block=%d, Mode=%s, Time=%.4f ms\n", N, block_size, mode.c_str(), milliseconds);

    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    writeMatrix("C_gpu.txt", h_C, N);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;

    return 0;
}