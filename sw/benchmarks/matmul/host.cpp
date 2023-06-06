// inspired by https://github.com/Xilinx/Vitis_Accel_Examples/blob/master/cpp_kernels/array_partition/src/matmul_partition.cpp#L26

// OpenCL utility layer include
#include "xcl2.hpp"
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>
#include <iomanip>

using std::default_random_engine;
using std::generate;
using std::uniform_int_distribution;
using std::vector;

// Coyote Includes
#include "cBench.hpp"
#include "cProcess.hpp"

using namespace fpga;

constexpr auto const targetRegion = 0;
constexpr auto const defReps = 1;
constexpr auto const nBenchRuns = 1;

void matmul(int* C, int* A, int* B, int M) {
    for (int k = 0; k < M; k++) {
        for (int j = 0; j < M; j++) {
            for (int i = 0; i < M; i++) {
                C[k * M + j] += A[k * M + i] * B[i * M + j];
            }
        }
    }
}

int gen_random() {
    static default_random_engine e;
    static uniform_int_distribution<int> dist(0, 10);

    return dist(e);
}

void print(int* data, int columns, int rows) {
    vector<int> out(columns * rows);
    for (int r = 0; r < 10; r++) {
        for (int c = 0; c < 10; c++) {
            std::cout << std::setw(4) << data[r * columns + c] << " ";
        }
        std::cout << "…\n";
    }
    for (int r = 0; r < 10; r++) {
        std::cout << "   … ";
    }
    std::cout << "⋱\n\n";
}

void verify(vector<int, aligned_allocator<int> >& gold, vector<int, aligned_allocator<int> >& output) {
    for (int i = 0; i < (int)output.size(); i++) {
        if (output[i] != gold[i]) {
            std::cout << "Mismatch " << i << ": gold: " << gold[i] << " device: " << output[i] << "\n";
            print(output.data(), 16, 16);
            exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char** argv) {
    static const int columns = 16;
    static const int rows = 16;

    size_t array_size = rows * columns;
    size_t array_size_bytes = array_size * sizeof(int);

    vector<int, aligned_allocator<int> > A(array_size);
    vector<int, aligned_allocator<int> > B(array_size);
    vector<int, aligned_allocator<int> > gold(array_size, 0);
    vector<int, aligned_allocator<int> > C(array_size, 0);

    generate(begin(A), end(A), gen_random);
    generate(begin(B), end(B), gen_random);

    std::cout << "A:\n";
    print(A.data(), columns, rows);
    std::cout << "B:\n";
    print(B.data(), columns, rows);
    matmul(gold.data(), A.data(), B.data(), columns);
    std::cout << "Gold:\n";
    print(gold.data(), columns, rows);

    // pages for merged A-B
    uint32_t n_pages_host = (array_size_bytes * 2 + pageSize - 1) / pageSize;
    uint32_t n_pages_rslt = (array_size_bytes + pageSize - 1) / pageSize;
    
    int* hMemAB = (int*) cproc.getMem({CoyoteAlloc::REG_4K, n_pages_host});
    int* rMemC;

    // fill coyote duplicates
    for (int j = 0; j < array_size; j++) {
        hMemAB[j] = A.data()[j];
        hMemAB[array_size + j] = B.data()[j];
    }

    cProcess cproc(targetRegion, getpid());
    cBench bench(nBenchRuns);

    auto benchmark_thr = [&]() {
        for(int i = 0; i < defReps; i++)
            cproc.invoke({CoyoteOper::TRANSFER, hMemAB, rMemC, array_size * 2, array_size});
    };

    bench.runtime(benchmark_thr);
    verify(gold, C);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << (bench.getAvg() / defReps) << " s" << std::endl << std::endl;

    return EXIT_SUCCESS;
}
