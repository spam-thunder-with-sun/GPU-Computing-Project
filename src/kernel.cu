#include <cuda_runtime.h>
#include <bitset>
#include <iostream>

#define literals 0
#define clauses 0

/**
 * 
 * @param n number of possible solutions
 * @param t number of threads overall
 */
__global__ bool sat(int n, int t) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ bool **M; //extract the formula matrix from the share memory

    bool *v = new bool[literals * 2 + 1];   //solution vector
    bool *res = new bool[clauses + 1];   //temp result of M * v

    for (int i = (n * tid) / t; i < (n * (tid - 1)) / t; ++i) {
        for (int j = 0; j < literals; ++j) {    //transforming number into a solution vector
            v[j] = (i >> j) & 1;
            v[j + literals + 1] = !((i >> j) & 1);
        }

        //we have the vector solution, now we have to try it!
        for (int i = 1; i <= clauses; ++i)
            for (int j = 1; j <= literals * 2; ++j)
                res[i] = res[i] || (M[i][j] && v[j]);   

        //check whether the solution is right
        if (is_valid_solution(res)) {
            print_solution(res);
            return true;
        }
    }

    delete v;   
    delete res;

    return false;
}

__device__ bool is_valid_solution(bool *res) {
    for (int i = 1; i <= clauses; ++i)
        if (res[i] == false)
            return false;
    return true;
}

__device__ void print_solution(bool *sol) {
    std::cout << "SAT: \n";
    for (int i = 1; i <= literals; ++i) {
        if (sol[i])
            std::cout << "x" << i << " ";
        else
            std::cout << "NOTx" << i << " ";
    }
    std::cout << std::endl;
}

int main() {

}