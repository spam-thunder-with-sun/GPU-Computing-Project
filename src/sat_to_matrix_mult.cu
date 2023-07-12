using namespace std;

#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include "create_matrix.h"
#include <bitset>

#define mytype unsigned short

int literals_ = 0;
int clauses_ = 0;

void print_solution(vector<bool> sol) 
{
    for (int i = 1; i <= sol.size() / 2; ++i)
        if (sol[i])
            cout << i << " ";
        else
            cout << "-" << i << " ";
    cout << endl;
}

bool sat(vector<vector<bool>> &M, vector<bool> &sol) 
{
    bool res;

    for (int i = 0; i < clauses_; ++i)
    {   
        res = false;

        for (int j = 1; j < literals_ * 2 + 1 && !res; ++j)
            res = M[i][j] && sol[j];

        if(!res)
            return false;
    }    

    return res;
}

void find_solution (vector<vector<bool>> &M) 
{
    vector<bool> vec (literals_ * 2 + 1, false);
    bool issat = false;

    for (int sol = 0; sol <= ((unsigned long long)1 << literals_) - 1; ++sol) 
    {

        for (int i = 0; i < literals_; ++i) 
        {
            vec[i + 1] = (sol >> i) & 1;
            vec[i + literals_ + 1] = !(vec[i + 1]);
        }

        if (sat(M, vec)) 
        {
            if(!issat)
            {
                issat = true;
                
            }
            cout << "SAT:";
            print_solution(vec);
        }
    }

    if(!issat)
        cout << "UNSAT!" << endl;
}

//Testato
pair<mytype *, mytype *> copy_to_gpu(vector<vector<int>> &matrix, int literals, int clauses)
{
    if(matrix.empty())
        return make_pair(nullptr, nullptr);

    int nChunksEveryHalfClause = ceil((float)literals / (sizeof(mytype) * 8));
    int nChunksEveryClause = nChunksEveryHalfClause * 2;
    size_t sizeMatrix = nChunksEveryClause * clauses * (sizeof(mytype));
    mytype *matrix_h, *matrix_d;

    //Alloco il vettore lato host
    matrix_h = (mytype*)malloc(sizeMatrix);
    //Riempio il vettore lato host
    for(int i = 0; i < matrix.size(); ++i)
    {
        //Inizializzo a tutti 0
        for(int j = 0; j < nChunksEveryClause; ++j)
            matrix_h[nChunksEveryClause * i + j] = 0;
        //Setto a 1 i bit corrispondenti
        for(int j = 0; j < matrix[i].size(); ++j)
        {
            int _value = abs(matrix[i][j]);
            int offset = (int)(_value-1) % (int)(sizeof(mytype) * 8);
            int chunk = nChunksEveryHalfClause - 1 - ((int)(_value-1) / (int)(sizeof(mytype) * 8));
            if(matrix[i][j] > 0)
                matrix_h[nChunksEveryClause * i + chunk] |= 1 << offset;
            else if(matrix[i][j] < 0)
                matrix_h[nChunksEveryClause * i + nChunksEveryHalfClause + chunk ] |= 1 << offset;
        }
    }  

    //Alloco il vettore lato device copiandolo da quello lato host
    cudaMalloc((void**)&matrix_d, sizeMatrix);
    cudaMemcpy(matrix_d, matrix_h, sizeMatrix, cudaMemcpyHostToDevice);

    return make_pair(matrix_h, matrix_d);
}

//Testato
void printCompressMatrix(mytype *matrix, int literals, int clauses)
{
    int nChunksEveryHalfClause = ceil((float)literals / (sizeof(mytype) * 8));
    int nChunksEveryClause = nChunksEveryHalfClause * 2;

    cout << "Print compress matrix: " << endl;
    cout << "nChunksEveryHalfClause: " << nChunksEveryHalfClause << endl;
    cout << "nChunksEveryClause: " << nChunksEveryClause << endl;
    cout << "literals: " << literals << endl;
    cout << "clauses: " << clauses << endl;
    cout << "sizeof(mytype): " << sizeof(mytype) << endl;
    cout << endl;

    for(int i = 0; i < clauses; ++i)
    {
        for(int j = 0; j < nChunksEveryClause; ++j)
        {
            bitset<sizeof(mytype) * 8> x(matrix[nChunksEveryClause * i + j]);
            cout << x << " ";
            //cout << x << "(" << matrix[nChunksEveryClause * i + j] << ") ";
        }
        cout << endl;
    }
    cout << endl;
}

int main() 
{
    cout << "-----------------------------------------------" << endl;
    cout << endl << "*** Start ***" << endl << endl;
    //input/dimacs/jnh1.cnf
    //input/3sat/uf20-01.cnf
    //input/small.cnf
    //input/tutorial.cnf
    //input/hole6.cnf
    CreateMatrix *matrix = new CreateMatrix("input/small.cnf", true);
    if (matrix->get_error())  return(1);
    vector<vector<bool>> bool_matrix = matrix->get_boolean_matrix();
    vector<vector<int>> int_matrix = matrix->get_int_matrix();
    literals_ = matrix->get_literals();
    clauses_ = matrix->get_clauses();

    //find_solution(bool_matrix);
    //cout << endl;

    cout << "Copy to gpu " <<  endl;
    pair<mytype *, mytype *> ref = copy_to_gpu(int_matrix, literals_, clauses_);
    printCompressMatrix(ref.first, literals_, clauses_);

    cout << endl << "*** End ***" << endl;
    cout << "-----------------------------------------------" << endl;

    return 0;
}

/*
I dati sono salvati in formato big endian, quindi il bit più significativo è il primo
*/
