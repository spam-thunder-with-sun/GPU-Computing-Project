using namespace std;

#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include "create_matrix.h"

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


int main() 
{
    //input/dimacs/jnh1.cnf
    //input/3sat/uf20-01.cnf
    //input/small.cnf
    //input/tutorial.cnf
    CreateMatrix *foo = new CreateMatrix("input/small.cnf", true);
    if (foo->get_error())  return(1);
    vector<vector<bool>> matrix = foo->get_matrix();
    literals_ = foo->get_literals();
    clauses_ = foo->get_clauses();

    find_solution(matrix);

    cout << "Fine" << endl;

    return 0;
}