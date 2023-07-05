using namespace std;

#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include "create_matrix.h"

int literals_ = 0;
int clauses_ = 0;


void print_solution(std::vector<bool> sol) {
    std::cout << "SAT: \n";
    for (int i = 1; i <= sol.size() / 2; ++i) {
        if (sol[i])
            std::cout << "x" << i << " ";
        else
            std::cout << "NOTx" << i << " ";
    }
    std::cout << std::endl;
}

bool is_valid_solution(std::vector<bool> res) {
    for (int i = 1; i < res.size(); ++i)
        if (res[i] == false)
            return false;
    return true;
}

bool sat(std::vector<std::vector<bool>> &M, std::vector<bool> &sol) {
    std::vector<bool> res(clauses_ + 1, false);

    for (int i = 1; i <= clauses_; ++i)
        for (int j = 1; j <= literals_ * 2; ++j)
            res[i] = res[i] || (M[i][j] && sol[j]);        

    return is_valid_solution(res);
}

void find_solution (std::vector<std::vector<bool>> &M) {
    std::vector<bool> vec (literals_ * 2 + 1, false);
    for (int sol = 0; sol <= (1 << literals_) - 1; ++sol) {
        for (int i = 0; i < literals_; ++i) {
            vec[i + 1] = (sol >> i) & 1;
            vec[i + literals_ + 1] = !(vec[i + 1]);
        }

        if (sat(M, vec)) {
            print_solution(vec);
            return;
        }
    }
    std::cout << "UNSAT!\n";
}


int main() 
{
    //input/jnh1.cnf
    //input/3sat/uf20-01.cnf
    CreateMatrix *foo = new CreateMatrix("input/small.cnf");

    if (foo->get_error()) 
        return(1);

    vector<vector<bool>> matrix = foo->get_matrix();
    literals_ = foo->get_literals();
    clauses_ = foo->get_clauses();
    foo->print_matrix();

    find_solution(matrix);

    return 0;
}