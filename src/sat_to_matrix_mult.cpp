#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>

std::ifstream in ("jnh1.cnf");
int literals = 0, clauses = 0;

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
    std::vector<bool> res(clauses + 1, false);

    for (int i = 1; i <= clauses; ++i)
        for (int j = 1; j <= literals * 2; ++j)
            res[i] = res[i] || (M[i][j] && sol[j]);        

    return is_valid_solution(res);
}

void find_solution (std::vector<std::vector<bool>> &M) {
    std::vector<bool> vec (literals * 2 + 1, false);
    for (int sol = 0; sol <= (1 << literals) - 1; ++sol) {
        for (int i = 0; i < literals; ++i) {
            vec[i + 1] = (sol >> i) & 1;
            vec[i + literals + 1] = !(vec[i + 1]);
        }

        if (sat(M, vec)) {
            print_solution(vec);
            return;
        }
    }
    std::cout << "UNSAT!\n";
}

void write_clause(std:: vector<std::vector<bool>> &M, std::string &line, int row) {
    for (int i = 0; i < line.size(); ++i) {
        while (!std::isdigit(line[i]) && line[i] != '-') ++i;
        
        int j = 0; 
        while (std::isdigit(line[i])) {
            ++j;    ++i;
        }

        int curr_literal = 0, mult = 1;
        if (i - j - 1 >= 0 && line[i - j - 1] == '-')
            mult = -1;

        for (int digit = 1; digit <= j; ++digit) {
            curr_literal += (line[i - digit] - '0') * mult;
            mult *= 10;
        }

        if (curr_literal > 0)
            M[row][curr_literal] = true;
        if (curr_literal < 0)
            M[row][literals - curr_literal] = true;
    } 
}

void allocate_matrix(std::vector<std::vector<bool>> &M) {
    M.resize(clauses + 1);
    for (int i = 1; i <= clauses; ++i)
        M[i].resize(literals * 2 + 1);
}

void read_literals_and_clauses(std::string &s) {
    int mult = 1, i = s.size() - 1;

    while (!std::isdigit(s[i]))--i;
    while (std::isdigit(s[i])) {
        clauses += mult * (s[i] - '0');
        mult *= 10;
        --i;
    }

    mult = 1;
    while (!std::isdigit(s[i])) --i;
    while (std::isdigit(s[i])) {
        literals += mult * (s[i] - '0');
        mult *= 10;
        --i;
    }
}

std::vector<std::vector<bool>> create_matrix() {    
    std::string line;
    std::vector<std::vector<bool>> M;
    int row = 1;

    if (in.is_open()) {
        while(std::getline(in, line)) {
            if (line[0] == 'c')
                continue;

            if (line[0] == 'p') {
                read_literals_and_clauses(line);
                allocate_matrix(M);
            } else {
                write_clause(M, line, row++);
            }
        }
    }
    return M;
}

int main() {
    std::vector<std::vector<bool>> M = create_matrix();
    find_solution(M);

    return 0;
}   