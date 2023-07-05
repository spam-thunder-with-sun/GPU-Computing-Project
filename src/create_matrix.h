#include <iostream>
#include <string>
#include <fstream>
#include <vector>

using namespace std;

class CreateMatrix
{
    public :
        CreateMatrix(string filename) 
        {
            string line;
            int row = 1;
            ifstream in (filename);

            if (!in.is_open()) 
            {
                cout << "Error opening file" << endl;
                error_ = true;
            }
            else
            {
                while(getline(in, line)) 
                {
                    if (line[0] == 'c')
                        continue;
                    if (line[0] == 'p') 
                    {
                        read_literals_and_clauses(line);
                        allocate_matrix();
                    } else 
                        write_clause(line, row++);

                }
                error_ = false;
            }
        }

        vector<vector<bool>> get_matrix() 
        {
            return matrix_;
        }

        int get_literals() 
        {
            return error_ ? -1 : literals_;
        }

        int get_clauses() 
        {
            return error_ ? -1 : clauses_;
        }

        bool get_error() 
        {
            return error_;
        }

        void print_matrix()
        {
            for (int i = 0; i < matrix_.size(); ++i) 
            {
                vector<bool> row = matrix_[i];
                for (int j = 0; j < row.size(); ++j)
                    cout << matrix_[i][j] << " ";
                cout << endl;
            }
        }

    
    private:
    vector<vector<bool>> matrix_;
    int literals_ = 0;
    int clauses_ = 0;
    bool error_ = true;

    void read_literals_and_clauses(string &s) 
    {
        int mult = 1, i = s.size() - 1;

        while (!isdigit(s[i]))--i;
        while (isdigit(s[i])) {
            clauses_ += mult * (s[i] - '0');
            mult *= 10;
            --i;
        }

        mult = 1;
        while (!isdigit(s[i])) --i;
        while (isdigit(s[i])) {
            literals_ += mult * (s[i] - '0');
            mult *= 10;
            --i;
        }
    }

    void allocate_matrix() 
    {
        matrix_.resize(clauses_ + 1);
        for (int i = 1; i <= clauses_; ++i)
            matrix_[i].resize(literals_ * 2 + 1);
    }

    void write_clause( string &line, int row) 
    {
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
                matrix_[row][curr_literal] = true;
            if (curr_literal < 0)
                matrix_[row][literals_ - curr_literal] = true;
        } 
    }
};
