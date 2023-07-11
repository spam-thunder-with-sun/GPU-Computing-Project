#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>

using namespace std;

class CreateMatrix
{
    public :
        CreateMatrix(string filename, bool print = false) 
        {
            //Aperura file
            ifstream in (filename);
            if (!in.is_open()) 
            {
                if(print)
                    cout << "Error opening file" << endl;
                error_ = true;
                return;
            }

            string line;                
            char c;
            int row = 0;
            int tmp;
            while(!in.eof())
            {
                //Leggo linea per linea
                getline(in, line);

                if(!line.empty())
                {
                    //Copio in un stringstream
                    istringstream ss(line);
                    //Leggo il primo carattere della linea
                    c = line[0];
                    switch(c)
                    {
                        case 'c':
                            //Commento
                            if(print)
                                cout << line << endl;
                            break;
                        case 'p':
                            //Dati del problema
                            ss >> c >> line >> literals_ >> clauses_;
                            if(print)
                                cout << "Data:" << line << " literals:" << literals_ << " clauses:" << clauses_ << endl;
                            //Alloco la matrice
                            matrix_.resize(clauses_);
                            for (int i = 0; i < clauses_; ++i)
                                matrix_[i].resize(literals_ * 2 + 1);
                            break;
                        default:
                            //Leggo il problema
                            while(!ss.eof())
                            {
                                ss >> tmp;

                                if(print)
                                    cout << tmp << " ";

                                if(tmp == 0)
                                {
                                    if(print)
                                        cout << endl;
                                    row++;
                                } else if(tmp > 0)
                                    matrix_[row][tmp] = true;
                                else if(tmp < 0)
                                    matrix_[row][(-tmp) + literals_] = true;  
                            }
                    }
                }
            }

            if(print)
            {
                cout << endl;
                print_matrix();
                cout << endl;
            }

            error_ = false;
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
            cout << "Literals: " << literals_ << endl;
            cout << "Clauses: " << clauses_ << endl;
            cout << "Matrix:" << endl;
            for (int i = 0; i < matrix_.size(); ++i) 
            {
                vector<bool> row = matrix_[i];
                for (int j = 1; j < row.size(); ++j)
                    cout << matrix_[i][j] << " ";
                cout << endl;
            }
        }

    private:
    vector<vector<bool>> matrix_;
    int literals_ = 0;
    int clauses_ = 0;
    bool error_ = true;
};
