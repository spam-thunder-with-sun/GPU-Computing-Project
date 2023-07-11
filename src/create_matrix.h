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
                            //Alloco le matrici
                            bool_matrix_.resize(clauses_);
                            int_matrix_.resize(clauses_);
                            for (int i = 0; i < clauses_; ++i)
                                bool_matrix_[i].resize(literals_ * 2 + 1);
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
                                {
                                    bool_matrix_[row][tmp] = true;
                                    int_matrix_[row].push_back(tmp);
                                }
                                else if(tmp < 0)
                                {
                                    bool_matrix_[row][(-tmp) + literals_] = true;
                                    int_matrix_[row].push_back(tmp);
                                } 
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

        vector<vector<bool>> get_boolean_matrix() 
        {
            return bool_matrix_;
        }

        vector<vector<int>> get_int_matrix() 
        {
            return int_matrix_;
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
            for (int i = 0; i < bool_matrix_.size(); ++i) 
            {
                vector<bool> row = bool_matrix_[i];
                for (int j = 1; j < row.size(); ++j)
                    cout << bool_matrix_[i][j] << " ";
                cout << endl;
            }
            for(int i = 0; i < int_matrix_.size(); ++i)
            {
                vector<int> row = int_matrix_[i];
                for (int j = 0; j < row.size(); ++j)
                    cout << int_matrix_[i][j] << " ";
                cout << endl;
            }
        }

    private:
    vector<vector<bool>> bool_matrix_;
    vector<vector<int>> int_matrix_;
    int literals_ = 0;
    int clauses_ = 0;
    bool error_ = true;
};
