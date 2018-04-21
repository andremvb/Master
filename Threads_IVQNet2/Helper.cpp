//
//  Helper.cpp
//  Red Neuronal IVQL
//
//  Created by Andre Valdivia on 15/12/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

#include "Helper.hpp"
#include <iostream>

//Convierte un vector en un estado
//Devuelve valores entre 0 y breaks-1
int vectorToState(vector<double>& input, int numBreaks){
    double state = 0;
    for (int i = 0, j = (int)input.size()-1; i < input.size(); i++,j--) {
        state += discretize(input[i], numBreaks, 0, 1) * pow(numBreaks, j);
    }
    if (state < 0) {
        std::cout<< "stop"<<endl;
    }
    
    return state;
}

int discretize(double number, int breaks, double min, double max){
    int numberInt = 0;
    number = number - min;
    max = max - min;
    double firstBreak = max/(double)breaks;
    numberInt = number/firstBreak;
    if (numberInt == breaks) {
        return breaks-1;
    }
    return numberInt;
}

//Devuelve un numero random entero entre un minimo y un maximo
int randInt(int min, int max){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min, max);
    
    return  dis(gen);
}

//Devueve un número random
double rand01(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    return dis(gen);
}

//Devuelve un número random con una semilla distinta especial para MLP
double randMLP(int numPesos){
    unsigned seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    double sigma = sqrt(double(2)/numPesos);
    normal_distribution<double> distribution(0,sigma);
    return distribution(generator);
}

void printToTxt(ofstream& file, double alpha, double beta, double temperature, vector<double> errores){
    
    file<<" Alpha: "<<alpha<< "Beta: "<<beta<<" Temperature: "<<temperature<<endl;
    for (int i = 0; i < errores.size() ; i++) {
        file<< errores[i]<<";";
    }
    file<<endl;
    for (int i = 0; i < errores.size() ; i++) {
        file<< i <<";";
    }
    file<<endl<<endl;
}

void printToTxt2(ofstream& file, double beta, double alpha, double temperature, vector<vector<double>> errores){
    
    file<< "Beta: "<<beta<<" Alpha: "<<alpha<<" Temperature: "<<temperature<<endl;
    for (int i = 0; i < errores.size() ; i++) {
        file<<"Fold: "<<i<<endl;
        for (int j = 0; j < errores[i].size(); j++) {
            file<< errores[i][j]<<";";
        }
        file<<endl;
    }
    file<<endl<<endl;
}

//Exporta un vector
void printVector(vector<double> vector, string name){
    ofstream file;
    file.open(name);
    for (int i = 0; i < vector.size() ; i++) {
            file<< vector[i]<<";";
    }
    file<<endl;
    file.close();
    file.clear();
}

//Exporta una matriz
void printMatrix(vector<vector<double>> matrix, string name){
    ofstream file;
    file.open(name);
    for (int i = 0; i < matrix.size() ; i++) {
        file<<"Fold: "<<i<<endl;
        for (int j = 0; j < matrix[i].size(); j++) {
            file<< matrix[i][j]<<";";
        }
        file<<endl;
    }
    file<<endl<<endl;
    file.close();
    file.clear();
}

