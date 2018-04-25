//
//  Helper.hpp
//  Red Neuronal IVQL
//
//  Created by Andre Valdivia on 15/12/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

#ifndef Helper_hpp
#define Helper_hpp

#include <stdio.h>
#include <math.h>
#include <random>
#include <fstream>
#include <vector>

#define PAlpha "\u03B1"
#define PBeta "\u03B2"
#define PGamma "\u03B3"
#define PEpsilon "\u03B5"
#define PDelta "\u03B4"
#define PSigma "\u03C3"

using namespace std;

enum class Politic {Softmax, e_greedy};
enum class Problem {spiral, iris, wine, glass, jain, flame, pathbased, compound, aggregation};


int randInt(int,int);
double rand01();
double randMLP(int numPesos);
int vectorToState(vector<double>& input, int numBreaks);
int discretize(double number, int breaks, double min, double max);
void printToTxt(ofstream& file, double alpha, double beta, double temperature, vector<double> errores);
void printToTxt2(ofstream& file, double beta, double alpha, double temperature, vector<vector<double>> errores);
void printVector(vector<double> vectorr, string name);
void printVector(vector<double> vectorr, string name, ofstream file);
void printMatrix(vector<vector<double>> matrix, string name);
void printMatrix(vector<vector<double>> matrix, string name, ofstream file);
#endif /* Helper_hpp */
