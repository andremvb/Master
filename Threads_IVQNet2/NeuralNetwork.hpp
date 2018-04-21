//
//  NeuralNetwork.hpp
//  Red Neuronal IVQL
//
//  Created by Andre Valdivia on 15/12/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

#ifndef NeuralNetwork_hpp
#define NeuralNetwork_hpp

#include "Data.hpp"
#include "Agent.hpp"
#include "Helper.hpp"

class NeuralNetwork {
    vector<double> rewards;
    
public:
    Params* p;
    
//    vector<int> dimentionsProblem;          //Vector de dimensiones por problema
    vector<int> dimentionsNet;              //Dimensiones por capa
    vector<int> classes;                    //Numero de clases por problema (loadIris, loadWine) = [3,3]
//    int totalClasses;                       //Numero total de clases
    
    vector<double> comparationLayer;        //Capa de comparacion para el train

    vector<long long unsigned int> states;  //Numero de estados por capa
    int numberProblems;                 //Numero de problemas
    int numActions = 0;                 //Numero de acciones de todos los agentes
    
    vector<vector<double>> actions;     //Acciones que realizan los agentes
    vector<vector<Agent>> net;          //Agentes
    
    int outputNet;              //Salida de la red
    
    //Setup
    void setNetwork(Data& data, Params& p);
    
    //Train
    void getReward(int output);                        //Setear el vector rewards con los rewards de la ultima capa de la red
    void setOutputNet();                                        //Setea el vector outputNet, es la sali
    void train(vector<double> &in, int out);           //Entrenamiento
    void test(vector<double> &in, int out);            //Test
    int actionsToState(vector<double>& actions, int numActions);//Convierte un vector en un numero
    void forget();                                              //Elimina el aprendizaje de la red
    
    //Otros
    void printDescription();        //Imprime la descripcion de la red
    void printQTable();             //Imprime la tabla Q
    void exportToPlot(string carpeta);            //Exporta los estados de cada agente
    void exportQMaxMinTemp(ofstream& file, double error);       //Exporta los Q max y min de toda la red
    void getQMinMax(double& qMax, double& qMin);
    void printStatesNotVisited(Data& data);
    
};

#endif /* NeuralNetwork_hpp */
