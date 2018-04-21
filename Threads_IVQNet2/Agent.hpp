//
//  Agente.hpp
//  Red Neuronal IVQL
//
//  Created by Andre Valdivia on 15/12/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

#ifndef Agente_hpp
#define Agente_hpp

#include <stdio.h>
#include <vector>
#include <map>
#include "Data.hpp"
#include "State.hpp"
#include "Helper.hpp"

using namespace std;

class Agent {
public:
    
    Params* p;
    int name = -1;
    int layer = -1;
    int actualState = -1;
    State* ptrActualState;
    int actionTaked = -1;
    double r_Q = -1;
    double distState = 0;
    double minDistance = -1;
    int numDim = -1;
    int numActions = -1;
    vector<double> posCentroids;                //Posicion de los centroides de los estados, para conjugarlos despues.
    vector<State> states;
    vector<vector<vector<double>>> oiTable;

    vector<Agent*> predecesores;
    vector<Agent*> sucesores;
    
    
    //Apoyo a algoritmos
    vector<double> max;
    vector<double> min;
    
    //Setear
    void setAgent(int name, int layer, Params& p, int numDimentions,int numActions, int numAgents, long long unsigned int numStates);
    
    //Entrenar
    //Selecciona una accion
    int selectAction(int state);
    int selectActionOptim(int state);
    void increaseOcurrence(int agent,int action);
    void updateQ(double reward);
    void updateQInfluence(double reward);
    double getOP(int name, int actionTaked);    //El agente retorna la OP sobre el agente cliente.
    double getIV();
    void forget();
    
    //Otros
    void printQTable();
    void getMaxMinQAll(double &max, double& min);
    bool printIsUsed(int state);
    
};






#endif /* Agente_hpp */
