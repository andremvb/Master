//
//  Agent.cpp
//  Red Neuronal IVQL
//
//  Created by Andre Valdivia on 15/12/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

#include "Agent.hpp"
#include <iostream>
#include "Helper.hpp"
#include <math.h>

using namespace std;
//numAgents = Numero de agentes de la capa anterior para setear la tabla oi
//numActions = Numero de acciones posibles por estado del agente
void Agent::setAgent(int name, int layer, Params& p, int numDimentions,int numActions, int numAgents, long long unsigned int numStates){
    
    //Eliminar numStatesPrev
    this->name = name;
    this->layer = layer;
    this->p = &p;
    this->numDim = numDimentions;
    this->numActions = numActions;
    
    states.resize(numStates);
    
    //Setear centroide de estados
    this->minDistance = 1/(double)numActions;
    
    //Setear los indices de los centros de los estados
    posCentroids.push_back(minDistance/2);
    for (int i = 0; i < numActions-1; i++) {
        posCentroids.push_back(posCentroids[i]+minDistance);
    }
    
    //Vector que setea el centro del estado
    vector<double> centro (numDimentions,posCentroids[0]);
    max.resize(numDimentions);
    min.resize(numDimentions);
    
    //Setear estados
    for (int i = 0; i < states.size(); i++) {
        states[i].setState(i, layer, name, numActions, numDimentions, centro, p);
        for (int index = numDimentions-1; index >= 0; index--) {
            if (centro[index] + minDistance <= 1) {
                centro[index] += minDistance;
                break;
            }else{
                centro[index] = posCentroids[0];
            }
            
        }
    }
    
    //Setear tabla oi... numAgentes/numStates/numActions+1
    oiTable.resize(numAgents);
    for (int i = 0; i < numAgents; i++) {
        oiTable[i].resize(numStates);
        for (int j = 0; j < numStates; j++) {
            oiTable[i][j].resize(numActions + 1);
        }
    }
    
    //Setear punteros de estados a vecinos (Poco util por ahora)
    int pos = 0;
    int ini = 0;
    int fin = 0;
    int intervalo = 0;
    for (int i = 0; i < numDimentions; i++) {
        pos = 0;
        int x = pow(numActions, numDimentions - i - 1);
        for (int j = 0; j < x; j++) {
            ini = pow(numActions, i + 1) * j;
            fin = ini +  pow(numActions, i + 1);
            intervalo = (fin - ini)/numActions;
            for (int k = ini; k < fin; k++, pos++) {
                if (pos >= ini + intervalo) {
                    states[pos].neight[i*2] = &states[pos - pow(numActions, i)];
                }
                if (pos < fin - intervalo) {
                    states[pos].neight[i*2+1] = &states[pos + pow(numActions, i)];
                }
            }
        }
    }
    
}


void Agent::forget(){
    for (int i = 0; i < states.size(); i++) {
        states[i].forget();
    }

    //oiTable
    for (int i = 0; i < oiTable.size(); i++) {
        for (int j = 0; j < oiTable[i].size(); j++) {
            std::fill(oiTable[i][j].begin(), oiTable[i][j].end(), 0);
        }
    }
}

void Agent::printQTable(){
    cout<< "Agente "<< name << ":" <<endl;
    for (int i = 0; i < states.size(); i++) {
        //Imprimir centroides como rotulo
        for (int j = 0; j < numDim; j++) {
            cout<< states[i].centroide[j]<<"\t";
        }
        cout<<":\t";
        //Imprimir los valores de accion
        for (int j = 0; j < states[i].actionVal.size(); j++) {
            
            cout<< states[i].actionVal[j] << "\t";
        }
        cout<< endl;
    }
    cout<<endl;
}

int Agent::selectAction(int state){
    actualState = state;
    ptrActualState = &states[state];
    actionTaked = states[state].selectAction();
    return actionTaked;
}

int Agent::selectActionOptim(int state){
    actualState = state;
    ptrActualState = &states[state];
    actionTaked = states[state].selectOptimAction();
    return actionTaked;
}



void Agent::increaseOcurrence(int agent, int action){
    this->oiTable[agent][actualState][action]++;
    oiTable[agent][actualState].back()++;
}

double Agent::getIV(){
    double iV=0;
    for (int i = 0; i < sucesores.size(); i++) {
        
//        int total = sucesores[i]->oiTable[name][actualState].back();
//        int o =sucesores[i]->oiTable[name][actualState][actionTaked];
//        double oi = (double)o/(double)total;
//        //NO saldra Nan porque se pida la opnion del estado actual, y ya tomo minimo una decisión en el forward de ese entrenamiento
//        if (isnan(oi)) {
//            cout<< "Salio NaN en getIV"<<endl;
//        }
//        
//        double op = -1;
//        if (sucesores[i]->r_Q < 0) {
//            op = sucesores[i]->r_Q * oi;
//        }else if(sucesores[i]->r_Q > 0){
//            op = sucesores[i]->r_Q * (1 - oi);
//        }else{
//            op = 0;
//        }
        double op = sucesores[i]->getOP(this->name, this->actionTaked);
        iV += p->beta * op;
    }
    return  iV;
}

double Agent::getOP(int name, int actionTaked){
//    double oi = (double)oiTable[name][actualState].back()/(double)oiTable[name][actualState][actionTaked];
    double oi = (double)oiTable[name][actualState][actionTaked]/(double)oiTable[name][actualState].back();
    if (isnan(oi)) {
        cout<< "Salion NAN en getOP"<<endl;
    }
    
    double op = 0;
    if (this->r_Q < 0)
        op = r_Q * oi;
    else if(r_Q > 0)
        op = r_Q * (1 - oi);
    else
        op = 0;
    
    return op;
}

void Agent::updateQ(double reward){
    double iV = getIV();
    r_Q = states[actualState].updateQ(reward, iV, actionTaked);
}

void Agent::updateQInfluence(double reward){
    double iV = getIV();
    
    r_Q = states[actualState].updateQ(reward, iV, actionTaked);
    
    if (r_Q < -20) {
        cout<<"Stop"<<endl;
    }
    double maxDist = p->alcanceState*minDistance;
    
    //Expansion de aprendizaje a estados aledaños
    //Setear max y min (limites de estados vecinos)
    vector<double> itState (numDim, minDistance/2);
    for (int i = 0; i < itState.size(); i++) {
        //Max
        if (states[actualState].centroide[i] + maxDist > 1) {
            max[i] = posCentroids.back();
        }else{
            max[i] = states[actualState].centroide[i] + maxDist;
        }
        //Min
        if (states[actualState].centroide[i] - maxDist < 0) {
            min[i] = posCentroids[0];
        }else{
            min[i] = states[actualState].centroide[i] - maxDist;
        }
    }
    
    //Actualizar Q de los estados vecinos a un radio
    double distance = -1;
    int i = numDim - 1;
    vector<double> tmpStateTuple = this->min;
    double tmpStatePosition = 0;
    while (i >= 0) {
        tmpStatePosition = vectorToState(tmpStateTuple, this->numActions);
        distance = states[actualState].distance(states[tmpStatePosition]);
        if (distance != 0) {
            states[tmpStatePosition].updateQNeight(r_Q*exp(-distance/(2*pow(p->sigma,2))), actionTaked);
//            states[tmpStatePosition].updateQNeight(r_Q*p->gamma/distance, actionTaked);
        }
        while (i >= 0){
            if (tmpStateTuple[i] + minDistance <= max[i]) {
                tmpStateTuple[i] += minDistance;
                i = numDim - 1;
                break;
            }else{
                tmpStateTuple[i] = min[i];
                i--;
            }
        }
    }
}

void Agent::getMaxMinQAll(double &max, double& min){
    max = -1000;
    min = 1000;
    double tmpMax;
    double tmpMin;
    
    for (int i = 0; i < states.size(); i++) {
        states[i].getMaxMinQ(tmpMax, tmpMin);
        if (tmpMax > max) {
            max = tmpMax;
        }
        if (tmpMin < min) {
            min = tmpMin;
        }
    }
}

bool Agent::printIsUsed(int state){
    return states[state].printIsUsed();
}
                     

