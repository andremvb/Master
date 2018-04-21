//
//  State.cpp
//  Version Intermedia
//
//  Created by Andre Valdivia on 5/07/17.
//  Copyright © 2017 Andre Valdivia. All rights reserved.
//

#include "State.hpp"
#include "Helper.hpp"
#include <iostream>

double State::distance(State& s){
    double dist = 0;
    for (int i = 0; i < centroide.size(); i++) {
        dist += pow(centroide[i] - s.centroide[i], 2);
    }
    return dist;
}

void State::setState(int name,int layer, int agent, int numActions, int numDimentions, vector<double> centro,Params& p){
    this->name = name;
    this->p = &p;
    this->centroide = centro;
    this->layer = layer;
    this->agent = agent;
    
    if (p.politic == Politic::Softmax) {
        changeProb = &State::softMax;
    }else if(p.politic == Politic::e_greedy){
        changeProb = &State::e_greedy;
    }else{
        changeProb = nullptr;
        cout<<"Politica aun no definida en el estado"<<endl;
    }
    
    actionVal.resize(numActions);
    actionProb.resize(numActions);
    neight.resize(numDimentions*2);
    
    (this->*changeProb)();
    
    
}

void State::forget(){
    for (int i = 0; i< actionVal.size(); i++) {
        actionVal[i] = 0;
        (this->*changeProb)();
    }
    used = false;
}

void State::softMax(){
    double eulers = 0;
    
    for (int i = 0; i < actionVal.size(); i++) {
        actionProb[i] = exp(actionVal[i]/p->temperature);
        eulers += actionProb[i];
    }
    
    if (eulers != 0 &&  !isnan(eulers) && !isinf(eulers)) {
        for (int i = 0; i < actionVal.size(); i++) {
            actionProb[i] = actionProb[i] / eulers;
            if (isnan(actionProb[i])) {
                actionProb[i] = 1;
                cout<< "wait, is nan en State::softmax"<<endl;
            }
        }
    }else{
        this->greedy();
    }

 }

void State::e_greedy(){
    double maxQ = -1000;
    int posMaxQ = -1;
    for (int i = 0; i < actionVal.size(); i++) {
        if (actionVal[i] > maxQ) {
            maxQ = actionVal[i];
            posMaxQ = i;
        }
    }
    double otherProb = p->ebsilon/( actionProb.size() -1);
    for (int i = 0 ; i < actionProb.size(); i++) {
        if (i != posMaxQ) {
            actionProb[i] = otherProb;
        }else{
            actionProb[i] = 1 - p->ebsilon;
        }
    }
}

void State::greedy(){
    double maxQ = -1000;
    int indexMaxQ = -1;
    for (int i = 0; i < actionVal.size(); i++) {
        if (actionVal[i] > maxQ) {
            indexMaxQ = i;
            maxQ = actionVal[i];
        }
    }
    
    for (int i = 0; i < actionProb.size(); i++) {
        if (i != indexMaxQ) {
            actionProb[i] = 0;
        }else{
            actionProb[i] = 1;
        }
    }
}

/*
 * Seleccionar politica en el entrenamiento de la red neuronal
 */
int State::selectAction(){
    int actionTaked = -1;
    double actRandom = rand01();
    double acumulador = 0;
    
    for (int i = 0; i < actionVal.size(); i++) {
        acumulador += actionProb[i];
        if (acumulador >= actRandom) {
            actionTaked = i;
            break;
        }
    }
    
    if (actionTaked == -1) {
        cout<< "La accion tomada salio -1"<<endl;
    }
    
    return actionTaked;
}

int State::selectOptimAction(){
    int actionTaked = -1;
    double max = -100;
    for (int i = 0; i < actionVal.size(); i++) {
        if (max < actionVal[i] && actionVal[i] != 0) {
            max = actionVal[i];
            actionTaked = i;
        }
    }
    if (actionTaked == -1) {
//        cout<< "No hay accion optima en: "<< "Agent:" <<  agent<< "\t Layer:" <<  layer<< endl;
//        cout<<"\t Centroid: "<<getCentroid()<<endl<<endl;
        return selectAction();
    }
    return actionTaked;
}

/*
 * Igual que selectOptimAction, pero en vez de que selecciona una accion greedy si es Nan, solo devuelve -1
 */
int State::selectOptimActionToPrint(){
    int actionTaked = -1;
    double max = -100;
    for (int i = 0; i < actionVal.size(); i++) {
        if (max < actionVal[i] && actionVal[i] != 0) {
            max = actionVal[i];
            actionTaked = i;
        }
    }
    return actionTaked;
}

//Actualiza el estado aledaño según el error del estado principal
double State::updateQNeight(double RV, int actionTaked){
    actionVal[actionTaked] = actionVal[actionTaked] + p->alpha * RV;
    used = true;
    
    //Expancion de acciones
    for (int i = 1; i <= p->alcanceActions; i++) {
        if(actionTaked + i < actionVal.size())
            actionVal[actionTaked + i] = actionVal[actionTaked + i] + p->alpha * RV * (p->gamma2/(i));
        
        if (actionTaked - i >= 0)
            actionVal[actionTaked - i] = actionVal[actionTaked - i] + p->alpha * RV * (p->gamma2/(i));
    }
    (this->*changeProb)();
    return RV;
}


double State::updateQ(double reward, double IV, int actionTaked){
    double RV = reward + IV - actionVal[actionTaked];
    actionVal[actionTaked] = actionVal[actionTaked] + p->alpha * (RV);
    used = true;
    for (int i = 1; i <= p->alcanceActions; i++) {
        if(actionTaked + i < actionVal.size())
        actionVal[actionTaked + i] = actionVal[actionTaked + i] + p->alpha * RV * (p->gamma2/(i));
        
        if (actionTaked - i >= 0)
        actionVal[actionTaked - i] = actionVal[actionTaked - i] + p->alpha * RV * (p->gamma2/(i));
    }
    (this->*changeProb)();
    return RV;
}

string State::getCentroid(){
    string n;
    for (int i = 0; i < centroide.size(); i++) {
        n.append(to_string(centroide[i]));
        n.append("\t");
    }
    return n;
}

void State::getMaxMinQ(double& max, double& min){
    max = -1000;
    min = +1000;
    for (int i = 0; i< actionVal.size(); i++) {
        if (actionVal[i] > max) {
            max = actionVal[i];
        }
        if (actionVal[i] < min) {
            min = actionVal[i];
        }
    }
}

bool State::printIsUsed(){
    if (!used) {
        cout<<"Estado no usado: ";
        for (int i = 0; i < centroide.size(); i++) {
            
        }
        for(double c:centroide){
            cout<<c<<" ";
        }
        cout<<" Agent: "<<agent;
        cout<<" Layer: "<<layer<<endl;
        return used;
    }
    return used;
}
