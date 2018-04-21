    //
//  NeuralNetwork.cpp
//  Red Neuronal IVQL
//
//  Created by Andre Valdivia on 15/12/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

#include "NeuralNetwork.hpp"
#include <iostream>
#include <math.h>
#include "Helper.hpp"

using namespace std;

void NeuralNetwork::setNetwork(Data& data, Params& p){
    this->numActions = p.numActions;
    this->classes = data.getClassesPerBatch();
//    this->dimentionsProblem = data.dimentions;
    this->numberProblems = (int)classes.size();
    this->p = &p;
//    this->totalClasses = data.totalClasses;
    
    dimentionsNet.push_back(data.getTotalDimentions());
    for (int i = 0; i < p.structure.size()-1; i++) {
        dimentionsNet.push_back(p.structure[i]);
    }
    
    for (int i = 0; i < dimentionsNet.size(); i++) {
        states.push_back(pow(numActions, dimentionsNet[i]));
        if (states[i] <= 0) {
            cout<< "Error, numero de estados mayor a permitido por un long long unsigned int"<<endl;
        }
    }
    
    actions.resize(p.structure.size());     //Setear las acciones por cada agente
    net.resize(p.structure.size());         //Setear numero de capas
    comparationLayer.resize(data.getTotalClasses());      //Vector de comparacion
    rewards.resize(data.getTotalClasses());               //Vector de rewards
    
    //Setear agentes
    for (int i = 0; i < p.structure.size(); i++) {    //Recorre las capas
        for (int j = 0; j < p.structure[i]; j++) {    //Recorre las neuronas
            net[i].push_back(Agent());
            actions[i].push_back(0);
            if (i == 0) {
                net[i][j].setAgent(j, i, p, dimentionsNet[i], numActions, 0, states[i]);
            }else{
                net[i][j].setAgent(j, i, p, dimentionsNet[i], numActions, p.structure[i-1], states[i]);
            }
            
        }
    }
    
    //Linkear agentes entre las capas. Sucesores y predecesores
    for (int i = 0; i < net.size(); i++) {
        for (int j = 0; j < net[i].size(); j++) {
            //Sucesores
            if (i != net.size()-1) {
                for (int k = 0; k < net[i+1].size(); k++) {
                    net[i][j].sucesores.push_back(&net[i+1][k]);
                }
            }
            //Predecesores
            if (i != 0) {
                for (int k = 0; k < net[i-1].size(); k++) {
                    net[i][j].predecesores.push_back(&net[i-1][k]);
                }
            }
        }
    }
}

void NeuralNetwork::forget(){
    for (int i = 0; i < net.size(); i++) {
        for (int j = 0; j < net[i].size(); j++) {
            net[i][j].forget();
        }
    }
}

//Imprime la descripcion de la red
void NeuralNetwork::printDescription(){
    cout<<"Numero de acciones: " << this->numActions<<endl;
    for (int i = 0; i < states.size(); i++) {
        cout<<"Numero estados capa "<<i<< " : "<<states[i]<<endl;
    }
}

//Imprime la tabla Q de cada Agente
void NeuralNetwork::printQTable(){
    for (int i = 0; i < net.size() ; i++) {
        cout<<"----- Capa " << i << " -----" << endl;
        for (int j = 0; j < net[i].size(); j++) {
            net[i][j].printQTable();
        }
    }
    
    cout<<"----- Decisiones -----" << endl;
    for (int i = 0; i < actions.size(); i++) {
        for (int j = 0; j < actions[i].size(); j++) {
            cout<< actions[i][j] << "\t" << endl;
        }
    }
    cout<< "..................................."<<endl;
}

/*
 * Convierte un vector de acciones discretas en un indice de estado
 * Por ejemplo:
 *      Si tenemos 10 acciones y 2 agentes con sus decisiones:
 *      Acciones            Estado
 *       [2, 3]       =      23
 */
int NeuralNetwork::actionsToState(vector<double>& actions, int numActions){
    int state = 0;
    for (int i = 0, j = (int)actions.size()-1; i < actions.size(); i++, j--) {
        state += actions[i] * pow(numActions, j);
    }
    return state;
}

//Setea el vector de rewards
void NeuralNetwork::getReward(int output){
    //Setear la capa de comparacion
    for (int i = 0; i < comparationLayer.size(); ++i) {
        if (i == output)
            comparationLayer[i] = 1;
        else
            comparationLayer[i] = 0;
    }
    
    for (int i = 0; i < comparationLayer.size(); i++) {
        rewards[i] = -abs((double)actions.back()[i]/(double)numActions - comparationLayer[i]);
    }
}


void NeuralNetwork::train(vector<double> &input, int output){
    //Set init state
    int state = vectorToState(input,numActions);
    
    //Seleccionar accion
    for (int i = 0; i < net.size(); i++) {
        for (int j = 0; j < net[i].size(); j++) {
            actions[i][j] = net[i][j].selectAction(state);
        }
        state = actionsToState(actions[i], numActions);
    }
    
    //Actualizar tabla de ocurrencia
    for (int i = 1; i < net.size(); i++) {
        for (int j = 0; j < net[i].size(); j++) {
                for (int k = 0 ; k < net[i-1].size(); k++) {
                    net[i][j].increaseOcurrence(k, actions[i-1][k]);
                }
        }
    }
    
    //Obtener reward
    getReward(output);
    setOutputNet();
    
    //Actualizar Q
    //Actuaizar Q de la ultima capa
    for (int i = 0; i < net.back().size(); i++) {
        net.back()[i].updateQInfluence(rewards[i]);
    }
    
    //Actualizar Q de las otras capas
    for (int i = (int)net.size()-2; i >=1; i--) {
        for (int j = 0; j < net[i].size(); j++) {
            net[i][j].updateQInfluence(0);
        }
    }
    
    //Actualizar Q de la primera capa
    for (int j = 0; j < net[0].size(); j++) {
        net[0][j].updateQInfluence(0);
    }
}

void NeuralNetwork::test(vector<double> &input, int output){
    //Set state
    int state = vectorToState(input,numActions);
    
    //Seleccionar accion
    for (int i = 0; i < net.size(); i++) {
        for (int j = 0; j < net[i].size(); j++) {
            actions[i][j] = net[i][j].selectActionOptim(state);
        }
        state = actionsToState(actions[i], numActions);
    }

    //Setaer outputNet
    setOutputNet();
}


void NeuralNetwork::exportToPlot(string carpeta){
    int name = 0;
    for (int i = 0; i < net.size(); i++) {
        for (int j = 0; j < net[i].size(); j++) {
            ofstream  file;
            file.open(carpeta + "Ag" + to_string(name)+ ".csv");
            for (int k = 0; k < net[i][j].states.size(); k++) {
                for (int x = 0; x < net[i][j].states[k].centroide.size(); x++) {
                    file << net[i][j].states[k].centroide[x] << ",";
                }
                file << net[i][j].states[k].selectOptimActionToPrint() << "\n";
            }
            file.close();
            file.clear();
            name++;
        }
    }
}

//Setear outputNet. Es el array de salidas de la red
void NeuralNetwork::setOutputNet(){
    int out = -1;
    double max = -1;
    for (int i = 0;  i < net.back().size(); i++) {

        if(max < actions.back()[i]){
            max = actions.back()[i];
            out = i;
        }

    }
    outputNet = out;
}

void NeuralNetwork::getQMinMax(double& qMax, double& qMin){
    qMax = -1000;
    qMin = 1000;
    double tmpMax;
    double tmpMin;
    for (int i = 0; i < net.size(); i++) {
        for (int j = 0; j < net[i].size(); j++) {
            net[i][j].getMaxMinQAll(tmpMax, tmpMin);
            if (tmpMax > qMax) {
                qMax = tmpMax;
            }
            if (tmpMin < qMin) {
                qMin = tmpMin;
            }
        }
    }
}

void NeuralNetwork::exportQMaxMinTemp(ofstream& file, double error){
    double Qmax = -1000;
    double Qmin = 1000;
    double tmpMax;
    double tmpMin;
    for (int i = 0; i < net.size(); i++) {
        for (int j = 0; j < net[i].size(); j++) {
            net[i][j].getMaxMinQAll(tmpMax, tmpMin);
            if (tmpMax > Qmax) {
                Qmax = tmpMax;
            }
            if (tmpMin < Qmin) {
                Qmin = tmpMin;
            }
        }
    }
    file<<p->temperature<<"\t" << Qmax<< "\t" << Qmin << "\t"<<error <<endl;
}

void NeuralNetwork::printStatesNotVisited(Data& data){
    double state = 0;
    vector<double> input(data.getTotalDimentions());
    int output;
    vector<vector<double>> actions;
    actions.resize(net.size());
    for (int i = 0; i < net.size(); i++) {
        actions[i].resize(net[i].size());
    }
    //Este seria el ideal
//    for (int i = 0; i < data.dataIndexTest.size(); i++) {
//        data.getTest(input, output);
//        state = vectorToState(input, numActions);
//        for (int j = 0; j < net.size(); j++) {
//            for (int k = 0; k < net[j].size(); k++) {
//                net[j][k].printIsUsed(state);
//                actions[j][k] = net[j][k].selectActionOptim(state);
//            }
//            state = actionsToState(actions[j], numActions);
//        }
//    }
    int statesNotVisited = 0;
    //Simplificado
    for (int i = 0; i < data.getTestSize(); i++) {
        data.getTest(input, output);
        state = vectorToState(input, numActions);
        for (int j = 0; j < net.size(); j++) {
            for (int k = 0; k < net[j].size(); k++) {
                actions[j][k] = net[j][k].selectActionOptim(state);
            }
            if (!net[j][0].printIsUsed(state)) {
                statesNotVisited++;
            }
            state = actionsToState(actions[j], numActions);
        }
    }
    data.resetTestPointer();
    cout<< "Numero de estados no visitados: "<<statesNotVisited;
    cout<<endl;
}
