//
//  main.cpp
//  Threads_IVQNet2
//
//  Created by Andre Valdivia on 24/01/18.
//  Copyright © 2018 Andre Valdivia. All rights reserved.
//

/*
 * Imprimir en leyend al principio el numero de combinacion.
 * Sugerencia: Imprimir el numero de estados sin haber visitado en el test.
 */

#include <iostream>
#include <map>
#include <math.h>
#include <fstream>
#include <string>
#include <iomanip>

#include "Agent.hpp"
#include "Helper.hpp"
#include "Data.hpp"
#include "NeuralNetwork.hpp"
#include "FileManager.hpp"
#include "Contingence.hpp"
#include "State.hpp"
#include "ErrorManager.hpp"

using namespace std;

std::ostream& operator<<(std::ostream& os, Politic const& politic){
    switch (politic) {
        case Politic::Softmax:
            os<<"Softmax";
            break;
        case Politic::e_greedy:
            os<<"E_greedy";
            break;
        default:
            os<<"CaseAunNoEn<<OPerator";
            break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, Problem const& problem){
    switch (problem) {
        case Problem::spiral:
            os<<"spiral";
            break;
        case Problem::iris:
            os<<"iris";
            break;
        case Problem::wine:
            os<<"wine";
            break;
        case Problem::jain:
            os<<"jain";
            break;
        case Problem::flame:
            os<<"flame";
            break;
        case Problem::pathbased:
            os<<"pathbased";
            break;
        case Problem::compound:
            os<<"compound";
            break;
        case Problem::aggregation:
            os<<"aggregation";
            break;
        default:
            os<<"CaseAunNoEn<<OPerator";
            break;
    }
    return os;
}


void exportParams(Params& p, string folder);

//Entrena la red en un problema, combinando parametros
void trainCombis();

//Entran la red en varios problemas.
void trainMultiModal();

int main(int argc, const char * argv[]) {
    Params param;                                   //Parametros
    NeuralNetwork neuralNetwork;                    //Modelo
    Data data;                                      //Data
    ContingenceTable contTableTest;                 //Tabla de contingencia para el test
    ContingenceTable contTableTrain;                //Tabla de contingencia para el train
    ErrorManager errorManager;
    
    //SETEAR PARAMETROS
    //Con que problema trabajaremos
    param.problems.push_back(Problem::jain);
//    param.problems.push_back(Problem::flame);
//    param.problems.push_back(Problem::spiral);
    //param.problems.push_back(Problem::pathbased);
    //param.problems.push_back(Problem::aggregation);
    
    //    p.ebsilon = 0.0005;
    param.baseTemp = 0.05;
    param.tempDec = true;
    param.temperature = 0.003;
    
    //Propagacion de Q-value
    param.gamma = 0.0005;
    param.timeConstant = 300;
    param.sigma = 0.03;
    param.gamma2 = 0.005;
    param.alcanceActions = 0;
    param.alcanceState = 2;
    
    param.politic = Politic::Softmax;
    
    param.numActions = 10;
    param.epochs = 300;
    
    param.numFolds = 10;
    
    param.alphas={0.05};        //    p.alphas = {0.05, 0.1, 0.2};
    param.betas = {0.75};       //    p.betas = {0.5, 0.75, 0.9, 0.95};
    param.factorTemps = {0.99}; //    p.factorTemps = {0.5, 0.85, 0.9, 0.99, 0.995};
    param.structure = {2};
    param.ebsilons = {};
    param.initChangeParam();
    
    
    data.loadProblems(param);
    data.initKFold(param.numFolds);
    data.printDescription();
    
    param.structure.push_back(data.getTotalClasses());
    
    //Initialize network
    neuralNetwork.setNetwork(data, param);
    neuralNetwork.printDescription();
    
    contTableTrain.setTable(data.getClassesPerBatch());
    contTableTest.setTable(data.getClassesPerBatch());
    
    vector<double> input (data.getTotalDimentions(),0);           //Vector de entrada a la red
    int output = 0;              //Etiquetas del vector de entrada.
    
    //EXPORT//
    string folder = "Estadisticas/AccionesDiscretas/";
    errorManager.init(&param, folder);
    
    cout<<"Numero de combinaciones: "<<param.getnumCombi()<<endl;
    
    ////////////// ENTRENAMIENTO DE LA RED ////////////////////////
    
    //Cambiando de parametros
    for (int combi = 0; combi < param.getnumCombi(); combi++) {
        cout<<"///// Generando combinacion: "<<combi<<"/////"<<endl;
        //Folds
        for (int fold = 0; fold < param.numFolds; fold++) {
            cout<<"F: "<<fold<<"  "<<endl;
            //Epocas
            for (int epoch = 0; epoch < param.epochs; epoch++) {
                param.decTemp(epoch);
                
                //Train
                for (int j = 0; j < data.getTrainSize(); j++) {
                    data.getTrain(input, output);
                    neuralNetwork.train(input, output);
                    contTableTrain.set(data.getActualBatch(), output, neuralNetwork.outputNet);
                }
                errorManager.setErrorTrain(contTableTrain, epoch, fold);
                data.shuffleDataTrain();
                data.resetTrainPointer();
                
                //Test
                for (int j = 0; j < data.getTestSize(); j++) {
                    data.getTest(input, output);
                    neuralNetwork.test(input, output);
                    contTableTest.set(data.getActualBatch(), output, neuralNetwork.outputNet);
                }
                data.resetTestPointer();
                errorManager.setErrorTest(contTableTest, epoch, fold);
                errorManager.exportFold(fold);
            }
            neuralNetwork.printStatesNotVisited(data);
            cout<<"Train: "<<errorManager.getTrainErrorFold(fold)<<"   Test: "<<errorManager.getTestErrorFold(fold)<<endl;
            data.nextFold();
            if (fold == 0) {
                neuralNetwork.exportToPlot(folder);
            }
            
            neuralNetwork.forget();
            cout<<"Promedio del train de los ultimas epocas: "<<errorManager.getTrainLastsErrors(param.epochs, fold, 100)<<endl;
            cout<<"Promedio del test de los ultimas epocas: "<<errorManager.getTestLastsErrors(param.epochs, fold, 100)<<endl<<endl;
        }
        data.resetFold();
        cout<<endl;
        errorManager.exportCombi(combi);
        errorManager.exportLeyend();
        cout<<"Error promedio en el train: "<<errorManager.getTrainErrorCombi(combi)<<endl;
        cout<<"Error promedio en el test: "<<errorManager.getTestErrorCombi(combi)<<endl<<endl;
        param.nextParam();
    }
    
    //Resetear para imprimir los parametros
    param.resetParam();
    
    //Export resumen
    exportParams(param, folder);
}

void trainCombis(){
    //Inicializando
    Params param;                                   //Parametros
    NeuralNetwork neuralNetwork;                    //Modelo
    Data data;                                      //Data
    ContingenceTable contTableTest;                 //Tabla de contingencia para el test
    ContingenceTable contTableTrain;                //Tabla de contingencia para el train
    ErrorManager errorManager;
    
    //Setear parametros
    //    p.problems.push_back(Problem::spiral);          //Con que problema trabajaremos
    param.problems.push_back(Problem::aggregation);          //Con que problema trabajaremos
    
    //    p.ebsilon = 0.0005;
    param.baseTemp = 0.05;
    param.tempDec = true;
    param.temperature = 0.003;
    
    //Propagacion de Q-value
    param.gamma = 0.0005;
    param.timeConstant = 300;
    param.sigma = 0.03;
    param.gamma2 = 0.005;
    param.alcanceActions = 0;
    param.alcanceState = 3;
    
    param.politic = Politic::Softmax;
    
    param.numActions = 15;
    param.epochs = 1000;
    
    param.numFolds = 10;
    
    param.alphas={0.05};        //    p.alphas = {0.05, 0.1, 0.2};
    param.betas = {0.75};       //    p.betas = {0.5, 0.75, 0.9, 0.95};
    param.factorTemps = {0.99}; //    p.factorTemps = {0.5, 0.85, 0.9, 0.99, 0.995};
    param.structure = {2};
    param.ebsilons = {};
    param.initChangeParam();
    
    data.loadProblems(param);
    data.initKFold(param.numFolds);
    data.printDescription();
    
    param.structure.push_back(data.getTotalClasses());
    
    //Initialize network
    neuralNetwork.setNetwork(data, param);
    neuralNetwork.printDescription();
    
    contTableTrain.setTable(data.getClassesPerBatch());
    contTableTest.setTable(data.getClassesPerBatch());
    
    vector<double> input (data.getTotalDimentions(),0);           //Vector de entrada a la red
    int output = 0;              //Etiquetas del vector de entrada.
    
    //EXPORT//
    string folder = "Estadisticas/AccionesDiscretas/";
    errorManager.init(&param, folder);
    
    cout<<"Numero de combinaciones: "<<param.getnumCombi()<<endl;
    
    ////////////// ENTRENAMIENTO DE LA RED ////////////////////////
    
    //Cambiando de parametros
    for (int combi = 0; combi < param.getnumCombi(); combi++) {
        cout<<"///// Generando combinacion: "<<combi<<"/////"<<endl;
        //Folds
        for (int fold = 0; fold < param.numFolds; fold++) {
            cout<<"F: "<<fold<<"  "<<endl;
            //Epocas
            for (int epoch = 0; epoch < param.epochs; epoch++) {
                param.decTemp(epoch);
                
                //Train
                for (int j = 0; j < data.getTrainSize(); j++) {
                    data.getTrain(input, output);
                    neuralNetwork.train(input, output);
                    contTableTrain.set(data.getActualBatch(), output, neuralNetwork.outputNet);
                }
                errorManager.setErrorTrain(contTableTrain, epoch, fold);
                data.shuffleDataTrain();
                
                //Test
                for (int j = 0; j < data.getTestSize(); j++) {
                    data.getTest(input, output);
                    neuralNetwork.test(input, output);
                    contTableTest.set(data.getActualBatch(), output, neuralNetwork.outputNet);
                }
                errorManager.setErrorTest(contTableTest, epoch, fold);
                errorManager.exportFold(fold);
            }
            neuralNetwork.printStatesNotVisited(data);
            cout<<"Train: "<<errorManager.getTrainErrorFold(fold)<<"   Test: "<<errorManager.getTestErrorFold(fold)<<endl;
            data.nextFold();
            if (fold == 0) {
                neuralNetwork.exportToPlot(folder);
            }
            
            neuralNetwork.forget();
            cout<<"Promedio del train de los ultimas epocas: "<<errorManager.getTrainLastsErrors(param.epochs, fold, 100)<<endl;
            cout<<"Promedio del test de los ultimas epocas: "<<errorManager.getTestLastsErrors(param.epochs, fold, 100)<<endl<<endl;
        }
        data.resetFold();
        cout<<endl;
        errorManager.exportCombi(combi);
        errorManager.exportLeyend();
        cout<<"Error promedio en el train: "<<errorManager.getTrainErrorCombi(combi)<<endl;
        cout<<"Error promedio en el test: "<<errorManager.getTestErrorCombi(combi)<<endl<<endl;
        param.nextParam();
        
    }
    
    //Resetear para imprimir los parametros
    param.resetParam();
    
    //Export resumen
    exportParams(param, folder);

}

/*
 * Crea un archivo Parametros.txt
 * Exporta todos los parametros usados en la red neuronal
 */
void exportParams(Params& p, string folder){
    ofstream notes;
    notes.open(folder + "Parametros.txt");
    notes << "Pruebas con las siguientes bases de datos: ";
    for (auto itr = p.problems.begin(); itr != p.problems.end(); itr++) {
        switch (*itr) {
            case Problem::spiral:
                notes << "loadSpiral"<<endl;
                break;
                
            case Problem::iris:
                notes << "loadIris" <<endl;
                break;
                
            case Problem::wine:
                notes << "loadWine" <<endl;
                break;
            
            case Problem::flame:
                notes << "Flame" <<endl;
                break;
                
            case Problem::jain:
                notes << "Jain" <<endl;
                break;
                
            case Problem::pathbased:
                notes << "Pathbased" <<endl;
                break;
                
            case Problem::compound:
                notes << "Compound" <<endl;
                break;
                
            case Problem::aggregation:
                notes << "Aggregation" <<endl;
                break;
            
            default:
                notes << "Problema aun no definido"<<endl;
                break;
        }
    }
    
    notes << "Epocas: " << p.epochs << endl;
    notes << "Numero de acciones: " << p.numActions <<endl;
    notes << "Numero de folds: " << p.numFolds << endl;
    notes << "Radio de estados: " << p.alcanceState <<endl;
    notes << "Sigma: " << p.sigma << endl;
    notes << "Alcance de acciones: " << p.alcanceActions << endl;
    notes << "Gamma de acciones: " << p.gamma2 << endl;
    
    notes << "Estructura: ";
    for(int i:p.structure){
        notes << i <<"\t";
    }
    notes<<endl;
    
    notes << "Alphas: ";
    for(double alpha: p.alphas){
        notes <<alpha<<"\t";
    }
    notes<<endl;
    
    notes <<"Betas: ";
    for(double beta: p.betas){
        notes <<beta<<"\t";
    }
    notes<<endl;
    
    notes << "Factor de temperaturas: ";
    for(double factor: p.factorTemps){
        notes <<factor<<"\t";
    }
    notes<<endl;
    notes << "Temperatura base: " << p.baseTemp <<endl;
    
}
