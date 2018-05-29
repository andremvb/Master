//
//  ErrorManager.cpp
//  Threads_IVQNet2
//
//  Created by Andre Valdivia on 24/01/18.
//  Copyright © 2018 Andre Valdivia. All rights reserved.
//

#include "ErrorManager.hpp"
#include <iostream>
#include <assert.h>
void ErrorManager::init(Params* param, string folder){
    this->param = param;
    this->folder = folder;
    this->errorEpochTrain = vector<vector<double>>(param->numFolds,vector<double>(param->epochs, 0));       //Errores por epoca
    this->errorPromCombiTrain = vector<double>(param->getnumCombi(), -1);          //Promedio de tests en folds por combinacion
    
    //Test
    this->errorEpochTest = vector<vector<double>>(param->numFolds, vector<double> (param->epochs, 0));    //Errores por epoca
    this->errorPromCombiTest = vector<double>(param->getnumCombi(), -1);          //Promedio de tests en folds por combinacion
    
    leyend.open(folder + "Leyend.txt");
}

void ErrorManager::initBatch(Params *param, string folder) {
    this->param = param;
    this->folder = folder;

    //1-Fold. 2-Epoch
    this->errorEpochTrain = vector<vector<double>>(1,vector<double>(param->epochs, 0));       //Errores por epoca
    this->errorEpochTest = vector<vector<double>>(1, vector<double> (param->epochs, 0));    //Errores por epoca
    this->errorEpochTestBase = vector<vector<double>>(1, vector<double> (param->epochs, 0));    //Errores por epoca

    this->errorBase = vector<double> (param->getnumBatchs());
    this->errorNewTest = vector<double> (param->getnumBatchs());
    this->errorNewTrain = vector<double> (param->getnumBatchs());

    leyend.open(folder + "Leyend.txt");
}

void ErrorManager::setErrorTrain(ContingenceTable &contTableTrain, int epoch, int fold, int batch) {
    errorEpochTrain[fold][epoch] = contTableTrain.getTotalError(batch);
    contTableTrain.forget();
}

void ErrorManager::setErrorTest(ContingenceTable &contTableTest, int epoch, int fold, int batch) {
    errorEpochTest[fold][epoch] = contTableTest.getTotalError(batch);
    contTableTest.forget();
}

void ErrorManager::setErrorTestBase(ContingenceTable &contTableTestBase, int epoch, int fold, int batch) {
    errorEpochTestBase[fold][epoch] = contTableTestBase.getTotalError(batch);
    contTableTestBase.forget();
}

void ErrorManager::exportCombi(int combi){
    ofstream file;
    file.open(folder + "Combi" + to_string(combi) + ".csv");
    if (file.bad()) {
        cout<<"Error creando el archivo"<<endl;
    }
    //Train
    vector<double> errorPromEpochTrain(param->epochs, 0);                 //Error promedio de cada epoca
    vector<double> errorDesvEpochTrain(param->epochs, 0);                 //Desviacion del promedio de cada epoca
    
    //Test
    vector<double> errorPromEpochTest(param->epochs, 0);                 //Error promedio de cada epoca
    vector<double> errorDesvEpochTest(param->epochs, 0);                 //Desviacion del promedio de cada epoca
    
    for (int epoch = 0; epoch < param->epochs; epoch++) {//
        //Calcular el promedio de train y test
        for (int test = 0; test < param->numFolds; test++) {
            errorPromEpochTrain[epoch] += errorEpochTrain[test][epoch];
            errorPromEpochTest[epoch] += errorEpochTest[test][epoch];
        }
        errorPromEpochTrain[epoch] = errorPromEpochTrain[epoch]/param->numFolds;
        errorPromEpochTest[epoch] = errorPromEpochTest[epoch]/param->numFolds;
        
        //Calcular la desviación
        for (int test = 0; test < param->numFolds; test++) {
            errorDesvEpochTrain[epoch] += pow(errorPromEpochTrain[epoch] - errorEpochTrain[test][epoch], 2);
            errorDesvEpochTest[epoch] += pow(errorPromEpochTest[epoch] - errorEpochTest[test][epoch], 2);
        }
        errorDesvEpochTrain[epoch] = pow(errorDesvEpochTrain[epoch]/(double)param->numFolds, 0.5);
        errorDesvEpochTest[epoch] = pow(errorDesvEpochTest[epoch]/(double)param->numFolds, 0.5);
        
    }
    
    
    for (int i = 0;i < param->epochs; i++) {
        param->decTemp(i);
        file<<i<<",";
        if (param->politic == Politic::Softmax) {
            file<<param->temperature<<",";
        }else if (param->politic == Politic::e_greedy){
            file<<param->ebsilon<<",";
        }
        file<<errorPromEpochTrain[i]<<",";
        file<<errorDesvEpochTrain[i]<<",";
        file<<errorPromEpochTest[i]<<",";
        file<<errorDesvEpochTest[i];
        file<<endl;
    }
    errorPromCombiTrain[combi] = errorPromEpochTrain.back();
    errorPromCombiTest[combi] = errorPromEpochTest.back();
    file.close();
    file.clear();
}

//Crea el archivo Fold_.txt
void ErrorManager::exportFold(int fold){
    ofstream file;
    file.open(folder + "Fold" + to_string(fold) + ".csv");
    if (file.bad()) {
        cout<<"Error creando el archivo"<<endl;
    }
    
    for (int i = 0; i < param->epochs; i++) {
        param->decTemp(i);
        file<<i<<",";
        file<<param->temperature<<",";
        file<<this->errorEpochTrain[fold][i]<<",";
        file<<this->errorEpochTest[fold][i]<<endl;
    }
    file.close();
    file.clear();
}

double ErrorManager::getTrainErrorCombi(int combi){
    assert(this->errorPromCombiTrain[combi] != -1);
    return this->errorPromCombiTrain[combi];
}

double ErrorManager::getTestErrorCombi(int combi){
    assert(this->errorPromCombiTest[combi] != -1);
    return this->errorPromCombiTest[combi];
}

double ErrorManager::getTrainErrorFold(int fold){
    return this->errorEpochTrain[fold].back();
}

double ErrorManager::getTestBaseErrorFold(int fold) {
    return this->errorEpochTestBase[fold].back();
}

    double ErrorManager::getTestErrorFold(int fold){
    return this->errorEpochTest[fold].back();
}

double ErrorManager::getTrainLastsErrors(int numEpochs, int fold, int numProm){
    double prom = 0;
    for (int i = numEpochs-numProm; i < numEpochs; i++) {
        prom += this->errorEpochTrain[fold][i];
    }
    return prom/numProm;
}

double ErrorManager::getTestLastsErrors(int numEpochs, int fold, int numProm){
    double prom = 0;
    for (int i = numEpochs-numProm; i < numEpochs; i++) {
        prom += this->errorEpochTest[fold][i];
    }
    return prom/numProm;
}

void ErrorManager::exportLeyend(){
//    leyend<<"Problema=iris"<<"\t";
    leyend<<"Problema="<<param->problems[0].name<<"\t";
    leyend<<"Numero de acciones"<<"="<<param->numActions<<"\t";
    leyend<<"Estructura"<<"="<<param->structure[0]<<"\t";
    leyend<<PAlpha<<"="<<param->alpha<<"\t";
    leyend<<PBeta<<"="<<param->beta<<"\t";
    if (param->politic == Politic::Softmax) {
        leyend<<"Temp base"<<"="<<param->baseTemp<<"\t";
        leyend<<"Temp factor"<<"="<<param->factorTemp<<"\t";
    }else if (param->politic == Politic::e_greedy){
        leyend<<PEpsilon<<"="<<param->ebsilon<<"\t";
        //        file<<"\t";
    }
    leyend<<"Prop. estado:"<<param->alcanceState<<"\t";
    leyend<<PSigma<<"="<<param->sigma<<"\t";
    leyend<<"Prop. acciones:"<<param->alcanceActions<<"\t";
    leyend<<"Gamma acciones:"<<param->gamma2<<"\t";
    leyend<<endl;
}

//Esta funcion no soporta k-folds. Solo se usara el fold 0
void ErrorManager::exportBatch(int batch) {
    ofstream file;
    file.open(folder + "Batch" + to_string(batch) + ".csv");
    if (file.bad()) {
        cout<<"Error creando el archivo export batch"<<endl;
    }

    file<<"Epoch,,Train,Test,Base"<<endl;
    for (int i = 0;i < param->epochs; i++) {
        param->decTemp(i);
        file<<i<<",";
        if (param->politic == Politic::Softmax) {
            file<<param->temperature<<",";
        }else if (param->politic == Politic::e_greedy){
            file<<param->ebsilon<<",";
        }
        file<<errorEpochTrain[0][i]<<",";
        file<<errorEpochTest[0][i]<<",";
        file<<errorEpochTestBase[0][i];
        file<<endl;
    }

    errorNewTrain[batch] =  errorEpochTrain[0].back();
    errorNewTest[batch] = errorEpochTest[0].back();
    errorBase[batch] = errorEpochTestBase[0].back();
    file.close();
    file.clear();
}



void exportSummary(vector<double> minErrors, vector<double> lastErrors, vector<double> testErrors, int combinaciones,string folder){
    ofstream resumen;
    resumen.open(folder + "resumen.csv");
    resumen<<"Errores Menores que 10:"<<endl;
    resumen<<"Errores minimos\t"<<"Errores ultimos\t"<<"Errores Test\t"<<endl;
    for (int i = 0; i < combinaciones; i++) {
        if (minErrors[i] < 10) {
            resumen<<i<<"\t";
            resumen<<minErrors[i]<<"\t";
            resumen<< lastErrors[i]<<"\t";
            resumen<<testErrors[i]<<"\t";
            resumen<<endl;
        }
    }
    resumen.close();
    resumen.clear();
}
