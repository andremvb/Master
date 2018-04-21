//
//  Data.cpp
//  Red Neuronal IVQL
//
//  Created by Andre Valdivia on 16/12/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

#include "Data.hpp"
#include <iostream>
#include "Helper.hpp"
#include <fstream>
#include <string>
#include <sstream>

oneData::oneData(vector<double> input,int output){
    this->output = output;
    this->input = input;
}

vector<string> Data::tokenize(string word, char separator){
    vector<string> vec;
    std::istringstream iss(word);
    std::string token;
    while (std::getline(iss, token, separator))
    {
        vec.push_back(token);
    }
    return vec;
}

void Data::printDescription(){
    vector<vector<int>> dataPerClass;
    dataPerClass.resize(data.size());
    
    cout<<"----------------------------"<<endl;
    cout<< "Clases:" <<endl;
    for (int i = 0; i < dataPerClass.size(); i++) {
        dataPerClass[i].resize(classes[i]);
    }
    
    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[i].size(); j++) {
            dataPerClass[i][data[i][j].output]++;
        }
    }
    
    for (int i = 0; i < dataPerClass.size(); i++) {
        cout<<"Data set "<<i<<": "<<endl;
        for (int j = 0; j < dataPerClass[i].size(); j++) {
            
            cout<< "Clase "<< j << " : "<<dataPerClass[i][j]<<endl;
        }

    }
    
    cout<<endl<< "Dimensiones:" <<endl;
    for (int i = 0 ; i < dimentions.size(); i++) {
        cout<<"Data set "<<i<<" : "<<dimentions[i]<<endl;
    }
    cout<<endl;
    
    cout<<"Descripcion de batchs: "<<k<<" folds"<<endl;
    for (int i = 0; i < batches.size(); ++i) {
        cout<<"Batch "<<i<<":"<<endl;
        cout<<"\t Train: "<<batches[i].train.size();
        cout<<"\t Test: "<<batches[i].test.size();
    }
    cout<<endl<<"----------------------------"<<endl;
}

//Imprime toda la data
void Data::printData(){
    for (int i = 0; i < data.size(); i++) {
        cout<<"++++++++++++"<<endl<<"DataSet "<<i<<": "<<endl<<"++++++++++++"<<endl;
        for (int j =0;  j < data[i].size(); j++) {
            cout<<"In: ";
            for (int k = 0; k < data[i][j].input.size(); k++) {
                cout<< data[i][j].input[k]<<"\t";
            }
            cout<< "|Out: "<<data[i][j].output<<endl;
        }
    }
}

void Data::shuffleDataTrain(){
    std::random_shuffle(batches[actualBatch].train.begin(), batches[actualBatch].train.end());
}

void Data::shuffleActualBatch(){
        std::random_shuffle(data[actualBatch].begin(), data[actualBatch].end());
}

void Data::insertOneData(vector<double> input,int output){
    data.back().push_back(oneData(input,output));
}

//Carga los problemas
void Data::loadProblems(Params &p){
    for(auto problem : p.problems){
        switch (problem) {
            case Problem::spiral:
                loadSpiral();
                break;
            case Problem::iris:
                loadIris();
                break;
                
            case Problem::wine:
                loadWine();
                break;
            
            case Problem::glass:
                loadGlass();
                break;
                
            case Problem::jain:
                loadJain();
                break;
                
            case Problem::flame:
                loadFlame();
                break;
                
            case Problem::pathbased:
                loadPathbased();
                break;
            
            case Problem::compound:
                loadCompound();
                break;
                
            case Problem::aggregation:
                loadAggregation();
                break;
            
            default:
                cout<< "No se definio problema"<<endl;
                break;
        }
    }
}

//Toda la adata entre 0 y 1
//Cada dataSet se normaliza independientemente
void Data::normalize(){
    //Para cada datSet
    for (int i = 0; i < data.size(); i++) {
        //Por cada dimension
        for (int j = 0; j < data[i][0].input.size(); j++) {
            double min = +1000000;
            double max = -1000000;
            for (int k = 0; k < data[i].size(); k++) {
                if (data[i][k].input[j] < min ) {
                    min = data[i][k].input[j];
                }
                if (data[i][k].input[j] > max) {
                    max = data[i][k].input[j];
                }
            }
            max = max - min;
                for (int k = 0; k < data[i].size(); k++) {
                    data[i][k].input[j] = (data[i][k].input[j] - min)/max;
                }
        }
    }
}

void Data::getTest(vector<double> &input, int &output){
    input = data[actualBatch][*itTest].input;
    output = data[actualBatch][*itTest].output;
    if (itTest != batches[actualBatch].test.end()-1) {
        itTest++;
    } else{
        itTest = batches[actualBatch].test.begin();
    }
}

void Data::getTrain(vector<double> &input, int &output) {
    input = data[actualBatch][*itTrain].input;
    output = data[actualBatch][*itTrain].output;
    if (itTrain != batches[actualBatch].train.end()-1) {
        itTrain++;
    } else{
        itTrain = batches[actualBatch].train.begin();
    }
}

void Data::initKFold(int k){
    this->k = k;
    batches.resize(data.size());
    for (int i = 0; i < data.size(); ++i) {
        if (k==1){
            batches[i].train.resize(data[i].size());
            batches[i].test.resize(data[i].size());
            for (int j = 0; j < data[i].size(); ++j) {
                batches[i].test[j] = j;
                batches[i].train[j] = j;
            }
        }else{
            batches[i].test.resize(data[i].size()/k);
            batches[i].train.resize(data[i].size()-(data[i].size()/k));
            int counter = 0;
            for (int &j : batches[i].test) {
                j = counter;
                data[i][j].first = true;
                counter++;
            }
            for(auto &j : batches[i].train){
                j = counter;
                counter++;
            }
        }
    }
    shuffleActualBatch();
    itTrain = batches[actualBatch].train.begin();
    itTest = batches[actualBatch].test.begin();
}

//No vuelve a hacer shuffle de la data
//Esto porque para probar diferentes combinaciones de parametros, se quiere que todos los folds usen la misma data
//Solamente vuelve a empezar los test desde el fold 0
void Data::resetFold(){
    actualFold = 0;
    batches[actualBatch].test.clear();
    batches[actualBatch].train.clear();
    for (int i = 0; i < data.size(); ++i) {
        batches[i].test.clear();
        batches[i].train.clear();
        if (k == 1){
            batches[i].train.resize(data[i].size());
            batches[i].test.resize(data[i].size());
            for (int j = 0; j < data[i].size(); ++j) {
                batches[i].test[j] = j;
                batches[i].train[j] = j;
            }
        }
        else{
            batches[i].test.resize(data[i].size()/k);
            batches[i].train.resize(data[i].size()-(data[i].size()/k));
            int counter = 0;
            for (int &j : batches[i].test) {
                j = counter;
                counter++;
            }
            for(auto &j : batches[i].train){
                j = counter;
                counter++;
            }
        }
    }
    itTrain = batches[actualBatch].train.begin();
    itTest = batches[actualBatch].test.begin();
}

void Data::nextFold(){
    actualFold++;
    if (actualFold == k) {
        actualFold = 0;
        cout<<"Se reinicio el fold en Data::nextFold"<<endl;
    }else{
        int range = data[actualBatch].size()/k;
        batches[actualBatch].test.clear();
        batches[actualBatch].train.clear();
        if (actualFold != k - 1) {
            int counter = 0;
            for (int j = 0; j < data[actualBatch].size(); ++j) {
                if (data[actualBatch][j].first == false && counter < range){
                    batches[actualBatch].test.push_back(j);
                    data[actualBatch][j].first = true;
                    counter++;
                }else{
                    batches[actualBatch].train.push_back(j);
                }
            }
        }else {
            for (int j = 0; j < data[actualBatch].size(); j++) {
                if (data[actualBatch][j].first) {
                    batches[actualBatch].train.push_back(j);
                }else{
                    batches[actualBatch].test.push_back(j);
                    data[actualBatch][j].first = true;
                }
            }
        }
    }
    itTrain = batches[actualBatch].train.begin();
    itTest = batches[actualBatch].test.begin();
}

void Data::XOR(){
    classes.push_back(2);
    insertOneData({0,0}, 0);
    insertOneData({0,1}, 1);
    insertOneData({1,0}, 1);
    insertOneData({1,1}, 0);
    shuffleActualBatch();
}

void Data::Seno(int numData,int numberClasses){
    data.resize(data.size()+1);
    classes.push_back(numberClasses);
    double max = M_PI * 2;
    double out = 0;
    for (int i = 0 ; i < numData; i++) {
        out = sin((double)i * max / numData);
        out = discretize(out, numberClasses, -1,1);
        insertOneData({static_cast<double>(i)}, out);
    }
}

void  Data::Ex0(){
    data.resize(data.size()+1);
    double out = 0;
    for (int i = 0; i<100; i++) {
        out = pow(i, 2);
        out = discretize(out, 10, 0,pow(100, 2));
        insertOneData({static_cast<double>(i)}, out);
    }
    shuffleActualBatch();
}

void Data::Ex1(int numData,int numberClasses){
    data.resize(data.size()+1);
    classes.push_back(numberClasses);
    double out = 0;
    int rangeMax = 100;
    int multiplicador = numData/rangeMax;
    for (int i = 0; i < rangeMax; i++) {
        // Normalizar: n - min/ max - min... -4 es el min
        //double min = -4;
        for (int j = 0; j < multiplicador; j++) {
            double tmp = ((double)i * (double)multiplicador + (double)j)/multiplicador;
            double in = tmp /(double)rangeMax;
            out = (2 * pow(tmp, 5) -  pow(tmp, 2) + 10 * tmp - 4 +4) / (2 * pow(100, 5) - pow(100, 2) + 10 * 100 - 4 +4);
            out = discretize(out, numberClasses, 0,1);

            insertOneData({static_cast<double>(in)}, out);
        }
    }
}

void Data::openDataSet(string name, positionClass pos, char separator){
    data.resize(data.size()+1);
    ifstream file;
    string line;
    file.open(name);
    if (file.is_open()) {
        while (!file.eof()) {
            file >> line;
            vector<string> tokens = tokenize(line, separator);
            vector<double> in;
            int out = -1;
            for (int i = 0; i < tokens.size(); i++) {
                if (pos == positionClass::first) {
                    if (i == 0) {
                        out = stof(tokens[i]) - 1;
                    }else{
                        in.push_back(stof(tokens[i]));
                    }
                }else{
                    if (i == tokens.size()-1) {
                        out = stof(tokens[i]) - 1;
                    }else{
                        in.push_back(stof(tokens[i]));
                    }
                }
            }
            insertOneData(in, out);
        }
    }else{
        cout<<"La base de datos no existe"<<endl;
    }
    
    file.close();
    normalize();
    if (k != 0) {
        initKFold(k);
    }
    dimentions.push_back((int)data[data.size()-1][0].second.input.size());
    //totalDimentions+=dimentions.back();
    //totalClasses += classes.back();
    
    //Con la nueva configuracion de olvido catastrofico
    if (totalDimentions < dimentions.back()) {
        totalDimentions = dimentions.back();
    }
    if (totalClasses < classes.back()) {
        totalClasses = classes.back();
    }
}

void Data::loadCara(){
    classes.push_back(8);
    openDataSet("cara.csv", positionClass::first,';');
}

void Data::loadWine(){
    classes.push_back(3);
    openDataSet("wine.csv", positionClass::first, ',');
}

void Data::loadIris(){
    classes.push_back(3);
    openDataSet("iris.txt", positionClass::last, ',');
}

void Data::loadSpiral(){
    classes.push_back(3);
    openDataSet("spiral.txt", positionClass::last, ',');
}

void Data::loadGlass(){
    classes.push_back(7);
    openDataSet("glass.txt", positionClass::last, ',');
}

void Data::loadJain(){
    classes.push_back(2);
    openDataSet("2Dim/jain.txt", positionClass::last, ',');
}

void Data::loadFlame(){
    classes.push_back(2);
    openDataSet("2Dim/flame.txt", positionClass::last, ',');
}

void Data::loadPathbased(){
    classes.push_back(3);
    openDataSet("2Dim/pathbased.txt", positionClass::last, ',');
}

void Data::loadCompound(){
    classes.push_back(6);
    openDataSet("2Dim/compound.txt", positionClass::last, ',');
}

void Data::loadAggregation(){
    classes.push_back(7);
    openDataSet("2Dim/aggregation.txt", positionClass::last, ',');
}

void Data::nextBatch() {
    if (actualBatch+1 < batches.size()){
        actualBatch++;
    }else{
        cout<<"Batches mayor a su rango"<<endl;
        actualBatch = 0;
    }
    itTrain = batches[actualBatch].train.begin();
    itTest = batches[actualBatch].test.begin();
    shuffleActualBatch();
}

int Data::getTrainSize() {
    return batches[actualBatch].train.size();
}

int Data::getTestSize() {
    return batches[actualBatch].test.size();
}

int Data::getTotalDimentions() {
    return totalDimentions;
}

int Data::getTotalClasses() {
    return totalClasses;
}

int Data::getActualBatch() {
    return actualBatch;
}

vector<int> Data::getClassesPerBatch() {
    return classes;
}

void Data::resetTestPointer() {
    itTest = batches[actualBatch].test.begin();
}

void Data::resetTrainPointer() {
    itTrain = batches[actualBatch].train.begin();
}
