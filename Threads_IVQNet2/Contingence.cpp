//
//  Contingence.cpp
//  MLP_variosProblemas
//
//  Created by Andre Valdivia on 12/03/17.
//  Copyright © 2017 Andre Valdivia. All rights reserved.
//

#include "Contingence.hpp"

#include <iostream>

void ContingenceTable::forget(){
    for (int i = 0; i < cT.size(); i++) {
        for (int j = 0; j < cT[i].size(); j++) {
            for (int k = 0; k < cT[i][j].size(); k++) {
                cT[i][j][k] = 0;
            }
        }
    }
    totalError = 0;
}

void ContingenceTable::printTable(){
    for (int i = 0; i < cT.size(); i++) {
        cout<<"--------"<<endl;
        cout<<"Para el problema "<< i <<" :"<<endl;
        //Imprimir top
        cout<<"\t";
        for (int j = 0; j < cT[i].size(); j++) {
            cout<<j<<"\t";
        }
        cout<<endl;
        for (int j = 0; j < cT[i].size(); j++) {
            cout<<j<<"\t";
            for (int k = 0; k < cT[i][j].size(); k++) {
                cout<<cT[i][j][k]<<"\t";
            }
            cout<<endl;
        }
    }
}

void ContingenceTable::init(vector<int> classes){
    cT.clear();
    cT.resize(classes.size());
    for (int i = 0; i < classes.size(); i++) {
        cT[i].resize(classes[i]);
        for (int j = 0; j < classes[i]; j++) {
            cT[i][j].resize(classes[i]);
        }
    }
}

void ContingenceTable::set(int batch,int real, int out){
    cT[batch][real][out]++;
}

//Devuelve el error total
double ContingenceTable::getTotalError(int batch) {
    totalError = 0;
    double aciertos = 0;
    double desaciertos = 0;
    double error = 0;
        for (int j = 0; j < cT[batch].size(); j++) {
            for (int k = 0; k < cT[batch][j].size(); k++) {
                if (j == k) {
                    aciertos += cT[batch][j][k];
                }else{
                    desaciertos += cT[batch][j][k];
                }
            }
        }

    error = (desaciertos * 100) / ( aciertos + desaciertos);
    return error;
}

void ContingenceTable::printTotalError(){
    vector<int> aciertos (cT.size(), 0);
    vector<int> desaciertos (cT.size(), 0);
    vector<double> errores (cT.size(), 0);
    for (int i = 0; i < cT.size(); i++) {
        for (int j = 0; j < cT[i].size(); j++) {
            for (int k = 0; k < cT[i][j].size(); k++) {
                if (j == k) {
                    aciertos[i] += cT[i][j][k];
                }else{
                    desaciertos[i] += cT[i][j][k];
                }
            }
        }
    }
    
    for (int i = 0; i < aciertos.size(); i++) {
        cout<<"------------"<<endl;
        cout<< "Aciertos en "<<i<< " : " << aciertos[i]<<endl;
        cout<< "Desaciertos en "<<i<< " : " << desaciertos[i]<<endl;
        errores[i] = ((double)desaciertos[i] * 100) / ( (double)aciertos[i] + (double)desaciertos[i]);
        cout<< "Error en "<<i<< " : " << errores[i]<<"%"<<endl;
    }
    for (int i = 0; i < errores.size(); i++) {
        totalError += errores[i];
    }
    totalError = totalError/errores.size();
    cout<<"------------"<<endl;
    cout<< "El error total es: "<< totalError<<"%"<<endl;
    
}
