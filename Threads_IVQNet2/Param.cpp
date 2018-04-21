//
//  Param.cpp
//  Acciones_Discretas
//
//  Created by Andre Valdivia on 26/09/17.
//  Copyright © 2017 Andre Valdivia. All rights reserved.
//

#include "Param.hpp"

/*
Para aumentar un parametro:
- Crear un vector de posibles parametros en Param.hpp
- Aumentar en la seccion Index
- Aumentar en la seccion Topes
- Aumentar en la seccion Param
    * El orden en index y topes debe ser el mismo
    * El primer parametro que cambia, es el ultimo parametro.
*/

void Params::nextParam(){
    
    for (int i = (int)indexParam.size()-1; i >= 0; i--) {
        if (indexParam[i] < topes[i]) {
            indexParam[i]++;
            break;
        }else{
            for (int j = i; j < indexParam.size(); j++) {
                indexParam[j] = 0;
            }
        }
    }
    //Sec: Index
    alpha = alphas[indexParam[0]];
    beta = betas[indexParam[1]];
    factorTemp = factorTemps[indexParam[2]];
}

void Params::initChangeParam(){
    //Sec: Topes
    topes.push_back(alphas.size()-1);
    topes.push_back(betas.size()-1);
    topes.push_back(factorTemps.size()-1);
    
    this->topeIterador = 1;
    for (int i = 0; i < topes.size(); i++) {
        topeIterador *= (topes[i] +1);
    }
    
    //Sec: Param
    alpha = alphas[0];
    beta = betas[0];
    factorTemp = factorTemps[0];
    
    indexParam = vector<int>(topes.size(), 0);   
}

void Params::resetParam(){
    for (int i = 0; i < indexParam.size(); i++) {
        indexParam[i] = 0;
    }
}

void Params::decTemp(int iteration){
    if (tempDec) {
        temperature = baseTemp * pow(factorTemp, iteration);
    }
//    sigma = sigma * exp(- iteration/timeConstant);
}

int Params::getnumCombi(){
    return topeIterador;
}
void Params::startOnParam(int comb){
    for (int i = 0; i < comb; i++) {
        nextParam();
    }
}


