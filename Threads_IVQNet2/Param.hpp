//
//  Param.hpp
//  Acciones_Discretas
//
//  Created by Andre Valdivia on 26/09/17.
//  Copyright © 2017 Andre Valdivia. All rights reserved.
//

#ifndef Param_hpp
#define Param_hpp

#include <stdio.h>
#include <vector>
#include "Helper.hpp"

using namespace std;

struct Params{
private:
    vector<int> indexParam;
    vector<double> topes;
    int topeIterador;

public:
    vector<Problem> problems;
    double alpha = 0;           //Coeficiente de aprendizaje
    double beta = 0;            //Coeficiente de comunicacion entre agentes
    double sigma = 0;           //Coeficiente de propagacion de estados
    double gamma = 0;           ///Primera version de estados vecinos
    double gamma2 = 0;           //Coeficiente de propagracion de acciones
    double temperature = 0;     //
    int alcanceState = 0;            //Numero de estados a los que influencia el estado
    int alcanceActions = 0;      //Numero de acciones que influencia al actualizar el Q
    double baseTemp = 0;        //
    double factorTemp = 0;
    double timeConstant = 0;    //Para el decaimiento del sigma en SOM
    bool tempDec = true;
    
    double ebsilon = 0;
    
    vector<int> structure = {0,0};
    
    int numActions = 0;
    int epochs = 0;
    
    int numFolds = 0;
    
    Politic politic;
    
    vector<double> alphas;
    vector<double> betas;
    vector<double> factorTemps;
    vector<double> ebsilons;

    void nextParam();
    void initChangeParam();
    void resetParam();
    void decTemp(int iteration);
    int getnumCombi(); //Devuelve el numero de combinaciones de parametros
    int getnumBatchs(); //Devuelve el numero de batches
    void addProblem(Problem problem);

    //Empieza el test en una combinacion de parametro comb
    void startOnParam(int comb);
};


#endif /* Param_hpp */
