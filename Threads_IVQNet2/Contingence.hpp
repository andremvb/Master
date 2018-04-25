//
//  Contingence.hpp
//  MLP_variosProblemas
//
//  Created by Andre Valdivia on 12/03/17.
//  Copyright © 2017 Andre Valdivia. All rights reserved.
//

#ifndef Contingence_hpp
#define Contingence_hpp
#include <vector>
using namespace std;

//  Clase para guardar la precision de el algoritmo.
//  Soporta varios problemas
//
//
//
class ContingenceTable {
    vector<vector<vector<int>>> cT;
    double totalError = 0;
public:
    
    
    
    
    //Necesario para crear la tabla de contingencia
    void init(vector<int> classes);
    
    //Inserta un valor a la tabla
    void set(int batch, int real, int out);
    
    //Devuelve el error total
    double getTotalError(int batch);
    
    //Imprime en consola el error total
    void printTotalError();
    
    //Imprime la tabla de contingencia
    void printTable();
    
    //Resetea a 0 todos los valores de la tabla de contingencia
    void forget();
};


#endif /* Contingence_hpp */
