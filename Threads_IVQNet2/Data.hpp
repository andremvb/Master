//
//  Data.hpp
//  Red Neuronal IVQL
//
//  Created by Andre Valdivia on 16/12/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

#ifndef Data_hpp
#define Data_hpp

#include <vector>
#include "Param.hpp"

enum class positionClass{first,last};

using namespace std;

//Estructura de una muestra
struct oneData{
public:
    int output;
    vector<double> input;
    int fold = -1;
    oneData(vector<double>,int);
};

/*
 *  Clase para cargar la data en la RAM
 *
 */
class Data {
private:
    struct Batch{
        vector<int> test;
        vector<int> train;
    };

    vector<int> classes;                                        //Vector del numero de clases por cada data set
    vector<int> dimentions;                                     //Vector del numero de dimensiones por cada data set
    int totalDimentions = 0;                                    //Numero total de dimensiones
    int totalClasses = 0;                                       //Numero total de clases

    int k = 0;                                                  //Numero de folds
    int actualFold = 0;                                         //Fold actual
    
    //Indices para recorrerer el vector de indices
    vector<int>::iterator itTrain;
    vector<int>::iterator itTest;
    int actualBatch = 0;

    vector<vector<oneData>> data;                               //1-Batches.2-
    vector<Batch> batches;                                      //Donde se enceuntran los indices de los indices

public:
    Data(){};

//    vector<int> dataIndexTrain;                                 //
//    vector<int> dataIndexTest;                                  //
    void insertOneData(vector<double> in,int out);              //Insertar una muestra al dataset del ultimo problema creado
    vector<string> tokenize(string word, char separator);
    void loadProblems(Params& p);
    void normalize();                                           //Normalizar la data
    void getTrain(vector<double> &input, int &output);   //Obtener vector de input y output de train
    void getTest(vector<double> &input, int &output);   //Obtener vector de input y output de test
    
    
    void shuffleDataTrain();                                    //Mezclar el train
    void shuffleActualBatch();
    void resetTestPointer();
    void resetTrainPointer();

    void openDataSet(string name, positionClass pos, char separator);
    
    //Estratificar
    void initKFold(int k = 10);
    void nextFold();
    void resetFold();
    void initBatch(int k);
    void nextBatch();
    
    //Conjunto de datos
    void XOR();
    void Ex0();
    void Ex1(int numData,int numberClasses);
    void Seno(int numData,int numberClasses);
    void loadSpiral();
    void loadJain();
    void loadFlame();
    void loadPathbased();
    void loadCompound();
    void loadAggregation();
    void loadIris();
    void loadGlass();
    void loadWine();
    void loadCara();
    
    
    //Otros
    void printData();
    void printDescription();

    int getTrainSize();
    int getTestSize();
    int getTotalClasses();
    int getTotalDimentions();

    int getActualBatch();

    vector<int> getClassesPerBatch();
};


#endif /* Data_hpp */

