//
//  ErrorManager.hpp
//  Threads_IVQNet2
//
//  Created by Andre Valdivia on 24/01/18.
//  Copyright © 2018 Andre Valdivia. All rights reserved.
//

#ifndef ErrorManager_hpp
#define ErrorManager_hpp

#include <stdio.h>
#include <vector>
#include "Param.hpp"
#include "Contingence.hpp"
#include <fstream>
using namespace std;


//Crea vector para almacenar los errores
class ErrorManager {
public:
    
    Params* param;
    string folder;
    ofstream leyend;
    
    //Train
    vector<vector<double>> errorEpochTrain;              //Errores por epoca    1.Fold, 2.Epoch
    vector<double> errorPromCombiTrain;                  //Errores de cada combinacion  1.PromCombi
    vector<double> errorNewTrain;                       //Error por batch 1.Batch-train

    //Test
    vector<vector<double>> errorEpochTest;              //Errores por epoca. 1.Fold, 2.Epoch
    vector<vector<double>> errorEpochTestBase;          //Errores por epoca. 1.Fold, 2.Epoch
    vector<double> errorPromCombiTest;                  //Promedio de tests en folds por combinacion    1.PromCombi
    vector<double> errorNewTest;                         //Error por batch en test   1.Batch-test
    vector<double> errorBase;                           //Error base por batch  1.Batch

    ~ErrorManager(){if(leyend.is_open()){leyend.close(); leyend.clear();}}
    
    /*
     * Inicializar ErrorManager
     */
    void init(Params* param, string folder);
    void initBatch(Params* param, string folder);
    
    /*
     * Setea el errores
     */
    void setErrorTrain(ContingenceTable &contTableTrain, int epoch, int fold, int batch);
    void setErrorTest(ContingenceTable &contTableTest, int epoch, int fold, int batch);
    
    /*
     * Exporta: temperatura - Error Promedio Train - Descivacion estadarTrain - Error Promedio Test - DesviacionestandarTest de todos los folds
     * En el archivo file (Que usualmente es Combi )
     */
    void exportCombi(int combi);
    void exportFold(int fold);
    void exportLeyend();
    void exportBatch(int batch);
    
    double getTrainErrorFold(int fold);
    double getTestErrorFold(int fold);
    
    /*
     * Usar cuando ya se exporto la combinacion
     */
    double getTrainErrorCombi(int combi);
    /*
     * Usar cuando ya se exporto la combinacion
     */
    double getTestErrorCombi(int combi);
    
    /*
     * Usar cuando ya se exporto la combinacion
     * Obtiene el promedio de los ultimos numEpochs errores (Train)
     */
    double getTrainLastsErrors(int numEpochs, int fold, int numProm);
    
    /*
     * Usar cuando ya se exporto la combinacion
     * Obtiene el promedio de los ultimos numEpochs errores (Test)
     */
    double getTestLastsErrors(int numEpochs, int fold, int numProm);

    double getTestBaseErrorFold(int fold);

    void setErrorTestBase(ContingenceTable &contTableTestBase, int epoch, int fold, int batch);
};

#endif /* ErrorManager_hpp */
