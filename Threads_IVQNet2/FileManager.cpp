//
//  FileManager.cpp
//  Red Neuronal IVQL
//
//  Created by Andre Valdivia on 3/01/17.
//  Copyright © 2017 Andre Valdivia. All rights reserved.
//

#include "FileManager.hpp"

//void FileManager::printToTxt(ofstream file, double beta, double alpha, double temperature, vector<double> errores){
//    
//    file<< "Beta: "<<beta<<" Alpha: "<<alpha<<" Temperature: "<<temperature<<endl;
//    for (int i = 0; i < errores.size() ; i++) {
//        file<< errores[i]<<";";
//    }
//    file<<endl;
//    for (int i = 0; i < errores.size() ; i++) {
//        file<< i <<";";
//    }
//    file<<endl<<endl;
//    
//}



//PRUEBAAAAAAAAAA
void FileManager::printToTxt2(ofstream file, double beta, double alpha, double temperature, vector<vector<double>> errores){
    
    file<< "Beta: "<<beta<<" Alpha: "<<alpha<<" Temperature: "<<temperature<<endl;
    for (int i = 0; i < errores.size() ; i++) {
        file<<"Fold: "<<i<<endl;
        for (int j = 0; j < errores[i].size(); j++) {
            file<< errores[i][j]<<";";
        }
        file<<endl;
    }
    file<<endl<<endl;
    
}
