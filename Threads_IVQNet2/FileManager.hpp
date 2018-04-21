//
//  FileManager.hpp
//  Red Neuronal IVQL
//
//  Created by Andre Valdivia on 3/01/17.
//  Copyright © 2017 Andre Valdivia. All rights reserved.
//

#ifndef FileManager_hpp
#define FileManager_hpp

#include <stdio.h>
#include <fstream>
#include <vector>
using namespace std;
class FileManager {
    
public:
    static void printToTxt2(ofstream file, double beta, double alpha, double temperature, vector<vector<double>> errores);
    
};

#endif /* FileManager_hpp */
