//
//  State.hpp
//  Version Intermedia
//
//  Created by Andre Valdivia on 5/07/17.
//  Copyright © 2017 Andre Valdivia. All rights reserved.
//

#ifndef State_hpp
#define State_hpp

#include <stdio.h>
#include <vector>
#include <math.h>
#include "Helper.hpp"
#include "Param.hpp"

using namespace std;

class State{
public:
    
    long long unsigned int name;
    Params* p;
    bool used;
    int layer;
    int agent;
    
    vector<double> actionVal;
    vector<double> actionProb;
    vector<double> centroide;
    vector<State*> neight;
    
    void setState(int name, int layer, int agent, int numActions,int numDimentions,vector<double> centro,Params& p);
    void forget();
    
    double distance(State& s);
    int selectAction();
    int selectOptimAction();
    int selectOptimActionToPrint();
    
    //Changes action probabilities vector
    void softMax();
    void e_greedy();
    void greedy();
    void (State::*changeProb)();
    
    double updateQ(double reward, double IV, int actionTaked);
    double updateQNeight(double RV, int actionTaked);
    
    string getCentroid();
    
    void getMaxMinQ(double& max, double& min);
    bool printIsUsed();
    
};



#endif /* State_hpp */
