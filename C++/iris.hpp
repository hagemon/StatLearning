//
//  iris.hpp
//  StatLearning
//
//  Created by 一折 on 2019/8/11.
//  Copyright © 2019 一折. All rights reserved.
//

#ifndef iris_hpp
#define iris_hpp

#include <stdio.h>
#include <vector>
using namespace std;

class Iris {
public:
    Iris();
    vector<vector<float>> train_x;
    vector<vector<float>> test_x;
    vector<float> train_y;
    vector<float> test_y;
};

#endif /* iris_hpp */
