//
//  svm.hpp
//  StatLearning
//
//  Created by 一折 on 2019/7/30.
//  Copyright © 2019 一折. All rights reserved.
//

#ifndef svm_hpp
#define svm_hpp

#include <stdio.h>
#include <vector>
#include <string>
using namespace std;


class SVM {
public:
    
    // construct function
    SVM(float e, float c, string kernel_fn);
    
    // properties
    vector<float> alpha;
    vector<float> w;
    float b;
    
    // functions
    void fit(vector<vector<float>> data, vector<float> label);
    float predict(vector<float> x);
    vector<float> predict(vector<vector<float>> x);

private:
    
    // properties
    float m_eps;
    float m_c;
    string m_kernel_fn;
    vector<vector<float>> m_data;
    vector<float> m_label;
    
    //kernel function
    float linear(vector<float> x, vector<float> z);
    float rbf(vector<float> x, vector<float> z);
    float kernel(vector<float> x, vector<float> z);
    
    // functions
    float predict(vector<vector<float>> gram, int k);  // predict for data in dataset.
    vector<vector<float>> cal_gram();
    
};

#endif /* svm_hpp */
