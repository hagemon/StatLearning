//
//  svm.cpp
//  StatLearning
//
//  Created by 一折 on 2019/7/30.
//  Copyright © 2019 一折. All rights reserved.
//

#include "svm.hpp"
#include <iostream>
#include "math.h"

SVM::SVM(float e, float c, string kernel_fn) {
    m_eps = e;
    m_c = c;
    m_kernel_fn = kernel_fn;
    alpha = {0};
    w = {0};
    b = 0;
}

float SVM::linear(vector<float> x, vector<float> z){
    assert(x.size() == z.size());
    float result = 0.0;
    for (int i=0; i<x.size(); i++) {
        result += x[i]*z[i];
    }
    return result;
}

float SVM::rbf(vector<float> x, vector<float> z){
    assert(x.size() == z.size());
    float result = 0.0;
    for (int i=0; i<x.size(); i++) {
        result += pow(x[i]-z[i], 2);
    }
    result = exp(-result)/2.0;
    return result;
}

float SVM::kernel(vector<float> x, vector<float> z) {
    if (m_kernel_fn == "linear") {
        return linear(x, z);
    }
    else if (m_kernel_fn == "rbf") {
        return rbf(x, z);
    }else{
        throw "No such kernel function";
    }
}

void SVM::fit(vector<vector<float>> data, vector<float> label) {
    vector<float> zeros(data.size());
    alpha.assign(zeros.begin(), zeros.end());
    b = 0;
    m_data.assign(data.begin(), data.end());
    m_label.assign(label.begin(), label.end());
    vector<vector<float>> gram = cal_gram();
    // select alpha.
    
    for (int i=0; i<data.size(); i++) {
        float gi = predict(gram, i);
        float ei = gi - label[i];
        float r = m_label[i] * ei;
        if ((r < -m_eps && alpha[i] < m_c) || (r > m_eps && alpha[i] > 0)) {
            // against kkt condition.
            for (int j=0; j < data.size(); j++){
                if (i == j) continue;
                
                float eta = gram[i][i] + gram[j][j] - 2 * gram[i][j];
                if (eta <= 0) continue;
                
                float gj = predict(gram, j);
                float ej = gj - label[j];
                float delta = label[j]*(ei-ej)/eta;
                if (delta < m_eps) continue;
                
                float aj_new = alpha[j] + delta;
                float l_bound, h_bound = 0.0;
                if (label[i] != label[j]) {
                    l_bound = max<float>(0, alpha[j]-alpha[i]);
                    h_bound = min<float>(m_c, m_c+alpha[j]-alpha[i]);
                } else {
                    l_bound = max<float>(0, alpha[i]+alpha[j]-m_c);
                    h_bound = min<float>(m_c, alpha[i]+alpha[j]);
                }
                
                
                if (aj_new < l_bound) aj_new = l_bound;
                if (aj_new > h_bound) aj_new = h_bound;
                
                float ai_new = alpha[i] + label[j]*label[i]*(alpha[j]-aj_new);
                alpha[i] = ai_new;
                alpha[j] = aj_new;
                
                float bi_new = -ei - label[i]*gram[i][i]*(ai_new-alpha[i]) - label[j]*gram[j][i]*(aj_new-alpha[j]) + b;
                float bj_new = -ei - label[j]*gram[i][j]*(ai_new-alpha[i]) - label[j]*gram[j][j]*(aj_new-alpha[j]) + b;
                
                if (ai_new > 0 && ai_new < m_c) b = bi_new;
                else if (aj_new > 0 && aj_new < m_c) b = bj_new;
                else b = (bi_new+bj_new) / 2;
                
            }
        }
    }
}

float SVM::predict(vector<float> x) {
    float result = 0.0;
    for (int i=0; i<alpha.size(); i++) {
        result += alpha[i] * m_label[i] * kernel(x, m_data[i]);
    }
    result += b;
    if (result > 0)
        result = 1;
    else
        result = -1;
    return result;
}

vector<float> SVM::predict(vector<vector<float>> x) {
    vector<float> result;
    for (int i=0; i<x.size(); i++) {
        result.push_back(predict(x[i]));
    }
    return result;
}

float SVM::predict(vector<vector<float>> gram, int k) {
    float result = 0.0;
    for (int i=0; i<alpha.size(); i++) {
        result += alpha[i] * m_label[i] * gram[i][k];
    }
    result += b;
    return result;
}


vector<vector<float>> SVM::cal_gram() {
    vector<vector<float>> gram(m_data.size(), vector<float>(m_data.size()));
    for (int i=0; i<m_data.size(); i++) {
        for (int j=i; j<m_data.size(); j++) {
            float k = kernel(m_data[i], m_data[j]);
            gram[i][j] = k;
            gram[j][i] = k;
        }
    }
    return gram;
}
