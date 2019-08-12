//
//  cart.hpp
//  StatLearning
//
//  Created by 一折 on 2019/8/11.
//  Copyright © 2019 一折. All rights reserved.
//

#ifndef cart_hpp
#define cart_hpp

#include <stdio.h>
#include <iostream>
#include <vector>
#define mat vector<vector<float>>
#define vec vector<float>

using namespace std;


struct CartNode {
    CartNode(float val, int dim, float loss, CartNode* left, CartNode* right, bool leaf);
    float m_val;
    int m_dim;
    CartNode* m_left;
    CartNode* m_right;
    float m_loss;
    float sub_tree_loss;
    int m_complex;
    bool is_leaf;
    
    void set_sub_loss(float loss);
    void set_complex(int complex);
};

//class CartNode {
//public:
//    CartNode(float val, int dim, float seg, float loss, CartNode* left, CartNode* right, bool leaf);
//    float m_val;
//    int m_dim;
//    float m_seg;
//    CartNode* m_left;
//    CartNode* m_right;
//    float m_loss;
//    float sub_tree_loss;
//    int m_complex;
//    bool is_leaf;
//
//    void set_sub_loss(float loss);
//    void set_complex(int complex);
//};

class Cart {
public:
    Cart(int n_min);
    int m_n_min;  // number of minimum data in a single node
    CartNode* root;
    int fold;
    bool m_cls;  // predicted value continuously or discrete
    
    void fit(mat data, vec label);
    vec predict(mat data);
    float pred(vec data, CartNode* node);
    
    // virtual functions
    virtual CartNode* build_tree(mat data, vec label) = 0;
    virtual CartNode* build_leaf(vec label) = 0;
    virtual void select_attr(mat data, vec label, float &loss, int &attr, float &seg) = 0;
};

class RegTree: public Cart {
public:
    RegTree(int n_min, bool cls);
    
    CartNode* build_tree(mat data, vec label);
    CartNode* build_leaf(vec label);
    void select_attr(mat data, vec label, float &loss, int &attr, float &seg);
};

class ClaTree: public Cart {
public:
    ClaTree(int n_min);
    
    CartNode* build_tree(mat data, vec label);
    CartNode* build_leaf(vec label);
    void select_attr(mat data, vec label, float &loss, int &attr, float &seg);
};


#endif /* cart_hpp */
