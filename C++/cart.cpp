//
//  cart.cpp
//  StatLearning
//
//  Created by 一折 on 2019/8/11.
//  Copyright © 2019 一折. All rights reserved.
//

#include "cart.hpp"
#include "math.h"
#include <numeric>
#include <map>
#include <set>


/*
 
 Node
 
 */

CartNode::CartNode(float val, int dim, float loss, CartNode* left, CartNode* right, bool leaf) {
    m_val = val;
    m_dim = dim;
    m_loss = loss;
    m_left = left;
    m_right = right;
    is_leaf = leaf;
}

void CartNode::set_sub_loss(float loss) {
    sub_tree_loss = loss;
}

void CartNode::set_complex(int complex) {
    m_complex = complex;
}

/*
 
 Base Tree
 
 */

Cart::Cart(int n_min) {
    m_n_min = n_min;
    fold = 10;
    root = nullptr;
}

void Cart::fit(mat data, vec label) {
    root = build_tree(data, label);
}

vec Cart::predict(mat data) {
    if ((data.size() == 0) || (root == nullptr)){
        vec res;
        return res;
    }
    vec res(data.size());
    for (int i=0; i<data.size(); i++) {
        res[i] = pred(data[i], root);
    }
    return res;
}

float Cart::pred(vec data, CartNode* node) {
    if(node->is_leaf)
        return node->m_val;
    if (data[node->m_dim] < node->m_val)
        return pred(data, node->m_left);
    return pred(data, node->m_right);
}

/*
 
 Regression Tree
 
 */

RegTree::RegTree(int n_min, bool cls):Cart(n_min){
    m_cls = cls;
}

CartNode* RegTree::build_tree(mat data, vec label) {
    if (data.size() <= m_n_min) {
        return build_leaf(label);
    }
    
    float loss = INFINITY;
    int attr=0;
    float seg=0.0;
    
    select_attr(data, label, loss, attr, seg);
    
    mat left_data, right_data;
    vec left_label, right_label;
    
    for (int i=0; i<data.size(); i++) {
        float val = data[i][attr];
        if (val < seg) {
            left_data.push_back(data[i]);
            left_label.push_back(label[i]);
        } else {
            right_data.push_back(data[i]);
            right_label.push_back(label[i]);
        }
    }
    
    if ((left_data.size() == 0) || (right_data.size() == 0))
        return build_leaf(label);
    
    CartNode* left = build_tree(left_data, left_label);
    CartNode* right = build_tree(right_data, right_label);
    CartNode* node = new CartNode(seg, attr, loss, left, right, false);
    
    return node;
    
}

CartNode* RegTree::build_leaf(vec label) {
    if (m_cls) {
        map<float, int> m;
        int max_count = 0, max_label = 0;
        float loss = 0.0;
        for(auto l:label){
            m[l] += 1;
            if (m[l] > max_count){
                max_count = m[l];
                max_label = l;
            }
        }
        
        CartNode* node = new CartNode(max_label, 0, loss, nullptr, nullptr, true);
        return node;
    } else {
        float val = accumulate(label.begin(), label.end(), 0.0)/label.size();
        float loss = 0.0;
        for (int i=0; i<label.size(); i++) {
            loss += pow(label[i]-val, 2);
        }
        CartNode* node = new CartNode(val, 0, loss, nullptr, nullptr, true);
        return node;
    }
}

void RegTree::select_attr(mat data, vec label, float &loss, int &attr, float &seg) {
    int n_attr = int(data[0].size());
    
    for (int i=0; i<n_attr; i++) {
        // get segmentations
        set<float> s_set;
        for (auto d:data)
            s_set.insert(d[i]);
        vec seg_vals;
        for (auto s:s_set)
            seg_vals.push_back(s);
        for (int j=0; j<seg_vals.size()-1; j++)
            seg_vals[j] = (seg_vals[j]+seg_vals[j+1]) / 2;
        seg_vals.pop_back();
        
        if (seg_vals.size() == 1)
            continue;
        
        // calculate loss and find segmentation
        vec r1, r2;
        for (auto s:seg_vals) {
            for (int j=0; j<data.size(); j++) {
                if (data[j][i] < s)
                    r1.push_back(label[j]);
                else
                    r2.push_back(label[j]);
            }
            float c1 = accumulate(r1.begin(), r1.end(), 0.0)/r1.size();
            float c2 = accumulate(r2.begin(), r2.end(), 0.0)/r2.size();
            float s_loss = 0.0;
            for (int j=0; j<r1.size(); j++)
                s_loss += pow(r1[j]-c1, 2);
            for (int j=0; j<r2.size(); j++)
                s_loss += pow(r2[j]-c2, 2);
            if (s_loss < loss) {
                loss = s_loss;
                attr = i;
                seg = s;
            }
        }
        
    }
    
}

/*
 
 Classification Tree
 
 */


ClaTree::ClaTree(int n_min): Cart(n_min) {
    m_cls = true;
}

CartNode* ClaTree::build_tree(mat data, vec label) {
    if (data.size() <= m_n_min) {
        return build_leaf(label);
    }
    
    float loss = INFINITY;
    int attr=0;
    float seg=0;
    
    select_attr(data, label, loss, attr, seg);
    
    
    mat left_data, right_data;
    vec left_label, right_label;
    
    for (int i=0; i<data.size(); i++) {
        float val = data[i][attr];
        if (val == seg) {
            left_data.push_back(data[i]);
            left_label.push_back(label[i]);
        } else {
            right_data.push_back(data[i]);
            right_label.push_back(label[i]);
        }
    }
    
    if ((left_data.size() == 0) || (right_data.size() == 0))
        return build_leaf(label);
    
    CartNode* left = build_tree(left_data, left_label);
    CartNode* right = build_tree(right_data, right_label);
    CartNode* node = new CartNode(seg, attr, loss, left, right, false);
    
    return node;
    
}

CartNode* ClaTree::build_leaf(vec label) {
    map<float, int> m;
    int max_count = 0;
    int max_label = 0;
    float loss = 0.0;
    for(auto l:label){
        m[l] += 1;
        if (m[l] > max_count){
            max_count = m[l];
            max_label = l;
        }
    }
    
    CartNode* node = new CartNode(max_label, 0, loss, nullptr, nullptr, true);
    return node;
}

void ClaTree::select_attr(mat data, vec label, float &loss, int &attr, float &seg) {
    int n_attr = int(data[0].size());
    
    for (int i=0; i<n_attr; i++) {
        // get segmentations
        set<float> s_set;
        for (auto d:data)
            s_set.insert(d[i]);
        vec seg_vals;
        for (auto s:s_set)
            seg_vals.push_back(s);
        
        if (seg_vals.size() == 1)
            continue;
        
        // calculate gini and find attribute
        
        for (auto s: seg_vals) {
            int count1=0, count2=0;
            map<float, int> m1, m2;
            for (int j=0; j < data.size(); j++) {
                if (data[j][i] == s) {
                    count1 += 1;
                    m1[label[j]] += 1;
                } else {
                    count2 += 1;
                    m2[label[j]] += 1;
                }
            }
            
            if ((count1 == 0) || count2 == 0)
                continue;
            
            float gini1 = 1.0, gini2 = 1.0;
            for (auto e:m1)
                gini1 -= pow(e.second/count1, 2);
            for (auto e:m2)
                gini2 -= pow(e.second/count2, 2);
            float gini = count1/data.size()*gini1 + count2/data.size()*gini2;
            if (gini < loss) {
                loss = gini;
                attr = i;
                seg = s;
            }
        }
        
    }
}

