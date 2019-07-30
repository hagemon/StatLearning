//
//  knn.h
//  StatLearning
//
//  Created by 一折 on 2019/7/30.
//  Copyright © 2019 一折. All rights reserved.
//

#ifndef knn_hpp
#define knn_hpp

#include <stdio.h>
#include <vector>

struct KDNode {
    std::vector<int> value;
    int dim;
    KDNode *left, *right;
    bool is_leaf;
};

struct val_dist_pair {
    std::vector<int> val;
    int dist;
};


class KDTree {
public:
    // Construct function
    KDTree(std::vector<std::vector<int>> d);
    
    // properties
    int n_ft;
    KDNode* root;
    std::vector<std::vector<int>> data;
    int min_dist;
    
    // building tree
    KDNode* build_tree(std::vector<std::vector<int>> d, int dim);
    KDNode* create_node(std::vector<int> d, int dim, KDNode* lnode, KDNode* rnode, bool leaf);
    std::vector<std::vector<int>> div(std::vector<std::vector<int>> &v, int dim);
    void find_k(std::vector<std::vector<int>> &v, int k, int dim, int start, int end);
    
    // util functions
    std::vector<std::vector<int>> slice(std::vector<std::vector<int>> &v, int start, int end);
    void pre(KDNode* node);
    int dist(std::vector<int> a, std::vector<int> b);
    
    // searching functions
    std::vector<std::vector<int>> search_knn(std::vector<int> target, int k);
    void search(std::vector<val_dist_pair>& heap, std::vector<int> target, int k, KDNode* node);
};

#endif /* knn_hpp */

