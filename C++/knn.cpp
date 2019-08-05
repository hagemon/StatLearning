//
//  knn.cpp
//  StatLearning
//
//  Created by 一折 on 2019/7/27.
//  Copyright © 2019 一折. All rights reserved.
//

#include "knn.hpp"
#include <iostream>
#include <algorithm>
#include "math.h"

#define INF 99999;

struct heap_greater {
    bool operator()(const val_dist_pair& a, const val_dist_pair& b) {
        return a.dist < b.dist;
    }
};

KDTree::KDTree(std::vector<std::vector<int>> d) {
    data.insert(data.begin(), d.begin(), d.end());
    n_ft = int(d[0].size());
    min_dist = INF;
    root = build_tree(data, 0);
}

KDNode* KDTree::create_node(std::vector<int> d, int dim, KDNode* lnode, KDNode* rnode, bool leaf){
    KDNode *node = new KDNode;
    node->value = d;
    node->dim = dim;
    node->left = lnode;
    node->right = rnode;
    node->is_leaf = leaf;
    return node;
}

KDNode* KDTree::build_tree(std::vector<std::vector<int>> d, int dim) {
    if (d.size() == 0) {
        return nullptr;
    }
    else if (d.size() == 1) {
        return create_node(d[0], dim+1, nullptr, nullptr, true);
    }
    
    div(d, dim);
    int next_dim = (dim + 1) % n_ft;
    int k = int(d.size()) / 2;
    std::vector<int> median = d[k];
    std::vector<std::vector<int>> left_data = slice(d, 0, k);
    std::vector<std::vector<int>> right_data = slice(d, k+1, int(d.size()));
    KDNode* node = create_node(median, dim, build_tree(left_data, next_dim), build_tree(right_data, next_dim), false);
    return node;
}

std::vector<std::vector<int>> KDTree::div(std::vector<std::vector<int>> &v, int dim) {
    int k = int(v.size() / 2);
    find_k(v, k, dim, 0, int(v.size())-1);
    return v;
}

std::vector<std::vector<int>> KDTree::slice(std::vector<std::vector<int>> &v, int start, int end) {
    int new_len = end-start;
    std::vector<std::vector<int>> nv(new_len);
    for (int i=0; i<new_len; i++) {
        nv[i] = v[start+i];
    }
    return nv;
}

void KDTree::find_k(std::vector<std::vector<int>> &v, int k, int dim, int start, int end) {
    std::vector<int> guard = v[start];
    int left = start + 1;
    int right = end;
    while (left <= right) {
        while (left <= right && v[left][dim] <= guard[dim]) {
            left++;
        }
        while (left <= right && v[right][dim] >= guard[dim]) {
            right--;
        }
        if (left < right){
            swap(v[left], v[right]);
            
        }
    }
    swap(v[start], v[right]);
    if (k > right) {
        find_k(v, k, dim, right+1, end);
    } else if (k < right) {
        find_k(v, k, dim, start, right-1);
    }
}

void KDTree::pre(KDNode* node) {
    for (auto val: node->value) {
        std::cout << val << " ";
    }
    std::cout<< std::endl;
    if (node->left != nullptr) {
        pre(node->left);
    }
    if (node->right != nullptr) {
        pre(node->right);
    }
}

void KDTree::search(std::vector<val_dist_pair>& heap, std::vector<int> target, int k, KDNode* node) {
    if (node == nullptr) {
        return;
    }
    if (node->is_leaf) {
        int d = dist(node->value, target);
        if (d < min_dist) {
            val_dist_pair pair = {.val = node->value, .dist = d};
            if (heap.size() < k) {
                heap.push_back(pair);
                std::push_heap(heap.begin(), heap.end(), heap_greater());
                if (heap.size() == k) min_dist = heap.front().dist;
            } else {
                std::pop_heap(heap.begin(), heap.end(), heap_greater());
                heap.pop_back();
                heap.push_back(pair);
                std::push_heap(heap.begin(), heap.end(), heap_greater());
                min_dist = heap.front().dist;
            }
        }
    } else {
        int dim = node->dim;
        if (target[dim] <= node->value[dim]) {
            search(heap, target, k, node->left);
            if(node->value[dim] - target[dim] <= min_dist) {
                search(heap, target, k, node->right);
            }
        } else {
            search(heap, target, k, node->right);
            if(target[dim] - node->value[dim] <= min_dist) {
                search(heap, target, k, node->left);
            }
        }
        int d = dist(node->value, target);
        val_dist_pair pair = {.val = node->value, .dist = d};
        if(heap.size() < k) {
            heap.push_back(pair);
            std::push_heap(heap.begin(), heap.end(), heap_greater());
            if (heap.size() == k) min_dist = heap.front().dist;
        } else if (d < min_dist) {
            std::pop_heap(heap.begin(), heap.end(), heap_greater());
            heap.pop_back();
            heap.push_back(pair);
            std::push_heap(heap.begin(), heap.end(), heap_greater());
            min_dist = heap.front().dist;
        }
    }
}


std::vector<std::vector<int>> KDTree::search_knn(std::vector<int> target, int k) {
    std::vector<val_dist_pair> heap;
    search(heap, target, k, root);
    std::vector<std::vector<int>> result(k);
    for(int i=0; i < k; i++) {
        result[i] = heap[i].val;
    }
    return result;
}


int KDTree::dist(std::vector<int> a, std::vector<int> b) {
    int d = 0;
    for (int i=0; i < a.size(); i++) {
        d += pow(a[i]-b[i], 2);
    }
    return d;
}
