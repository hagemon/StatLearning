//
//  main.cpp
//  StatLearning
//
//  Created by 一折 on 2019/7/30.
//  Copyright © 2019 一折. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include <vector>
#include "knn.hpp"


int main() {
    std::vector<std::vector<int>> data = {{2, 3}, {5, 4}, {9, 6}, {4, 7}, {8, 1}, {7, 2}};
    KDTree tree = KDTree(data);
    std::cout<<"Pre Order of Tree:"<<std::endl;
    tree.pre(tree.root);
    std::cout<<"Top K data To Target:"<<std::endl;
    std::vector<int> target = {2, 3};
    std::vector<std::vector<int>> result = tree.search_knn(target, 3);
    for (auto d: result) {
        for(auto v: d) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }
}
