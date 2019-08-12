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
#include "svm.hpp"
#include "iris.hpp"
#include "cart.hpp"
#include <string>


int main() {
    
    // Uncomment lines to use corresponding models.
    
    /*
     
     Iris Dataset
     
     */
    
    Iris iris = Iris();
    
    /*
     
     KD Tree
     
     */
    
    /*
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
     */
    
    
    /*
     
     SVM
     
     */
    
    /*
    SVM svm = SVM(0.00001, 1, "linear");

    svm.fit(iris.train_x, iris.train_y);
    for (auto a: svm.alpha) cout << a << " ";
    cout << endl;

    vector<float> result = svm.predict(iris.test_x);

    float hit = 0;

    for (int i=0; i<iris.test_y.size(); i++) {
        if (result[i] == iris.test_y[i]) {
            hit += 1;
        }
    }

    cout << "Precision: " << hit / iris.test_y.size() << endl;
    */
    
    
    /*
     
     CART
     
     */
    
    // ClaTree tree = ClaTree(5);
    RegTree tree = RegTree(5, true);
    tree.fit(iris.train_x, iris.train_y);
    vec result = tree.predict(iris.test_x);
    
    float hit = 0;
    
    for (int i=0; i<iris.test_y.size(); i++) {
        if (result[i] == iris.test_y[i]) {
            hit += 1;
        }
    }
    
    cout << "Precision: " << hit / iris.test_y.size() << endl;
    
}
