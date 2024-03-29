//
//  iris.cpp
//  StatLearning
//
//  Created by 一折 on 2019/8/11.
//  Copyright © 2019 一折. All rights reserved.
//

#include "iris.hpp"

Iris::Iris() {
    train_x = {{5.6, 2.1}, {1.5, 0.4}, {5.8, 2.2}, {1.5, 0.3}, {1.3, 0.3}, {1.5, 0.4}, {4.7, 1.4}, {1.1, 0.1}, {1.0, 0.2}, {5.0, 1.9}, {5.3, 2.3}, {5.1, 1.9}, {1.4, 0.2}, {1.5, 0.2}, {5.7, 2.5}, {4.7, 1.6}, {4.0, 1.3}, {1.5, 0.1}, {3.6, 1.3}, {1.4, 0.2}, {1.3, 0.2}, {4.4, 1.2}, {4.4, 1.4}, {5.6, 1.4}, {5.7, 2.3}, {5.8, 1.6}, {1.4, 0.2}, {5.5, 1.8}, {5.1, 1.8}, {6.1, 2.3}, {4.6, 1.4}, {4.5, 1.5}, {4.7, 1.4}, {1.4, 0.1}, {6.9, 2.3}, {4.2, 1.3}, {5.1, 2.3}, {1.6, 0.2}, {6.3, 1.8}, {3.3, 1.0}, {6.0, 2.5}, {1.4, 0.2}, {5.1, 2.4}, {1.6, 0.2}, {1.5, 0.2}, {1.5, 0.2}, {6.1, 1.9}, {1.4, 0.1}, {1.6, 0.4}, {1.5, 0.2}, {1.6, 0.2}, {5.1, 1.5}, {1.7, 0.3}, {4.0, 1.3}, {4.5, 1.5}, {5.1, 2.0}, {5.0, 1.7}, {1.4, 0.2}, {1.3, 0.4}, {4.8, 1.8}, {5.0, 2.0}, {1.4, 0.3}, {1.3, 0.2}, {4.8, 1.8}, {5.2, 2.3}, {4.2, 1.2}, {1.4, 0.3}, {4.9, 1.5}, {3.9, 1.1}, {5.9, 2.1}, {6.1, 2.5}, {5.5, 2.1}, {3.9, 1.4}, {1.9, 0.4}, {4.7, 1.2}, {5.8, 1.8}, {4.5, 1.6}, {1.7, 0.5}, {4.5, 1.7}, {4.6, 1.3}, {4.3, 1.3}, {4.5, 1.5}, {5.7, 2.1}, {1.5, 0.2}, {6.7, 2.2}, {5.2, 2.0}, {5.0, 1.5}, {1.7, 0.2}, {4.2, 1.3}, {6.6, 2.1}, {5.1, 1.9}, {6.0, 1.8}, {1.4, 0.2}, {4.0, 1.0}, {5.6, 2.4}, {1.4, 0.2}, {4.9, 1.8}, {4.1, 1.3}, {6.4, 2.0}, {5.1, 1.6}};
    train_y = {1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
    
    test_x = {{4.2, 1.3}, {5.1, 1.8}, {5.1, 1.9}, {3.9, 1.4}, {1.5, 0.2}, {5.4, 2.3}, {5.5, 1.8}, {4.9, 1.8}, {5.7, 2.1}, {4.2, 1.5}, {1.9, 0.4}, {3.5, 1.0}, {1.6, 0.2}, {5.1, 2.4}, {1.6, 0.6}, {4.6, 1.3}, {1.5, 0.4}, {6.0, 1.8}, {4.9, 2.0}, {1.3, 0.3}, {1.9, 0.2}, {4.3, 1.3}, {1.5, 0.1}, {4.9, 1.5}, {1.4, 0.2}, {1.5, 0.2}, {4.8, 1.8}, {4.0, 1.3}, {5.1, 1.6}, {1.5, 0.2}, {1.4, 0.3}, {4.5, 1.5}, {3.3, 1.0}, {1.4, 0.2}, {6.6, 2.1}, {4.1, 1.0}, {3.9, 1.2}, {4.6, 1.5}, {1.4, 0.1}, {5.1, 2.3}, {1.6, 0.2}, {1.7, 0.4}, {6.1, 2.5}, {1.3, 0.3}, {1.5, 0.3}, {1.4, 0.2}, {5.3, 1.9}, {1.7, 0.2}, {4.9, 1.5}, {4.9, 1.8}};
    test_y = {-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0};
}
