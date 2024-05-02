#include "dense.Layer.h"



int main (void){
    Layer_Dense Input, l1, l2, output;

    if(initLayer(&Input, 0, 50, 4, ReLU) < 0){
        perror("Failed to initialise layer");
    }

    if(initLayer(&l1, 50, 6, 4, ReLU) < 0){
        perror("Failed to initialise layer");
    }


    if(initLayer(&l2, 6, 6, 4, ReLU) < 0){
        perror("Failed to initialise layer");
    }

    if(initLayer(&output, 6, 3, 4, NULL) < 0){
        perror("Failed to initialise layer");
    }

    //printLayer(&l1);
    //printLayer(&l2);


    forward_pass(&Input, &l1);
    forward_pass(&l1, &l2);
    forward_pass(&l2, &output);

    int expected[4] = {1,2,2,0};


    printLayer(&output);

    float learning_rate = 0.000001; // 42%

    backward_pass (&l2, &output, NULL, expected, learning_rate);
    backward_pass (&l1, &l2, &output, NULL, learning_rate);
    backward_pass (&Input, &l1, &l2, NULL, learning_rate);


    printLayer(&Input);
    printLayer(&l1);
    printLayer(&l2);
    printLayer(&output);

    return 0;
}
