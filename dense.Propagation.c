#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "dense.Layer.h"


void Softmax(Layer_Dense *l){

    for(int row = 0; row < l->outputs_dim[0]; row++){

        float row_max = l->output_z[row][0];
        for(int col = 1; col < l->outputs_dim[1]; col++){
            if(l->output_z[row][col] > row_max)
                row_max = l->output_z[row][col];
        }

        float row_sum = 0;/* Exponentiated sum */
        for(int col = 1; col < l->outputs_dim[1]; col++){
            l->output_a[row][col] = expf(l->output_z[row][col] - row_max);
            row_sum += l->output_a[row][col];
        }

        for(int col = 1; col < l->outputs_dim[1]; col++){
            l->output_a[row][col] /= row_sum;
        }

    }

}


void forward_pass (Layer_Dense *l1, Layer_Dense *l2){
    if(l1->outputs_dim[1] != l2->weights_dim[0]){
        printf("Invalid layer dimensions\n");
        return;
    }

    for(int row = 0; row < l1->outputs_dim[0]; row++){
        for(int col = 0; col < l2->weights_dim[1]; col++){
            for(int i = 0; i < l1->outputs_dim[1]; i++){
                l2->output_z[row][col] += l1->output_a[row][i] * l2->weights[i][col];
            }
            l2->output_z[row][col] += l2->biases[col];
            if(l2->activ != NULL) // Only null for the output layer
                l2->output_a[row][col] =  l2->activ(l2->output_z[row][col]);
        }
    }

    if(l2->activ == NULL) Softmax(l2);

}
