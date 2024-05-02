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


void Soft_loss_idx(Layer_Dense *l, int *expected_idx, float **loss_out){

    for(int row = 0; row < l->outputs_dim[0]; row++ ){
        memset(loss_out[row], 0, sizeof(float) * l->outputs_dim[1]);
        loss_out[row][expected_idx[row]] = -logf(l->output_a[row][expected_idx[row]]);
        
        /* To clip infinite values */
        if (loss_out[row][expected_idx[row]] == 0x7f800000){ //(+ve infinity float)
            loss_out[row][expected_idx[row]] = powf(10, -7);
        } else if (loss_out[row][expected_idx[row]] == 0xff800000){ //(-ve infinity float)
            loss_out[row][expected_idx[row]] = 1 - powf(10, -7);
        }
    }
}

/* l3 is left as NULL for output layer back propagation */
void backward_pass (Layer_Dense *l1, Layer_Dense *l2, Layer_Dense *l3, int *expected, float alpha){ 
    if(l1->outputs_dim[1] != l2->weights_dim[0]){
        printf("Invalid layer dimensions\n");
        return;
    }

    float m = (float) l1->outputs_dim[1]; // Number of batches


    if(l2->activ == ReLU){
        for(int row = 0; row < l2->outputs_dim[0]; row++){
            for(int col = 0; col < l2->outputs_dim[1]; col++){
                for(int i = 0; i < l2->outputs_dim[1]; i++){
                    l2->dZ[row][col] = l3->weights[col][i] - l3->dZ[row][i];
                }
                l2->dZ[row][col] *= ReLU_Derivative(l2->output_z[row][col]);                
            }
        }
    } else {
        for(int row = 0; row < l2->outputs_dim[0]; row++){
            for(int col = 0; col < l2->outputs_dim[1]; col++){
                l2->dZ[row][col] = l2->output_z[row][col] - ((col == expected[row])? 1: 0); 
            }
        }
        //Soft_loss_idx(l2, expected, l2->dZ);

    }

    float dW2[l2->weights_dim[0]][l2->weights_dim[1]];

    for(int row = 0; row < l2->weights_dim[0]; row++){
        for(int col = 0; col < l2->weights_dim[1]; col++){
            for(int i = 0; i < l1->outputs_dim[0]; i++){
                dW2[row][col] += (l2->dZ[i][col] * l1->output_a[i][row]) / m;
            }
            l2->weights[row][col] -= dW2[row][col] * alpha; 
        }
    }
    

    float dB2[l2->weights_dim[1]];

    for(int row = 0; row < l2->outputs_dim[0]; row++){
        for(int col = 0; col < l2->outputs_dim[1]; col++){
            dB2[row] += l2->dZ[row][col];
        }
        dB2[row] /= m;
        l2->biases[row] -= dB2[row] * alpha;
    }

}
