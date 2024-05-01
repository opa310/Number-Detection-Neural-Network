#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "dense.Layer.h"




void printLayer(Layer_Dense *l){
    printf("\nweights_dims : (%d,%d) ", l->weights_dim[0], l->weights_dim[1]);
    printf("\noutputs_dims : (%d,%d) ", l->outputs_dim[0], l->outputs_dim[1]);
    printf("\n\nweights :\n"); 
    for(int row = 0; row < l->weights_dim[0]; row++){
        for(int col = 0; col < l->weights_dim[1]; col++){
            printf("%0.7f, ", l->weights[row][col]);
        }
        printf("\n");
    }
    printf("\n\nbiases:\n"); 
    for(int col = 0; col < l->weights_dim[1]; col++){
            printf("%0.7f, ", l->biases[col]);
        printf("\n");
    }
    printf("\n\noutput_z :\n"); 
    for(int row = 0; row < l->outputs_dim[0]; row++){
        for(int col = 0; col < l->outputs_dim[1]; col++){
            printf("%0.7f, ", l->output_z[row][col]);
        }
        printf("\n");
    }
    printf("\n\noutput_a :\n"); 
    for(int row = 0; row < l->outputs_dim[0]; row++){
        for(int col = 0; col < l->outputs_dim[1]; col++){
            printf("%0.7f, ", l->output_a[row][col]);
        }
        printf("\n");
    }

    printf("\nActiv : ");
    if(l->activ == NULL){
        printf("Output Layer\n");
    } else if(l->activ == ReLU){
        printf("ReLU\n");
    }

}

int initLayer(Layer_Dense *l, int prev_layer_size, int layer_size, int batch_size, Activation function){
    l->weights_dim[0] = prev_layer_size;
    l->weights_dim[1] = layer_size;

    l->outputs_dim[0] = batch_size;
    l->outputs_dim[1] = layer_size;
   

    /* Weights */
    if((l->weights = (float **) malloc(sizeof(float *) * prev_layer_size)) == NULL){
        goto freeall;      
    }

    for(int row = 0; row < l->weights_dim[0]; row++){
        if((l->weights[row] = (float *) malloc(sizeof(float) * layer_size)) == NULL){
            goto freeall; 
        }
        for(int col = 0; col < l->weights_dim[1]; col++){
            /* Generates a random number between -1 and 1 */
            l->weights[row][col] = (float)rand()/(float)(RAND_MAX) * ((rand()&0x1)? 1:-1); 
        }
    }


    /* Biases */
    if((l->biases = (float *) malloc(sizeof(float) * layer_size)) == NULL){
        goto freeall; 
    }

    for(int col = 0; col < layer_size; col++){
            l->biases[col] = (float)rand()/(float)(RAND_MAX) * ((rand()&0x1)? 1:-1);
    }

    /* Output z */
    if((l->output_z = (float **) malloc(sizeof(float *) * batch_size)) == NULL){
        goto freeall; 
    }

    for(int row = 0; row < batch_size; row++){
        if((l->output_z[row] = (float *) malloc(sizeof(float) * layer_size)) == NULL){
            goto freeall; 
        }
        memset(l->output_z[row], 0, sizeof(float) * layer_size);
    }
    

    /* Output a */
    if((l->output_a = (float **) malloc(sizeof(float *) * batch_size)) == NULL){
        goto freeall; 
    }

    for(int row = 0; row < batch_size; row++){
        if((l->output_a[row] = (float *) malloc(sizeof(float) * layer_size)) == NULL){
            goto freeall; 
        }
        memset(l->output_a[row], 0, sizeof(float) * layer_size);
    }

    /* Activation function */
    l->activ = function;

    return 0;

    freeall:
        if(l->weights)
        for(int row = 0; row < l->weights_dim[0]; row++){
            free(l->weights[row]);
        }
        free(l->weights);

        free(l->biases);


        if(l->output_z)
        for(int row = 0; row < l->outputs_dim[0]; row++){
            free(l->output_z[row]);
        }
        free(l->output_z);

        if(l->output_a)
        for(int row = 0; row < l->outputs_dim[0]; row++){
            free(l->output_a[row]);
        }
        free(l->output_a);

        return -1;


}




int main (void){
    Layer_Dense d;

    if(initLayer(&d, 2, 8, 4, ReLU) < 0){
        perror("Failed to initialise layer");
    }

    printLayer(&d);

    return 0;
}
