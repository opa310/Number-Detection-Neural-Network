#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>


typedef float (*Activation) (float value);

typedef struct _dense_layer{
    int weights_dim[2]; //{Rows, Columns}
    int outputs_dim[2]; //{Rows, Columns}
    float **weights;
    float *biases;
    float **output_z; //output before apply activation
    float **dZ;
    float **output_a; //output after applying activation
    Activation activ;
} Layer_Dense;


/* Activation functions */
float ReLU(float x)__attribute__((unused));
float ReLU_Derivative(float x)__attribute__((unused));
void Softmax(Layer_Dense *l);

float correct;
float total;


void printLayer(Layer_Dense *l);
int initLayer(Layer_Dense *l, int prev_layer_size, int layer_size, int batch_size, Activation function);
void layer_dense_to_csv(Layer_Dense* layer, char* filename);
void readLayerFromCSV(Layer_Dense ***layers, char *filename);


void forward_pass (Layer_Dense *l1, Layer_Dense *l2);
void backward_pass (Layer_Dense *l1, Layer_Dense *l2, Layer_Dense *l3, int *expected, float alpha);

//#define malloc(...) NULL //For testing the init functions

#endif /* DENSE_LAYER_H */
