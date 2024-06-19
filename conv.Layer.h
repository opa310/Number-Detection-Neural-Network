#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <activations.h>

typedef float (*Pooling) (float ***input, int *kernels_dim, int channel, int input_row, int input_col);

float Max_Pooling(float ***input, int *kernels_dim, int channel, int input_row, int input_col)__attribute__((unused));
float Min_Pooling(float ***input, int *kernels_dim, int channel, int input_row, int input_col)__attribute__((unused));
float Avg_Pooling(float ***input, int *kernels_dim, int channel, int input_row, int input_col)__attribute__((unused));

typedef struct _conv_input_layer{
    int inputs_dim[3]; //{Channel, rows, columns}
    float ***inputs;
} Input_Layer_Conv;


typedef struct _conv_layer{
    int kernels_dim[3]; //{kernel_count, rows, columns}
    int outputs_dim[3]; //{Channel, rows, columns}
    int stride;
    float ***kernels;
    float *biases; // biases[kernel_count]
    //vvv Feature Maps(Output Matrixes) depth * number of kernels 
    float ***output_z; //output before apply activation
    float ***dZ;
    float ***output_a; //output after applying activation*/
    Activation activ;
} Layer_Conv;


/* TODO */
typedef struct _pool_layer{
    int kernels_dim[3]; //{rows, columns)}
    int outputs_dim[3]; //{Channel, rows, columns}
    int stride;
    float ***output; //output before apply activation
    // POOLING FUNCTION POINTER
    Pooling pool;
} Layer_Pool;



/* Activation functions 
float ReLU(float x)__attribute__((unused));
float ReLU_Derivative(float x)__attribute__((unused));
void Softmax(Layer_Dense *l);

float correct;
float total;

*/
void printLayer_conv(Layer_Conv *l);
void printLayer_conv_input(Input_Layer_Conv *l);
void printLayer_pool(Layer_Pool *l);

int initLayer_conv(Layer_Conv *l, int prev_layer_row, int prev_layer_col, 
                int kernel_count, int kernel_row, int kernel_col, int stride, Activation function);
int initLayer_conv_input(Input_Layer_Conv *l, int channels, int row, int col);
int initLayer_pool(Layer_Pool *l, int prev_layer_channels, int prev_layer_row, int prev_layer_col,
                     int kernel_row, int kernel_col, int stride, Pooling function);
/*void layer_dense_to_csv(Layer_Dense* layer, char* filename);
void readLayerFromCSV(Layer_Dense ***layers, char *filename);

*/
void forward_pass_conv (int *inputs_dim, float ***inputs, Layer_Conv *l);
void forward_pass_pool (int *inputs_dim, float ***inputs, Layer_Pool *l);
/*
void backward_pass (Layer_Dense *l1, Layer_Dense *l2, Layer_Dense *l3, int *expected, float alpha);
*/
//#define malloc(...) NULL //For testing the init functions

#endif /* DENSE_LAYER_H */
