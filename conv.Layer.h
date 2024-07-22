#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "dense.Layer.h"

typedef float (*Pooling) (float ***input, int *kernels_dim, int channel, int input_row, int input_col);
typedef void (*Pooling_Derivative)(float ***input, float ***pooling_grad, float output_grad,
                             int *kernels_dim, int channel, int input_row, int input_col);

float Max_Pooling(float ***input, int *kernels_dim, int channel, int input_row, int input_col)__attribute__((unused));

void Max_Pooling_Derivative(float ***input, float ***pooling_grad, float output_grad,
                             int *kernels_dim, int channel, int input_row, int input_col)__attribute__((unused));

float Min_Pooling(float ***input, int *kernels_dim, int channel, int input_row, int input_col)__attribute__((unused));

void Min_Pooling_Derivative(float ***input, float ***pooling_grad, float output_grad,
                             int *kernels_dim, int channel, int input_row, int input_col)__attribute__((unused));

float Avg_Pooling(float ***input, int *kernels_dim, int channel, int input_row, int input_col)__attribute__((unused));

void Avg_Pooling_Derivative(float ***input, float ***pooling_grad, float output_grad,
                             int *kernels_dim, int channel, int input_row, int input_col)__attribute__((unused));

typedef struct _conv_input_layer{
    int inputs_dim[4]; //{Batch, Channel, rows, columns}
    float ****inputs;
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
    Activation activ_deriv;
} Layer_Conv;


/* TODO */
typedef struct _pool_layer{
    int kernels_dim[3]; //{rows, columns)}
    int outputs_dim[3]; //{Channel, rows, columns}
    int stride;
    float ***dZ;
    float ***output; //output before apply activation
    Pooling pool;
    Pooling_Derivative pool_deriv;
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
int initLayer_conv_input(Input_Layer_Conv *l, int batches, int channels, int rows, int cols);
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
void backward_pass_conv (int *inputs_dim, float ***inputs, float ***input_grad, Layer_Conv *l_conv, Layer_Pool *l_pool, float alpha);
//#define malloc(...) NULL //For testing the init functions

void flatten_pool_to_dense ( Layer_Pool *l_pool, Layer_Dense *l_dense, int dense_layer_batch_idx);
void unflatten_dense_to_pool(Layer_Dense *l_dense, Layer_Pool *l_pool, int dense_layer_batch_idx);

#endif /* DENSE_LAYER_H */
