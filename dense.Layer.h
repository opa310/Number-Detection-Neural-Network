#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

typedef float (*Activation) (float value);

typedef struct _dense_layer{
    int weights_dim[2]; //{Rows, Columns}
    int outputs_dim[2]; //{Rows, Columns}
    float **weights;
    float *biases;
    float **output_z; //output before apply activation
    float **output_a; //output after applying activation
    Activation activ;
} Layer_Dense;


float ReLU(float x){
    return (x<0) ? 0: x;
}

float ReLU_Derivative(float x){
    return (float) x > 0;
}


#endif /* DENSE_LAYER_H */
