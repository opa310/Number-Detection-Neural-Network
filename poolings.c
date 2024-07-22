#include "conv.Layer.h"

float Max_Pooling(float ***input, int *kernels_dim, int channel, int input_row, int input_col){
    float max = input[channel][input_row][input_col];

    for(int row = 0; row < kernels_dim[1]; row++){
        for(int col = 0; col < kernels_dim[2]; col++){
            if(input[channel][input_row + row][input_col + col] > max){
                max = input[channel][input_row + row][input_col + col];
            }
        }
    }

    return max;
}


void Max_Pooling_Derivative(float ***input, float ***pooling_grad, float output_grad,
                            int *kernels_dim, int channel, int input_row, int input_col) {

    // Find the position of the max value in the pooling window
    float max_value = input[channel][input_row][input_col];
    int max_row = 0;
    int max_col = 0;

    for (int row = 0; row < kernels_dim[1]; row++) {
        for (int col = 0; col < kernels_dim[2]; col++) {
            float value = input[channel][input_row + row][input_col + col];
            if (value > max_value) {
                max_value = value;
                max_row = row;
                max_col = col;
            }
        }
    }

    // Set the gradient for the max position
    pooling_grad[channel][input_row + max_row][input_col + max_col] += output_grad;
}



float Min_Pooling(float ***input, int *kernels_dim, int channel, int input_row, int input_col){
    float min = input[channel][input_row][input_col];

    for(int row = 0; row < kernels_dim[1]; row++){
        for(int col = 0; col < kernels_dim[2]; col++){
            if(input[channel][input_row + row][input_col + col] < min){
                min = input[channel][input_row + row][input_col + col];
            }
        }
    }

    return min;
}

void Min_Pooling_Derivative(float ***input, float ***pooling_grad, float output_grad,
                             int *kernels_dim, int channel, int input_row, int input_col){
    // Find the position of the min value in the pooling window
    float min_value = input[channel][input_row][input_col];
    int min_row = 0;
    int min_col = 0;

    for (int row = 0; row < kernels_dim[1]; row++) {
        for (int col = 0; col < kernels_dim[2]; col++) {
            float value = input[channel][input_row + row][input_col + col];
            if (value < min_value) {
                min_value = value;
                min_row = row;
                min_col = col;
            }
        }
    }

    // Set the gradient for the max position
    pooling_grad[channel][input_row + min_row][input_col + min_col] += output_grad;

}


float Avg_Pooling(float ***input, int *kernels_dim, int channel, int input_row, int input_col){
    float sum = 0.0;

    for(int row = 0; row < kernels_dim[1]; row++){
        for(int col = 0; col < kernels_dim[2]; col++){
            sum += input[channel][input_row + row][input_col + col];
        }
    }

    return sum / (kernels_dim[1] * kernels_dim[2]); //Average or Mean
}


void Avg_Pooling_Derivative(float ***input, float ***pooling_grad, float output_grad,
                             int *kernels_dim, int channel, int input_row, int input_col){

    input = (void *)input; // To suppress the unused param warning


    float avg_grad = output_grad / (kernels_dim[1] * kernels_dim[2]);

    for (int row = 0; row < kernels_dim[1]; row++) {
        for (int col = 0; col < kernels_dim[2]; col++) {
            pooling_grad[channel][input_row + row][input_col + col] += avg_grad;
        }
    }

}
