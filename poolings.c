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


float Avg_Pooling(float ***input, int *kernels_dim, int channel, int input_row, int input_col){
    float sum = 0.0;

    for(int row = 0; row < kernels_dim[1]; row++){
        for(int col = 0; col < kernels_dim[2]; col++){
            sum += input[channel][input_row + row][input_col + col];
        }
    }

    return sum / (kernels_dim[1] * kernels_dim[2]); //Average or Mean
}
