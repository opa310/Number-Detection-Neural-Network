#include "conv.Layer.h"

void print3DMatrix(float ***matrix, int depth, int rows, int cols) {
    for (int d = 0; d < depth; d++) {
        printf("Depth %d:\n", d);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                printf("%f ", matrix[d][r][c]);
            }
            printf("\n");
        }
        printf("\n");
    }
}


void forward_pass_conv (int *inputs_dim, float ***inputs, Layer_Conv *l){
    int output_dim_height = ((inputs_dim[1] - l->kernels_dim[1]) / l->stride) + 1;
    int output_dim_width = ((inputs_dim[2] - l->kernels_dim[2]) / l->stride) + 1;

    // Check if calculated dimensions match layer's output dimensions
    if (output_dim_height != l->outputs_dim[1] || output_dim_width != l->outputs_dim[2]) {
        printf("Mismatched layer dimensions: expected (%d, %d), got (%d, %d)\n", 
               l->outputs_dim[1], l->outputs_dim[2], output_dim_height, output_dim_width);
        return;
    }
    
    for(int kernel_num = 0; kernel_num < l->kernels_dim[0]; kernel_num++){
        int output_row = 0;
        for(int input_row = 0; input_row <= inputs_dim[1] - l->kernels_dim[1]; input_row += l->stride){
            int output_col = 0;
            for(int input_col = 0; input_col <= inputs_dim[2] - l->kernels_dim[2]; input_col += l->stride){
                l->output_z[kernel_num][output_row][output_col] = 0.0;

                for(int channel = 0; channel < inputs_dim[0]; channel++){
                    for(int kernel_row = 0; kernel_row < l->kernels_dim[1]; kernel_row++){
                        for(int kernel_col = 0; kernel_col < l->kernels_dim[2]; kernel_col++){
                            l->output_z[kernel_num][output_row][output_col] += 
                                inputs[channel][input_row + kernel_row][input_col + kernel_col] * 
                                l->kernels[kernel_num][kernel_row][kernel_col];
                        }
                    }
                }

                l->output_z[kernel_num][output_row][output_col] += l->biases[kernel_num];

                l->output_a[kernel_num][output_row][output_col] = 
                        l->activ(l->output_z[kernel_num][output_row][output_col]);
                output_col++;
            }
            output_row++;
        }
    }
    
}



void forward_pass_pool (int *inputs_dim, float ***inputs, Layer_Pool *l){
    int output_dim_height = ((inputs_dim[1] - l->kernels_dim[1]) / l->stride) + 1;
    int output_dim_width = ((inputs_dim[2] - l->kernels_dim[2]) / l->stride) + 1;

    // Check if calculated dimensions match layer's output dimensions
    if (output_dim_height != l->outputs_dim[1] || output_dim_width != l->outputs_dim[2]) {
        printf("Mismatched layer dimensions: expected (%d, %d), got (%d, %d)\n", 
               l->outputs_dim[1], l->outputs_dim[2], output_dim_height, output_dim_width);
        return;
    }
    

    int output_row = 0;
    for(int input_row = 0; input_row <= inputs_dim[1] - l->kernels_dim[1]; input_row += l->stride){
        int output_col = 0;
        for(int input_col = 0; input_col <= inputs_dim[2] - l->kernels_dim[2]; input_col += l->stride){
            for(int channel = 0; channel < inputs_dim[0]; channel++){
                l->output[channel][output_row][output_col] = l->pool(inputs, l->kernels_dim, channel, output_row, output_col);
            }
            output_col++;
        }
        output_row++;
    }
    
    
}

void flatten_pool_to_dense ( Layer_Pool *l_pool, Layer_Dense *l_dense, int dense_layer_batch_idx){

    if((l_pool->outputs_dim[0] * l_pool->outputs_dim[1] * l_pool->outputs_dim[2]) != l_dense->outputs_dim[1]){
        printf("Invalid flattening layer dimensions: %d\n", l_pool->outputs_dim[0] * l_pool->outputs_dim[1] * l_pool->outputs_dim[2]);
        return;
    }

    int output_idx = 0;

    //print3DMatrix(&l_dense->output_a, 1, l_dense->outputs_dim[0], l_dense->outputs_dim[1]);

    for(int channel = 0; channel < l_pool->outputs_dim[0]; channel++){
        for(int input_row = 0; input_row < l_pool->outputs_dim[1]; input_row++){
            for(int input_col = 0; input_col < l_pool->outputs_dim[2]; input_col++){
                l_dense->output_a[dense_layer_batch_idx][output_idx] = l_pool->output[channel][input_row][input_col];
                l_dense->output_z[dense_layer_batch_idx][output_idx] = l_pool->output[channel][input_row][input_col];

                output_idx++;
            }
        }
    }

}

void unflatten_dense_to_pool(Layer_Dense *l_dense, Layer_Pool *l_pool, int dense_layer_batch_id) {
    if((l_pool->outputs_dim[0] * l_pool->outputs_dim[1] * l_pool->outputs_dim[2]) != l_dense->outputs_dim[1]){
        printf("Invalid unflattening layer dimensions: %d\n", l_pool->outputs_dim[0] * l_pool->outputs_dim[1] * l_pool->outputs_dim[2]);
        return;
    }

    //print3DMatrix(&l_dense->dZ, 1, l_dense->outputs_dim[0], l_dense->outputs_dim[1]);

    int output_idx = 0;

    for(int channel = 0; channel < l_pool->outputs_dim[0]; channel++){
        for(int input_row = 0; input_row < l_pool->outputs_dim[1]; input_row++){
            for(int input_col = 0; input_col < l_pool->outputs_dim[2]; input_col++){
                l_pool->dZ[channel][input_row][input_col] = l_dense->dZ[dense_layer_batch_id][output_idx];
                l_pool->output[channel][input_row][input_col] = l_dense->output_a[dense_layer_batch_id][output_idx];
                output_idx++;
            }
        }
    }
    //print3DMatrix(&l_pool->dZ[0], l_pool->outputs_dim[0], l_pool->outputs_dim[1], l_pool->outputs_dim[2]);
}



void backward_pass_conv(int *inputs_dim, float ***inputs, float ***input_grad, Layer_Conv *l_conv, Layer_Pool *l_pool, float alpha) {
    // Allocate memory for dC
    float ***dC = (float ***)malloc(l_conv->outputs_dim[0] * sizeof(float **));
    for (int i = 0; i < l_conv->outputs_dim[0]; i++) {
        dC[i] = (float **)malloc(l_conv->outputs_dim[1] * sizeof(float *));
        for (int j = 0; j < l_conv->outputs_dim[1]; j++) {
            dC[i][j] = (float *)malloc(l_conv->outputs_dim[2] * sizeof(float));
            memset(dC[i][j], 0, l_conv->outputs_dim[2] * sizeof(float));
        }
    }

    // Perform backpropagation through the pooling layer
    int output_row = 0;
    for (int input_row = 0; input_row <= l_conv->outputs_dim[1] - l_pool->kernels_dim[1]; input_row += l_pool->stride) {
        int output_col = 0;
        for (int input_col = 0; input_col <= l_conv->outputs_dim[2] - l_pool->kernels_dim[2]; input_col += l_pool->stride) {
            for (int channel = 0; channel < l_conv->outputs_dim[0]; channel++) {
                //printf("%f\n", l_pool->dZ[channel][output_row][output_col]);
                l_pool->pool_deriv(l_conv->output_z, dC, l_pool->dZ[channel][output_row][output_col], l_pool->kernels_dim, channel, input_row, input_col);
            }
            output_col++;
        }
        output_row++;
    }

    //print3DMatrix(l_pool->dZ, l_pool->outputs_dim[0], l_pool->outputs_dim[1], l_pool->outputs_dim[2]);
    //print3DMatrix(dC, l_conv->outputs_dim[0], l_conv->outputs_dim[1], l_conv->outputs_dim[2]);




    // Calculate dZ for the convolutional layer
    for (int channel = 0; channel < l_pool->outputs_dim[0]; channel++) {
        for (int row = 0; row < l_pool->outputs_dim[1]; row++) {
            for (int col = 0; col < l_pool->outputs_dim[2]; col++) {
                l_conv->dZ[channel][row][col] = 0;
                for (int i = 0; i < l_pool->outputs_dim[2]; i++) {
                    l_conv->dZ[channel][row][col] += dC[channel][row][i] * l_conv->activ_deriv(l_conv->output_z[channel][i][col]);
                }
            }
        }
    }



    // Free memory allocated for dC
    for (int i = 0; i < l_conv->outputs_dim[0]; i++) {
        for (int j = 0; j < l_conv->outputs_dim[1]; j++) {
            free(dC[i][j]);
        }
        free(dC[i]);
    }
    free(dC);


    //printf("\nInputs dim 1 %d - outputs dim 1 %d", inputs_dim[1], l_conv->outputs_dim[1]);
    //printf("\nInputs dim 2 %d - outputs dim 2 %d", inputs_dim[2], l_conv->outputs_dim[2]);

    //Update kernels and biases
    for (int kernel_channel = 0; kernel_channel < l_conv->kernels_dim[0]; kernel_channel++) {
        float dB = 0;
        for (int input_row = 0; input_row <= inputs_dim[1] - l_conv->outputs_dim[1]; input_row += l_conv->stride) {
            for (int input_col = 0; input_col <= inputs_dim[2] - l_conv->outputs_dim[2]; input_col += l_conv->stride) {
                float dK = 0;
                for (int input_channel = 0; input_channel < inputs_dim[0]; input_channel++) {
                    for (int dZ_row = 0; dZ_row < l_conv->outputs_dim[1]; dZ_row++) {
                        for (int dZ_col = 0; dZ_col < l_conv->outputs_dim[2]; dZ_col++) {
                            dK += inputs[input_channel][input_row + dZ_row][input_col + dZ_col] * l_conv->dZ[kernel_channel][dZ_row][dZ_col];
                            if (input_channel == 0) {
                                dB += l_conv->dZ[kernel_channel][dZ_row][dZ_col];
                            }
                        }
                    }
                }

                //printf("\n (%d,%d,%d)", kernel_channel, input_row/l_conv->stride, input_col/l_conv->stride);
                l_conv->kernels[kernel_channel][input_row/l_conv->stride][input_col/l_conv->stride] -= dK * alpha;
            }
        }
        //printf("dB = %f\n", dB);
        l_conv->biases[kernel_channel] -= dB * alpha;
        //printf("Bias %d = %f\n", kernel_channel, l_conv->biases[kernel_channel]);
    }



    if (input_grad == NULL) return; // The previous layer is likely an input layer for the conv network

    // Define padding and dilation
    int padding_vert = l_conv->kernels_dim[1] - 1;
    int padding_hor = l_conv->kernels_dim[2] - 1;
    int dilation = l_conv->stride - 1;

    // Calculate dZ_Padded dimensions
    int dZ_Padded_dim[3]; 
    dZ_Padded_dim[0] = inputs_dim[0];
    dZ_Padded_dim[1] = inputs_dim[1] + (padding_vert * 2) + (dilation) * (l_conv->outputs_dim[1] - 1);
    dZ_Padded_dim[2] = inputs_dim[2] + (padding_hor * 2) + (dilation) * (l_conv->outputs_dim[2] - 1);


    // dZ_Padded_dim[1] = (padding_vert * 2) + l_conv->outputs_dim[1] + (l_conv->outputs_dim[1] - 1) * dilation;
    // dZ_Padded_dim[2] = (padding_hor * 2) + l_conv->outputs_dim[2] + (l_conv->outputs_dim[2] - 1) * dilation;


    // Allocate memory for dZ_Padded and initialize
    float ***dZ_Padded = (float ***)malloc(dZ_Padded_dim[0] * sizeof(float **));
    for (int i = 0; i < dZ_Padded_dim[0]; i++) {
        dZ_Padded[i] = (float **)malloc(dZ_Padded_dim[1] * sizeof(float *));
        for (int j = 0; j < dZ_Padded_dim[1]; j++) {
            dZ_Padded[i][j] = (float *)malloc(dZ_Padded_dim[2] * sizeof(float));
            memset(dZ_Padded[i][j], 0, dZ_Padded_dim[2] * sizeof(float));
        }
    }


    // Create the Padded and Dilated gradient matrix
    for (int channel = 0; channel < l_conv->outputs_dim[0]; channel++) {
        for (int row = 0; row < l_conv->outputs_dim[1]; row++) {
            for (int col = 0; col < l_conv->outputs_dim[2]; col++) {
                dZ_Padded[channel][padding_vert + row * (dilation + 1)][padding_hor + col * (dilation + 1)] = l_conv->dZ[channel][row][col];
            }
        }
    }


    // Calculate input gradients
    for (int input_channel = 0; input_channel < inputs_dim[0]; input_channel++) {
        int input_row = 0;
        for (int dZ_Padded_row = 0; dZ_Padded_row <= dZ_Padded_dim[1] - l_conv->kernels_dim[1]; dZ_Padded_row += l_conv->stride) {
            int input_col = 0;
            for (int dZ_Padded_col = 0; dZ_Padded_col <= dZ_Padded_dim[2] - l_conv->kernels_dim[2]; dZ_Padded_col += l_conv->stride) {
                input_grad[input_channel][input_row][input_col] = 0.0;
                for (int kernel_channel = 0; kernel_channel < l_conv->kernels_dim[0]; kernel_channel++) {
                    for (int kernel_row = 0; kernel_row < l_conv->kernels_dim[1]; kernel_row++) {
                        for (int kernel_col = 0; kernel_col < l_conv->kernels_dim[2]; kernel_col++) {
                            input_grad[input_channel][input_row][input_col] += dZ_Padded[input_channel][dZ_Padded_row][dZ_Padded_col] *
                                                                               l_conv->kernels[kernel_channel][l_conv->kernels_dim[1] - 1 - kernel_row][l_conv->kernels_dim[2] - 1 - kernel_col];
                        }
                    }
                }
                input_col++;
            }
            input_row++;
        }
    }


    // Free memory allocated for dZ_Padded
    for (int i = 0; i < dZ_Padded_dim[0]; i++) {
        for (int j = 0; j < dZ_Padded_dim[1]; j++) {
            free(dZ_Padded[i][j]);
        }
        free(dZ_Padded[i]);
    }
    free(dZ_Padded);
}
