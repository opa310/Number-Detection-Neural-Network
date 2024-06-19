#include "conv.Layer.h"


void forward_pass_conv (int *inputs_dim, float ***inputs, Layer_Conv *l){
    int output_dim_height = ((inputs_dim[1] - l->kernels_dim[1]) / l->stride) + 1;
    int output_dim_width = ((inputs_dim[2] - l->kernels_dim[2]) / l->stride) + 1;

    // Check if calculated dimensions match layer's output dimensions
    if (output_dim_height != l->outputs_dim[1] || output_dim_width != l->outputs_dim[2]) {
        printf("Mismatched layer dimensions\n");
        return;
    }
    
    for(int kernel_num = 0; kernel_num < l->kernels_dim[0]; kernel_num++){
        int output_row = 0;
        for(int input_row = 0; input_row <= inputs_dim[1] - l->kernels_dim[1]; input_row += l->stride){
            if(output_row > l->outputs_dim[1]) break;
            int output_col = 0;
            for(int input_col = 0; input_col <= inputs_dim[2] - l->kernels_dim[2]; input_col += l->stride){
                if(output_row > l->outputs_dim[2]) break;
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
        printf("Mismatched layer dimensions\n");
        return;
    }
    

    int output_row = 0;
    for(int input_row = 0; input_row <= inputs_dim[1] - l->kernels_dim[1]; input_row += l->stride){
        if(output_row > l->outputs_dim[1]) break;
        int output_col = 0;
        for(int input_col = 0; input_col <= inputs_dim[2] - l->kernels_dim[2]; input_col += l->stride){
            if(output_row > l->outputs_dim[2]) break;
            for(int channel = 0; channel < inputs_dim[0]; channel++){
                l->output[channel][output_row][output_col] = l->pool(inputs, l->kernels_dim, channel, output_row, output_col);
            }
            output_col++;
        }
        output_row++;
    }
    
    
}
