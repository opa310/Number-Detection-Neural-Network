#include "conv.Layer.h"

void printLayer_conv(Layer_Conv *l){
    printf("\nkernels_dims : (%d,%d,%d) (kernel_count, rows, columns)",
     l->kernels_dim[0], l->kernels_dim[1], l->kernels_dim[2]);

    printf("\nStride : %d",l->stride);

    printf("\n\nKernels / Filters :\n"); 
    for(int kernel_num = 0; kernel_num < l->kernels_dim[0]; kernel_num++){
        for(int row = 0; row < l->kernels_dim[1]; row++){
            for(int col = 0; col < l->kernels_dim[2]; col++){
                printf("%0.7f, ", l->kernels[kernel_num][row][col]);
            }
            printf("\n");
        }
        printf("\n\n");
    }

    printf("\n\nbiases:\n"); 
    for(int kernel_num = 0; kernel_num < l->kernels_dim[0]; kernel_num++){
            printf("%0.7f, ", l->biases[kernel_num]);
        printf("\n");
    }
    printf("\n\ndZ gradient :\n"); 
    for(int channel = 0; channel < l->outputs_dim[0]; channel++){
        for(int row = 0; row < l->outputs_dim[1]; row++){
            for(int col = 0; col < l->outputs_dim[2]; col++){
                printf("%0.7f, ", l->dZ[channel][row][col]);
            }
            printf("\n");
        }
        printf("\n\n");
    }

    printf("\n\noutput_z :\n"); 
    for(int channel = 0; channel < l->outputs_dim[0]; channel++){
        for(int row = 0; row < l->outputs_dim[1]; row++){
            for(int col = 0; col < l->outputs_dim[2]; col++){
                printf("%0.7f, ", l->output_z[channel][row][col]);
            }
            printf("\n");
        }
        printf("\n\n");
    }

    printf("\n\noutput_a :\n"); 
    for(int channel = 0; channel < l->outputs_dim[0]; channel++){
        for(int row = 0; row < l->outputs_dim[1]; row++){
            for(int col = 0; col < l->outputs_dim[2]; col++){
                printf("%0.7f, ", l->output_a[channel][row][col]);
            }
            printf("\n");
        }
        printf("\n\n");
    }

    printf("\nActiv : ");
    if(l->activ == NULL){
        printf("SoftMax\n");
    } else if(l->activ == ReLU){
        printf("ReLU\n");
    }

}


void printLayer_conv_input(Input_Layer_Conv *l){

    printf("\ninputs_dim : (%d,%d,%d) (channels, rows, columns)",
     l->inputs_dim[0], l->inputs_dim[1], l->inputs_dim[2]);



    printf("\n\ninput :\n"); 
    for(int channel = 0; channel < l->inputs_dim[0]; channel++){
        for(int row = 0; row < l->inputs_dim[1]; row++){
            for(int col = 0; col < l->inputs_dim[2]; col++){
                printf("%0.7f, ", l->inputs[channel][row][col]);
            }
            printf("\n");
        }
        printf("\n\n");
    }

}


void printLayer_pool(Layer_Pool *l){

    printf("\nkernels_dims : (%d,%d,%d) (kernel_count, rows, columns)",
     l->kernels_dim[0], l->kernels_dim[1], l->kernels_dim[2]);


    printf("\nStride : %d",l->stride);

    

    printf("\n\noutput :\n"); 
    for(int channel = 0; channel < l->outputs_dim[0]; channel++){
        for(int row = 0; row < l->outputs_dim[1]; row++){
            for(int col = 0; col < l->outputs_dim[2]; col++){
                printf("%0.7f, ", l->output[channel][row][col]);
            }
            printf("\n");
        }
        printf("\n\n");
    }


    printf("\nPooling : ");
    if(l->pool == Min_Pooling){
        printf("Min Pooling\n");
    } else if(l->pool == Max_Pooling){
        printf("Max Pooling\n");
    } else if(l->pool == Avg_Pooling){
        printf("Average Pooling\n");
    }



}




int initLayer_conv(Layer_Conv *l, int prev_layer_row, int prev_layer_col, 
                int kernel_count, int kernel_row, int kernel_col, int stride, Activation function){
    l->kernels_dim[0] = kernel_count;
    l->kernels_dim[1] = kernel_row;
    l->kernels_dim[2] = kernel_col;

    l->stride = stride;

    l->outputs_dim[0] = kernel_count;
    l->outputs_dim[1] = ((prev_layer_row - kernel_row)/stride) +1; // Integar division Floors this value
    l->outputs_dim[2] = ((prev_layer_col - kernel_col)/stride) +1;

   

    /* Kernels */
    if((l->kernels = (float ***) malloc(sizeof(float **) * l->kernels_dim[0])) == NULL){
        goto freeall;      
    }
    for(int kernel_num = 0; kernel_num < l->kernels_dim[0]; kernel_num++){
        if((l->kernels[kernel_num] = (float **) malloc(sizeof(float *) * l->kernels_dim[1])) == NULL){
            goto freeall; 
        }
        for(int row = 0; row < l->kernels_dim[1]; row++){
            if((l->kernels[kernel_num][row] = (float *) malloc(sizeof(float) * l->kernels_dim[2])) == NULL){
                goto freeall; 
            }
            for(int col = 0; col < l->kernels_dim[2]; col++){
                /* Generates a random number between -1 and 1 */
                l->kernels[kernel_num][row][col] = (float)rand()/(float)(RAND_MAX) * ((rand()&0x1)? 1:-1);
            }
        }
    }
    


    /* Biases */
    if((l->biases = (float *) malloc(sizeof(float) * l->kernels_dim[0])) == NULL){
        goto freeall; 
    }
    /* Zeroed Biases*/
    memset(l->biases, 0, sizeof(float) * l->kernels_dim[0]);

    /* Random Biases 
    for(int col = 0; col < l->kernels_dim[0]; col++){
       l->biases[col] = (float)rand()/(float)(RAND_MAX) * ((rand()&0x1)? 1:-1);
    }*/

    /* Output z */
    if((l->output_z = (float ***) malloc(sizeof(float **) * l->outputs_dim[0])) == NULL){
        goto freeall;      
    }
    for(int channel = 0; channel < l->outputs_dim[0]; channel++){
        if((l->output_z[channel] = (float **) malloc(sizeof(float *) * l->outputs_dim[1])) == NULL){
            goto freeall; 
        }
        for(int row = 0; row < l->outputs_dim[1]; row++){
            if((l->output_z[channel][row] = (float *) malloc(sizeof(float) * l->outputs_dim[2])) == NULL){
            goto freeall; 
            }
            memset(l->output_z[channel][row], 0, sizeof(float) * l->outputs_dim[2]);
        }
    }

    

   /* change in z (dZ) */
    if((l->dZ = (float ***) malloc(sizeof(float **) * l->outputs_dim[0])) == NULL){
        goto freeall;      
    }
    for(int channel = 0; channel < l->outputs_dim[0]; channel++){
        if((l->dZ[channel] = (float **) malloc(sizeof(float *) * l->outputs_dim[1])) == NULL){
            goto freeall; 
        }
        for(int row = 0; row < l->outputs_dim[1]; row++){
            if((l->dZ[channel][row] = (float *) malloc(sizeof(float) * l->outputs_dim[2])) == NULL){
            goto freeall; 
            }
            memset(l->dZ[channel][row], 0, sizeof(float) * l->outputs_dim[2]);
        }
    }


    

    /* Output a */
    if((l->output_a = (float ***) malloc(sizeof(float **) * l->outputs_dim[0])) == NULL){
        goto freeall;      
    }
    for(int channel = 0; channel < l->outputs_dim[0]; channel++){
        if((l->output_a[channel] = (float **) malloc(sizeof(float *) * l->outputs_dim[1])) == NULL){
            goto freeall; 
        }
        for(int row = 0; row < l->outputs_dim[1]; row++){
            if((l->output_a[channel][row] = (float *) malloc(sizeof(float) * l->outputs_dim[2])) == NULL){
            goto freeall; 
            }
            memset(l->output_a[channel][row], 0, sizeof(float) * l->outputs_dim[2]);
        }
    }


    /* Activation function */
    l->activ = function;

    return 0;


    freeall:
        if(l->kernels)
        for(int kernel_num = 0; kernel_num < l->kernels_dim[0]; kernel_num++){
            for(int row = 0; row < l->kernels_dim[1]; row++){
                free(l->kernels[kernel_num][row]);
            }
            free(l->kernels[kernel_num]);
        }
        free(l->kernels);

        free(l->biases);


        if(l->output_z)
        for(int channel = 0; channel < l->outputs_dim[0]; channel++){
            for(int row = 0; row < l->outputs_dim[1]; row++){
                free(l->output_z[channel][row]);
            }
            free(l->output_z[channel]);
        }
        free(l->output_z);

        if(l->dZ)
        for(int channel = 0; channel < l->outputs_dim[0]; channel++){
            for(int row = 0; row < l->outputs_dim[1]; row++){
                free(l->dZ[channel][row]);
            }
            free(l->dZ[channel]);
        }
        free(l->dZ);


        if(l->output_a)
        for(int channel = 0; channel < l->outputs_dim[0]; channel++){
            for(int row = 0; row < l->outputs_dim[1]; row++){
                free(l->output_a[channel][row]);
            }
            free(l->output_a[channel]);
        }
        free(l->output_a);

        return -1;


}


int initLayer_conv_input(Input_Layer_Conv *l, int channels, int row, int col){

    l->inputs_dim[0] = channels;
    l->inputs_dim[1] = row;
    l->inputs_dim[2] = col;

    if((l->inputs = (float ***) malloc(sizeof(float **) * l->inputs_dim[0])) == NULL){
        goto freeall;      
    }
    for(int channel = 0; channel < l->inputs_dim[0]; channel++){
        if((l->inputs[channel] = (float **) malloc(sizeof(float *) * l->inputs_dim[1])) == NULL){
            goto freeall; 
        }
        for(int row = 0; row < l->inputs_dim[1]; row++){
            if((l->inputs[channel][row] = (float *) malloc(sizeof(float) * l->inputs_dim[2])) == NULL){
            goto freeall; 
            }
            memset(l->inputs[channel][row], 0, sizeof(float) * l->inputs_dim[2]);

            for(int col = 0; col < l->inputs_dim[1]; col++){
                l->inputs[channel][row][col] = (float)rand()/(float)(RAND_MAX) * ((rand()&0x1)? 1:-1);
            }
        }
    }

    return 0;

    freeall:
        if(l->inputs)
        for(int kernel_num = 0; kernel_num < l->inputs_dim[0]; kernel_num++){
            for(int row = 0; row < l->inputs_dim[1]; row++){
                free(l->inputs[kernel_num][row]);
            }
            free(l->inputs[kernel_num]);
        }
        free(l->inputs);

        return -1;

}


int initLayer_pool(Layer_Pool *l, int prev_layer_channels, int prev_layer_row, int prev_layer_col,
                     int kernel_row, int kernel_col, int stride, Pooling function){

    l->kernels_dim[0] = prev_layer_channels;
    l->kernels_dim[1] = kernel_row;
    l->kernels_dim[2] = kernel_col;

    l->stride = stride;

    l->outputs_dim[0] = prev_layer_channels;
    l->outputs_dim[1] = ((prev_layer_row - kernel_row)/stride) +1; // Integar division Floors this value
    l->outputs_dim[2] = ((prev_layer_col - kernel_col)/stride) +1;
    

    /* Output a */
    if((l->output = (float ***) malloc(sizeof(float **) * l->outputs_dim[0])) == NULL){
        goto freeall;      
    }
    for(int channel = 0; channel < l->outputs_dim[0]; channel++){
        if((l->output[channel] = (float **) malloc(sizeof(float *) * l->outputs_dim[1])) == NULL){
            goto freeall; 
        }
        for(int row = 0; row < l->outputs_dim[1]; row++){
            if((l->output[channel][row] = (float *) malloc(sizeof(float) * l->outputs_dim[2])) == NULL){
            goto freeall; 
            }
            memset(l->output[channel][row], 0, sizeof(float) * l->outputs_dim[2]);
        }
    }


    /* Activation function */
    l->pool = function;

    return 0;


    freeall:
        if(l->output)
        for(int channel = 0; channel < l->outputs_dim[0]; channel++){
            for(int row = 0; row < l->outputs_dim[1]; row++){
                free(l->output[channel][row]);
            }
            free(l->output[channel]);
        }
        free(l->output);

        return -1;


}

/*
void layer_conv_to_csv(Layer_Conv* layer, char* filename) {
    FILE* fp = fopen(filename, "a");
    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    // Write the header row
    fprintf(fp, "weights_dim,outputs_dim,weights,biases,output_z,dZ,output_a\n");

    // Write the dimensions of the weights arrays
    fprintf(fp, "%d,%d\n", layer->weights_dim[0], layer->weights_dim[1]);

    // Write the dimensions of the outputs arrays
    fprintf(fp, "%d,%d\n",layer->outputs_dim[0], layer->outputs_dim[1]);

    // Write the weights
    for (int i = 0; i < layer->weights_dim[0]; i++) {
        for (int j = 0; j < layer->weights_dim[1]; j++) {
            fprintf(fp, "%f,", layer->weights[i][j]);
        }
        fprintf(fp, "\n");
    }

    // Write the biases
    for (int i = 0; i < layer->weights_dim[1]; i++) {
        fprintf(fp, "%f,", layer->biases[i]);
    }
    fprintf(fp, "\n");


    // Write the output_z
    for (int i = 0; i < layer->outputs_dim[0]; i++) {
        for (int j = 0; j < layer->outputs_dim[1]; j++) {
            fprintf(fp, "%f,", layer->output_z[i][j]);
        }
        fprintf(fp, "\n");
    }

    // Write the dZ
    for (int i = 0; i < layer->outputs_dim[0]; i++) {
        for (int j = 0; j < layer->outputs_dim[1]; j++) {
            fprintf(fp, "%f,", layer->dZ[i][j]);
        }
        fprintf(fp, "\n");
    }

    // Write the output_a
    for (int i = 0; i < layer->outputs_dim[0]; i++) {
        for (int j = 0; j < layer->outputs_dim[1]; j++) {
            fprintf(fp, "%f,", layer->output_a[i][j]);
        }
        fprintf(fp, "\n");
    }

    if (layer->activ == ReLU) {
        fprintf(fp, "ReLU\n"); 
    } else {
        fprintf(fp, "NULL\n");
    }


    fclose(fp);
}


void readLayerFromCSV(Layer_Dense ***layers, char *filename) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    if((*layers = (Layer_Dense **) malloc(sizeof(Layer_Dense *))) == NULL) {
        printf("Could not allocate Memmory for Layer\n");
        exit(1);
    }


    int num_layers = 1;
    char header[100];


    while (fgets(header, sizeof(header), fp) != NULL) {

        // Check if it's a new layer
        if (strstr(header, "weights_dim") != NULL) {
            if( (*layers = realloc(*layers, num_layers * sizeof(Layer_Dense *))) == NULL){
                printf("Could not reallocate Memmory for Layer\n");
                exit(1);
            }


            int prev_layer_size, layer_size, batch_size;

            // Read weights_dim
            fscanf(fp, "%d,%d\n", &prev_layer_size, &layer_size);

            // Read outputs_dim
            fscanf(fp, "%d,%d\n", &batch_size, &layer_size);

            (*layers)[num_layers - 1] = (Layer_Dense *) malloc(sizeof(Layer_Dense));
            // Initialize current_layer
            if(initLayer((*layers)[num_layers - 1], prev_layer_size, layer_size, batch_size, NULL) < 0)
            {
                perror("Failed to initialise layer");
                exit(EXIT_FAILURE);
            }

            Layer_Dense *current_layer = (*layers)[num_layers - 1];


            // Read weights
            for (int j = 0; j < current_layer->weights_dim[0]; j++) {
                for (int k = 0; k < current_layer->weights_dim[1]; k++) {
                    fscanf(fp, "%f,", &(current_layer->weights[j][k]));
                }
                fscanf(fp, "\n");
            }


            // Read biases
            for (int j = 0; j < current_layer->weights_dim[1]; j++) {
                fscanf(fp, "%f,", &(current_layer->biases[j]));
            }
            fscanf(fp, "\n");


            // Read output_z
            for (int j = 0; j < current_layer->outputs_dim[0]; j++) {
                for (int k = 0; k < current_layer->outputs_dim[1]; k++) {
                    fscanf(fp, "%f,", &(current_layer->output_z[j][k]));
                }
                fscanf(fp, "\n");
            }


            // Read dZ
            for (int j = 0; j < current_layer->outputs_dim[0]; j++) {
                for (int k = 0; k < current_layer->outputs_dim[1]; k++) {
                    fscanf(fp, "%f,", &(current_layer->dZ[j][k]));
                }
                fscanf(fp, "\n");
            }


            // Read output_a
            for (int j = 0; j < current_layer->outputs_dim[0]; j++) {
                for (int k = 0; k < current_layer->outputs_dim[1]; k++) {
                    fscanf(fp, "%f,", &(current_layer->output_a[j][k]));
                }
                fscanf(fp, "\n");
            }

            // Read activation function
            char activation[10];
            fscanf(fp, "%s\n", activation);
            if (strcmp(activation, "ReLU") == 0) {
                current_layer->activ = ReLU;
            } else {
                current_layer->activ = NULL;
            }


            // Reallocate memory for layers
            num_layers++;
        }
    }

    fclose(fp);
}
*/
