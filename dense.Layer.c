#include "dense.Layer.h"



void printLayer(Layer_Dense *l){
    printf("\nDense Layer");

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
    printf("\n\ndZ gradient :\n"); 
    for(int row = 0; row < l->outputs_dim[0]; row++){
        for(int col = 0; col < l->outputs_dim[1]; col++){
            printf("%0.7f, ", l->dZ[row][col]);
        }
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
        printf("SoftMax\n");
    } else if(l->activ == ReLU){
        printf("ReLU\n");
    }  else if (l->activ == Leaky_ReLU){
        printf("Leaky ReLU\n");
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
    /* Zeroed Biases*/
    memset(l->biases, 0, sizeof(float) * layer_size);

    /* Random Biases 
    for(int col = 0; col < layer_size; col++){
       l->biases[col] = (float)rand()/(float)(RAND_MAX) * ((rand()&0x1)? 1:-1);
    }*/

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

   /* change in z (dZ) */
    if((l->dZ = (float **) malloc(sizeof(float *) * batch_size)) == NULL){
        goto freeall; 
    }

    for(int row = 0; row < batch_size; row++){
        if((l->dZ[row] = (float *) malloc(sizeof(float) * layer_size)) == NULL){
            goto freeall; 
        }
        memset(l->dZ[row], 0, sizeof(float) * layer_size);
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

    if(function == ReLU){
            l->activ_deriv = ReLU_Derivative;
    }else if(function == Leaky_ReLU){
            l->activ_deriv = Leaky_ReLU_Derivative;
    } else {
            l->activ_deriv = NULL;
    }

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

        if(l->dZ)
        for(int row = 0; row < l->outputs_dim[0]; row++){
            free(l->dZ[row]);
        }
        free(l->dZ);


        if(l->output_a)
        for(int row = 0; row < l->outputs_dim[0]; row++){
            free(l->output_a[row]);
        }
        free(l->output_a);

        return -1;


}


void layer_dense_to_csv(Layer_Dense* layer, char* filename) {
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
    } else if (layer->activ == Leaky_ReLU){
        fprintf(fp, "LeakyReLU\n");
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
            }else if(strcmp(activation, "LeakyReLU") == 0){
                current_layer->activ = Leaky_ReLU;
            } else {
                current_layer->activ = NULL;
            }


            // Reallocate memory for layers
            num_layers++;
        }
    }

    fclose(fp);
}
