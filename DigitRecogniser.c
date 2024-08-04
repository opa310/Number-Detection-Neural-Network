#include "dense.Layer.h"
#include "conv.Layer.h"


#define BATCH_SIZE 10 // Adjust batch size as needed
#define MAXCHAR 30000  // Define maximum characters for file reading
#define LEARNING_RATE 0.0097 // Learning rate for training


#include <SDL2/SDL.h>



void render_image(const unsigned char *pixels) {

    const int width = 28;
    const int height = 28;
    const int pixel_size = 10;

    SDL_Window *window = NULL;
    SDL_Renderer *renderer = NULL;

    SDL_Init(SDL_INIT_VIDEO);

    // Create a window
    window = SDL_CreateWindow("Image Renderer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width * pixel_size, height * pixel_size, 0);
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    // Render the image
    SDL_Rect rect;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            rect.x = x * pixel_size;
            rect.y = y * pixel_size;
            rect.w = pixel_size;
            rect.h = pixel_size;

            // Get grayscale value
            unsigned char value = pixels[y * width + x];

            // Set color based on grayscale value
            SDL_SetRenderDrawColor(renderer, value, value, value, 255);

            // Fill pixel square
            SDL_RenderFillRect(renderer, &rect);
        }
    }

    // Render to screen
    SDL_RenderPresent(renderer);

    // Event handling loop
    SDL_Event event;

    while (SDL_WaitEvent(&event)) {
        if (event.type == SDL_QUIT) {
            break;
        }
    }

    // Cleanup

    SDL_DestroyWindow(window);
    SDL_DestroyRenderer(renderer);
    SDL_Quit();
}


void calc_accuracy(Layer_Dense *output, int *expected){


    for (int i = 0; i < output->outputs_dim[0]; i++) {
        // Find the index of the maximum predicted value
        int max_index = 0;
        float max_value = output->output_a[i][0];
        for (int j = 1; j < output->outputs_dim[0]; j++) {
            if (output->output_a[i][j] > max_value) {
                max_value = output->output_a[i][j];
                max_index = j;
            }
        }

        // Check if the predicted class matches the expected class
        if (max_index == expected[i]) {
            correct++;
        }
        total++;
    }

}


void printOutputAndExpected(Layer_Dense *output, int *expected)
{
    printf("\nOutput Layer\n");
    printLayer(output);

    printf("\nExpected Output\n");
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        printf("%d\n", expected[i]);
    }
}

void printProgressBar(int progress, int total)
{
    const int barWidth = 70;
    float percent = (float)progress / total;
    int pos = barWidth * percent;

    printf("[");
    for (int i = 0; i < barWidth; ++i)
    {
        if (i < pos)
            printf("=");
        else if (i == pos)
            printf(">");
        else
            printf(" ");
    }
    printf("] %.2f%% Complete, %.2f%% Accuracy\r", percent * 100, (correct/total) * 100);
    fflush(stdout);
}

int main(void)
{

    Input_Layer_Conv input_conv;
    Layer_Conv conv0;
    Layer_Pool pool0;
    Layer_Dense dummy, Input, output;

    if (initLayer_conv_input(&input_conv, BATCH_SIZE, 1, 28, 28) < 0||
        initLayer_conv(&conv0, 28, 28, 8, 3, 3, 1, Leaky_ReLU) < 0 ||
        initLayer_pool(&pool0, conv0.outputs_dim[0], conv0.outputs_dim[1], conv0.outputs_dim[2], 2, 2, 2, Avg_Pooling) < 0||
        initLayer(&dummy, 0, 0, 1, ReLU) < 0 ||
        initLayer(&Input, 0, pool0.outputs_dim[0] * pool0.outputs_dim[1] * pool0.outputs_dim[2], BATCH_SIZE, ReLU) < 0 ||
        initLayer(&output, pool0.outputs_dim[0] * pool0.outputs_dim[1] * pool0.outputs_dim[2], 10, BATCH_SIZE, NULL) < 0)
    {
        perror("Failed to initialise layers");
        exit(EXIT_FAILURE);
    }


    FILE *fp, *test_output;
    Layer_Dense** model;

    while(1){

        char task[7];
        printf("\nEnter \"train\" to train the model, \"gen\" to generate the testing output, and \"test\" to do predictions (use q to quit): ");

        if (scanf("%s", task) != 0)
        {
            if (strcmp(task,"train") == 0)
            {
                goto TRAIN;
            } else if (strcmp(task,"gen") == 0)
            {
                goto GEN;
            } else if (strcmp(task,"test") == 0){
                goto TEST;
            } else if (strcmp(task,"q") == 0){
                exit(0);
            } else {
                printf("Invalid Input\n");
            }
        }
    }

TRAIN:


    if ((fp = fopen("digit-recognizer/train.csv", "r")) == NULL)
    {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    // Count the total number of lines in the file
    int total_lines = 0;
    char ch;
    while ((ch = fgetc(fp)) != EOF)
    {
        if (ch == '\n')
            total_lines++;
    }

    rewind(fp); // Reset file pointer to beginning

    // Train the model
    printf("Training the model...\n");
    int epochs = 30; // Number of epochs for training 
    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        correct = 0;
        total = 0;


        printf("Epoch %d/%d\n", epoch, epochs);
        int processed_lines = 0;
        int expected[BATCH_SIZE];

        rewind(fp); // Reset file pointer to beginning
        while (processed_lines < total_lines)
        {
            // Process a batch of lines
            int batch_count = 0;
            while (batch_count < BATCH_SIZE && !feof(fp))
            {
                // Read the line
                char row[MAXCHAR];
                if (fgets(row, MAXCHAR, fp) == NULL)
                    break;

                // Process the line data
                unsigned char user_data[784]; 

                // Extract expected value and input data using strtok
                char *token = strtok(row, ",");
                if (token == NULL)
                {
                    printf("Error parsing line\n");
                    break;
                }

                // Convert first token to integer
                // This is the expected output idx
                expected[batch_count] = atoi(token); 
                for (int i = 0; i < 784; i++)
                {
                    token = strtok(NULL, ",");
                    if (token == NULL)
                    {
                        printf("Error parsing line\n");
                        break;
                    }
                    user_data[i] = atof(token);
                }

                //render_image(user_data);

                // Populate input 
                for (int j = 0; j < 784; j++)
                {
                    input_conv.inputs[batch_count][0][j/28][j%28] = (float) user_data[j] / 255; 
                }


                //printLayer_conv_input(&input_conv);

                forward_pass_conv(&input_conv.inputs_dim[1], input_conv.inputs[batch_count], &conv0);
                //printLayer_conv(&conv0);

                forward_pass_pool(conv0.outputs_dim, conv0.output_a, &pool0);
                //printLayer_pool(&pool0);

                flatten_pool_to_dense(&pool0, &Input, batch_count);
                //printLayer(&Input);

                batch_count++;
                processed_lines++;
            }

            forward_pass(&Input, &output);

            calc_accuracy(&output, expected);
            //printOutputAndExpected(&output, expected);
            //printLayer(&output);

            // Backpropagation
            backward_pass(&Input, &output, NULL, expected, LEARNING_RATE);
            backward_pass(&dummy, &Input, &output, NULL, LEARNING_RATE);

            //printLayer(&output);

            for(int dense_batch = 0; dense_batch < BATCH_SIZE; dense_batch++){
                unflatten_dense_to_pool(&Input, &pool0, dense_batch);
                //printLayer(&Input);
                //printLayer_pool(&pool0);
                backward_pass_conv(&input_conv.inputs_dim[1], input_conv.inputs[dense_batch], NULL, &conv0, &pool0, 0.00001/*LEARNING_RATE / BATCH_SIZE*/);
            }

            //printLayer_conv(&conv0);
            


            // Updates progress bar
            printProgressBar(processed_lines, total_lines);
        }
        printf("\n");
    }


    fclose(fp);






    // Store model into csv 
    int result = remove("Digit-Recogniser.csv");

    if (result == 0) {
        printf("File \"%s\" deleted successfully.\n", "Digit-Recogniser.csv");
    } else {
        // Protects for case when file does not exist
        if (errno != ENOENT) {
            perror("Error deleting file");
        }
    }

    layer_dense_to_csv(&Input, "Digit-Recogniser.csv");
    layer_dense_to_csv(&output, "Digit-Recogniser.csv");


GEN:


    readLayerFromCSV(&model, "Digit-Recogniser.csv");

    //printf("Got here 1\n");
    printLayer(model[0]);
    //printf("Got here 2\n");
    printLayer(model[1]);



    if ((fp = fopen("digit-recognizer/test.csv", "r")) == NULL)
    {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }


    // Testing phase
    printf("\nGenerating output on testing dataset ...\n");

    if ((test_output = fopen("digit-recognizer/test_output.csv", "w")) == NULL)
    {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }
    fprintf(test_output, "ImageId,Label\n");

    char row[MAXCHAR];
    int line_idx = 0;
    while (fgets(row, MAXCHAR, fp) != NULL)
    {
        if(line_idx != 0){
            // Process the line data
            unsigned char user_data[Input.outputs_dim[1]]; 


            char *token = strtok(row, ",");

            for (int i = 0; i < Input.outputs_dim[1]; i++)
            {

                if (token == NULL)
                {
                    printf("Error parsing line\n");
                    break;
                }

                user_data[i] = atof(token);

                token = strtok(NULL, ",");
            }


            // Populate input
            for (int j = 0; j < Input.outputs_dim[1]; j++)
            {
                model[0]->output_a[0][j] = (float)user_data[j] / 255; 
            }

            forward_pass(model[0], model[1]);


            // Get the predicted digit (index of maximum output)
            int predicted_digit = 0;
            float max_value = model[1]->output_a[0][0];
            for (int i = 1; i < model[1]->outputs_dim[1]; i++)
            {
                if (model[1]->output_a[0][i] > max_value)
                {
                    max_value = model[1]->output_a[0][i];
                    predicted_digit = i;
                }
            }

            fprintf(test_output, "%d,%d\n", line_idx, predicted_digit);

        }
        line_idx++;

    }

    fclose(test_output);
    printf("Testing complete\n");

TEST:
    readLayerFromCSV(&model, "Digit-Recogniser.csv");

    if ((fp = fopen("digit-recognizer/test.csv", "r")) == NULL)
    {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    // Prediction phase
    while (1)
    {
        int line_index;
        printf("\nEnter a line index between 1 and 28000 (or 'q' to quit): ");


        if (scanf("%d", &line_index) != 1)
        {
            // Check for quit command
            if (getchar() == 'q')
            {
                break;
            }


            printf("Invalid input. Please enter a number.\n");
            while (getchar() != '\n')
                ; // Clear input buffer
            continue;
        }



        rewind(fp); 
        for (int i = 0; i < line_index; i++)
        {
            char row[MAXCHAR];
            if (fgets(row, MAXCHAR, fp) == NULL)
            {
                printf("Error reading from file\n");
                break;
            } 
        }

        // Read the line
        char row[MAXCHAR];
        if (fgets(row, MAXCHAR, fp) == NULL)
        {
            perror("Error reading from file");
            break;
        }

        // Process the line data
        unsigned char user_data[Input.outputs_dim[1]]; 


        char *token = strtok(row, ",");

        for (int i = 0; i < Input.outputs_dim[1]; i++)
        {

            if (token == NULL)
            {
                printf("Error parsing line\n");
                break;
            }

            user_data[i] = atof(token);

            token = strtok(NULL, ",");
        }

        // Populate input
        for (int j = 0; j < Input.outputs_dim[1]; j++)
        {
            model[0]->output_a[0][j] = (float)user_data[j] / 255; 
        }

        forward_pass(model[0], model[1]);


        // Get the predicted digit (index of maximum output)
        int predicted_digit = 0;
        float max_value = model[1]->output_a[0][0];
        for (int i = 1; i < model[1]->outputs_dim[1]; i++)
        {
            if (model[1]->output_a[0][i] > max_value)
            {
                max_value = model[1]->output_a[0][i];
                predicted_digit = i;
            }
        }

        printf("Predicted Digit: %d\n", predicted_digit);

        render_image(user_data);


    }

    fclose(fp);

    return 0;
}

