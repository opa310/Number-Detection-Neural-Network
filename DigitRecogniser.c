#include "dense.Layer.h"


#define BATCH_SIZE 10 // Adjust batch size as needed
#define MAXCHAR 30000  // Define maximum characters for file reading
#define LEARNING_RATE 0.0076 // Learning rate for training


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

    // Wait for user to close window
    SDL_Event event;
    while (SDL_WaitEvent(&event) && event.type != SDL_QUIT) {
        
    }

    // Cleanup
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
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
    Layer_Dense Input, l1, output;

    if (initLayer(&Input, 0, 784, BATCH_SIZE, ReLU) < 0 ||
        initLayer(&l1, 784, 10, BATCH_SIZE, ReLU) < 0 ||
        initLayer(&output, 10, 10, BATCH_SIZE, NULL) < 0)
    {
        perror("Failed to initialise layer");
        exit(EXIT_FAILURE);
    }


    FILE *fp, *test_output;

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
                float user_data[Input.outputs_dim[1]]; 

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
                for (int i = 0; i < Input.outputs_dim[1]; i++)
                {
                    token = strtok(NULL, ",");
                    if (token == NULL)
                    {
                        printf("Error parsing line\n");
                        break;
                    }
                    user_data[i] = atof(token); 
                }

                // Populate input 
                for (int j = 0; j < Input.outputs_dim[1]; j++)
                {
                    Input.output_a[batch_count][j] = user_data[j] / 255; 
                }

                batch_count++;
                processed_lines++;
            }

            forward_pass(&Input, &l1);
            forward_pass(&l1, &output);

            calc_accuracy(&output, expected);

            // Backpropagation
            backward_pass(&l1, &output, NULL, expected, LEARNING_RATE);
            backward_pass(&Input, &l1, &output, NULL, LEARNING_RATE);

            // Updates progress bar
            printProgressBar(processed_lines, total_lines);
        }
        printf("\n");
    }


    fclose(fp);

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
            Input.output_a[0][j] = (float)user_data[j] / 255; 
        }

        forward_pass(&Input, &l1);
        forward_pass(&l1, &output);

        // Get the predicted digit (index of maximum output)
        int predicted_digit = 0;
        float max_value = output.output_a[0][0];
        for (int i = 1; i < output.outputs_dim[1]; i++)
        {
            if (output.output_a[0][i] > max_value)
            {
                max_value = output.output_a[0][i];
                predicted_digit = i;
            }
        }

        fprintf(test_output, "%d,%d\n", line_idx, predicted_digit);
                
        }
        line_idx++;

    }

    fclose(test_output);
    printf("Testing complete\n");



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
            Input.output_a[0][j] = (float)user_data[j] / 255; 
        }

        forward_pass(&Input, &l1);
        forward_pass(&l1, &output);

        // Get the predicted digit (index of maximum output)
        int predicted_digit = 0;
        float max_value = output.output_a[0][0];
        for (int i = 1; i < output.outputs_dim[1]; i++)
        {
            if (output.output_a[0][i] > max_value)
            {
                max_value = output.output_a[0][i];
                predicted_digit = i;
            }
        }

        printf("Predicted Digit: %d\n", predicted_digit);

        render_image(user_data);


    }

    fclose(fp);

    return 0;
}

