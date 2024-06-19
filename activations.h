#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

typedef float (*Activation) (float value);

float ReLU(float x)__attribute__((unused));
float ReLU_Derivative(float x)__attribute__((unused));

#endif
