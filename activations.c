#include <activations.h>

#define LEAKY_RELU_ALPHA 0.01

inline float ReLU(float x) {
    return (x < 0) ? 0 : x;
}

inline float ReLU_Derivative(float x) {
    return (x > 0) ? 1 : 0;
}

inline float Leaky_ReLU(float x) {
    return (x >= 0) ? x : LEAKY_RELU_ALPHA * x;
}

inline float Leaky_ReLU_Derivative(float x) {
    return (x >= 0) ? 1 : LEAKY_RELU_ALPHA;
}
