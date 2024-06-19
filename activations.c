#include <activations.h>

inline float ReLU(float x){
    return (x<0) ? 0: x;
}

inline float ReLU_Derivative(float x){
    return (float) x > 0;
}
