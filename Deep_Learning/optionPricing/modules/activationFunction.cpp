class ActivationFunction
{   
public:
    static double heaviside(double x) {
        return (x >= 0.0) ? 1.0 : 0.0;
    }
    static double heaviside_derivative(double x) {
        return 1.0; 
    }

    static double relu(double x) {
        return x > 0 ? x : 0;
    }
    static double relu_derivative(double x) {
        return (x > 0) ? 1.0 : 0.0;
    }

    static double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    static double sigmoid_derivative(double activated_value) {
        return activated_value * (1.0 - activated_value);
    } 
};