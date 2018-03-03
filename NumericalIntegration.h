#ifndef NUMERICALINTEGRATION_H
#define NUMERICALINTEGRATION_H

namespace KalmanFilter {

enum IntegrationMode {
    Euler,
    RK4
};

template<typename Functor,
         typename ValueType,
         typename Scalar,
         Scalar TimeStep,
         IntegrationMode mode = Euler>
class NumericalIntegration {
private:
    Functor f;

    inline void EulerForward(Scalar tspan, const ValueType &x0, ValueType &x)
    {
        ValueType dx;
        const Scalar h = TimeStep;
        for (x = x0; tspan > 0; tspan -= h) {
            // Euler forward
            f(x, dx);
            x += h * dx;
        }
    }

    inline void RK4(Scalar tspan, const ValueType &x0, ValueType &x)
    {
        ValueType k1, k2, k3, k4;
        const Scalar h = TimeStep;
        for (; tspan > 0; tspan -= h) {
            // classical Runge-Kutta (RK4)
            f(x, k1);
            f(x + h / 2 * k1, k2);
            f(x + h / 2 * k2, k3);
            f(x + h  * k3, k4);
            x += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
        }
    }

public:
    void integrate(Scalar tspan, const ValueType &x0, ValueType &x)
    {
        switch (mode) {
            case Euler:
                EulerForward(tspan, x0, x);
                break;

            case RK4:
                RK4(tspan, x0, x);
                break;

            default:
                // XXX TODO: error handling
                break;
        }
    }
};

} // end namespace KalmanFilter

#endif // NUMERICALINTEGRATION_H
