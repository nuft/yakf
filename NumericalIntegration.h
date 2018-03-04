#ifndef NUMERICALINTEGRATION_H
#define NUMERICALINTEGRATION_H

namespace KalmanFilter {

enum IntegrationMode {
    Euler,
    RungeKutta
};

template<typename Functor,
         typename ValueType,
         typename Scalar,
         IntegrationMode mode = Euler>
class NumericalIntegration {
private:
    const Scalar h;
    Functor f;

    // forward Euler method
    inline void ForwardEuler(Scalar time_span, const ValueType &x0, ValueType &x)
    {
        ValueType dx;
        for (x = x0; time_span > 0; time_span -= h) {
            f(x, dx);
            x += h * dx;
        }
    }

    // 4th order Runge-Kutta (RK4) method
    inline void RK4(Scalar time_span, const ValueType &x0, ValueType &x)
    {
        ValueType k1, k2, k3, k4;
        for (x = x0; time_span > 0; time_span -= h) {
            f(x, k1);
            f(x + h / 2 * k1, k2);
            f(x + h / 2 * k2, k3);
            f(x + h  * k3, k4);
            x += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
        }
    }

public:
    NumericalIntegration(Scalar time_step) : h(time_step) {}

    void integrate(Scalar time_span, const ValueType &x0, ValueType &x)
    {
        switch (mode) {
            case Euler:
                ForwardEuler(time_span, x0, x);
                break;

            case RungeKutta:
                RK4(time_span, x0, x);
                break;

            default:
                // XXX TODO: error handling
                break;
        }
    }
};

} // end namespace KalmanFilter

#endif // NUMERICALINTEGRATION_H
