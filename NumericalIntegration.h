#ifndef NUMERICALINTEGRATION_H
#define NUMERICALINTEGRATION_H

namespace kalmanfilter {

enum IntegrationMode {
    EULER,
    RK4
};

/** Numerical integrator class template
 * \param Functor function to be integrated
 * \param ValueType input and output type of the functor
 * \param Scalar scalar type, double or float
 * \param IntegrationMode choose EULER or RK4 method
 */
template<typename Functor,
         typename ValueType,
         typename Scalar,
         IntegrationMode mode = EULER>
class NumericalIntegration {
private:
    const Scalar h;
    Functor f;

    // Explicit Euler method
    inline void Euler(Scalar time_span, const ValueType &x0, ValueType &xout)
    {
        ValueType dx;
        ValueType &x = xout;
        for (x = x0; time_span > 0; time_span -= h) {
            dx = f(x);
            x += h * dx;
        }
    }

    // 4th order Runge-Kutta (RK4) method
    inline void RungeKutta4(Scalar time_span, const ValueType &x0, ValueType &xout)
    {
        ValueType k1, k2, k3, k4;
        ValueType &x = xout;
        for (x = x0; time_span > 0; time_span -= h) {
            k1 = f(x);
            k2 = f(x + h / 2 * k1);
            k3 = f(x + h / 2 * k2);
            k4 = f(x + h  * k3);
            x += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
        }
    }

public:
    /** Numerical integrator constructor
     * \param time_step integration step size
     * \note In case of Runke-Kutta method the intermediate step size is half of
             time_step size.
     */
    NumericalIntegration(Scalar time_step) : h(time_step) {}

    /** Integrate given functor
     * \param time_span integration time span
     * \param x0 initial value
     * \param xout integration result
     */
    void integrate(Scalar time_span, const ValueType &x0, ValueType &xout)
    {
        switch (mode) {
            case EULER:
                Euler(time_span, x0, xout);
                break;

            case RK4:
                RungeKutta4(time_span, x0, xout);
                break;

            default:
                // XXX TODO: error handling
                break;
        }
    }
};

} // end namespace kalmanfilter

#endif // NUMERICALINTEGRATION_H
