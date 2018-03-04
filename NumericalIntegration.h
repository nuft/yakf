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
class NumericalIntegration : public Functor {
private:
    const Scalar h;
    Functor f;

    // Explicit Euler method
    inline ValueType Euler(Scalar time_span, const ValueType &x0)
    {
        ValueType x, dx;
        for (x = x0; time_span > 0; time_span -= h) {
            dx = f(x);
            x += h * dx;
        }
        return x;
    }

    // 4th order Runge-Kutta (RK4) method
    inline ValueType RungeKutta4(Scalar time_span, const ValueType &x0)
    {
        ValueType x, k1, k2, k3, k4;
        for (x = x0; time_span > 0; time_span -= h) {
            k1 = f(x);
            k2 = f(x + h / 2 * k1);
            k3 = f(x + h / 2 * k2);
            k4 = f(x + h  * k3);
            x += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
        }
        return x;
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
     * \return integration result
     */
    ValueType integrate(Scalar time_span, const ValueType &x0)
    {
        switch (mode) {
            case EULER:
                return Euler(time_span, x0);

            case RK4: // fall through
            default:
                return RungeKutta4(time_span, x0);
        }
    }
};

} // end namespace kalmanfilter

#endif // NUMERICALINTEGRATION_H
