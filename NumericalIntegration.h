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
template<typename _Functor, IntegrationMode mode = EULER>
class NumericalIntegration : public _Functor {
public:
    using Functor = _Functor;
    using ValueType = typename Functor::ValueType;
    using Scalar = typename Functor::Scalar;
private:
    Functor f;

    // Explicit Euler method
    inline ValueType Euler(Scalar time_span, Scalar step, const ValueType &x0)
    {
        ValueType x, dx;
        for (x = x0; time_span > 0; time_span -= step) {
            dx = f(x);
            x += step * dx;
        }
        return x;
    }

    // 4th order Runge-Kutta (RK4) method
    inline ValueType RungeKutta4(Scalar time_span, Scalar step, const ValueType &x0)
    {
        ValueType x, k1, k2, k3, k4;
        for (x = x0; time_span > 0; time_span -= step) {
            k1 = f(x);
            k2 = f(x + step / 2 * k1);
            k3 = f(x + step / 2 * k2);
            k4 = f(x + step  * k3);
            x += step / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
        }
        return x;
    }

public:
    /** Integrate given functor
     * \param time_span integration time span
     * \param step integration time step
     * \param x0 initial value
     * \return integration result
     */
    ValueType integrate(Scalar time_span, Scalar step, const ValueType &x0)
    {
        switch (mode) {
            case EULER:
                return Euler(time_span, step, x0);

            case RK4: // fall through
            default:
                return RungeKutta4(time_span, step, x0);
        }
    }
};

} // end namespace kalmanfilter

#endif // NUMERICALINTEGRATION_H
