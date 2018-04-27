#ifndef EXTENDED_KALMAN_FILTER_H
#define EXTENDED_KALMAN_FILTER_H

#include <Eigen/Dense>
#include "NumericalIntegration.h"

namespace kalmanfilter {

enum DiffMode {
    ANALYTIC,
    AUTOMATIC, // not supported
    NUMERIC    // not supported
};

template <typename Dynamics,
          typename Observation,
          IntegrationMode int_mode = EULER,
          DiffMode diff_mode = ANALYTIC>
class ExtendedKalmanFilter {
public:
    using Scalar = typename Dynamics::State::Scalar;
    using State = typename Dynamics::State;
    using Control = typename Dynamics::Control;
    using Measurement = typename Observation::Measurement;
    using StateCov = Eigen::Matrix<Scalar, State::RowsAtCompileTime, State::RowsAtCompileTime>;
    enum {
        nx = State::RowsAtCompileTime,
        nz = Measurement::RowsAtCompileTime,
    };

private:
    static_assert(diff_mode == ANALYTIC, "Unsupported DiffMode");
    NumericalIntegration<Dynamics, int_mode> f;
    Observation h;
    State x; // state vector
    StateCov P; // state covariance
    Eigen::Matrix<Scalar, nz, nz> S; // innovation covariance
    Eigen::Matrix<Scalar, nx, nz> K; // Kalman gain
    Eigen::Matrix<Scalar, nx, nx> I; // identity

public:
    ExtendedKalmanFilter(State x0, StateCov P0) : x(x0), P(P0)
    {
        I.setIdentity();
    }

    State get_state()
    {
        return x;
    }

    StateCov get_covariance()
    {
        return P;
    }

    void predict(const Control &u, Scalar delta_t, unsigned integration_steps = 1)
    {
        Eigen::Matrix<Scalar, nx, nx> A;

        A = I + delta_t * f.jacobian(x, u);

        x = f.integrate(delta_t, x, u, integration_steps);
        P = A * P * A.transpose() + f.Q;
    }

    void correct(const Measurement &z)
    {
        Measurement y;                      // innovation
        Eigen::Matrix<Scalar, nz, nx> H;    // jacobian of h
        Eigen::Matrix<Scalar, nx, nx> IKH;  // temporary matrix

        H = h.jacobian(x);
        S = H * P * H.transpose() + h.R;

        // efficiently compute: K = P * H.transpose() * S.inverse();
        K = S.llt().solve(H * P).transpose();

        y = z - h(x);
        IKH = (I - K * H);

        // measurement update
        x = x + K * y;
        P = IKH * P * IKH.transpose() + K * h.R * K.transpose();
    }

    State update(const Control &u,
                 const Measurement &z,
                 Scalar delta_t,
                 unsigned integration_steps = 1)
    {
        predict(u, delta_t, integration_steps);
        correct(z);
        return x;
    }
};

} // end namespace kalmanfilter

#endif /* EXTENDED_KALMAN_FILTER_H */
