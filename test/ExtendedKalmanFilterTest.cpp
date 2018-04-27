#include "gtest/gtest.h"
#include "ExtendedKalmanFilter.h"
#include "Eigen/Dense"

using namespace kalmanfilter;


struct Dynamics {
    // mandatory fields
    using Scalar = double;
    using State = Eigen::Matrix<double, 2, 1>;
    using Control = double;
    Eigen::Matrix2d Q = (Eigen::Matrix2d() << 0.01,0.0,0.0,0.01).finished();

    State operator()(const State &x, const Control &u)
    {
        State xd;
        xd.setZero();
        return xd;
    }

    Eigen::Matrix2d jacobian(const State &x, const Control &u)
    {
        Eigen::Matrix2d F;
        F.setIdentity();
        return F;
    }
};

struct Observation {
    // mandatory fields
    using Measurement = Eigen::Matrix<double, 2, 1>;
    Eigen::Matrix2d R = (Eigen::Matrix2d() << 0.01,0.0,0.0,0.01).finished();

    Measurement operator()(const Dynamics::State &x)
    {
        return x;
    }

    Eigen::Matrix2d jacobian(const Dynamics::State &x)
    {
        Eigen::Matrix2d H;
        H.setIdentity();
        return H;
    }
};

using EKF = ExtendedKalmanFilter<Dynamics, Observation, EULER, ANALYTIC>;

TEST(EKFTestCase, TestInitialization) {
    EKF::StateCov P;
    P.setZero();
    P.diagonal() << 0.1, 0.1;
    EKF::State x;
    x << 0, 0;
    EKF ekf(x, P);
    ASSERT_EQ(x, ekf.get_state());
    ASSERT_EQ(P, ekf.get_covariance());
}
