#include "gtest/gtest.h"
#include "NumericalIntegration.h"
#include "Eigen/Dense"

using namespace KalmanFilter;

struct ZeroDynamics
{
    void operator()(const double &x, double &dx) const {dx = 0.0;}
};

TEST(EulerTest, TestZeroDynamics) {
    NumericalIntegration<ZeroDynamics, double, double, Euler> ni(1e-3);
    double res;
    const double x0 = 42;
    ni.integrate(1, x0, res);
    ASSERT_EQ(x0, res);
}

struct ConstDynamics
{
    void operator()(const double &x, double &dx) const {dx = 21.0;}
};

TEST(EulerTest, TestConstDynamics) {
    NumericalIntegration<ConstDynamics, double, double, Euler> ni(1e-3);
    double res;
    const double x0 = 0;
    ni.integrate(2, x0, res);
    EXPECT_NEAR(42.0, res, 0.1);
}

using Vec2d = Eigen::Vector2d;
struct FreeFall
{
    void operator()(const Vec2d &x, Vec2d &dx) const {
        dx[0] = x[1];
        dx[1] = -9.81;
    }
};

TEST(EulerTest, TestMultivariateIntegration) {
    NumericalIntegration<FreeFall, Vec2d, double, Euler> ni(1e-3);
    Vec2d res;
    const Vec2d x0(0, 0);
    double delta_t = 3.0;
    ni.integrate(3, x0, res);
    EXPECT_NEAR(-0.5*9.81*delta_t*delta_t, res[0], 1.0);
    EXPECT_NEAR(-9.81*delta_t, res[1], 1.0);
}
