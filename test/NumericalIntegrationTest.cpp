#include "gtest/gtest.h"
#include "NumericalIntegration.h"
#include "Eigen/Dense"

using namespace kalmanfilter;

struct ZeroDynamics
{
    void operator()(const double &x, double &dx) const {dx = 0.0;}
};

TEST(EulerTestCase, TestZeroDynamics) {
    NumericalIntegration<ZeroDynamics, double, double, EULER> ni(1e-3);
    double res;
    const double x0 = 42;
    ni.integrate(1, x0, res);
    ASSERT_EQ(x0, res);
}

TEST(RungeKuttaTestCase, TestZeroDynamics) {
    NumericalIntegration<ZeroDynamics, double, double, RK4> ni(1e-3);
    double res;
    const double x0 = 42;
    ni.integrate(1, x0, res);
    ASSERT_EQ(x0, res);
}

struct ConstDynamics
{
    void operator()(const double &x, double &dx) const {dx = 21.0;}
};

TEST(EulerTestCase, TestConstDynamics) {
    NumericalIntegration<ConstDynamics, double, double, EULER> ni(1e-3);
    double res;
    const double x0 = 0;
    ni.integrate(2, x0, res);
    EXPECT_NEAR(42.0, res, 0.1);
}

TEST(RungeKuttaTestCase, TestConstDynamics) {
    NumericalIntegration<ConstDynamics, double, double, RK4> ni(1e-3);
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

TEST(EulerTestCase, TestMultivariateIntegration) {
    NumericalIntegration<FreeFall, Vec2d, double, EULER> ni(0.01);
    Vec2d res;
    const Vec2d x0(0, 0);
    double delta_t = 3.0;
    ni.integrate(delta_t, x0, res);
    EXPECT_NEAR(-0.5*9.81*delta_t*delta_t, res[0], 1.0);
    EXPECT_NEAR(-9.81*delta_t, res[1], 1.0);
}

TEST(RungeKuttaTestCase, TestMultivariateIntegration) {
    NumericalIntegration<FreeFall, Vec2d, double, RK4> ni(0.1);
    Vec2d res;
    const Vec2d x0(0, 0);
    double delta_t = 3.0;
    ni.integrate(delta_t, x0, res);
    EXPECT_NEAR(-0.5*9.81*delta_t*delta_t, res[0], 1e-6);
    EXPECT_NEAR(-9.81*delta_t, res[1], 1e-6);
}
