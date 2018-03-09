#include "gtest/gtest.h"
#include "NumericalIntegration.h"
#include "Eigen/Dense"

using namespace kalmanfilter;

struct ZeroDynamics
{
    using State = double;
    using Control = double;
    using Scalar = double;
    double operator()(const double &x, const double &u) const {return 0.0;}
};

TEST(EulerTestCase, TestZeroDynamics) {
    NumericalIntegration<ZeroDynamics, EULER> ni;
    double res;
    const double x0 = 42;
    res = ni.integrate(1, 1e-3, x0, 0);
    ASSERT_EQ(x0, res);
}

TEST(RungeKuttaTestCase, TestZeroDynamics) {
    NumericalIntegration<ZeroDynamics, RK4> ni;
    double res;
    const double x0 = 42;
    res = ni.integrate(1, 1e-3, x0, 0);
    ASSERT_EQ(x0, res);
}

struct ConstDynamics
{
    using State = double;
    using Control = double;
    using Scalar = double;
    double operator()(const double &x, const double &u) const {return 21.0;}
};

TEST(EulerTestCase, TestConstDynamics) {
    NumericalIntegration<ConstDynamics, EULER> ni;
    double res;
    const double x0 = 0;
    res = ni.integrate(2, 1e-3, x0, 0);
    EXPECT_NEAR(42.0, res, 0.1);
}

TEST(RungeKuttaTestCase, TestConstDynamics) {
    NumericalIntegration<ConstDynamics, RK4> ni;
    double res;
    const double x0 = 0;
    res = ni.integrate(2, 1e-3, x0, 0);
    EXPECT_NEAR(42.0, res, 0.1);
}

using Vec2d = Eigen::Vector2d;
struct FreeFall
{
    using State = Vec2d;
    using Control = double;
    using Scalar = double;
    Vec2d operator()(const Vec2d &x, const Control &u) const {
        Vec2d dx(x[1], -9.81);
        return dx;
    }
};

TEST(EulerTestCase, TestMultivariateIntegration) {
    NumericalIntegration<FreeFall, EULER> ni;
    Vec2d res;
    const Vec2d x0(0, 0);
    double delta_t = 3.0;
    res = ni.integrate(delta_t, 0.01, x0, 0);
    EXPECT_NEAR(-0.5*9.81*delta_t*delta_t, res[0], 1.0);
    EXPECT_NEAR(-9.81*delta_t, res[1], 1.0);
}

TEST(RungeKuttaTestCase, TestMultivariateIntegration) {
    NumericalIntegration<FreeFall, RK4> ni;
    Vec2d res;
    const Vec2d x0(0, 0);
    double delta_t = 3.0;
    res = ni.integrate(delta_t, 0.1, x0, 0);
    EXPECT_NEAR(-0.5*9.81*delta_t*delta_t, res[0], 1e-6);
    EXPECT_NEAR(-9.81*delta_t, res[1], 1e-6);
}
