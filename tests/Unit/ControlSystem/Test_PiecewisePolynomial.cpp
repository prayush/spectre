// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "ControlSystem/PiecewisePolynomial.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.ControlSystem.FunctionsOfTime.PiecewisePolynomial",
                  "[ControlSystem][Unit]") {
  double t = 0.0;
  const double dt = 0.6;
  const double final_time = 4.0;
  const constexpr size_t deriv_order = 3;
  const constexpr size_t ret_deriv_order = 1;

  // test two component system (x**3 and x**2)
  const std::array<DataVector, deriv_order + 1> init_func{
      {{0.0, 0.0}, {0.0, 0.0}, {0.0, 2.0}, {6.0, 0.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(t, init_func);

  while (t < final_time) {
    const auto& lambdas0 = f_of_t(t);
    CHECK(approx(lambdas0[0][0]) == cube(t));
    CHECK(approx(lambdas0[0][1]) == square(t));
    CHECK(approx(lambdas0[1][0]) == 3.0 * square(t));
    CHECK(approx(lambdas0[1][1]) == 2.0 * t);
    CHECK(approx(lambdas0[2][0]) == 6.0 * t);
    CHECK(approx(lambdas0[2][1]) == 2.0);
    CHECK(approx(lambdas0[3][0]) == 6.0);
    CHECK(approx(lambdas0[3][1]) == 0.0);

    const auto& lambdas1 = f_of_t.operator()<ret_deriv_order>(t);
    CHECK(lambdas1.size() == ret_deriv_order + 1);
    CHECK(approx(lambdas1[0][0]) == cube(t));
    CHECK(approx(lambdas1[0][1]) == square(t));
    CHECK(approx(lambdas1[1][0]) == 3.0 * square(t));
    CHECK(approx(lambdas1[1][1]) == 2.0 * t);

    t += dt;
    f_of_t.update(t, {6.0, 0.0});
  }
  // test time_bounds function
  const auto& t_bounds = f_of_t.time_bounds();
  CHECK(t_bounds[0] == 0.0);
  CHECK(t_bounds[1] == 4.2);
}

SPECTRE_TEST_CASE(
    "Unit.ControlSystem.FunctionsOfTime.PiecewisePolynomial.NonConstDeriv",
    "[ControlSystem][Unit]") {
  double t = 0.0;
  const double dt = 0.6;
  const double final_time = 4.0;
  const constexpr size_t deriv_order = 2;

  // initally x**2, but update with non-constant 2nd deriv
  const std::array<DataVector, deriv_order + 1> init_func{
      {{0.0}, {0.0}, {2.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(t, init_func);

  while (t < final_time) {
    t += dt;
    f_of_t.update(t, {3.0 + t});
  }
  const auto& lambdas = f_of_t(t);
  CHECK(approx(lambdas[0][0]) == 33.948);
  CHECK(approx(lambdas[1][0]) == 19.56);
  CHECK(approx(lambdas[2][0]) == 7.2);
}

SPECTRE_TEST_CASE(
    "Unit.ControlSystem.FunctionsOfTime.PiecewisePolynomial.WithinRoundoff",
    "[ControlSystem][Unit]") {
  const constexpr size_t deriv_order = 3;
  const size_t n_derivs = 3;
  const std::array<DataVector, deriv_order + 1> init_func{
      {{1.0, 1.0}, {3.0, 2.0}, {6.0, 2.0}, {6.0, 0.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(1.0, init_func);
  f_of_t.update(2.0, {6.0, 0.0});
  const auto& lambdas = f_of_t(1.0 - 5.0e-16);
  CHECK_ITERABLE_APPROX(lambdas, init_func);
}

// [[OutputRegex, t must be increasing from call to call. Attempted to update at
// time 1, which precedes the previous update time of 2.]]
SPECTRE_TEST_CASE(
    "Unit.ControlSystem.FunctionsOfTime.PiecewisePolynomial.BadUpdateTime",
    "[ControlSystem][Unit]") {
  ERROR_TEST();
  // two component system (x**3 and x**2)
  const constexpr size_t deriv_order = 3;
  const std::array<DataVector, deriv_order + 1> init_func{
      {{0.0, 0.0}, {0.0, 0.0}, {0.0, 2.0}, {6.0, 0.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(0.0, init_func);
  f_of_t.update(2.0, {6.0, 0.0});
  f_of_t.update(1.0, {6.0, 0.0});
}

// [[OutputRegex, the number of components trying to be updated \(3\) does not
// match the number of components \(2\) in the PiecewisePolynomial.]]
SPECTRE_TEST_CASE(
    "Unit.ControlSystem.FunctionsOfTime.PiecewisePolynomial.BadUpdateSize",
    "[ControlSystem][Unit]") {
  ERROR_TEST();
  // two component system (x**3 and x**2)
  const constexpr size_t deriv_order = 3;
  const std::array<DataVector, deriv_order + 1> init_func{
      {{0.0, 0.0}, {0.0, 0.0}, {0.0, 2.0}, {6.0, 0.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(0.0, init_func);
  f_of_t.update(1.0, {6.0, 0.0, 0.0});
}

// [[OutputRegex, requested time 0.5 precedes earliest time 1 of times.]]
SPECTRE_TEST_CASE(
    "Unit.ControlSystem.FunctionsOfTime.PiecewisePolynomial.TimeOutOfRange",
    "[ControlSystem][Unit]") {
  ERROR_TEST();
  // two component system (x**3 and x**2)
  const constexpr size_t deriv_order = 3;
  const size_t n_derivs = 3;
  const std::array<DataVector, deriv_order + 1> init_func{
      {{1.0, 1.0}, {3.0, 2.0}, {6.0, 2.0}, {6.0, 0.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t(1.0, init_func);
  f_of_t.update(2.0, {6.0, 0.0});
  const auto& lambdas = f_of_t(0.5);
}
