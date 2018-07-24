// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <cmath>

#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

inline tnsr::I<double, 1, Frame::Inertial> extract_point_from_coords(
    const size_t offset, const tnsr::I<DataVector, 1>& x) {
  return tnsr::I<double, 1, Frame::Inertial>{
      std::array<double, 1>{{x.get(0)[offset]}}};
}

inline tnsr::I<double, 2, Frame::Inertial> extract_point_from_coords(
    const size_t offset, const tnsr::I<DataVector, 2>& x) {
  return tnsr::I<double, 2, Frame::Inertial>{
      std::array<double, 2>{{x.get(0)[offset], x.get(1)[offset]}}};
}

inline tnsr::I<double, 3, Frame::Inertial> extract_point_from_coords(
    const size_t offset, const tnsr::I<DataVector, 3>& x) {
  return tnsr::I<double, 3, Frame::Inertial>{std::array<double, 3>{
      {x.get(0)[offset], x.get(1)[offset], x.get(2)[offset]}}};
}

template <size_t Dim>
void check_solution_x(const double kx, const double omega, const DataVector& u,
                      const ScalarWave::Solutions::PlaneWave<Dim>& pw,
                      const tnsr::I<DataVector, Dim>& x, const double t) {
  const DataVector psi = cube(u);
  const DataVector dpsi_dt = -3.0 * omega * square(u);
  const DataVector dpsi_dx = 3.0 * kx * square(u);
  const DataVector d2psi_dt2 = 6.0 * square(omega) * u;
  const DataVector d2psi_dtdx = -6.0 * omega * kx * u;
  const DataVector d2psi_dxdx = 6.0 * square(kx) * u;
  for (size_t s = 0; s < u.size(); ++s) {
    CHECK(approx(psi[s]) == pw.psi(x, t).get()[s]);
    CHECK(approx(dpsi_dt[s]) == pw.dpsi_dt(x, t).get()[s]);
    CHECK(approx(d2psi_dt2[s]) == pw.d2psi_dt2(x, t).get()[s]);
    CHECK(approx(dpsi_dx[s]) == pw.dpsi_dx(x, t).get(0)[s]);
    CHECK(approx(d2psi_dtdx[s]) == pw.d2psi_dtdx(x, t).get(0)[s]);
    CHECK(approx(d2psi_dxdx[s]) == pw.d2psi_dxdx(x, t).get(0, 0)[s]);
    const auto p = extract_point_from_coords(s, x);
    CHECK(approx(psi[s]) == pw.psi(p, t).get());
    CHECK(approx(dpsi_dt[s]) == pw.dpsi_dt(p, t).get());
    CHECK(approx(d2psi_dt2[s]) == pw.d2psi_dt2(p, t).get());
    CHECK(approx(dpsi_dx[s]) == pw.dpsi_dx(p, t).get(0));
    CHECK(approx(d2psi_dtdx[s]) == pw.d2psi_dtdx(p, t).get(0));
    CHECK(approx(d2psi_dxdx[s]) == pw.d2psi_dxdx(p, t).get(0, 0));
  }
}

template <size_t Dim>
void check_solution_y(const double kx, const double ky, const double omega,
                      const DataVector& u,
                      const ScalarWave::Solutions::PlaneWave<Dim>& pw,
                      const tnsr::I<DataVector, Dim>& x, const double t) {
  const DataVector dpsi_dy = 3.0 * ky * square(u);
  const DataVector d2psi_dtdy = -6.0 * omega * ky * u;
  const DataVector d2psi_dxdy = 6.0 * kx * ky * u;
  const DataVector d2psi_dydy = 6.0 * square(ky) * u;
  for (size_t s = 0; s < u.size(); ++s) {
    CHECK(approx(dpsi_dy[s]) == pw.dpsi_dx(x, t).get(1)[s]);
    CHECK(approx(d2psi_dtdy[s]) == pw.d2psi_dtdx(x, t).get(1)[s]);
    CHECK(approx(d2psi_dxdy[s]) == pw.d2psi_dxdx(x, t).get(0, 1)[s]);
    CHECK(approx(d2psi_dxdy[s]) == pw.d2psi_dxdx(x, t).get(1, 0)[s]);
    CHECK(approx(d2psi_dydy[s]) == pw.d2psi_dxdx(x, t).get(1, 1)[s]);
    const auto p = extract_point_from_coords(s, x);
    CHECK(approx(dpsi_dy[s]) == pw.dpsi_dx(p, t).get(1));
    CHECK(approx(d2psi_dtdy[s]) == pw.d2psi_dtdx(p, t).get(1));
    CHECK(approx(d2psi_dxdy[s]) == pw.d2psi_dxdx(p, t).get(0, 1));
    CHECK(approx(d2psi_dxdy[s]) == pw.d2psi_dxdx(p, t).get(1, 0));
    CHECK(approx(d2psi_dydy[s]) == pw.d2psi_dxdx(p, t).get(1, 1));
  }
}

void check_solution_z(const double kx, const double ky, const double kz,
                      const double omega, const DataVector& u,
                      const ScalarWave::Solutions::PlaneWave<3>& pw,
                      const tnsr::I<DataVector, 3>& x, const double t) {
  const DataVector dpsi_dz = 3.0 * kz * square(u);
  const DataVector d2psi_dtdz = -6.0 * omega * kz * u;
  const DataVector d2psi_dxdz = 6.0 * kx * kz * u;
  const DataVector d2psi_dydz = 6.0 * ky * kz * u;
  const DataVector d2psi_dzdz = 6.0 * square(kz) * u;
  for (size_t s = 0; s < u.size(); ++s) {
    CHECK(approx(dpsi_dz[s]) == pw.dpsi_dx(x, t).get(2)[s]);
    CHECK(approx(d2psi_dtdz[s]) == pw.d2psi_dtdx(x, t).get(2)[s]);
    CHECK(approx(d2psi_dxdz[s]) == pw.d2psi_dxdx(x, t).get(0, 2)[s]);
    CHECK(approx(d2psi_dxdz[s]) == pw.d2psi_dxdx(x, t).get(2, 0)[s]);
    CHECK(approx(d2psi_dydz[s]) == pw.d2psi_dxdx(x, t).get(2, 1)[s]);
    CHECK(approx(d2psi_dydz[s]) == pw.d2psi_dxdx(x, t).get(1, 2)[s]);
    CHECK(approx(d2psi_dzdz[s]) == pw.d2psi_dxdx(x, t).get(2, 2)[s]);
    const auto p = extract_point_from_coords(s, x);
    CHECK(approx(dpsi_dz[s]) == pw.dpsi_dx(p, t).get(2));
    CHECK(approx(d2psi_dtdz[s]) == pw.d2psi_dtdx(p, t).get(2));
    CHECK(approx(d2psi_dxdz[s]) == pw.d2psi_dxdx(p, t).get(0, 2));
    CHECK(approx(d2psi_dxdz[s]) == pw.d2psi_dxdx(p, t).get(2, 0));
    CHECK(approx(d2psi_dydz[s]) == pw.d2psi_dxdx(p, t).get(2, 1));
    CHECK(approx(d2psi_dydz[s]) == pw.d2psi_dxdx(p, t).get(1, 2));
    CHECK(approx(d2psi_dzdz[s]) == pw.d2psi_dxdx(p, t).get(2, 2));
  }
}

void test_1d() {
  const double k = -1.5;
  const double center_x = 2.4;
  const double omega = std::abs(k);
  const double t = 3.1;
  const double x1 = -0.2;
  const double x2 = 8.7;
  const tnsr::I<DataVector, 1> x(DataVector({x1, x2}));
  const DataVector u(
      {k * (x1 - center_x) - omega * t, k * (x2 - center_x) - omega * t});
  const ScalarWave::Solutions::PlaneWave<1> pw(
      {{k}}, {{center_x}}, std::make_unique<MathFunctions::PowX>(3));
  check_solution_x(k, omega, u, pw, x, t);
}

void test_2d() {
  const double kx = 1.5;
  const double ky = -7.2;
  const double center_x = 2.4;
  const double center_y = -4.8;
  const double omega = std::sqrt(square(kx) + square(ky));
  const double t = 3.1;
  const double x1 = -10.2;
  const double x2 = 8.7;
  const double y1 = -1.98;
  const double y2 = 48.27;
  const tnsr::I<DataVector, 2> x{
      std::array<DataVector, 2>{{DataVector({x1, x2}), DataVector({y1, y2})}}};
  const DataVector u({kx * (x1 - center_x) + ky * (y1 - center_y) - omega * t,
                      kx * (x2 - center_x) + ky * (y2 - center_y) - omega * t});
  const ScalarWave::Solutions::PlaneWave<2> pw(
      {{kx, ky}}, {{center_x, center_y}},
      std::make_unique<MathFunctions::PowX>(3));
  check_solution_x(kx, omega, u, pw, x, t);
  check_solution_y(kx, ky, omega, u, pw, x, t);
}

void test_3d() {
  const double kx = 1.5;
  const double ky = -7.2;
  const double kz = 2.7;
  const double center_x = 2.4;
  const double center_y = -4.8;
  const double center_z = 8.4;
  const double omega = std::sqrt(square(kx) + square(ky) + square(kz));
  const double t = 3.1;
  const double x1 = -10.2;
  const double x2 = 8.7;
  const double y1 = -1.98;
  const double y2 = 48.27;
  const double z1 = 2.2;
  const double z2 = 1.1;
  const tnsr::I<DataVector, 3> x{std::array<DataVector, 3>{
      {DataVector({x1, x2}), DataVector({y1, y2}), DataVector({z1, z2})}}};
  const DataVector u({kx * (x1 - center_x) + ky * (y1 - center_y) +
                          kz * (z1 - center_z) - omega * t,
                      kx * (x2 - center_x) + ky * (y2 - center_y) +
                          kz * (z2 - center_z) - omega * t});
  const ScalarWave::Solutions::PlaneWave<3> pw(
      {{kx, ky, kz}}, {{center_x, center_y, center_z}},
      std::make_unique<MathFunctions::PowX>(3));
  check_solution_x(kx, omega, u, pw, x, t);
  check_solution_y(kx, ky, omega, u, pw, x, t);
  check_solution_z(kx, ky, kz, omega, u, pw, x, t);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.WaveEquation.PlaneWave",
    "[PointwiseFunctions][Unit]") {
  test_1d();
  test_2d();
  test_3d();
}
