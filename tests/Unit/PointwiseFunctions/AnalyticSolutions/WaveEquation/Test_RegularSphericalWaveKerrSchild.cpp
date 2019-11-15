// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <memory>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/ScalarWave/System.hpp"      // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/RegularSphericalWaveKerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
struct RegularSphWaveKerr {
  using type = CurvedScalarWave::Solutions::RegularSphericalWaveKerrSchild;
  static constexpr OptionString help{
      "A regular spherical wave in Kerr spacetime"};
};

void test_construct_from_options() noexcept {
  Options<tmpl::list<RegularSphWaveKerr>> opts("");
  opts.parse(
      "RegularSphWaveKerr:\n"
      "  BlackHoleMass: 1.\n"
      "  BlackHoleSpin: [0.1,0.2,0.3]\n"
      "  BlackHoleCenter: [0.,0.,0.]\n"
      "  WaveProfile:\n"
      "    Gaussian:\n"
      "      Amplitude: 1.\n"
      "      Width: 1.\n"
      "      Center: 0.");
  CHECK(opts.get<RegularSphWaveKerr>() ==
        CurvedScalarWave::Solutions::RegularSphericalWaveKerrSchild(
            1.0, {{0.1, 0.2, 0.3}}, {{0.0, 0.0, 0.0}},
            std::make_unique<MathFunctions::Gaussian>(1., 1., 0.)));
}

void test_serialize() noexcept {
  CurvedScalarWave::Solutions::RegularSphericalWaveKerrSchild solution(
      1.0, {{0.1, 0.2, 0.3}}, {{0.0, 0.0, 0.0}},
      std::make_unique<MathFunctions::Gaussian>(1., 1., 0.));
  test_serialization(solution);
}

void test_move() noexcept {
  CurvedScalarWave::Solutions::RegularSphericalWaveKerrSchild solution(
      1.0, {{0.1, 0.2, 0.3}}, {{0.0, 0.0, 0.0}},
      std::make_unique<MathFunctions::Gaussian>(1., 1., 0.));
  // since it can't actually be copied
  CurvedScalarWave::Solutions::RegularSphericalWaveKerrSchild copy_of_solution(
      1.0, {{0.1, 0.2, 0.3}}, {{0.0, 0.0, 0.0}},
      std::make_unique<MathFunctions::Gaussian>(1., 1., 0.));
  test_move_semantics(std::move(solution), copy_of_solution);
}

void test_no_hole() noexcept {
  const ScalarWave::Solutions::RegularSphericalWave flat_solution{
      std::make_unique<MathFunctions::Gaussian>(1., 1., 0.)};
  const CurvedScalarWave::Solutions::RegularSphericalWaveKerrSchild solution(
      1., {{0., 0., 0.}}, {{1.e20, 1., 1.}},
      std::make_unique<MathFunctions::Gaussian>(1., 1., 0.));

  const tnsr::I<DataVector, 3> x{std::array<DataVector, 3>{
      {DataVector({0., 1., 2., 3.}), DataVector({0., 0., 0., 0.}),
       DataVector({0., 0., 0., 0.})}}};
  auto vars = solution.variables(
      x, 0.,
      tmpl::list<CurvedScalarWave::Pi, CurvedScalarWave::Phi<3>,
                 CurvedScalarWave::Psi>{});
  auto flat_vars = flat_solution.variables(
      x, 0., tmpl::list<ScalarWave::Pi, ScalarWave::Phi<3>, ScalarWave::Psi>{});
  CHECK_ITERABLE_APPROX(get<CurvedScalarWave::Psi>(vars).get(),
                        get<ScalarWave::Psi>(flat_vars).get());
  CHECK_ITERABLE_APPROX(get<CurvedScalarWave::Pi>(vars).get(),
                        get<ScalarWave::Pi>(flat_vars).get());
  CHECK_ITERABLE_APPROX(get<CurvedScalarWave::Phi<3>>(vars).get(0),
                        get<ScalarWave::Phi<3>>(flat_vars).get(0));
  CHECK_ITERABLE_APPROX(get<CurvedScalarWave::Phi<3>>(vars).get(1),
                        get<ScalarWave::Phi<3>>(flat_vars).get(1));
  CHECK_ITERABLE_APPROX(get<CurvedScalarWave::Phi<3>>(vars).get(2),
                        get<ScalarWave::Phi<3>>(flat_vars).get(2));
}

void test_kerr() noexcept {
  const double mass = 1.7;
  const std::array<double, 3> spin{{0.1, 0.2, 0.3}};
  const std::array<double, 3> center{{0.3, 0.2, 0.4}};
  const CurvedScalarWave::Solutions::RegularSphericalWaveKerrSchild solution{
      mass, spin, center,
      std::make_unique<MathFunctions::Gaussian>(1., 1., 0.)};

  const tnsr::I<DataVector, 3> x{std::array<DataVector, 3>{
      {DataVector({0., 1., 2., 3.}), DataVector({0., 0., 0., 0.}),
       DataVector({0., 0., 0., 0.})}}};
  auto vars = solution.variables(
      x, 0.,
      tmpl::list<CurvedScalarWave::Pi, CurvedScalarWave::Phi<3>,
                 CurvedScalarWave::Psi>{});

  // Now construct the vars in-situ
  const ScalarWave::Solutions::RegularSphericalWave flat_wave_solution{
      std::make_unique<MathFunctions::Gaussian>(1., 1., 0.)};
  const auto flat_wave_vars = flat_wave_solution.variables(
      x, 0., ScalarWave::System<3>::variables_tag::tags_list{});
  const auto flat_wave_dt_vars = flat_wave_solution.variables(
      x, 0.,
      tmpl::list<::Tags::dt<ScalarWave::Pi>, ::Tags::dt<ScalarWave::Phi<3>>,
                 ::Tags::dt<ScalarWave::Psi>>{});

  const gr::Solutions::KerrSchild bh_solution(mass, spin, center);
  const auto kerr_variables = bh_solution.variables(
      x, 0., gr::Solutions::KerrSchild::tags<DataVector>{});

  const auto& local_psi = get<ScalarWave::Psi>(flat_wave_vars);
  const auto& local_phi = get<ScalarWave::Phi<3>>(flat_wave_vars);
  auto local_pi = make_with_value<Scalar<DataVector>>(x, 0.);
  {
    const auto shift_dot_dpsi = dot_product(
        get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(kerr_variables),
        get<ScalarWave::Phi<3>>(flat_wave_vars));
    get(local_pi) = (get(shift_dot_dpsi) -
                     get(get<::Tags::dt<ScalarWave::Psi>>(flat_wave_dt_vars))) /
                    get(get<gr::Tags::Lapse<DataVector>>(kerr_variables));
  }

  CHECK_ITERABLE_APPROX(get<CurvedScalarWave::Psi>(vars).get(),
                        local_psi.get());
  CHECK_ITERABLE_APPROX(get<CurvedScalarWave::Pi>(vars).get(), local_pi.get());
  CHECK_ITERABLE_APPROX(get<CurvedScalarWave::Phi<3>>(vars).get(0),
                        local_phi.get(0));
  CHECK_ITERABLE_APPROX(get<CurvedScalarWave::Phi<3>>(vars).get(1),
                        local_phi.get(1));
  CHECK_ITERABLE_APPROX(get<CurvedScalarWave::Phi<3>>(vars).get(2),
                        local_phi.get(2));
}

void test_kerr_schild_vars() noexcept {
  const double mass = 1.7;
  const std::array<double, 3> spin{{0.1, 0.2, 0.3}};
  const std::array<double, 3> center{{0.3, 0.2, 0.4}};
  const tnsr::I<DataVector, 3> x{std::array<DataVector, 3>{
      {DataVector({0., 1., 2., 3.}), DataVector({0., 0., 0., 0.}),
       DataVector({0., 0., 0., 0.})}}};

  // Now get Kerr background vars directly
  const CurvedScalarWave::Solutions::RegularSphericalWaveKerrSchild solution{
      mass, spin, center,
      std::make_unique<MathFunctions::Gaussian>(1., 1., 0.)};
  auto vars =
      solution.variables(x, 0., gr::Solutions::KerrSchild::tags<DataVector>{});

  // Now get Kerr background vars directly
  const gr::Solutions::KerrSchild bh_solution(mass, spin, center);
  const auto kerr_variables = bh_solution.variables(
      x, 0., gr::Solutions::KerrSchild::tags<DataVector>{});

  tmpl::for_each<gr::Solutions::KerrSchild::tags<DataVector>>(
      [&vars, &kerr_variables](auto x) {
        using tag = typename decltype(x)::type;
        CHECK_ITERABLE_APPROX(get<tag>(vars), get<tag>(kerr_variables));
      });
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.AnalyticSolutions.WaveEquation.RegularSphericalWaveKerrSchild",
    "[PointwiseFunctions][Unit]") {
  test_construct_from_options();
  test_serialize();
  test_move();
  test_no_hole();
  test_kerr();
  test_kerr_schild_vars();
}

/// Check restrictions on input options below

// [[OutputRegex, Spin magnitude must be < 1]]
SPECTRE_TEST_CASE(
    "Unit.AnalyticSolutions.WaveEquation.RegularSphericalWaveKerrSchildSpin",
    "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  CurvedScalarWave::Solutions::RegularSphericalWaveKerrSchild solution(
      1.0, {{1.0, 1.0, 1.0}}, {{0.0, 0.0, 0.0}},
      std::make_unique<MathFunctions::Gaussian>(1., 1., 0.));
}

// [[OutputRegex, Mass must be non-negative]]
SPECTRE_TEST_CASE(
    "Unit.AnalyticSolutions.WaveEquation.RegularSphericalWaveKerrSchildMass",
    "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  CurvedScalarWave::Solutions::RegularSphericalWaveKerrSchild solution(
      -1.0, {{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}},
      std::make_unique<MathFunctions::Gaussian>(1., 1., 0.));
}

// [[OutputRegex, In string:.*At line 2 column 18:.Value -0.5 is below the lower
// bound of 0]]
SPECTRE_TEST_CASE(
    "Unit.AnalyticSolutions.WaveEquation.RegularSphericalWaveKerrSchildOptM",
    "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  Options<tmpl::list<RegularSphWaveKerr>> opts("");
  opts.parse(
      "RegularSphWaveKerr:\n"
      "  BlackHoleMass: -0.5\n"
      "  BlackHoleSpin: [0.1,0.2,0.3]\n"
      "  BlackHoleCenter: [1.0,3.0,2.0]\n"
      "  WaveProfile:\n"
      "    Gaussian:\n"
      "      Amplitude: 1.\n"
      "      Width: 1.\n"
      "      Center: 0.");
  opts.get<RegularSphWaveKerr>();
}

// [[OutputRegex, Spin magnitude must be < 1]]
SPECTRE_TEST_CASE(
    "Unit.AnalyticSolutions.WaveEquation.RegularSphericalWaveKerrSchildOptS",
    "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  Options<tmpl::list<RegularSphWaveKerr>> opts("");
  opts.parse(
      "RegularSphWaveKerr:\n"
      "  BlackHoleMass: 0.5\n"
      "  BlackHoleSpin: [1.1,0.9,0.3]\n"
      "  BlackHoleCenter: [1.0,3.0,2.0]\n"
      "  WaveProfile:\n"
      "    Gaussian:\n"
      "      Amplitude: 1.\n"
      "      Width: 1.\n"
      "      Center: 0.");
  opts.get<RegularSphWaveKerr>();
}
