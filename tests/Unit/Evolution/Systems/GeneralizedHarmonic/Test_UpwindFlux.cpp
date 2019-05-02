// Distributed under the MIT License.
// See LICENSE.txt for details.
#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <random>
#include <utility>

#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/StrahlkorperGr.hpp"
#include "ApparentHorizons/Tags.hpp"  // IWYU pragma: keep
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DivideBy.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/TestHelpers.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::Pi
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::Phi
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::UPsi
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::UZero
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::UMinus
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::UPlus
// IWYU pragma: no_forward_declare StrahlkorperTags::CartesianCoords
// IWYU pragma: no_forward_declare StrahlkorperTags::NormalOneForm
// IWYU pragma: no_forward_declare Tags::CharSpeed
// IWYU pragma: no_forward_declare Tags::dt
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

namespace GeneralizedHarmonic {
namespace GeneralizedHarmonic_detail {
template <size_t Dim>
db::item_type<Tags::CharacteristicFields<Dim, Frame::Inertial>>
weight_char_fields(
    const db::item_type<Tags::CharacteristicFields<Dim, Frame::Inertial>>&
        char_fields_int,
    const db::item_type<Tags::CharacteristicSpeeds<Dim, Frame::Inertial>>&
        char_speeds_int,
    const db::item_type<Tags::CharacteristicFields<Dim, Frame::Inertial>>&
        char_fields_ext,
    const db::item_type<Tags::CharacteristicSpeeds<Dim, Frame::Inertial>>&
        char_speeds_ext) noexcept;
}  // namespace GeneralizedHarmonic_detail
}  // namespace GeneralizedHarmonic

namespace {
// Test GH upwind flux using random fields
void test_upwind_flux_random(
    const Strahlkorper<Frame::Inertial>& strahlkorper) noexcept {
  const size_t spatial_dim = 3;

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  std::uniform_real_distribution<> dist_positive(1.0, 2.0);
  std::uniform_real_distribution<> dist_pert(-0.1, 0.1);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_dist = make_not_null(&dist);
  const auto nn_dist_pert = make_not_null(&dist_pert);

  const auto box = db::create<
      db::AddSimpleTags<StrahlkorperTags::items_tags<Frame::Inertial>>,
      db::AddComputeTags<
          StrahlkorperTags::compute_items_tags<Frame::Inertial>>>(strahlkorper);
  const auto& unnormalized_normal =
      db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box);

  const auto one =
      make_with_value<Scalar<DataVector>>(unnormalized_normal, 1.0);
  const auto minus_one =
      make_with_value<Scalar<DataVector>>(unnormalized_normal, -1.0);
  const auto five =
      make_with_value<Scalar<DataVector>>(unnormalized_normal, 5.0);
  const auto minus_five =
      make_with_value<Scalar<DataVector>>(unnormalized_normal, -5.0);

  // Choose spacetime_metric randomly, but make sure the result is
  // still invertible. To do this, start with
  // Minkowski, and then add a 10% random perturbation.
  auto spacetime_metric_int = make_with_random_values<
      tnsr::aa<DataVector, spatial_dim, Frame::Inertial>>(
      nn_generator, nn_dist_pert, unnormalized_normal);
  get<0, 0>(spacetime_metric_int) += get(minus_one);
  get<1, 1>(spacetime_metric_int) += get(one);
  get<2, 2>(spacetime_metric_int) += get(one);
  get<3, 3>(spacetime_metric_int) += get(one);

  auto spacetime_metric_ext = make_with_random_values<
      tnsr::aa<DataVector, spatial_dim, Frame::Inertial>>(
      nn_generator, nn_dist_pert, unnormalized_normal);
  get<0, 0>(spacetime_metric_ext) += get(minus_one);
  get<1, 1>(spacetime_metric_ext) += get(one);
  get<2, 2>(spacetime_metric_ext) += get(one);
  get<3, 3>(spacetime_metric_ext) += get(one);

  // Set pi, phi to be random (phi, pi should not need to be consistent with
  // spacetime_metric for the flux consistency tests to pass)
  const auto phi_int = make_with_random_values<
      tnsr::iaa<DataVector, spatial_dim, Frame::Inertial>>(
      nn_generator, nn_dist_pert, unnormalized_normal);
  const auto pi_int = make_with_random_values<
      tnsr::aa<DataVector, spatial_dim, Frame::Inertial>>(
      nn_generator, nn_dist_pert, unnormalized_normal);
  const auto phi_ext = make_with_random_values<
      tnsr::iaa<DataVector, spatial_dim, Frame::Inertial>>(
      nn_generator, nn_dist_pert, unnormalized_normal);
  const auto pi_ext = make_with_random_values<
      tnsr::aa<DataVector, spatial_dim, Frame::Inertial>>(
      nn_generator, nn_dist_pert, unnormalized_normal);

  const auto& spatial_metric_int = gr::spatial_metric(spacetime_metric_int);
  const auto& inverse_spatial_metric_int =
      determinant_and_inverse(spatial_metric_int).second;
  const DataVector one_over_one_form_magnitude_int =
      get(magnitude(unnormalized_normal, inverse_spatial_metric_int));
  const auto unit_normal_one_form_int = StrahlkorperGr::unit_normal_one_form(
      unnormalized_normal, one_over_one_form_magnitude_int);
  const auto shift_int =
      gr::shift(spacetime_metric_int, inverse_spatial_metric_int);
  const auto lapse_int = gr::lapse(shift_int, spacetime_metric_int);

  const auto& spatial_metric_ext = gr::spatial_metric(spacetime_metric_ext);
  const auto& inverse_spatial_metric_ext =
      determinant_and_inverse(spatial_metric_ext).second;
  const DataVector one_over_one_form_magnitude_ext =
      get(magnitude(unnormalized_normal, inverse_spatial_metric_ext));
  const auto unit_normal_one_form_ext = StrahlkorperGr::unit_normal_one_form(
      divide_by(unnormalized_normal, get(minus_one)),
      one_over_one_form_magnitude_ext);
  const auto shift_ext =
      gr::shift(spacetime_metric_ext, inverse_spatial_metric_ext);
  const auto lapse_ext = gr::lapse(shift_int, spacetime_metric_ext);

  const auto gamma_1 = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_dist, unnormalized_normal);
  const auto gamma_2 = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_dist, unnormalized_normal);

  // Get the characteristic fields and speeds
  const auto char_fields_int = GeneralizedHarmonic::CharacteristicFieldsCompute<
      spatial_dim, Frame::Inertial>::function(gamma_2,
                                              inverse_spatial_metric_int,
                                              spacetime_metric_int, pi_int,
                                              phi_int,
                                              unit_normal_one_form_int);
  const auto char_fields_ext = GeneralizedHarmonic::CharacteristicFieldsCompute<
      spatial_dim, Frame::Inertial>::function(gamma_2,
                                              inverse_spatial_metric_ext,
                                              spacetime_metric_ext, pi_ext,
                                              phi_ext,
                                              unit_normal_one_form_int);

  std::array<DataVector, 4> char_speeds_one{
      {get(one), get(one), get(one), get(one)}};
  std::array<DataVector, 4> char_speeds_minus_one{
      {get(minus_one), get(minus_one), get(minus_one), get(minus_one)}};
  std::array<DataVector, 4> char_speeds_five{
      {get(five), get(five), get(five), get(five)}};
  std::array<DataVector, 4> char_speeds_minus_five{
      {get(minus_five), get(minus_five), get(minus_five), get(minus_five)}};

  GeneralizedHarmonic::UpwindFlux<spatial_dim> flux_computer{};

  // If all the char speeds are +1, the weighted fields should just
  // be the interior fields
  const auto weighted_char_fields_one =
      GeneralizedHarmonic::GeneralizedHarmonic_detail::weight_char_fields<3>(
          char_fields_int, char_speeds_one, char_fields_ext, char_speeds_one);
  CHECK(weighted_char_fields_one == char_fields_int);

  // If all the char speeds are -1, the weighted fields should just be
  // the exterior fields up to a sign
  const auto weighted_char_fields_minus_one =
      GeneralizedHarmonic::GeneralizedHarmonic_detail::weight_char_fields<3>(
          char_fields_int, char_speeds_minus_one, char_fields_ext,
          char_speeds_minus_one);
  CHECK(weighted_char_fields_minus_one == -1.0 * char_fields_ext);

  // Check scaling by 5 instead of 1
  const auto weighted_char_fields_minus_five =
      GeneralizedHarmonic::GeneralizedHarmonic_detail::weight_char_fields<3>(
          char_fields_int, char_speeds_minus_five, char_fields_ext,
          char_speeds_minus_five);
  const auto weighted_char_fields_five =
      GeneralizedHarmonic::GeneralizedHarmonic_detail::weight_char_fields<3>(
          char_fields_int, char_speeds_five, char_fields_ext, char_speeds_five);
  CHECK(weighted_char_fields_minus_five == -5.0 * char_fields_ext);
  CHECK(weighted_char_fields_five == 5.0 * char_fields_int);

  // Consistency checks
  Variables<typename GeneralizedHarmonic::UpwindFlux<spatial_dim>::package_tags>
  packaged_data_int(get(one).size(),
                    std::numeric_limits<double>::signaling_NaN());
  flux_computer.package_data(make_not_null(&packaged_data_int),
                             spacetime_metric_int, pi_int, phi_int, lapse_int,
                             shift_int, inverse_spatial_metric_int, gamma_1,
                             gamma_2, unit_normal_one_form_int);
  Variables<typename GeneralizedHarmonic::UpwindFlux<spatial_dim>::package_tags>
  packaged_data_ext(get(one).size(),
                    std::numeric_limits<double>::signaling_NaN());
  flux_computer.package_data(make_not_null(&packaged_data_ext),
                             spacetime_metric_ext, pi_ext, phi_ext, lapse_ext,
                             shift_ext, inverse_spatial_metric_ext, gamma_1,
                             gamma_2, unit_normal_one_form_ext);

  const auto minus_unit_normal_one_form_int =
      divide_by(unit_normal_one_form_int, get(minus_one));
  Variables<typename GeneralizedHarmonic::UpwindFlux<spatial_dim>::package_tags>
  packaged_data_int_opposite_normal(
      get(one).size(), std::numeric_limits<double>::signaling_NaN());
  flux_computer.package_data(make_not_null(&packaged_data_int_opposite_normal),
                             spacetime_metric_int, pi_int, phi_int, lapse_int,
                             shift_int, inverse_spatial_metric_int, gamma_1,
                             gamma_2, minus_unit_normal_one_form_int);
  const auto minus_unit_normal_one_form_ext =
      divide_by(unit_normal_one_form_ext, get(minus_one));
  Variables<typename GeneralizedHarmonic::UpwindFlux<spatial_dim>::package_tags>
  packaged_data_ext_opposite_normal(
      get(one).size(), std::numeric_limits<double>::signaling_NaN());
  flux_computer.package_data(make_not_null(&packaged_data_ext_opposite_normal),
                             spacetime_metric_ext, pi_ext, phi_ext, lapse_ext,
                             shift_ext, inverse_spatial_metric_ext, gamma_1,
                             gamma_2, minus_unit_normal_one_form_ext);

  // Check that if the same fields are given for the interior and exterior
  // (except that the normal vector gets multiplied by -1.0) that the
  // numerical flux reduces to the flux
  auto psi_normal_dot_numerical_flux = make_with_value<
      db::item_type<::Tags::NormalDotNumericalFlux<gr::Tags::SpacetimeMetric<
          spatial_dim, Frame::Inertial, DataVector>>>>(one, 0.0);
  auto pi_normal_dot_numerical_flux =
      make_with_value<db::item_type<::Tags::NormalDotNumericalFlux<
          GeneralizedHarmonic::Tags::Pi<spatial_dim, Frame::Inertial>>>>(one,
                                                                         0.0);
  auto phi_normal_dot_numerical_flux =
      make_with_value<db::item_type<::Tags::NormalDotNumericalFlux<
          GeneralizedHarmonic::Tags::Phi<spatial_dim, Frame::Inertial>>>>(one,
                                                                          0.0);
  ::TestHelpers::NumericalFluxes::apply_numerical_flux(
      flux_computer, packaged_data_int, packaged_data_int_opposite_normal,
      make_not_null(&psi_normal_dot_numerical_flux),
      make_not_null(&pi_normal_dot_numerical_flux),
      make_not_null(&phi_normal_dot_numerical_flux));

  ::GeneralizedHarmonic::ComputeNormalDotFluxes<spatial_dim>
      normal_dot_flux_computer{};
  auto psi_normal_dot_flux = make_with_value<
      db::item_type<::Tags::NormalDotNumericalFlux<gr::Tags::SpacetimeMetric<
          spatial_dim, Frame::Inertial, DataVector>>>>(one, 0.0);
  auto pi_normal_dot_flux =
      make_with_value<db::item_type<::Tags::NormalDotNumericalFlux<
          GeneralizedHarmonic::Tags::Pi<spatial_dim, Frame::Inertial>>>>(one,
                                                                         0.0);
  auto phi_normal_dot_flux =
      make_with_value<db::item_type<::Tags::NormalDotNumericalFlux<
          GeneralizedHarmonic::Tags::Phi<spatial_dim, Frame::Inertial>>>>(one,
                                                                          0.0);
  normal_dot_flux_computer.apply(
      make_not_null(&psi_normal_dot_flux), make_not_null(&pi_normal_dot_flux),
      make_not_null(&phi_normal_dot_flux), spacetime_metric_int, pi_int,
      phi_int, gamma_1, gamma_2, lapse_int, shift_int,
      inverse_spatial_metric_int, unit_normal_one_form_int);

  CHECK_ITERABLE_APPROX(psi_normal_dot_numerical_flux, psi_normal_dot_flux);
  CHECK_ITERABLE_APPROX(pi_normal_dot_numerical_flux, pi_normal_dot_flux);
  CHECK_ITERABLE_APPROX(phi_normal_dot_numerical_flux, phi_normal_dot_flux);
}

// Test GH upwind flux by comparing to Schwarzschild
template <typename Solution>
void test_upwind_flux_analytic(
    const Solution& solution_int, const Solution& solution_ext,
    const Strahlkorper<Frame::Inertial>& strahlkorper) noexcept {
  // Set up grid
  const size_t spatial_dim = 3;

  const auto box = db::create<
      db::AddSimpleTags<StrahlkorperTags::items_tags<Frame::Inertial>>,
      db::AddComputeTags<
          StrahlkorperTags::compute_items_tags<Frame::Inertial>>>(strahlkorper);

  const auto& x =
      db::get<StrahlkorperTags::CartesianCoords<Frame::Inertial>>(box);
  const double t = std::numeric_limits<double>::signaling_NaN();

  const auto one = make_with_value<Scalar<DataVector>>(x, 1.0);
  const auto minus_one = make_with_value<Scalar<DataVector>>(x, -1.0);
  const auto five = make_with_value<Scalar<DataVector>>(x, 5.0);
  const auto minus_five = make_with_value<Scalar<DataVector>>(x, -5.0);

  // Evaluate analytic solution for interior
  const auto vars_int = solution_int.variables(
      x, t, typename Solution::template tags<DataVector>{});
  const auto& lapse_int = get<gr::Tags::Lapse<>>(vars_int);
  const auto& dt_lapse_int = get<Tags::dt<gr::Tags::Lapse<>>>(vars_int);
  const auto& d_lapse_int =
      get<typename Solution::template DerivLapse<DataVector>>(vars_int);
  const auto& shift_int = get<gr::Tags::Shift<spatial_dim>>(vars_int);
  const auto& d_shift_int =
      get<typename Solution::template DerivShift<DataVector>>(vars_int);
  const auto& dt_shift_int =
      get<Tags::dt<gr::Tags::Shift<spatial_dim>>>(vars_int);
  const auto& spatial_metric_int =
      get<gr::Tags::SpatialMetric<spatial_dim>>(vars_int);
  const auto& dt_spatial_metric_int =
      get<Tags::dt<gr::Tags::SpatialMetric<spatial_dim>>>(vars_int);
  const auto& d_spatial_metric_int =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars_int);

  const auto inverse_spatial_metric_int =
      determinant_and_inverse(spatial_metric_int).second;
  const auto spacetime_metric_int =
      gr::spacetime_metric(lapse_int, shift_int, spatial_metric_int);
  const auto phi_int =
      GeneralizedHarmonic::phi(lapse_int, d_lapse_int, shift_int, d_shift_int,
                               spatial_metric_int, d_spatial_metric_int);
  const auto pi_int = GeneralizedHarmonic::pi(
      lapse_int, dt_lapse_int, shift_int, dt_shift_int, spatial_metric_int,
      dt_spatial_metric_int, phi_int);

  // Evaluate analytic solution for exterior (i.e. neighbor)
  const auto vars_ext = solution_ext.variables(
      x, t, typename Solution::template tags<DataVector>{});
  const auto& lapse_ext = get<gr::Tags::Lapse<>>(vars_ext);
  const auto& dt_lapse_ext = get<Tags::dt<gr::Tags::Lapse<>>>(vars_ext);
  const auto& d_lapse_ext =
      get<typename Solution::template DerivLapse<DataVector>>(vars_ext);
  const auto& shift_ext = get<gr::Tags::Shift<spatial_dim>>(vars_ext);
  const auto& d_shift_ext =
      get<typename Solution::template DerivShift<DataVector>>(vars_ext);
  const auto& dt_shift_ext =
      get<Tags::dt<gr::Tags::Shift<spatial_dim>>>(vars_ext);
  const auto& spatial_metric_ext =
      get<gr::Tags::SpatialMetric<spatial_dim>>(vars_ext);
  const auto& dt_spatial_metric_ext =
      get<Tags::dt<gr::Tags::SpatialMetric<spatial_dim>>>(vars_ext);
  const auto& d_spatial_metric_ext =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars_ext);

  const auto inverse_spatial_metric_ext =
      determinant_and_inverse(spatial_metric_ext).second;
  const auto spacetime_metric_ext =
      gr::spacetime_metric(lapse_ext, shift_ext, spatial_metric_ext);
  const auto phi_ext =
      GeneralizedHarmonic::phi(lapse_ext, d_lapse_ext, shift_ext, d_shift_ext,
                               spatial_metric_ext, d_spatial_metric_ext);
  const auto pi_ext = GeneralizedHarmonic::pi(
      lapse_ext, dt_lapse_ext, shift_ext, dt_shift_ext, spatial_metric_ext,
      dt_spatial_metric_ext, phi_ext);

  // More ingredients to get the char fields
  const size_t n_pts = x.begin()->size();
  const auto gamma_1 = make_with_value<Scalar<DataVector>>(x, 0.4);
  const auto gamma_2 = make_with_value<Scalar<DataVector>>(x, 0.1);

  // Get surface normal vectors
  const DataVector one_over_one_form_magnitude_int =
      1.0 / get(magnitude(
                db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box),
                inverse_spatial_metric_int));
  const auto unit_normal_one_form_int = StrahlkorperGr::unit_normal_one_form(
      db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box),
      one_over_one_form_magnitude_int);

  const auto minus_unnormalized_normal_one_form =
      divide_by(db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box),
                get(minus_one));
  const DataVector one_over_one_form_magnitude_ext =
      1.0 / get(magnitude(minus_unnormalized_normal_one_form,
                          inverse_spatial_metric_ext));
  const auto unit_normal_one_form_ext = StrahlkorperGr::unit_normal_one_form(
      minus_unnormalized_normal_one_form, one_over_one_form_magnitude_ext);

  // Get the characteristic fields and speeds
  const auto char_fields_int = GeneralizedHarmonic::CharacteristicFieldsCompute<
      spatial_dim, Frame::Inertial>::function(gamma_2,
                                              inverse_spatial_metric_int,
                                              spacetime_metric_int, pi_int,
                                              phi_int,
                                              unit_normal_one_form_int);
  const auto char_fields_ext = GeneralizedHarmonic::CharacteristicFieldsCompute<
      spatial_dim, Frame::Inertial>::function(gamma_2,
                                              inverse_spatial_metric_ext,
                                              spacetime_metric_ext, pi_ext,
                                              phi_ext,
                                              unit_normal_one_form_int);

  std::array<DataVector, 4> char_speeds_one{
      {get(one), get(one), get(one), get(one)}};
  std::array<DataVector, 4> char_speeds_minus_one{
      {get(minus_one), get(minus_one), get(minus_one), get(minus_one)}};
  std::array<DataVector, 4> char_speeds_five{
      {get(five), get(five), get(five), get(five)}};
  std::array<DataVector, 4> char_speeds_minus_five{
      {get(minus_five), get(minus_five), get(minus_five), get(minus_five)}};

  GeneralizedHarmonic::UpwindFlux<spatial_dim> flux_computer{};

  // If all the char speeds are +1, the weighted fields should just
  // be the interior fields
  const auto weighted_char_fields_one =
      GeneralizedHarmonic::GeneralizedHarmonic_detail::weight_char_fields<3>(
          char_fields_int, char_speeds_one, char_fields_ext, char_speeds_one);
  CHECK(weighted_char_fields_one == char_fields_int);

  // If all the char speeds are -1, the weighted fields should just be
  // the exterior fields up to a sign
  const auto weighted_char_fields_minus_one =
      GeneralizedHarmonic::GeneralizedHarmonic_detail::weight_char_fields<3>(
          char_fields_int, char_speeds_minus_one, char_fields_ext,
          char_speeds_minus_one);
  CHECK(weighted_char_fields_minus_one == -1.0 * char_fields_ext);

  // Check scaling by 5 instead of 1
  const auto weighted_char_fields_minus_five =
      GeneralizedHarmonic::GeneralizedHarmonic_detail::weight_char_fields<3>(
          char_fields_int, char_speeds_minus_five, char_fields_ext,
          char_speeds_minus_five);
  const auto weighted_char_fields_five =
      GeneralizedHarmonic::GeneralizedHarmonic_detail::weight_char_fields<3>(
          char_fields_int, char_speeds_five, char_fields_ext, char_speeds_five);
  CHECK(weighted_char_fields_minus_five == -5.0 * char_fields_ext);
  CHECK(weighted_char_fields_five == 5.0 * char_fields_int);

  // Consistency checks of the upwind flux
  // First, package interior and exterior quantities for the flux
  Variables<typename GeneralizedHarmonic::UpwindFlux<spatial_dim>::package_tags>
  packaged_data_int(get(one).size(),
                    std::numeric_limits<double>::signaling_NaN());
  flux_computer.package_data(make_not_null(&packaged_data_int),
                             spacetime_metric_int, pi_int, phi_int, lapse_int,
                             shift_int, inverse_spatial_metric_int, gamma_1,
                             gamma_2, unit_normal_one_form_int);
  Variables<typename GeneralizedHarmonic::UpwindFlux<spatial_dim>::package_tags>
  packaged_data_ext(get(one).size(),
                    std::numeric_limits<double>::signaling_NaN());
  flux_computer.package_data(make_not_null(&packaged_data_ext),
                             spacetime_metric_ext, pi_ext, phi_ext, lapse_ext,
                             shift_ext, inverse_spatial_metric_ext, gamma_1,
                             gamma_2, unit_normal_one_form_ext);

  const auto minus_unit_normal_one_form_int =
      divide_by(unit_normal_one_form_int, get(minus_one));
  Variables<typename GeneralizedHarmonic::UpwindFlux<spatial_dim>::package_tags>
  packaged_data_int_opposite_normal(
      get(one).size(), std::numeric_limits<double>::signaling_NaN());
  flux_computer.package_data(make_not_null(&packaged_data_int_opposite_normal),
                             spacetime_metric_int, pi_int, phi_int, lapse_int,
                             shift_int, inverse_spatial_metric_int, gamma_1,
                             gamma_2, minus_unit_normal_one_form_int);
  const auto minus_unit_normal_one_form_ext =
      divide_by(unit_normal_one_form_ext, get(minus_one));
  Variables<typename GeneralizedHarmonic::UpwindFlux<spatial_dim>::package_tags>
  packaged_data_ext_opposite_normal(
      get(one).size(), std::numeric_limits<double>::signaling_NaN());
  flux_computer.package_data(make_not_null(&packaged_data_ext_opposite_normal),
                             spacetime_metric_ext, pi_ext, phi_ext, lapse_ext,
                             shift_ext, inverse_spatial_metric_ext, gamma_1,
                             gamma_2, minus_unit_normal_one_form_ext);

  // Check that if the same fields are given for the interior and exterior
  // (except that the normal vector gets multiplied by -1.0) that the
  // numerical flux reduces to the flux
  auto psi_normal_dot_numerical_flux = make_with_value<
      db::item_type<::Tags::NormalDotNumericalFlux<gr::Tags::SpacetimeMetric<
          spatial_dim, Frame::Inertial, DataVector>>>>(x, 0.0);
  auto pi_normal_dot_numerical_flux =
      make_with_value<db::item_type<::Tags::NormalDotNumericalFlux<
          GeneralizedHarmonic::Tags::Pi<spatial_dim, Frame::Inertial>>>>(x,
                                                                         0.0);
  auto phi_normal_dot_numerical_flux =
      make_with_value<db::item_type<::Tags::NormalDotNumericalFlux<
          GeneralizedHarmonic::Tags::Phi<spatial_dim, Frame::Inertial>>>>(x,
                                                                          0.0);
  ::TestHelpers::NumericalFluxes::apply_numerical_flux(
      flux_computer, packaged_data_int, packaged_data_int_opposite_normal,
      make_not_null(&psi_normal_dot_numerical_flux),
      make_not_null(&pi_normal_dot_numerical_flux),
      make_not_null(&phi_normal_dot_numerical_flux));

  ::GeneralizedHarmonic::ComputeNormalDotFluxes<spatial_dim>
      normal_dot_flux_computer{};
  auto psi_normal_dot_flux = make_with_value<
      db::item_type<::Tags::NormalDotNumericalFlux<gr::Tags::SpacetimeMetric<
          spatial_dim, Frame::Inertial, DataVector>>>>(x, 0.0);
  auto pi_normal_dot_flux =
      make_with_value<db::item_type<::Tags::NormalDotNumericalFlux<
          GeneralizedHarmonic::Tags::Pi<spatial_dim, Frame::Inertial>>>>(x,
                                                                         0.0);
  auto phi_normal_dot_flux =
      make_with_value<db::item_type<::Tags::NormalDotNumericalFlux<
          GeneralizedHarmonic::Tags::Phi<spatial_dim, Frame::Inertial>>>>(x,
                                                                          0.0);
  normal_dot_flux_computer.apply(
      make_not_null(&psi_normal_dot_flux), make_not_null(&pi_normal_dot_flux),
      make_not_null(&phi_normal_dot_flux), spacetime_metric_int, pi_int,
      phi_int, gamma_1, gamma_2, lapse_int, shift_int,
      inverse_spatial_metric_int, unit_normal_one_form_int);

  CHECK_ITERABLE_APPROX(psi_normal_dot_numerical_flux, psi_normal_dot_flux);
  CHECK_ITERABLE_APPROX(pi_normal_dot_numerical_flux, pi_normal_dot_flux);
  CHECK_ITERABLE_APPROX(phi_normal_dot_numerical_flux, phi_normal_dot_flux);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.UpwindFlux",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GeneralizedHarmonic/"};

  const size_t l_max = 2;
  const std::array<double, 3> center{{0.3, 0.2, 0.1}};
  const double radius_inside_horizons = 1.0;
  const auto strahlkorper = Strahlkorper<Frame::Inertial>(
      l_max, l_max, radius_inside_horizons, center);
  test_upwind_flux_random(strahlkorper);

  // Test GH upwind flux against Kerr Schild
  const double mass = 2.;
  const std::array<double, 3> spin{{0.1, 0.2, 0.3}};
  const gr::Solutions::KerrSchild solution_1(mass, spin, center);
  const gr::Solutions::KerrSchild solution_2(2.0 * mass, spin, center);
  const std::array<double, 3> lower_bound{{0.82, 1.22, 1.32}};
  const std::array<double, 3> upper_bound{{0.78, 1.18, 1.28}};
  test_upwind_flux_analytic(solution_1, solution_2, strahlkorper);
}
