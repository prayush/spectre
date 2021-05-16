// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Bjorhus.hpp"

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BjorhusImpl.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/InterfaceNullNormal.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/ProjectionOperators.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace GeneralizedHarmonic::BoundaryConditions {
namespace helpers {
double min_characteristic_speed(
    const std::array<DataVector, 4>& char_speeds) noexcept {
  std::array<double, 4> min_speeds{
      {min(char_speeds.at(0)), min(char_speeds.at(1)), min(char_speeds.at(2)),
       min(char_speeds.at(3))}};
  return *std::min_element(min_speeds.begin(), min_speeds.end());
}
template <typename T>
void set_bc_corr_zero_when_char_speed_is_positive(
    const gsl::not_null<T*> dt_v_corr,
    const DataVector& char_speed_u) noexcept {
  auto it = dt_v_corr->begin();
  for (; it != dt_v_corr->end(); ++it) {
    for (size_t i = 0; i < it->size(); ++i) {
      if (char_speed_u[i] > 0.) {
        (*it)[i] = 0.;
      }
    }
  }
}
}  // namespace helpers

namespace detail {
ConstraintPreservingBjorhusType
convert_constraint_preserving_bjorhus_type_from_yaml(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  if (type_read == "ConstraintPreserving") {
    return ConstraintPreservingBjorhusType::ConstraintPreserving;
  } else if (type_read == "ConstraintPreservingPhysical") {
    return ConstraintPreservingBjorhusType::ConstraintPreservingPhysical;
  }
  PARSE_ERROR(
      options.context(),
      "Failed to convert \""
          << type_read
          << "\" to ConstraintPreservingBjorhusType::Type. Must "
             "be one of ConstraintPreserving or ConstraintPreservingPhysical");
}
}  // namespace detail

template <size_t Dim>
ConstraintPreservingBjorhus<Dim>::ConstraintPreservingBjorhus(
    const detail::ConstraintPreservingBjorhusType type) noexcept
    : type_(type) {}

template <size_t Dim>
ConstraintPreservingBjorhus<Dim>::ConstraintPreservingBjorhus(
    CkMigrateMessage* const msg) noexcept
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
ConstraintPreservingBjorhus<Dim>::get_clone() const noexcept {
  return std::make_unique<ConstraintPreservingBjorhus>(*this);
}

template <size_t Dim>
void ConstraintPreservingBjorhus<Dim>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
  p | type_;
}

template <size_t Dim>
std::optional<std::string> ConstraintPreservingBjorhus<Dim>::dg_time_derivative(
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        dt_spacetime_metric_correction,
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        dt_pi_correction,
    const gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
        dt_phi_correction,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& /*normal_vector*/,
    // c.f. dg_interior_evolved_variables_tags
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,
    // c.f. dg_interior_temporary_tags
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
    const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
    const Scalar<DataVector>& lapse,
    const tnsr::a<DataVector, Dim, Frame::Inertial>& gauge_source,
    const tnsr::ab<DataVector, Dim, Frame::Inertial>&
        spacetime_deriv_gauge_source,
    // c.f. dg_interior_dt_vars_tags
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& dt_spacetime_metric,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& dt_pi,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& dt_phi,
    // c.f. dg_interior_deriv_vars_tags
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& d_spacetime_metric,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& d_pi,
    const tnsr::ijaa<DataVector, Dim, Frame::Inertial>& d_phi) const noexcept {
  TempBuffer<tmpl::list<::Tags::TempI<0, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempiaa<0, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempiaa<1, Dim, Frame::Inertial, DataVector>,
                        ::Tags::TempII<0, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempii<0, Dim, Frame::Inertial, DataVector>,
                        ::Tags::TempAA<0, Dim, Frame::Inertial, DataVector>,
                        ::Tags::TempA<0, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempa<0, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempa<1, Dim, Frame::Inertial, DataVector>,
                        ::Tags::TempA<1, Dim, Frame::Inertial, DataVector>,
                        ::Tags::TempA<2, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempaa<0, Dim, Frame::Inertial, DataVector>,
                        ::Tags::TempAb<0, Dim, Frame::Inertial, DataVector>,
                        ::Tags::TempAA<1, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempaa<1, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempiaa<2, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempaa<2, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempa<2, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempa<3, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempaa<3, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempiaa<3, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempaa<4, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempaa<5, Dim, Frame::Inertial, DataVector>>>
      local_buffer(get_size(get<0>(normal_covector)), 0.);

  auto& unit_interface_normal_vector =
      get<::Tags::TempI<0, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& three_index_constraint =
      get<::Tags::Tempiaa<0, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& four_index_constraint =
      get<::Tags::Tempiaa<1, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& inverse_spatial_metric =
      get<::Tags::TempII<0, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& extrinsic_curvature =
      get<::Tags::Tempii<0, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& inverse_spacetime_metric =
      get<::Tags::TempAA<0, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& spacetime_unit_normal_vector =
      get<::Tags::TempA<0, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& incoming_null_one_form =
      get<::Tags::Tempa<0, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& outgoing_null_one_form =
      get<::Tags::Tempa<1, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& incoming_null_vector =
      get<::Tags::TempA<1, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& outgoing_null_vector =
      get<::Tags::TempA<2, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& projection_ab =
      get<::Tags::Tempaa<0, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& projection_Ab =
      get<::Tags::TempAb<0, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& projection_AB =
      get<::Tags::TempAA<1, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& char_projected_rhs_dt_v_psi =
      get<::Tags::Tempaa<1, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& char_projected_rhs_dt_v_zero =
      get<::Tags::Tempiaa<2, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& char_projected_rhs_dt_v_minus =
      get<::Tags::Tempaa<2, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& constraint_char_zero_plus =
      get<::Tags::Tempa<2, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& constraint_char_zero_minus =
      get<::Tags::Tempa<3, Dim, Frame::Inertial, DataVector>>(local_buffer);

  typename Tags::CharacteristicSpeeds<Dim, Frame::Inertial>::type char_speeds;

  auto& bc_dt_v_psi =
      get<::Tags::Tempaa<3, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& bc_dt_v_zero =
      get<::Tags::Tempiaa<3, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& bc_dt_v_plus =
      get<::Tags::Tempaa<4, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& bc_dt_v_minus =
      get<::Tags::Tempaa<5, Dim, Frame::Inertial, DataVector>>(local_buffer);

  compute_intermediate_vars(
      make_not_null(&unit_interface_normal_vector),
      make_not_null(&three_index_constraint),
      make_not_null(&four_index_constraint),
      make_not_null(&inverse_spatial_metric),
      make_not_null(&extrinsic_curvature),
      make_not_null(&inverse_spacetime_metric),
      make_not_null(&spacetime_unit_normal_vector),
      make_not_null(&incoming_null_one_form),
      make_not_null(&outgoing_null_one_form),
      make_not_null(&incoming_null_vector),
      make_not_null(&outgoing_null_vector), make_not_null(&projection_ab),
      make_not_null(&projection_Ab), make_not_null(&projection_AB),
      make_not_null(&char_projected_rhs_dt_v_psi),
      make_not_null(&char_projected_rhs_dt_v_zero),
      make_not_null(&char_projected_rhs_dt_v_minus),
      make_not_null(&constraint_char_zero_plus),
      make_not_null(&constraint_char_zero_minus), make_not_null(&char_speeds),
      face_mesh_velocity, normal_covector, pi, phi, spacetime_metric, coords,
      gamma1, gamma2, lapse, gauge_source, spacetime_deriv_gauge_source, dt_pi,
      dt_phi, dt_spacetime_metric, d_pi, d_phi, d_spacetime_metric);

  // Account for moving mesh: char speeds -> cher speeds - n_i v^i_g
  if (face_mesh_velocity.has_value()) {
    const auto negative_lambda0 =
        get(dot_product(normal_covector, *face_mesh_velocity));
    for (size_t a = 0; a < 4; ++a) {
      char_speeds.at(a) -= negative_lambda0;
    }
  }

  detail::set_dt_v_psi_constraint_preserving(
      make_not_null(&bc_dt_v_psi), unit_interface_normal_vector,
      three_index_constraint, char_projected_rhs_dt_v_psi, char_speeds);

  detail::set_dt_v_zero_constraint_preserving(
      make_not_null(&bc_dt_v_zero), unit_interface_normal_vector,
      four_index_constraint, char_projected_rhs_dt_v_zero, char_speeds);

  if (type_ == detail::ConstraintPreservingBjorhusType::ConstraintPreserving) {
    detail::set_dt_v_minus_constraint_preserving(
        make_not_null(&bc_dt_v_minus), gamma2, coords, incoming_null_one_form,
        outgoing_null_one_form, incoming_null_vector, outgoing_null_vector,
        projection_ab, projection_Ab, projection_AB,
        char_projected_rhs_dt_v_psi, char_projected_rhs_dt_v_minus,
        constraint_char_zero_plus, constraint_char_zero_minus, char_speeds);
  } else if (type_ == detail::ConstraintPreservingBjorhusType::
                          ConstraintPreservingPhysical) {
    detail::set_dt_v_minus_constraint_preserving_physical(
        make_not_null(&bc_dt_v_minus), gamma2, coords, normal_covector,
        unit_interface_normal_vector, spacetime_unit_normal_vector,
        incoming_null_one_form, outgoing_null_one_form, incoming_null_vector,
        outgoing_null_vector, projection_ab, projection_Ab, projection_AB,
        inverse_spatial_metric, extrinsic_curvature, spacetime_metric,
        inverse_spacetime_metric, three_index_constraint,
        char_projected_rhs_dt_v_psi, char_projected_rhs_dt_v_minus,
        constraint_char_zero_plus, constraint_char_zero_minus, phi, d_phi, d_pi,
        char_speeds);
  } else {
    ERROR("Failed to set dtVMinus. Input option \""
          << "\" Must be one of ConstraintPreserving or "
             "ConstraintPreservingPhysical");
  }

  // Only add corrections at grid points where the char speeds are negative
  helpers::set_bc_corr_zero_when_char_speed_is_positive(
      make_not_null(&bc_dt_v_psi), char_speeds.at(0));
  helpers::set_bc_corr_zero_when_char_speed_is_positive(
      make_not_null(&bc_dt_v_zero), char_speeds.at(1));
  helpers::set_bc_corr_zero_when_char_speed_is_positive(
      make_not_null(&bc_dt_v_minus), char_speeds.at(3));

  // Convert corrections to dt<evolved variables>
  auto dt_evolved_vars = evolved_fields_from_characteristic_fields(
      gamma2, bc_dt_v_psi, bc_dt_v_zero, bc_dt_v_plus, bc_dt_v_minus,
      normal_covector);

  *dt_pi_correction = get<Tags::Pi<Dim, Frame::Inertial>>(dt_evolved_vars);
  *dt_phi_correction = get<Tags::Phi<Dim, Frame::Inertial>>(dt_evolved_vars);
  *dt_spacetime_metric_correction =
      get<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>>(
          dt_evolved_vars);

  // Subtract out original dt<vars>
  for (size_t a = 0; a <= Dim; ++a) {
    for (size_t b = a; b <= Dim; ++b) {
      dt_pi_correction->get(a, b) -= dt_pi.get(a, b);
      dt_spacetime_metric_correction->get(a, b) -=
          dt_spacetime_metric.get(a, b);
      for (size_t i = 0; i < Dim; ++i) {
        dt_phi_correction->get(i, a, b) -= dt_phi.get(i, a, b);
      }
    }
  }

  if (face_mesh_velocity.has_value()) {
    const auto negative_lambda0 =
        dot_product(normal_covector, *face_mesh_velocity);
    if (min(-get(negative_lambda0)) < 0) {
      return {
          "Incoming characteristic speeds for constraint preserving "
          "radiation boundary. Its unclear if proper boundary conditions"
          "are imposed in this case."};
    }
  }

  return {};
}

template <size_t Dim>
void ConstraintPreservingBjorhus<Dim>::compute_intermediate_vars(
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        unit_interface_normal_vector,
    const gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
        three_index_constraint,
    const gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
        four_index_constraint,
    const gsl::not_null<tnsr::II<DataVector, Dim, Frame::Inertial>*>
        inverse_spatial_metric,
    const gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
        extrinsic_curvature,
    const gsl::not_null<tnsr::AA<DataVector, Dim, Frame::Inertial>*>
        inverse_spacetime_metric,
    const gsl::not_null<tnsr::A<DataVector, Dim, Frame::Inertial>*>
        spacetime_unit_normal_vector,
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
        incoming_null_one_form,
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
        outgoing_null_one_form,
    const gsl::not_null<tnsr::A<DataVector, Dim, Frame::Inertial>*>
        incoming_null_vector,
    const gsl::not_null<tnsr::A<DataVector, Dim, Frame::Inertial>*>
        outgoing_null_vector,
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        projection_ab,
    const gsl::not_null<tnsr::Ab<DataVector, Dim, Frame::Inertial>*>
        projection_Ab,
    const gsl::not_null<tnsr::AA<DataVector, Dim, Frame::Inertial>*>
        projection_AB,
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        char_projected_rhs_dt_v_psi,
    const gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
        char_projected_rhs_dt_v_zero,
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        char_projected_rhs_dt_v_minus,
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
        constraint_char_zero_plus,
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*>
        constraint_char_zero_minus,
    const gsl::not_null<std::array<DataVector, 4>*> char_speeds,

    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
    /* face_mesh_velocity */,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& /* coords */,
    const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
    const Scalar<DataVector>& lapse,
    const tnsr::a<DataVector, Dim, Frame::Inertial>& gauge_source,
    const tnsr::ab<DataVector, Dim, Frame::Inertial>&
        spacetime_deriv_gauge_source,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& dt_pi,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& dt_phi,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& dt_spacetime_metric,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& d_pi,
    const tnsr::ijaa<DataVector, Dim, Frame::Inertial>& d_phi,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& d_spacetime_metric)
    const noexcept {
  TempBuffer<tmpl::list<::Tags::TempScalar<0, DataVector>,
                        ::Tags::TempI<0, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempii<0, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempa<0, Dim, Frame::Inertial, DataVector>,
                        ::Tags::Tempia<0, Dim, Frame::Inertial, DataVector>>>
      local_buffer(get_size(get<0>(normal_covector)));

  //   auto& lapse = get<::Tags::TempScalar<0, DataVector>>(local_buffer);
  auto& shift =
      get<::Tags::TempI<0, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& spatial_metric =
      get<::Tags::Tempii<0, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& spacetime_unit_normal_one_form =
      get<::Tags::Tempa<0, Dim, Frame::Inertial, DataVector>>(local_buffer);
  auto& two_index_constraint =
      get<::Tags::Tempia<0, Dim, Frame::Inertial, DataVector>>(local_buffer);

  *inverse_spacetime_metric = determinant_and_inverse(spacetime_metric).second;
  gr::spatial_metric(make_not_null(&spatial_metric), spacetime_metric);
  *inverse_spatial_metric = determinant_and_inverse(spatial_metric).second;
  raise_or_lower_index(unit_interface_normal_vector, normal_covector,
                       *inverse_spatial_metric);

  gr::shift(make_not_null(&shift), spacetime_metric, *inverse_spatial_metric);
  //   gr::lapse(make_not_null(&lapse), shift, spacetime_metric);
  gr::spacetime_normal_vector(spacetime_unit_normal_vector, lapse, shift);
  gr::spacetime_normal_one_form(make_not_null(&spacetime_unit_normal_one_form),
                                lapse);

  GeneralizedHarmonic::extrinsic_curvature(
      extrinsic_curvature, *spacetime_unit_normal_vector, pi, phi);

  GeneralizedHarmonic::three_index_constraint(three_index_constraint,
                                              d_spacetime_metric, phi);
  if (LIKELY(Dim == 3)) {
    GeneralizedHarmonic::four_index_constraint(four_index_constraint, d_phi);
  } else if (UNLIKELY(Dim == 2)) {
    for (size_t a = 0; a <= Dim; ++a) {
      for (size_t b = 0; b <= Dim; ++b) {
        four_index_constraint->get(0, a, b) =
            d_phi.get(0, 1, a, b) - d_phi.get(1, 0, a, b);
        four_index_constraint->get(1, a, b) =
            -four_index_constraint->get(0, a, b);
      }
    }
  } else {
    std::fill(four_index_constraint->begin(), four_index_constraint->end(), 0.);
  }

  gr::interface_null_normal(incoming_null_one_form,
                            spacetime_unit_normal_one_form, normal_covector,
                            -1.);
  gr::interface_null_normal(outgoing_null_one_form,
                            spacetime_unit_normal_one_form, normal_covector,
                            1.);
  gr::interface_null_normal(incoming_null_vector, *spacetime_unit_normal_vector,
                            *unit_interface_normal_vector, -1.);
  gr::interface_null_normal(outgoing_null_vector, *spacetime_unit_normal_vector,
                            *unit_interface_normal_vector, 1.);

  gr::transverse_projection_operator(projection_ab, spacetime_metric,
                                     spacetime_unit_normal_one_form,
                                     normal_covector);
  gr::transverse_projection_operator(
      projection_Ab, *spacetime_unit_normal_vector,
      spacetime_unit_normal_one_form, *unit_interface_normal_vector,
      normal_covector);
  gr::transverse_projection_operator(projection_AB, *inverse_spacetime_metric,
                                     *spacetime_unit_normal_vector,
                                     *unit_interface_normal_vector);

  const auto dt_char_fields = characteristic_fields(
      gamma2, *inverse_spatial_metric, dt_spacetime_metric, dt_pi, dt_phi,
      normal_covector);
  *char_projected_rhs_dt_v_psi =
      get<Tags::VSpacetimeMetric<Dim, Frame::Inertial>>(dt_char_fields);
  *char_projected_rhs_dt_v_zero =
      get<Tags::VZero<Dim, Frame::Inertial>>(dt_char_fields);
  *char_projected_rhs_dt_v_minus =
      get<Tags::VMinus<Dim, Frame::Inertial>>(dt_char_fields);

  // c^{\hat{0}-}_a = F_a + n^k C_{ka}
  GeneralizedHarmonic::two_index_constraint(
      make_not_null(&two_index_constraint), spacetime_deriv_gauge_source,
      spacetime_unit_normal_one_form, *spacetime_unit_normal_vector,
      *inverse_spatial_metric, *inverse_spacetime_metric, pi, phi, d_pi, d_phi,
      gamma2, *three_index_constraint);
  f_constraint(constraint_char_zero_plus, gauge_source,
               spacetime_deriv_gauge_source, spacetime_unit_normal_one_form,
               *spacetime_unit_normal_vector, *inverse_spatial_metric,
               *inverse_spacetime_metric, pi, phi, d_pi, d_phi, gamma2,
               *three_index_constraint);
  for (size_t a = 0; a < Dim + 1; ++a) {
    constraint_char_zero_minus->get(a) = constraint_char_zero_plus->get(a);
    for (size_t i = 0; i < Dim; ++i) {
      constraint_char_zero_plus->get(a) -=
          unit_interface_normal_vector->get(i) * two_index_constraint.get(i, a);
      constraint_char_zero_minus->get(a) +=
          unit_interface_normal_vector->get(i) * two_index_constraint.get(i, a);
    }
  }

  characteristic_speeds(char_speeds, gamma1, lapse, shift, normal_covector);
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID ConstraintPreservingBjorhus<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template class ConstraintPreservingBjorhus<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace GeneralizedHarmonic::BoundaryConditions
