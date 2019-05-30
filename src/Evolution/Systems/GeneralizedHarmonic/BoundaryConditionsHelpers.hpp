// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/DataOnSlice.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InterfaceActionHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
namespace Tags {
template <typename Tag>
struct Magnitude;
}  // namespace Tags
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace GeneralizedHarmonic {
namespace Actions {

namespace BoundaryConditions_detail {}  // namespace BoundaryConditions_detail

namespace BoundaryConditions_detail {
enum class PsiBcMethod {
  AnalyticBc,
  Freezing,
  ConstraintPreservingNeumann,
  ConstraintPreservingDirichlet,
  Unknown
};
enum class PhiBcMethod { AnalyticBc, Freezing, Unknown };
enum class PiBcMethod { AnalyticBc, Freezing, Unknown };

// \brief This function computes intermediate variables needed for
// Bjorhus-type constraint preserving boundary conditions for the
// GeneralizedHarmonic system
template <size_t VolumeDim, typename TagsList, typename DbTags,
          typename VarsTagsList, typename DtVarsTagsList>
void local_variables(
    gsl::not_null<TempBuffer<TagsList>*> buffer, db::DataBox<DbTags>& box,
    const Direction<VolumeDim>& direction, const size_t& dimension,
    const db::item_type<::Tags::Mesh<VolumeDim>>& mesh,
    const Variables<VarsTagsList>& /* vars */,
    const Variables<DtVarsTagsList>& /* dt_vars_volume */,
    const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
        unit_interface_normal_one_form,
    const db::item_type<GeneralizedHarmonic::Tags::CharacteristicSpeeds<
        VolumeDim, Frame::Inertial>>& char_speeds) noexcept {
  //
  // NOTE: variable names below closely follow \cite Lindblom2005qh
  //
  // 1) Extract quantities from databox that are needed to compute
  // intermediate variables
  using tags_needed_on_slice = tmpl::list<
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<VolumeDim, Frame::Inertial, DataVector>,
      gr::Tags::SpatialMetric<VolumeDim, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<VolumeDim, Frame::Inertial, DataVector>,
      gr::Tags::SpacetimeNormalOneForm<VolumeDim, Frame::Inertial, DataVector>,
      gr::Tags::SpacetimeNormalVector<VolumeDim, Frame::Inertial, DataVector>,
      gr::Tags::SpacetimeMetric<VolumeDim, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpacetimeMetric<VolumeDim, Frame::Inertial, DataVector>,
      // ---- derivs of Psi, Pi, and Phi.
      // gr::Tags::DerivSpacetimeMetric<VolumeDim, Frame::Inertial, DataVector>,
      // ::Tags::deriv<GeneralizedHarmonic::Tags::Pi<VolumeDim,
      // Frame::Inertial>,
      //               tmpl::size_t<VolumeDim>, Frame::Inertial>,
      // ::Tags::deriv<GeneralizedHarmonic::Tags::Phi<VolumeDim,
      // Frame::Inertial>,
      //               tmpl::size_t<VolumeDim>, Frame::Inertial>,
      // ---- constraint damping parameters
      // GeneralizedHarmonic::Tags::ConstraintGamma0,
      // GeneralizedHarmonic::Tags::ConstraintGamma1,
      // GeneralizedHarmonic::Tags::ConstraintGamma2,
      // Constraints
      GeneralizedHarmonic::Tags::TwoIndexConstraint<VolumeDim, Frame::Inertial>,
      GeneralizedHarmonic::Tags::ThreeIndexConstraint<VolumeDim,
                                                      Frame::Inertial>,
      GeneralizedHarmonic::Tags::FourIndexConstraint<VolumeDim,
                                                     Frame::Inertial>,
      GeneralizedHarmonic::Tags::FConstraint<VolumeDim, Frame::Inertial>
      //
      >;
  const auto vars_on_this_slice = db::data_on_slice(
      box, mesh.extents(), dimension,
      index_to_slice_at(mesh.extents(), direction), tags_needed_on_slice{});

  // 2) name quantities as its just easier
  const auto& shift =
      get<gr::Tags::Shift<VolumeDim, Frame::Inertial, DataVector>>(
          vars_on_this_slice);
  // const auto& spatial_metric =
  //     get<gr::Tags::SpatialMetric<VolumeDim, Frame::Inertial, DataVector>>(
  //         vars_on_this_slice);
  const auto& inverse_spatial_metric = get<
      gr::Tags::InverseSpatialMetric<VolumeDim, Frame::Inertial, DataVector>>(
      vars_on_this_slice);
  const auto& spacetime_normal_one_form = get<
      gr::Tags::SpacetimeNormalOneForm<VolumeDim, Frame::Inertial, DataVector>>(
      vars_on_this_slice);
  const auto& spacetime_normal_vector = get<
      gr::Tags::SpacetimeNormalVector<VolumeDim, Frame::Inertial, DataVector>>(
      vars_on_this_slice);
  const auto& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<VolumeDim, Frame::Inertial, DataVector>>(
          vars_on_this_slice);
  const auto& inverse_spacetime_metric = get<
      gr::Tags::InverseSpacetimeMetric<VolumeDim, Frame::Inertial, DataVector>>(
      vars_on_this_slice);
  /*const auto& dspacetime_metric = get<
      gr::Tags::DerivSpacetimeMetric<VolumeDim, Frame::Inertial, DataVector>>(
      vars_on_this_slice);
  const auto& dpi = get<
      ::Tags::deriv<GeneralizedHarmonic::Tags::Pi<VolumeDim, Frame::Inertial>,
                    tmpl::size_t<VolumeDim>, Frame::Inertial>>(
      vars_on_this_slice);
  const auto& dphi = get<
      ::Tags::deriv<GeneralizedHarmonic::Tags::Phi<VolumeDim, Frame::Inertial>,
                    tmpl::size_t<VolumeDim>, Frame::Inertial>>(
      vars_on_this_slice);
  const auto& gamma0 =
      get<GeneralizedHarmonic::Tags::ConstraintGamma0>(vars_on_this_slice);
  const auto& gamma1 =
      get<GeneralizedHarmonic::Tags::ConstraintGamma1>(vars_on_this_slice);
  const auto& gamma2 =
      get<GeneralizedHarmonic::Tags::ConstraintGamma2>(vars_on_this_slice);*/
  const auto& two_index_constraint =
      get<GeneralizedHarmonic::Tags::TwoIndexConstraint<VolumeDim,
                                                        Frame::Inertial>>(
          vars_on_this_slice);
  const auto& three_index_constraint =
      get<GeneralizedHarmonic::Tags::ThreeIndexConstraint<VolumeDim,
                                                          Frame::Inertial>>(
          vars_on_this_slice);
  const auto& four_index_constraint =
      get<GeneralizedHarmonic::Tags::FourIndexConstraint<VolumeDim,
                                                         Frame::Inertial>>(
          vars_on_this_slice);
  const auto& f_constraint =
      get<GeneralizedHarmonic::Tags::FConstraint<VolumeDim, Frame::Inertial>>(
          vars_on_this_slice);

  // 3) Extract variable storage out of the buffer now
  // timelike and spacelike SPACETIME vectors, l^a and k^a
  auto& local_outgoing_null_one_form =
      get<::Tags::TempA<0, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  auto& local_incoming_null_one_form =
      get<::Tags::TempA<1, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  // timelike and spacelike SPACETIME oneforms, l_a and k_a
  auto& local_outgoing_null_vector =
      get<::Tags::Tempa<2, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  auto& local_incoming_null_vector =
      get<::Tags::Tempa<3, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  // SPACETIME form of interface normal (vector and oneform)
  auto& local_interface_normal_one_form =
      get<::Tags::Tempa<4, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  auto& local_interface_normal_vector =
      get<::Tags::TempA<5, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  // spacetime null form t_a and vector t^a
  get<::Tags::Tempa<6, VolumeDim, Frame::Inertial, DataVector>>(*buffer) = get<
      gr::Tags::SpacetimeNormalOneForm<VolumeDim, Frame::Inertial, DataVector>>(
      vars_on_this_slice);
  get<::Tags::TempA<7, VolumeDim, Frame::Inertial, DataVector>>(*buffer) = get<
      gr::Tags::SpacetimeNormalVector<VolumeDim, Frame::Inertial, DataVector>>(
      vars_on_this_slice);
  // interface normal dot shift: n_k N^k
  auto& interface_normal_dot_shift =
      get<::Tags::TempScalar<8, DataVector>>(*buffer);
  // spacetime projection operator P_ab, P^ab, and P^a_b
  auto& local_projection_ab =
      get<::Tags::TempAA<9, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  auto& local_projection_AB =
      get<::Tags::Tempaa<10, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  auto& local_projection_Ab =
      get<::Tags::TempAb<11, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  // Char speeds
  auto& local_char_speed_u_psi =
      get<::Tags::TempScalar<12, DataVector>>(*buffer);
  auto& local_char_speed_u_plus =
      get<::Tags::TempScalar<13, DataVector>>(*buffer);
  auto& local_char_speed_u_minus =
      get<::Tags::TempScalar<14, DataVector>>(*buffer);
  auto& local_char_speed_u_zero =
      get<::Tags::TempScalar<15, DataVector>>(*buffer);
  // constraint characteristics
  auto& local_constraint_char_zero_minus =
      get<::Tags::Tempa<16, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  auto& local_constraint_char_three =
      get<::Tags::Tempiaa<17, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  auto& local_constraint_char_four =
      get<::Tags::Tempiaa<18, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  // lapse, shift and inverse spatial_metric
  get<::Tags::TempScalar<19, DataVector>>(*buffer) =
      get<gr::Tags::Lapse<DataVector>>(vars_on_this_slice);
  get<::Tags::TempI<20, VolumeDim, Frame::Inertial, DataVector>>(*buffer) =
      get<gr::Tags::Shift<VolumeDim, Frame::Inertial, DataVector>>(
          vars_on_this_slice);
  get<::Tags::TempII<21, VolumeDim, Frame::Inertial, DataVector>>(*buffer) =
      get<gr::Tags::InverseSpatialMetric<VolumeDim, Frame::Inertial,
                                         DataVector>>(vars_on_this_slice);

  // 4) Compute intermediate variables now
  // 4.1) Spacetime form of interface normal (vector and oneform)
  const auto unit_interface_normal_vector = raise_or_lower_index(
      unit_interface_normal_one_form, inverse_spatial_metric);
  get<0>(local_interface_normal_one_form) = 0.;
  get<0>(local_interface_normal_vector) = 0.;
  for (size_t i = 0; i < VolumeDim; ++i) {
    local_interface_normal_one_form.get(1 + i) =
        unit_interface_normal_one_form.get(i);
    local_interface_normal_vector.get(1 + i) =
        unit_interface_normal_vector.get(i);
  }
  // 4.2) timelike and spacelike SPACETIME vectors, l^a and k^a,
  //      without (1/sqrt(2))
  for (size_t a = 0; a < VolumeDim + 1; ++a) {
    local_outgoing_null_one_form.get(a) =
        spacetime_normal_one_form.get(a) +
        local_interface_normal_one_form.get(a);
    local_incoming_null_one_form.get(a) =
        spacetime_normal_one_form.get(a) -
        local_interface_normal_one_form.get(a);
    local_outgoing_null_vector.get(a) =
        spacetime_normal_vector.get(a) + local_interface_normal_vector.get(a);
    local_incoming_null_vector.get(a) =
        spacetime_normal_vector.get(a) - local_interface_normal_vector.get(a);
  }
  //       interface_normal_dot_shift = n_i N^i
  for (size_t i = 0; i < VolumeDim; ++i) {
    get(interface_normal_dot_shift) =
        shift.get(i) * local_interface_normal_one_form.get(i + 1);
  }
  // 4.3) Spacetime projection operators P_ab, P^ab and P^a_b
  for (size_t a = 0; a < VolumeDim + 1; ++a) {
    for (size_t b = 0; b < VolumeDim + 1; ++b) {
      local_projection_ab.get(a, b) =
          spacetime_metric.get(a, b) +
          spacetime_normal_one_form.get(a) * spacetime_normal_one_form.get(b) +
          local_interface_normal_one_form.get(a) *
              local_interface_normal_one_form.get(b);
      local_projection_Ab.get(a, b) =
          spacetime_normal_one_form.get(a) * spacetime_normal_vector.get(b) +
          local_interface_normal_one_form.get(a) *
              local_interface_normal_vector.get(b);
      if (UNLIKELY(a == b)) {
        local_projection_Ab.get(a, b) += 1.;
      }
      local_projection_AB.get(a, b) =
          inverse_spacetime_metric.get(a, b) +
          spacetime_normal_vector.get(a) * spacetime_normal_vector.get(b) +
          local_interface_normal_vector.get(a) *
              local_interface_normal_vector.get(b);
    }
  }
  // 4.4) Characteristic speeds
  get(local_char_speed_u_psi) = char_speeds.at(0);
  get(local_char_speed_u_zero) = char_speeds.at(1);
  get(local_char_speed_u_plus) = char_speeds.at(2);
  get(local_char_speed_u_minus) = char_speeds.at(3);

  // 4.5) c^{\hat{0}-}_a = F_a + n^k C_{ka}
  for (size_t a = 0; a < VolumeDim + 1; ++a) {
    local_constraint_char_zero_minus.get(a) = f_constraint.get(a);
    for (size_t i = 0; i < VolumeDim; ++i) {
      local_constraint_char_zero_minus.get(a) +=
          unit_interface_normal_vector.get(i) * two_index_constraint.get(i, a);
    }
  }
  // 4.6) c^\hat{3}_{jab} = C_{jab} = \partial_j\psi_{ab} - \Phi_{jab}
  for (size_t i = 0; i < VolumeDim; ++i) {
    for (size_t a = 0; a < VolumeDim + 1; ++a) {
      // due to symmetry, loop over b \in [a, Dim + 1] instead of [0, Dim + 1]
      for (size_t b = a; b < VolumeDim + 1; ++b) {
        local_constraint_char_three.get(i, a, b) =
            three_index_constraint.get(i, a, b);
      }
    }
  }
  // 4.7) c^\hat{4}_{ijab} = C_{ijab}
  for (size_t i = 0; i < VolumeDim; ++i) {
    for (size_t a = 0; a < VolumeDim + 1; ++a) {
      // due to symmetry, loop over b \in [a, Dim + 1] instead of [0, Dim + 1]
      for (size_t b = a; b < VolumeDim + 1; ++b) {
        local_constraint_char_four.get(i, a, b) =
            four_index_constraint.get(i, a, b);
      }
    }
  }
}

// \brief This struct sets boundary condition on dt<UPsi>
template <typename ReturnType, size_t SpatialDim, typename Frame>
struct set_dt_u_psi {
  static ReturnType apply(
      const PsiBcMethod Method, const gsl::not_null<ReturnType*> bc_dt_u_psi,
      const tnsr::i<DataVector, SpatialDim, Frame>& unit_normal_one_form,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, SpatialDim, Frame>& shift,
      const tnsr::II<DataVector, SpatialDim, Frame>& inverse_spatial_metric,
      const tnsr::aa<DataVector, SpatialDim, Frame>& pi,
      const tnsr::iaa<DataVector, SpatialDim, Frame>& phi,
      const tnsr::iaa<DataVector, SpatialDim, Frame>& deriv_spacetime_metric,
      const tnsr::aa<DataVector, SpatialDim, Frame>& dt_u_psi,
      const std::array<DataVector, 4>& char_speeds) noexcept {
    switch (Method) {
      case PsiBcMethod::ConstraintPreservingNeumann:
        return apply_neumann_constraint_preserving(
            bc_dt_u_psi, unit_normal_one_form, lapse, shift, pi, phi, dt_u_psi,
            char_speeds);
      case PsiBcMethod::ConstraintPreservingDirichlet:
        return apply_neumann_constraint_preserving(
            bc_dt_u_psi, unit_normal_one_form, lapse, shift,
            inverse_spatial_metric, pi, phi, deriv_spacetime_metric, dt_u_psi,
            char_speeds);
      case PsiBcMethod::Unknown:
      default:
        ASSERT(false, "Requested BC method fo UPsi not implemented!");
    }
  }

 private:
  static ReturnType apply_neumann_constraint_preserving(
      const gsl::not_null<ReturnType*> bc_dt_u_psi,
      const tnsr::i<DataVector, SpatialDim, Frame>& unit_normal_one_form,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, SpatialDim, Frame>& shift,
      const tnsr::II<DataVector, SpatialDim, Frame>& inverse_spatial_metric,
      const tnsr::aa<DataVector, SpatialDim, Frame>& pi,
      const tnsr::iaa<DataVector, SpatialDim, Frame>& phi,
      const tnsr::iaa<DataVector, SpatialDim, Frame>& deriv_spacetime_metric,
      const tnsr::aa<DataVector, SpatialDim, Frame>& dt_u_psi,
      const std::array<DataVector, 4>& char_speeds) noexcept;
  static ReturnType apply_dirichlet_consrtraint_preserving(
      const gsl::not_null<ReturnType*> bc_dt_u_psi,
      const tnsr::i<DataVector, SpatialDim, Frame>& unit_normal_one_form,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, SpatialDim, Frame>& shift,
      const tnsr::aa<DataVector, SpatialDim, Frame>& pi,
      const tnsr::iaa<DataVector, SpatialDim, Frame>& phi,
      const tnsr::aa<DataVector, SpatialDim, Frame>& dt_u_psi,
      const std::array<DataVector, 4>& char_speeds) noexcept;
};

template <typename ReturnType, size_t SpatialDim, typename Frame>
ReturnType set_dt_u_psi<ReturnType, SpatialDim, Frame>::
    apply_neumann_constraint_preserving(
        const gsl::not_null<ReturnType*> bc_dt_u_psi,
        const tnsr::i<DataVector, SpatialDim, Frame>& unit_normal_one_form,
        const Scalar<DataVector>& lapse,
        const tnsr::I<DataVector, SpatialDim, Frame>& shift,
        const tnsr::II<DataVector, SpatialDim, Frame>& inverse_spatial_metric,
        const tnsr::aa<DataVector, SpatialDim, Frame>& pi,
        const tnsr::iaa<DataVector, SpatialDim, Frame>& phi,
        const tnsr::iaa<DataVector, SpatialDim, Frame>& deriv_spacetime_metric,
        const tnsr::aa<DataVector, SpatialDim, Frame>& dt_u_psi,
        const std::array<DataVector, 4>& char_speeds) noexcept {
  if (UNLIKELY(get_size(get<0, 0>(*bc_dt_u_psi)) != get_size(get(lapse)))) {
    *bc_dt_u_psi = ReturnType(get_size(get(lapse)));
  }
  const auto unit_normal_vector =
      raise_or_lower_index(unit_normal_one_form, inverse_spatial_metric);
  for (size_t a = 0; a <= SpatialDim; ++a) {
    for (size_t b = a; b <= SpatialDim; ++b) {
      bc_dt_u_psi.get(a, b) = dt_u_psi.get(a, b);
      for (size_t i = 0; i < SpatialDim; ++i) {
        bc_dt_u_psi.get(a, b) +=
            char_speeds[0] * unit_normal_vector.get(i) *
            (deriv_spacetime_metric.get(i, a, b) - phi.get(i, a, b));
      }
    }
  }
  return *bc_dt_u_psi;
}

template <typename ReturnType, size_t SpatialDim, typename Frame>
ReturnType set_dt_u_psi<ReturnType, SpatialDim, Frame>::
    apply_dirichlet_consrtraint_preserving(
        const gsl::not_null<ReturnType*> bc_dt_u_psi,
        const tnsr::i<DataVector, SpatialDim, Frame>& unit_normal_one_form,
        const Scalar<DataVector>& lapse,
        const tnsr::I<DataVector, SpatialDim, Frame>& shift,
        const tnsr::aa<DataVector, SpatialDim, Frame>& pi,
        const tnsr::iaa<DataVector, SpatialDim, Frame>& phi,
        const tnsr::aa<DataVector, SpatialDim, Frame>& dt_u_psi,
        const std::array<DataVector, 4>& char_speeds) noexcept {
  if (UNLIKELY(get_size(get<0, 0>(*bc_dt_u_psi)) != get_size(get(lapse)))) {
    *bc_dt_u_psi = ReturnType(get_size(get(lapse)));
  }
  for (size_t a = 0; a <= SpatialDim; ++a) {
    for (size_t b = a; b <= SpatialDim; ++b) {
      bc_dt_u_psi.get(a, b) = -get(lapse) * pi.get(a, b);
      for (size_t i = 0; i < SpatialDim; ++i) {
        bc_dt_u_psi.get(a, b) += shift.get(i) * phi.get(i, a, b);
      }
    }
  }
  return *bc_dt_u_psi;
}

}  // namespace BoundaryConditions_detail
}  // namespace Actions
}  // namespace GeneralizedHarmonic
