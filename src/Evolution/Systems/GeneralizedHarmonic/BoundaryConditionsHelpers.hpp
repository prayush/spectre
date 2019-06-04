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
#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InterfaceActionHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
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
enum class UPsiBcMethod {
  AnalyticBc,
  Freezing,
  ConstraintPreservingBjorhus,
  ConstraintPreservingDirichlet,
  Unknown
};
enum class UZeroBcMethod {
  AnalyticBc,
  Freezing,
  ConstraintPreservingBjorhus,
  ConstraintPreservingDirichlet,
  ConstraintPreservingRealDirichlet,
  Unknown
};
enum class UPlusBcMethod { AnalyticBc, Freezing, Unknown };
enum class UMinusBcMethod { AnalyticBc, Freezing, Unknown };

template <size_t VolumeDim>
using all_local_vars = tmpl::list<
    // timelike and spacelike SPACETIME vectors, l^a and k^a
    ::Tags::TempA<0, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::TempA<1, VolumeDim, Frame::Inertial, DataVector>,
    // timelike and spacelike SPACETIME oneforms, l_a and k_a
    ::Tags::Tempa<2, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::Tempa<3, VolumeDim, Frame::Inertial, DataVector>,
    // SPACETIME form of interface normal (vector and oneform)
    ::Tags::Tempa<4, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::TempA<5, VolumeDim, Frame::Inertial, DataVector>,
    // spacetime null form t_a and vector t^a
    ::Tags::Tempa<6, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::TempA<7, VolumeDim, Frame::Inertial, DataVector>,
    // interface normal dot shift: n_k N^k
    ::Tags::TempScalar<8, DataVector>,
    // spacetime projection operator P_ab, P^ab, and P^a_b
    ::Tags::TempAA<9, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::Tempaa<10, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::TempAb<11, VolumeDim, Frame::Inertial, DataVector>,
    // Char speeds
    ::Tags::TempScalar<12, DataVector>, ::Tags::TempScalar<13, DataVector>,
    ::Tags::TempScalar<14, DataVector>, ::Tags::TempScalar<15, DataVector>,
    // Constraints
    ::Tags::Tempa<16, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::Tempiaa<17, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::Tempiaa<18, VolumeDim, Frame::Inertial, DataVector>,
    // 3+1 geometric quantities: lapse, shift, inverse
    // 3-metric
    ::Tags::TempScalar<19, DataVector>,
    ::Tags::TempI<20, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::TempII<21, VolumeDim, Frame::Inertial, DataVector>,
    // Characteristic projected time derivatives of evolved
    // fields
    ::Tags::Tempaa<22, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::Tempiaa<23, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::Tempaa<24, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::Tempaa<25, VolumeDim, Frame::Inertial, DataVector>,
    // Constraint damping parameter
    ::Tags::TempScalar<26, DataVector>,
    // Preallocated memory to store boundary conditions
    ::Tags::Tempaa<27, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::Tempiaa<28, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::Tempaa<29, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::Tempaa<30, VolumeDim, Frame::Inertial, DataVector>,
    // derivatives of psi, pi, phi
    ::Tags::Tempiaa<31, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::Tempiaa<32, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::Tempijaa<33, VolumeDim, Frame::Inertial, DataVector>>;

// \brief This function computes intermediate variables needed for
// Bjorhus-type constraint preserving boundary conditions for the
// GeneralizedHarmonic system
template <size_t VolumeDim, typename TagsList, typename DbTags,
          typename VarsTagsList, typename DtVarsTagsList>
void local_variables(
    gsl::not_null<TempBuffer<TagsList>*> buffer, const db::DataBox<DbTags>& box,
    const Direction<VolumeDim>& direction, const size_t& dimension,
    const typename ::Tags::Mesh<VolumeDim>::type& mesh,
    const Variables<VarsTagsList>& /* vars */,
    const Variables<DtVarsTagsList>& dt_vars,
    const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
        unit_interface_normal_one_form,
    const typename Tags::CharacteristicSpeeds<VolumeDim, Frame::Inertial>::type&
        char_speeds) noexcept {
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
      gr::Tags::DerivSpacetimeMetric<VolumeDim, Frame::Inertial, DataVector>,
      ::Tags::deriv<Tags::Pi<VolumeDim, Frame::Inertial>,
                    tmpl::size_t<VolumeDim>, Frame::Inertial>,
      ::Tags::deriv<Tags::Phi<VolumeDim, Frame::Inertial>,
                    tmpl::size_t<VolumeDim>, Frame::Inertial>,
      // ---- constraint damping parameters
      // Tags::ConstraintGamma0,
      // Tags::ConstraintGamma1,
      Tags::ConstraintGamma2,
      // Characteristics
      // Constraints
      Tags::TwoIndexConstraint<VolumeDim, Frame::Inertial>,
      Tags::ThreeIndexConstraint<VolumeDim, Frame::Inertial>,
      Tags::FourIndexConstraint<VolumeDim, Frame::Inertial>,
      Tags::FConstraint<VolumeDim, Frame::Inertial>
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
  /*const auto& gamma0 =
      get<Tags::ConstraintGamma0>(vars_on_this_slice);
  const auto& gamma1 =
      get<Tags::ConstraintGamma1>(vars_on_this_slice);*/
  const auto& gamma2 = get<Tags::ConstraintGamma2>(vars_on_this_slice);
  const auto& two_index_constraint =
      get<Tags::TwoIndexConstraint<VolumeDim, Frame::Inertial>>(
          vars_on_this_slice);
  const auto& three_index_constraint =
      get<Tags::ThreeIndexConstraint<VolumeDim, Frame::Inertial>>(
          vars_on_this_slice);
  const auto& four_index_constraint =
      get<Tags::FourIndexConstraint<VolumeDim, Frame::Inertial>>(
          vars_on_this_slice);
  const auto& f_constraint =
      get<Tags::FConstraint<VolumeDim, Frame::Inertial>>(vars_on_this_slice);
  // storage for DT<UChar> = CharProjection(dt<U>)
  const auto& rhs_dt_psi = get<::Tags::dt<
      gr::Tags::SpacetimeMetric<VolumeDim, Frame::Inertial, DataVector>>>(
      dt_vars);
  const auto& rhs_dt_pi =
      get<::Tags::dt<Tags::Pi<VolumeDim, Frame::Inertial>>>(dt_vars);
  const auto& rhs_dt_phi =
      get<::Tags::dt<Tags::Phi<VolumeDim, Frame::Inertial>>>(dt_vars);
  const auto char_projected_dt_u = characteristic_fields(
      gamma2, inverse_spatial_metric, rhs_dt_psi, rhs_dt_pi, rhs_dt_phi,
      unit_interface_normal_one_form);

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
  // Characteristic projected time derivatives of evolved fields
  get<::Tags::Tempaa<22, VolumeDim, Frame::Inertial, DataVector>>(*buffer) =
      get<Tags::UPsi<VolumeDim, Frame::Inertial>>(char_projected_dt_u);
  get<::Tags::Tempiaa<23, VolumeDim, Frame::Inertial, DataVector>>(*buffer) =
      get<Tags::UZero<VolumeDim, Frame::Inertial>>(char_projected_dt_u);
  get<::Tags::Tempaa<24, VolumeDim, Frame::Inertial, DataVector>>(*buffer) =
      get<Tags::UPlus<VolumeDim, Frame::Inertial>>(char_projected_dt_u);
  get<::Tags::Tempaa<25, VolumeDim, Frame::Inertial, DataVector>>(*buffer) =
      get<Tags::UMinus<VolumeDim, Frame::Inertial>>(char_projected_dt_u);
  // Spatial derivatives of evolved variables: Psi, Pi and Phi
  get<::Tags::Tempiaa<31, VolumeDim, Frame::Inertial, DataVector>>(*buffer) =
      get<gr::Tags::DerivSpacetimeMetric<VolumeDim, Frame::Inertial,
                                         DataVector>>(vars_on_this_slice);
  get<::Tags::Tempiaa<32, VolumeDim, Frame::Inertial, DataVector>>(*buffer) =
      get<::Tags::deriv<Tags::Pi<VolumeDim, Frame::Inertial>,
                        tmpl::size_t<VolumeDim>, Frame::Inertial>>(
          vars_on_this_slice);
  get<::Tags::Tempijaa<33, VolumeDim, Frame::Inertial, DataVector>>(*buffer) =
      get<::Tags::deriv<Tags::Phi<VolumeDim, Frame::Inertial>,
                        tmpl::size_t<VolumeDim>, Frame::Inertial>>(
          vars_on_this_slice);
  // Constraint damping parameters
  get<::Tags::TempScalar<26, DataVector>>(*buffer) =
      get<Tags::ConstraintGamma2>(vars_on_this_slice);
  // Preallocated memory for output
  auto& _bc_upsi =
      get<::Tags::Tempaa<27, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  auto& _bc_uzero =
      get<::Tags::Tempiaa<28, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  auto& _bc_uplus =
      get<::Tags::Tempaa<29, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  auto& _bc_uminus =
      get<::Tags::Tempaa<30, VolumeDim, Frame::Inertial, DataVector>>(*buffer);

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
  // fill preallocated memory with zeros
  std::fill(_bc_upsi.begin(), _bc_upsi.end(), 0.);
  std::fill(_bc_uzero.begin(), _bc_uzero.end(), 0.);
  std::fill(_bc_uplus.begin(), _bc_uplus.end(), 0.);
  std::fill(_bc_uminus.begin(), _bc_uminus.end(), 0.);
}

// \brief This struct sets boundary condition on dt<UPsi>
template <typename ReturnType, size_t VolumeDim>
struct set_dt_u_psi {
  template <typename TagsList, typename VarsTagsList, typename DtVarsTagsList>
  static ReturnType apply(const UPsiBcMethod Method,
                          TempBuffer<TagsList>& buffer,
                          const Variables<VarsTagsList>& vars,
                          const Variables<DtVarsTagsList>& /* dt_vars */,
                          const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
                              unit_normal_one_form) noexcept {
    // Not using auto below to enforce a loose test on the quantity being
    // fetched from the buffer
    const typename Tags::ThreeIndexConstraint<VolumeDim, Frame::Inertial>::type&
        three_index_constraint =
            get<::Tags::Tempiaa<17, VolumeDim, Frame::Inertial, DataVector>>(
                buffer);
    const tnsr::A<DataVector, VolumeDim,
                  Frame::Inertial>& unit_interface_normal_vector =
        get<::Tags::TempA<5, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    const typename gr::Tags::Lapse<DataVector>::type& lapse =
        get<::Tags::TempScalar<19, DataVector>>(buffer);
    const typename gr::Tags::Shift<VolumeDim, Frame::Inertial,
                                   DataVector>::type& shift =
        get<::Tags::TempI<20, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    const typename Tags::Pi<VolumeDim, Frame::Inertial>::type& pi =
        get<Tags::Pi<VolumeDim, Frame::Inertial>>(vars);
    const typename Tags::Phi<VolumeDim, Frame::Inertial>::type& phi =
        get<Tags::Phi<VolumeDim, Frame::Inertial>>(vars);
    const typename Tags::UPsi<VolumeDim, Frame::Inertial>::type&
        char_projected_rhs_dt_u_psi =
            get<::Tags::Tempaa<22, VolumeDim, Frame::Inertial, DataVector>>(
                buffer);
    const typename Tags::CharacteristicSpeeds<VolumeDim, Frame::Inertial>::type
        char_speeds{{get(get<::Tags::TempScalar<12, DataVector>>(buffer)),
                     get(get<::Tags::TempScalar<13, DataVector>>(buffer)),
                     get(get<::Tags::TempScalar<14, DataVector>>(buffer)),
                     get(get<::Tags::TempScalar<15, DataVector>>(buffer))}};
    // Memory allocated for return type
    ReturnType& bc_dt_u_psi =
        get<::Tags::Tempaa<27, VolumeDim, Frame::Inertial, DataVector>>(buffer);

    // Switch on prescribed boundary condition method
    switch (Method) {
      case UPsiBcMethod::Freezing:
        return make_with_value<ReturnType>(unit_normal_one_form, 0.);
      case UPsiBcMethod::ConstraintPreservingBjorhus:
        return apply_bjorhus_constraint_preserving(
            make_not_null(&bc_dt_u_psi), unit_interface_normal_vector,
            three_index_constraint, char_projected_rhs_dt_u_psi, char_speeds);
      case UPsiBcMethod::ConstraintPreservingDirichlet:
        return apply_dirichlet_constraint_preserving(
            make_not_null(&bc_dt_u_psi), lapse, shift, pi, phi);
      case UPsiBcMethod::Unknown:
      default:
        ASSERT(false, "Requested BC method fo UPsi not implemented!");
    }
  }

 private:
  static ReturnType apply_bjorhus_constraint_preserving(
      const gsl::not_null<ReturnType*> bc_dt_u_psi,
      const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
          unit_interface_normal_vector,
      const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>&
          three_index_constraint,
      const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>&
          char_projected_rhs_dt_u_psi,
      const std::array<DataVector, 4>& char_speeds) noexcept;
  static ReturnType apply_dirichlet_constraint_preserving(
      const gsl::not_null<ReturnType*> bc_dt_u_psi,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, VolumeDim, Frame::Inertial>& shift,
      const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>& pi,
      const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>& phi) noexcept;
};

template <typename ReturnType, size_t VolumeDim>
ReturnType
set_dt_u_psi<ReturnType, VolumeDim>::apply_bjorhus_constraint_preserving(
    const gsl::not_null<ReturnType*> bc_dt_u_psi,
    const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
        unit_interface_normal_vector,
    const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>&
        three_index_constraint,
    const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_u_psi,
    const std::array<DataVector, 4>& char_speeds) noexcept {
  if (UNLIKELY(get_size(get<0, 0>(*bc_dt_u_psi)) !=
               get_size(get<0>(unit_interface_normal_vector)))) {
    *bc_dt_u_psi = ReturnType(get_size(get<0>(unit_interface_normal_vector)));
  }
  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      bc_dt_u_psi->get(a, b) = char_projected_rhs_dt_u_psi.get(a, b);
      for (size_t i = 0; i < VolumeDim; ++i) {
        bc_dt_u_psi->get(a, b) += char_speeds.at(0) *
                                  unit_interface_normal_vector.get(i + 1) *
                                  three_index_constraint.get(i, a, b);
      }
    }
  }
  return *bc_dt_u_psi;
}

template <typename ReturnType, size_t VolumeDim>
ReturnType
set_dt_u_psi<ReturnType, VolumeDim>::apply_dirichlet_constraint_preserving(
    const gsl::not_null<ReturnType*> bc_dt_u_psi,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, VolumeDim, Frame::Inertial>& shift,
    const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>& phi) noexcept {
  if (UNLIKELY(get_size(get<0, 0>(*bc_dt_u_psi)) != get_size(get(lapse)))) {
    *bc_dt_u_psi = ReturnType(get_size(get(lapse)));
  }
  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      bc_dt_u_psi->get(a, b) = -get(lapse) * pi.get(a, b);
      for (size_t i = 0; i < VolumeDim; ++i) {
        bc_dt_u_psi->get(a, b) += shift.get(i) * phi.get(i, a, b);
      }
    }
  }
  return *bc_dt_u_psi;
}

// \brief This struct sets boundary condition on dt<UZero>
template <typename ReturnType, size_t VolumeDim>
struct set_dt_u_zero {
  template <typename TagsList, typename VarsTagsList, typename DtVarsTagsList>
  static ReturnType apply(const UZeroBcMethod Method,
                          TempBuffer<TagsList>& buffer,
                          const Variables<VarsTagsList>& vars,
                          const Variables<DtVarsTagsList>& /* dt_vars */,
                          const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
                              unit_normal_one_form) noexcept {
    // Not using auto below to enforce a loose test on the quantity being
    // fetched from the buffer
    const tnsr::A<DataVector, VolumeDim,
                  Frame::Inertial>& unit_interface_normal_vector =
        get<::Tags::TempA<5, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    const typename Tags::FourIndexConstraint<VolumeDim, Frame::Inertial>::type&
        four_index_constraint =
            get<::Tags::Tempiaa<18, VolumeDim, Frame::Inertial, DataVector>>(
                buffer);
    const typename Tags::UZero<VolumeDim, Frame::Inertial>::type&
        char_projected_rhs_dt_u_zero =
            get<::Tags::Tempiaa<23, VolumeDim, Frame::Inertial, DataVector>>(
                buffer);
    const typename Tags::CharacteristicSpeeds<VolumeDim, Frame::Inertial>::type
        char_speeds{{get(get<::Tags::TempScalar<12, DataVector>>(buffer)),
                     get(get<::Tags::TempScalar<13, DataVector>>(buffer)),
                     get(get<::Tags::TempScalar<14, DataVector>>(buffer)),
                     get(get<::Tags::TempScalar<15, DataVector>>(buffer))}};

    // spacetime unit vector t^a
    const auto& spacetime_unit_normal_vector =
        get<::Tags::TempA<7, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    const typename gr::Tags::Lapse<DataVector>::type& lapse =
        get<::Tags::TempScalar<19, DataVector>>(buffer);
    const typename gr::Tags::Shift<VolumeDim, Frame::Inertial,
                                   DataVector>::type& shift =
        get<::Tags::TempI<20, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    const auto& inverse_spatial_metric =
        get<::Tags::TempII<21, VolumeDim, Frame::Inertial, DataVector>>(buffer);

    const typename Tags::Pi<VolumeDim, Frame::Inertial>::type& pi =
        get<Tags::Pi<VolumeDim, Frame::Inertial>>(vars);
    const typename Tags::Phi<VolumeDim, Frame::Inertial>::type& phi =
        get<Tags::Phi<VolumeDim, Frame::Inertial>>(vars);
    const auto& d_spacetime_metric =
        get<::Tags::Tempiaa<31, VolumeDim, Frame::Inertial, DataVector>>(
            buffer);
    const auto& d_pi =
        get<::Tags::Tempiaa<32, VolumeDim, Frame::Inertial, DataVector>>(
            buffer);
    const auto& d_phi =
        get<::Tags::Tempijaa<33, VolumeDim, Frame::Inertial, DataVector>>(
            buffer);

    // Memory allocated for return type
    ReturnType& bc_dt_u_zero =
        get<::Tags::Tempiaa<28, VolumeDim, Frame::Inertial, DataVector>>(
            buffer);
    switch (Method) {
      case UZeroBcMethod::Freezing:
        return make_with_value<ReturnType>(unit_normal_one_form, 0.);
      case UZeroBcMethod::ConstraintPreservingBjorhus:
        return apply_bjorhus_constraint_preserving(
            make_not_null(&bc_dt_u_zero), unit_interface_normal_vector,
            four_index_constraint, char_projected_rhs_dt_u_zero, char_speeds);
      case UZeroBcMethod::ConstraintPreservingDirichlet:
        return apply_dirichlet_constraint_preserving(
            make_not_null(&bc_dt_u_zero), unit_interface_normal_vector,
            unit_normal_one_form, spacetime_unit_normal_vector, lapse, shift,
            inverse_spatial_metric, pi, phi, d_spacetime_metric, d_pi, d_phi);
      case UZeroBcMethod::ConstraintPreservingRealDirichlet:
        return apply_real_dirichlet_constraint_preserving(
            make_not_null(&bc_dt_u_zero), unit_interface_normal_vector,
            unit_normal_one_form, spacetime_unit_normal_vector, lapse, shift,
            inverse_spatial_metric, pi, phi, d_pi, d_phi);
      case UZeroBcMethod::Unknown:
      default:
        ASSERT(false, "Requested BC method fo UZero not implemented!");
    }
  }

 private:
  static ReturnType apply_bjorhus_constraint_preserving(
      const gsl::not_null<ReturnType*> bc_dt_u_zero,
      const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
          unit_interface_normal_vector,
      const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>&
          four_index_constraint,
      const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>&
          char_projected_rhs_dt_u_zero,
      const std::array<DataVector, 4>& char_speeds) noexcept;
  static ReturnType apply_dirichlet_constraint_preserving(
      const gsl::not_null<ReturnType*> bc_dt_u_zero,
      const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
          unit_interface_normal_vector,
      const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
          unit_interface_normal_one_form,
      const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
          spacetime_unit_normal_vector,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, VolumeDim, Frame::Inertial>& shift,
      const tnsr::II<DataVector, VolumeDim, Frame::Inertial>&
          inverse_spatial_metric,
      const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>& pi,
      const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>& phi,
      const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>& d_psi,
      const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>& d_pi,
      const tnsr::ijaa<DataVector, VolumeDim, Frame::Inertial>& d_phi) noexcept;
  static ReturnType apply_real_dirichlet_constraint_preserving(
      const gsl::not_null<ReturnType*> bc_dt_u_zero,
      const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
          unit_interface_normal_vector,
      const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
          unit_interface_normal_one_form,
      const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
          spacetime_unit_normal_vector,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, VolumeDim, Frame::Inertial>& shift,
      const tnsr::II<DataVector, VolumeDim, Frame::Inertial>&
          inverse_spatial_metric,
      const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>& pi,
      const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>& phi,
      const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>& d_pi,
      const tnsr::ijaa<DataVector, VolumeDim, Frame::Inertial>& d_phi) noexcept;
};

template <typename ReturnType, size_t VolumeDim>
ReturnType
set_dt_u_zero<ReturnType, VolumeDim>::apply_bjorhus_constraint_preserving(
    const gsl::not_null<ReturnType*> bc_dt_u_zero,
    const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
        unit_interface_normal_vector,
    const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>&
        four_index_constraint,
    const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_u_zero,
    const std::array<DataVector, 4>& char_speeds) noexcept {
  if (UNLIKELY(get_size(get<0, 0, 0>(*bc_dt_u_zero)) !=
               get_size(get<0>(unit_interface_normal_vector)))) {
    *bc_dt_u_zero = ReturnType(get_size(get<0>(unit_interface_normal_vector)));
  }
  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      for (size_t i = 0; i < VolumeDim; ++i) {
        bc_dt_u_zero->get(i, a, b) = char_projected_rhs_dt_u_zero.get(i, a, b);
      }
      // Lets say this term is T2_{jab} := - n_l N^l n^i C_{ijab}.
      // But we store C4_{iab} = LeviCivita^{ijk} C_{jkab},
      // which means  C_{ijab} = LeviCivita^{ijk} C4_{kab}
      // where C4 is `four_index_constraint`.
      // therefore, T2_{jab} =  char_speed<UZero> n^i C_{ijab},
      // = char_speed<UZero> n^i LeviCivita^{ijk} C4_{kab}; i.e.
      // if LeviCivitaIterator it is indexed by
      // it[0] <--> i,
      // it[1] <--> j,
      // it[2] <--> k, then
      // T2_{it[1], ab} += char_speed<UZero> n^it[0] it.sign() C4_{it[2], ab};
      for (LeviCivitaIterator<VolumeDim> it; it; ++it) {
        bc_dt_u_zero->get(it[1], a, b) +=
            it.sign() * char_speeds.at(1) *
            unit_interface_normal_vector.get(it[0] + 1) *
            four_index_constraint.get(it[2], a, b);
      }
    }
  }
  return *bc_dt_u_zero;
}

template <typename ReturnType, size_t VolumeDim>
ReturnType
set_dt_u_zero<ReturnType, VolumeDim>::apply_dirichlet_constraint_preserving(
    const gsl::not_null<ReturnType*> bc_dt_u_zero,
    const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
        unit_interface_normal_vector,
    const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
        unit_interface_normal_one_form,
    const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
        spacetime_unit_normal_vector,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, VolumeDim, Frame::Inertial>& shift,
    const tnsr::II<DataVector, VolumeDim, Frame::Inertial>&
        inverse_spatial_metric,
    const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>& phi,
    const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>& d_psi,
    const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>& d_pi,
    const tnsr::ijaa<DataVector, VolumeDim, Frame::Inertial>& d_phi) noexcept {
  if (UNLIKELY(get_size(get<0, 0, 0>(*bc_dt_u_zero)) != get_size(get(lapse)))) {
    *bc_dt_u_zero = ReturnType(get_size(get(lapse)));
  }
  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = 0; b <= VolumeDim; ++b) {
      // For a chosen (a, b):
      //  tmp_i = -N \partial_i \Pi_{ab} (T1)
      //        + 0.5 N * t^c t^d \partial_i \psi_{cd} Pi_{ab} (T2)
      //        + N^j \partial_i \Phi_{jab} (T3)
      //        + N t^e g^{mj} \partial_i \psi_{ej} \Phi_{mab} (T4)
      // and,
      //  bc_dt_u_zero_{iab} = tmp_i - n_i n^k tmp_k (for a chosen (a, b))
      tnsr::i<DataVector, VolumeDim, Frame::Inertial> tmp(get_size(get(lapse)));
      for (size_t i = 0; i < VolumeDim; ++i) {
        tmp.get(i) = -get(lapse) * d_pi.get(i, a, b);  // T1
        // (subtract dLapse*Pi)
        for (size_t c = 0; c <= VolumeDim; ++c) {
          for (size_t d = 0; d <= VolumeDim; ++d) {
            tmp.get(i) += 0.5 * get(lapse) *  // T2
                          spacetime_unit_normal_vector.get(c) *
                          spacetime_unit_normal_vector.get(d) *
                          d_psi.get(i, c, d) * pi.get(a, b);
          }
        }
        for (size_t j = 0; j < VolumeDim; ++j) {
          tmp.get(i) += shift.get(j) * d_phi.get(i, j, a, b);  // T3
          // (add d_k Shift^j \Phi_{jab})
          for (size_t k = 0; k < VolumeDim; ++k) {
            for (size_t e = 0; e <= VolumeDim; ++e) {
              tmp.get(i) += get(lapse) * spacetime_unit_normal_vector.get(e) *
                            inverse_spatial_metric.get(j, k) *
                            d_psi.get(i, e, k + 1) * phi.get(j, a, b);  // T4
            }
          }
        }
      }
      DataVector normal_dot_tmp(get_size(get(lapse)), 0.);
      for (size_t i = 0; i < VolumeDim; ++i) {
        normal_dot_tmp += unit_interface_normal_vector.get(i + 1) * tmp.get(i);
      }
      for (size_t i = 0; i < VolumeDim; ++i) {
        bc_dt_u_zero->get(i, a, b) =
            tmp.get(i) - unit_interface_normal_one_form.get(i) * normal_dot_tmp;
      }
    }
  }
  return *bc_dt_u_zero;
}

template <typename ReturnType, size_t VolumeDim>
ReturnType set_dt_u_zero<ReturnType, VolumeDim>::
    apply_real_dirichlet_constraint_preserving(
        const gsl::not_null<ReturnType*> bc_dt_u_zero,
        const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
            unit_interface_normal_vector,
        const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
            unit_interface_normal_one_form,
        const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
            spacetime_unit_normal_vector,
        const Scalar<DataVector>& lapse,
        const tnsr::I<DataVector, VolumeDim, Frame::Inertial>& shift,
        const tnsr::II<DataVector, VolumeDim, Frame::Inertial>&
            inverse_spatial_metric,
        const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>& pi,
        const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>& phi,
        const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>& d_pi,
        const tnsr::ijaa<DataVector, VolumeDim, Frame::Inertial>&
            d_phi) noexcept {
  if (UNLIKELY(get_size(get<0, 0, 0>(*bc_dt_u_zero)) != get_size(get(lapse)))) {
    *bc_dt_u_zero = ReturnType(get_size(get(lapse)));
  }
  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = 0; b <= VolumeDim; ++b) {
      // For a chosen (a, b):
      //  tmp_i = -N \partial_i \Pi_{ab} (T1)
      //        + 0.5 N * t^c t^d \phi_{icd} Pi_{ab} (T2)
      //        + N^j \partial_i \Phi_{jab} (T3)
      //        + N t^e g^{jk} \psi_{iek} \Phi_{jab} (T4)
      // and,
      //  bc_dt_u_zero_{iab} = tmp_i - n_i n^k tmp_k (for a chosen (a, b))
      tnsr::i<DataVector, VolumeDim, Frame::Inertial> tmp(get_size(get(lapse)));
      for (size_t i = 0; i < VolumeDim; ++i) {
        tmp.get(i) = -get(lapse) * d_pi.get(i, a, b);  // T1
        // (subtract dLapse*Pi)
        for (size_t c = 0; c <= VolumeDim; ++c) {
          for (size_t d = 0; d <= VolumeDim; ++d) {
            tmp.get(i) += 0.5 * get(lapse) *  // T2
                          spacetime_unit_normal_vector.get(c) *
                          spacetime_unit_normal_vector.get(d) *
                          phi.get(i, c, d) * pi.get(a, b);
          }
        }
        for (size_t j = 0; j < VolumeDim; ++j) {
          tmp.get(i) += shift.get(j) * d_phi.get(i, j, a, b);  // T3
          // (add d_k Shift^j \Phi_{jab})
          for (size_t k = 0; k < VolumeDim; ++k) {
            for (size_t e = 0; e <= VolumeDim; ++e) {
              tmp.get(i) += get(lapse) * spacetime_unit_normal_vector.get(e) *
                            inverse_spatial_metric.get(j, k) *
                            phi.get(i, e, k + 1) * phi.get(j, a, b);  // T4
            }
          }
        }
      }
      DataVector normal_dot_tmp(get_size(get(lapse)), 0.);
      for (size_t i = 0; i < VolumeDim; ++i) {
        normal_dot_tmp += unit_interface_normal_vector.get(i + 1) * tmp.get(i);
      }
      for (size_t i = 0; i < VolumeDim; ++i) {
        bc_dt_u_zero->get(i, a, b) =
            tmp.get(i) - unit_interface_normal_one_form.get(i) * normal_dot_tmp;
      }
    }
  }
  return *bc_dt_u_zero;
}

// \brief This struct sets boundary condition on dt<UPlus>
template <typename ReturnType, size_t VolumeDim>
struct set_dt_u_plus {
  template <typename TagsList, typename VarsTagsList, typename DtVarsTagsList>
  static ReturnType apply(const UPlusBcMethod Method,
                          TempBuffer<TagsList>& /* buffer */,
                          const Variables<VarsTagsList>& /* vars */,
                          const Variables<DtVarsTagsList>& /* dt_vars */,
                          const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
                              unit_normal_one_form) noexcept {
    switch (Method) {
      case UPlusBcMethod::Freezing:
        return make_with_value<ReturnType>(unit_normal_one_form, 0.);
      case UPlusBcMethod::Unknown:
      default:
        ASSERT(false, "Requested BC method fo UPlus not implemented!");
    }
  }

 private:
};

// \brief This struct sets boundary condition on dt<UMinus>
template <typename ReturnType, size_t VolumeDim>
struct set_dt_u_minus {
  template <typename TagsList, typename VarsTagsList, typename DtVarsTagsList>
  static ReturnType apply(const UMinusBcMethod Method,
                          TempBuffer<TagsList>& /* buffer */,
                          const Variables<VarsTagsList>& /* vars */,
                          const Variables<DtVarsTagsList>& /* dt_vars */,
                          const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
                              unit_normal_one_form) noexcept {
    switch (Method) {
      case UMinusBcMethod::Freezing:
        return make_with_value<ReturnType>(unit_normal_one_form, 0.);
      case UMinusBcMethod::Unknown:
      default:
        ASSERT(false, "Requested BC method fo UMinus not implemented!");
    }
  }

 private:
};
}  // namespace BoundaryConditions_detail
}  // namespace Actions
}  // namespace GeneralizedHarmonic
