// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

// debugPK
#include <iomanip>
#include <iostream>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/DataOnSlice.hpp"
#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditionsHelpers.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"
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

template <typename T1, typename T2>
void print_rank1_tensor_at_point(const std::string& name, const T1& tensor,
                                 const T2& coord_x, const T2& coord_y,
                                 const T2& coord_z, const size_t idx,
                                 const size_t VolumeDim = 3) noexcept {
  std::cout << "--- OUTPUT for " << name << "[" << coord_x[idx] << ", "
            << coord_y[idx] << ", " << coord_z[idx] << "] ---" << std::endl
            << std::flush;
  for (size_t a = 0; a <= VolumeDim; ++a) {
    std::cout << "  " << tensor.get(a)[idx] << "  " << std::flush;
  }
  std::cout << std::endl << std::flush;
}

template <typename T1, typename T2>
void print_rank2_tensor_at_point(const std::string& name, const T1& tensor,
                                 const T2& coord_x, const T2& coord_y,
                                 const T2& coord_z, const size_t idx,
                                 const size_t VolumeDim = 3) noexcept {
  std::cout << "--- OUTPUT for " << name << "[" << coord_x[idx] << ", "
            << coord_y[idx] << ", " << coord_z[idx] << "] ---" << std::endl
            << std::flush;
  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = 0; b <= VolumeDim; ++b) {
      std::cout << "  " << tensor.get(a, b)[idx] << "  " << std::flush;
    }
    std::cout << std::endl << std::flush;
  }
  std::cout << std::endl << std::flush;
}

template <typename T1, typename T2>
void print_rank3_tensor_at_point(const std::string& name, const T1& tensor,
                                 const T2& coord_x, const T2& coord_y,
                                 const T2& coord_z, const size_t idx,
                                 const size_t VolumeDim = 3) noexcept {
  std::cout << "--- OUTPUT for " << name << "[" << coord_x[idx] << ", "
            << coord_y[idx] << ", " << coord_z[idx] << "] ---" << std::endl
            << std::flush;
  for (size_t j = 0; j <= VolumeDim; ++j) {
    for (size_t a = 0; a <= VolumeDim; ++a) {
      for (size_t b = 0; b <= VolumeDim; ++b) {
        std::cout << "  " << tensor.get(j, a, b)[idx] << "  " << std::flush;
      }
      std::cout << std::endl << std::flush;
    }
    std::cout << std::endl << std::flush;
  }
  std::cout << std::endl << std::flush;
}
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
enum class UMinusBcMethod {
  AnalyticBc,
  Freezing,
  ConstraintPreservingBjorhus,
  ConstraintPreservingPhysicalBjorhus,
  Unknown
};

template <size_t VolumeDim>
using all_local_vars = tmpl::list<
    // timelike and spacelike SPACETIME oneforms, l_a and k_a
    ::Tags::Tempa<0, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::Tempa<1, VolumeDim, Frame::Inertial, DataVector>,
    // timelike and spacelike SPACETIME vectors, l^a and k^a
    ::Tags::TempA<2, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::TempA<3, VolumeDim, Frame::Inertial, DataVector>,
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
    // Characteristics of Constraints
    ::Tags::Tempa<16, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::Tempa<34, VolumeDim, Frame::Inertial, DataVector>,
    // ::Tags::Tempia<35, VolumeDim, Frame::Inertial, DataVector>,   // C_{ja}
    // ::Tags::Tempa<36, VolumeDim, Frame::Inertial, DataVector>,    // F_a
    ::Tags::Tempiaa<17, VolumeDim, Frame::Inertial, DataVector>,  // C_{jab}
    // e^{ijk} C_{jkab}
    ::Tags::Tempiaa<18, VolumeDim, Frame::Inertial, DataVector>,
    // 3+1 geometric quantities: lapse, shift, inverse
    // 3-metric, extrinsic curvature K_ij
    ::Tags::TempScalar<19, DataVector>,
    ::Tags::TempI<20, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::TempII<21, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::Tempii<35, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::TempAA<36, VolumeDim, Frame::Inertial, DataVector>,
    // Characteristic projected time derivatives of evolved
    // fields
    ::Tags::Tempaa<22, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::Tempiaa<23, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::Tempaa<24, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::Tempaa<25, VolumeDim, Frame::Inertial, DataVector>,
    // Constraint damping parameter gamma2
    ::Tags::TempScalar<26, DataVector>,
    // Preallocated memory to store boundary conditions
    ::Tags::Tempaa<27, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::Tempiaa<28, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::Tempaa<29, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::Tempaa<30, VolumeDim, Frame::Inertial, DataVector>,
    // derivatives of psi, pi, phi
    ::Tags::Tempiaa<31, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::Tempiaa<32, VolumeDim, Frame::Inertial, DataVector>,
    ::Tags::Tempijaa<33, VolumeDim, Frame::Inertial, DataVector>,
    // <34>, <35>, <36> defined above
    // mapped coordinates
    ::Tags::TempI<37, VolumeDim, Frame::Inertial, DataVector>>;

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
      gr::Tags::ExtrinsicCurvature<VolumeDim, Frame::Inertial, DataVector>,
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
      Tags::FConstraint<VolumeDim, Frame::Inertial>,
      ::Tags::MappedCoordinates<::Tags::ElementMap<3, Frame::Inertial>,
                                ::Tags::Coordinates<3, Frame::Logical>>
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
      get<::Tags::Tempa<0, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  auto& local_incoming_null_one_form =
      get<::Tags::Tempa<1, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  // timelike and spacelike SPACETIME oneforms, l_a and k_a
  auto& local_outgoing_null_vector =
      get<::Tags::TempA<2, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  auto& local_incoming_null_vector =
      get<::Tags::TempA<3, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
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
  auto& local_projection_AB =
      get<::Tags::TempAA<9, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  auto& local_projection_ab =
      get<::Tags::Tempaa<10, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  auto& local_projection_Ab =
      get<::Tags::TempAb<11, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  // 4.4) Characteristic speeds
  get(get<::Tags::TempScalar<12, DataVector>>(*buffer)) = char_speeds.at(0);
  get(get<::Tags::TempScalar<13, DataVector>>(*buffer)) = char_speeds.at(1);
  get(get<::Tags::TempScalar<14, DataVector>>(*buffer)) = char_speeds.at(2);
  get(get<::Tags::TempScalar<15, DataVector>>(*buffer)) = char_speeds.at(3);
  // constraint characteristics
  auto& local_constraint_char_zero_minus =
      get<::Tags::Tempa<16, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  auto& local_constraint_char_zero_plus =
      get<::Tags::Tempa<34, VolumeDim, Frame::Inertial, DataVector>>(*buffer);
  // 4.6) c^\hat{3}_{jab} = C_{jab} = \partial_j\psi_{ab} - \Phi_{jab}
  get<::Tags::Tempiaa<17, VolumeDim, Frame::Inertial, DataVector>>(*buffer) =
      get<Tags::ThreeIndexConstraint<VolumeDim, Frame::Inertial>>(
          vars_on_this_slice);
  // 4.7) c^\hat{4}_{ijab} = C_{ijab}
  get<::Tags::Tempiaa<18, VolumeDim, Frame::Inertial, DataVector>>(*buffer) =
      get<Tags::FourIndexConstraint<VolumeDim, Frame::Inertial>>(
          vars_on_this_slice);
  // lapse, shift and inverse spatial_metric
  get<::Tags::TempScalar<19, DataVector>>(*buffer) =
      get<gr::Tags::Lapse<DataVector>>(vars_on_this_slice);
  get<::Tags::TempI<20, VolumeDim, Frame::Inertial, DataVector>>(*buffer) =
      get<gr::Tags::Shift<VolumeDim, Frame::Inertial, DataVector>>(
          vars_on_this_slice);
  get<::Tags::TempII<21, VolumeDim, Frame::Inertial, DataVector>>(*buffer) =
      get<gr::Tags::InverseSpatialMetric<VolumeDim, Frame::Inertial,
                                         DataVector>>(vars_on_this_slice);
  get<::Tags::Tempii<35, VolumeDim, Frame::Inertial, DataVector>>(*buffer) =
      get<gr::Tags::ExtrinsicCurvature<VolumeDim, Frame::Inertial, DataVector>>(
          vars_on_this_slice);
  get<::Tags::TempAA<36, VolumeDim, Frame::Inertial, DataVector>>(*buffer) =
      get<gr::Tags::InverseSpacetimeMetric<VolumeDim, Frame::Inertial,
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

  // Coordinates
  get<::Tags::TempI<37, VolumeDim, Frame::Inertial, DataVector>>(*buffer) =
      get<::Tags::MappedCoordinates<
          ::Tags::ElementMap<VolumeDim, Frame::Inertial>,
          ::Tags::Coordinates<VolumeDim, Frame::Logical>>>(vars_on_this_slice);

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
          spacetime_normal_one_form.get(a) * spacetime_normal_one_form.get(b) -
          local_interface_normal_one_form.get(a) *
              local_interface_normal_one_form.get(b);
      local_projection_Ab.get(a, b) =
          spacetime_normal_one_form.get(a) * spacetime_normal_vector.get(b) -
          local_interface_normal_one_form.get(a) *
              local_interface_normal_vector.get(b);
      if (UNLIKELY(a == b)) {
        local_projection_Ab.get(a, b) += 1.;
      }
      local_projection_AB.get(a, b) =
          inverse_spacetime_metric.get(a, b) +
          spacetime_normal_vector.get(a) * spacetime_normal_vector.get(b) -
          local_interface_normal_vector.get(a) *
              local_interface_normal_vector.get(b);
    }
  }
  // 4.5) c^{\hat{0}-}_a = F_a + n^k C_{ka}
  for (size_t a = 0; a < VolumeDim + 1; ++a) {
    local_constraint_char_zero_minus.get(a) = f_constraint.get(a);
    local_constraint_char_zero_plus.get(a) = f_constraint.get(a);
    for (size_t i = 0; i < VolumeDim; ++i) {
      local_constraint_char_zero_minus.get(a) +=
          unit_interface_normal_vector.get(i) * two_index_constraint.get(i, a);
      local_constraint_char_zero_plus.get(a) -=
          unit_interface_normal_vector.get(i) * two_index_constraint.get(i, a);
    }
  }
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
                          /* unit_normal_one_form */) noexcept {
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
    // --------------------------------------- TESTS
#if 0
    // POPULATE various tensors needed to compute BcDtUpsi
    // EXACTLY as done in SpEC
    auto local_inertial_coords =
        get<::Tags::TempI<37, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto local_three_index_constraint = three_index_constraint;
    auto local_unit_interface_normal_vector = unit_interface_normal_vector;
    auto local_lapse = lapse;
    auto local_shift = shift;
    auto local_pi = pi;
    auto local_phi = phi;
    auto local_char_projected_rhs_dt_u_psi = char_projected_rhs_dt_u_psi;
    auto local_char_speeds = char_speeds;
    // Setting 3idxConstraint
    for (size_t i = 0; i < get(lapse).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = 0; b <= VolumeDim; ++b) {
          local_three_index_constraint.get(0, a, b)[i] = 11. - 3.;
          local_three_index_constraint.get(1, a, b)[i] = 13. - 5.;
          local_three_index_constraint.get(2, a, b)[i] = 17. - 7.;
        }
      }
    }
    // Setting unit_interface_normal_Vector
    for (size_t i = 0; i < get(lapse).size(); ++i) {
      local_unit_interface_normal_vector.get(0)[i] = 1.e300;
      local_unit_interface_normal_vector.get(1)[i] = -1.;
      local_unit_interface_normal_vector.get(2)[i] = 1.;
      local_unit_interface_normal_vector.get(3)[i] = 1.;
    }
    // Setting lapse
    for (size_t i = 0; i < get(lapse).size(); ++i) {
      get(local_lapse)[i] = 2.;
    }
    // Setting shift
    for (size_t i = 0; i < get(lapse).size(); ++i) {
      local_shift.get(0)[i] = 1.;
      local_shift.get(1)[i] = 2.;
      local_shift.get(2)[i] = 3.;
    }
    // Setting pi AND phi
    for (size_t i = 0; i < get(lapse).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = 0; b <= VolumeDim; ++b) {
          local_pi.get(a, b)[i] = 1.;
          local_phi.get(0, a, b)[i] = 3.;
          local_phi.get(1, a, b)[i] = 5.;
          local_phi.get(2, a, b)[i] = 7.;
        }
      }
    }
    // Setting local_RhsUPsi
    for (size_t i = 0; i < get(lapse).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_char_projected_rhs_dt_u_psi.get(0, a)[i] = 23.;
        local_char_projected_rhs_dt_u_psi.get(1, a)[i] = 29.;
        local_char_projected_rhs_dt_u_psi.get(2, a)[i] = 31.;
        local_char_projected_rhs_dt_u_psi.get(3, a)[i] = 37.;
      }
    }
    // Setting char speeds
    for (size_t i = 0; i < get(lapse).size(); ++i) {
      local_char_speeds.at(0)[i] = -0.3;
      local_char_speeds.at(1)[i] = -0.1;
    }

    // debugPK
    std::fill(bc_dt_u_psi.begin(), bc_dt_u_psi.end(), 0.);
    auto _ = apply_bjorhus_constraint_preserving(
        make_not_null(&bc_dt_u_psi), local_unit_interface_normal_vector,
        local_three_index_constraint, local_char_projected_rhs_dt_u_psi,
        local_char_speeds);
    // DISPLAY results of the TEST
    if (debugPKon) {
      for (size_t i = 0; i < get(lapse).size(); ++i) {
        print_rank2_tensor_at_point(
            "BcDtUpsi", bc_dt_u_psi, local_inertial_coords.get(0),
            local_inertial_coords.get(1), local_inertial_coords.get(2), i);
      }
    }
#endif
    // --------------------------------------- TESTS
    std::fill(bc_dt_u_psi.begin(), bc_dt_u_psi.end(), 0.);
    // Switch on prescribed boundary condition method
    switch (Method) {
      case UPsiBcMethod::Freezing:
        return bc_dt_u_psi;
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
    // dummy return to suppress compiler warning
    return ReturnType{};
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
  ASSERT(get_size(get<0, 0>(*bc_dt_u_psi)) ==
             get_size(get<0>(unit_interface_normal_vector)),
         "Size of input variables and temporary memory do not match.");
  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      bc_dt_u_psi->get(a, b) += char_projected_rhs_dt_u_psi.get(a, b);
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
  ASSERT(get_size(get<0, 0>(*bc_dt_u_psi)) == get_size(get(lapse)),
         "Size of input variables and temporary memory do not match.");
  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      bc_dt_u_psi->get(a, b) -= get(lapse) * pi.get(a, b);
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
// --------------------------------------- TESTS
#if 0
    std::fill(bc_dt_u_zero.begin(), bc_dt_u_zero.end(), 0.);
    std::cout << "1. INSIDE set_dt_u_zero --- " << std::endl;
    // POPULATE various tensors needed to compute BcDtU0
    // EXACTLY as done in SpEC
    auto local_inertial_coords =
        get<::Tags::TempI<37, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto local_unit_interface_normal_vector = unit_interface_normal_vector;
    auto local_lapse = lapse;
    auto local_shift = shift;
    auto local_pi = pi;
    auto local_phi = phi;
    auto local_char_projected_rhs_dt_u_zero = char_projected_rhs_dt_u_zero;
    auto local_char_speeds = char_speeds;
    // Setting 4idxConstraint:
    // initialize dPhi (with same values as for SpEC) and compute C4 from it
    for (size_t a = 0; a <= VolumeDim; ++a) {
      for (size_t b = 0; b <= VolumeDim; ++b) {
        for (size_t i = 0; i < get(lapse).size(); ++i) {
          local_dphi.get(0, 0, a, b)[i] = 3.;
          local_dphi.get(0, 1, a, b)[i] = 5.;
          local_dphi.get(0, 2, a, b)[i] = 7.;
          local_dphi.get(1, 0, a, b)[i] = 59.;
          local_dphi.get(1, 1, a, b)[i] = 61.;
          local_dphi.get(1, 2, a, b)[i] = 67.;
          local_dphi.get(2, 0, a, b)[i] = 73.;
          local_dphi.get(2, 1, a, b)[i] = 79.;
          local_dphi.get(2, 2, a, b)[i] = 83.;
        }
      }
    }
    // But we store C4_{iab} = LeviCivita^{ijk} dphi_{jkab}
    auto local_four_index_constraint =
        GeneralizedHarmonic::four_index_constraint<VolumeDim, Frame::Inertial,
                                                   DataVector>(local_dphi);

    if (debugPKon) {
      for (size_t i = 0; i < get(lapse).size(); ++i) {
        print_rank3_tensor_at_point("local_4idxC", local_four_index_constraint,
                                    local_inertial_coords.get(0),
                                    local_inertial_coords.get(1),
                                    local_inertial_coords.get(2), i);
      }
    }
    std::cout << "2. INSIDE set_dt_u_zero --- " << std::endl;

    if (debugPKoff) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = 0; b <= VolumeDim; ++b) {
          std::cout << "a = " << a << ", b = " << b << std::endl
                    << "\t No of grid points: " << get(lapse).size() << ", "
                    << get(lapse).size() << ", " << shift.size() << ", "
                    << shift.get(0).size() << std::endl
                    << "\t local_lapse: " << local_lapse.get() << std::endl
                    << "\t local_shift: " << local_shift.get(0) << std::endl
                    << "\t local_unit_int_normal: "
                    << local_unit_interface_normal_vector.get(0) << std::endl
                    << "\t local_dphi: " << local_dphi.get(0, 0, a, b)
                    << std::endl
                    << "\t local_4idxC: "
                    << local_four_index_constraint.get(0, a, b) << std::endl
                    << std::flush;
        }
      }
    }
    // Setting unit_interface_normal_Vector
    for (size_t i = 0; i < get(lapse).size(); ++i) {
      local_unit_interface_normal_vector.get(0)[i] = 1.e300;
      local_unit_interface_normal_vector.get(1)[i] = -1.;
      local_unit_interface_normal_vector.get(2)[i] = 1.;
      local_unit_interface_normal_vector.get(3)[i] = 1.;
    }
    // Setting lapse
    for (size_t i = 0; i < get(lapse).size(); ++i) {
      get(local_lapse)[i] = 2.;
    }
    // Setting shift
    for (size_t i = 0; i < get(lapse).size(); ++i) {
      local_shift.get(0)[i] = 1.;
      local_shift.get(1)[i] = 2.;
      local_shift.get(2)[i] = 3.;
    }
    // Setting pi AND phi
    for (size_t i = 0; i < get(lapse).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = 0; b <= VolumeDim; ++b) {
          local_pi.get(a, b)[i] = 1.;
          local_phi.get(0, a, b)[i] = 3.;
          local_phi.get(1, a, b)[i] = 5.;
          local_phi.get(2, a, b)[i] = 7.;
        }
      }
    }
    // Setting local_RhsU0
    for (size_t i = 0; i < get(lapse).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          local_char_projected_rhs_dt_u_zero.get(0, a, b)[i] = 91.;
          local_char_projected_rhs_dt_u_zero.get(1, a, b)[i] = 97.;
          local_char_projected_rhs_dt_u_zero.get(2, a, b)[i] = 101.;
        }
      }
    }
    // Setting char speeds
    for (size_t i = 0; i < get(lapse).size(); ++i) {
      local_char_speeds.at(0)[i] = -0.3;
      local_char_speeds.at(1)[i] = -0.1;
    }
    std::cout << "3. INSIDE set_dt_u_zero --- " << std::endl;

    // debugPK
    auto _ = apply_bjorhus_constraint_preserving(
        make_not_null(&bc_dt_u_zero), local_unit_interface_normal_vector,
        local_four_index_constraint, local_char_projected_rhs_dt_u_zero,
        local_char_speeds);
    std::cout << "4. INSIDE set_dt_u_zero --- " << std::endl;
    // DISPLAY results of the TEST
    if (debugPKon) {
      for (size_t i = 0; i < get(lapse).size(); ++i) {
        print_rank3_tensor_at_point(
            "BcDtUZero", bc_dt_u_zero, local_inertial_coords.get(0),
            local_inertial_coords.get(1), local_inertial_coords.get(2), i);
      }
    }
#endif
    // --------------------------------------- TESTS
    std::fill(bc_dt_u_zero.begin(), bc_dt_u_zero.end(), 0.);
    // Switch on prescribed boundary condition method
    switch (Method) {
      case UZeroBcMethod::Freezing:
        return bc_dt_u_zero;
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
    // dummy return to suppress compiler warning
    return ReturnType{};
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
  ASSERT(get_size(get<0, 0, 0>(*bc_dt_u_zero)) ==
             get_size(get<0>(unit_interface_normal_vector)),
         "Size of input variables and temporary memory do not match.");

  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      for (size_t i = 0; i < VolumeDim; ++i) {
        bc_dt_u_zero->get(i, a, b) += char_projected_rhs_dt_u_zero.get(i, a, b);
      }
      // Lets say this term is T2_{kab} := - n_l N^l n^j C_{jkab}.
      // But we store C4_{iab} = LeviCivita^{ijk} dphi_{jkab},
      // which means  C_{jkab} = LeviCivita^{ijk} C4_{iab}
      // where C4 is `four_index_constraint`.
      // therefore, T2_{iab} =  char_speed<UZero> n^j C_{jiab}
      // (since char_speed<UZero> = - n_l N^l), and therefore:
      // T2_{iab} = char_speed<UZero> n^k LeviCivita^{ijk} C4_{jab}.
      // Let LeviCivitaIterator be indexed by
      // it[0] <--> i,
      // it[1] <--> j,
      // it[2] <--> k, then
      // T2_{it[0], ab} += char_speed<UZero> n^it[2] it.sign() C4_{it[1], ab};
      for (LeviCivitaIterator<VolumeDim> it; it; ++it) {
        bc_dt_u_zero->get(it[0], a, b) +=
            it.sign() * char_speeds.at(1) *
            unit_interface_normal_vector.get(it[2] + 1) *
            four_index_constraint.get(it[1], a, b);
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
  ASSERT(get_size(get<0, 0, 0>(*bc_dt_u_zero)) == get_size(get(lapse)),
         "Size of input variables and temporary memory do not match.");
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
        bc_dt_u_zero->get(i, a, b) +=
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
  ASSERT(get_size(get<0, 0, 0>(*bc_dt_u_zero)) == get_size(get(lapse)),
         "Size of input variables and temporary memory do not match.");
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
        bc_dt_u_zero->get(i, a, b) +=
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
                          TempBuffer<TagsList>& buffer,
                          const Variables<VarsTagsList>& /* vars */,
                          const Variables<DtVarsTagsList>& /* dt_vars */,
                          const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
                          /* unit_normal_one_form */) noexcept {
    // Memory allocated for return type
    ReturnType& bc_dt_u_plus =
        get<::Tags::Tempaa<29, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    std::fill(bc_dt_u_plus.begin(), bc_dt_u_plus.end(), 0.);
    // Switch on prescribed boundary condition method
    switch (Method) {
      case UPlusBcMethod::Freezing:
        return bc_dt_u_plus;
      case UPlusBcMethod::Unknown:
      default:
        ASSERT(false, "Requested BC method fo UPlus not implemented!");
    }
    // dummy return to suppress compiler warning
    return ReturnType{};
  }

 private:
};

// \brief This struct sets boundary condition on dt<UMinus>
template <typename ReturnType, size_t VolumeDim>
struct set_dt_u_minus {
  template <typename TagsList, typename VarsTagsList, typename DtVarsTagsList>
  static ReturnType apply(
      const UMinusBcMethod Method, TempBuffer<TagsList>& buffer,
      const Variables<VarsTagsList>& vars,
      const Variables<DtVarsTagsList>& /* dt_vars */,
      const typename ::Tags::Coordinates<VolumeDim, Frame::Inertial>::type&
          inertial_coords,
      const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
      /* unit_normal_one_form */) noexcept {
    // Not using auto below to enforce a loose test on the quantity being
    // fetched from the buffer
    const typename Tags::ConstraintGamma2::type& constraint_gamma2 =
        get<::Tags::TempScalar<26, DataVector>>(buffer);
    const typename Tags::ThreeIndexConstraint<VolumeDim, Frame::Inertial>::type&
        three_index_constraint =
            get<::Tags::Tempiaa<17, VolumeDim, Frame::Inertial, DataVector>>(
                buffer);
    const tnsr::a<DataVector, VolumeDim,
                  Frame::Inertial>& unit_interface_normal_one_form =
        get<::Tags::Tempa<4, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    const tnsr::A<DataVector, VolumeDim,
                  Frame::Inertial>& unit_interface_normal_vector =
        get<::Tags::TempA<5, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    const tnsr::A<DataVector, VolumeDim,
                  Frame::Inertial>& spacetime_unit_normal_vector =
        get<::Tags::TempA<7, VolumeDim, Frame::Inertial, DataVector>>(buffer);

    const typename Tags::UPsi<VolumeDim, Frame::Inertial>::type&
        char_projected_rhs_dt_u_psi =
            get<::Tags::Tempaa<22, VolumeDim, Frame::Inertial, DataVector>>(
                buffer);
    const typename Tags::UMinus<VolumeDim, Frame::Inertial>::type&
        char_projected_rhs_dt_u_minus =
            get<::Tags::Tempaa<25, VolumeDim, Frame::Inertial, DataVector>>(
                buffer);
    const typename Tags::CharacteristicSpeeds<VolumeDim, Frame::Inertial>::type
        char_speeds{{get(get<::Tags::TempScalar<12, DataVector>>(buffer)),
                     get(get<::Tags::TempScalar<13, DataVector>>(buffer)),
                     get(get<::Tags::TempScalar<14, DataVector>>(buffer)),
                     get(get<::Tags::TempScalar<15, DataVector>>(buffer))}};

    // timelike and spacelike SPACETIME vectors, l^a and k^a
    const auto& outgoing_null_one_form =
        get<::Tags::Tempa<0, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    const auto& incoming_null_one_form =
        get<::Tags::Tempa<1, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    // timelike and spacelike SPACETIME oneforms, l_a and k_a
    const auto& outgoing_null_vector =
        get<::Tags::TempA<2, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    const auto& incoming_null_vector =
        get<::Tags::TempA<3, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    // spacetime projection operator P_ab, P^ab, and P^a_b
    const auto& projection_AB =
        get<::Tags::TempAA<9, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    const auto& projection_ab =
        get<::Tags::Tempaa<10, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    const auto& projection_Ab =
        get<::Tags::TempAb<11, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    // constraint characteristics
    const auto& constraint_char_zero_minus =
        get<::Tags::Tempa<16, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    const auto& constraint_char_zero_plus =
        get<::Tags::Tempa<34, VolumeDim, Frame::Inertial, DataVector>>(buffer);

    const auto& inverse_spatial_metric =
        get<::Tags::TempII<21, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    const auto& extrinsic_curvature =
        get<::Tags::Tempii<35, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    const auto& inverse_spacetime_metric =
        get<::Tags::TempAA<36, VolumeDim, Frame::Inertial, DataVector>>(buffer);

    const typename gr::Tags::SpacetimeMetric<
        VolumeDim, Frame::Inertial, DataVector>::type& spacetime_metric =
        get<gr::Tags::SpacetimeMetric<VolumeDim, Frame::Inertial, DataVector>>(
            vars);
    const typename Tags::Phi<VolumeDim, Frame::Inertial>::type& phi =
        get<Tags::Phi<VolumeDim, Frame::Inertial>>(vars);
    const auto& d_pi =
        get<::Tags::Tempiaa<32, VolumeDim, Frame::Inertial, DataVector>>(
            buffer);
    const auto& d_phi =
        get<::Tags::Tempijaa<33, VolumeDim, Frame::Inertial, DataVector>>(
            buffer);
    // Memory allocated for return type
    ReturnType& bc_dt_u_minus =
        get<::Tags::Tempaa<30, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    // --------------------------------------- TESTS
#if 1
    std::cout << std::setprecision(16);
    // POPULATE various tensors needed to compute BcDtUMinus
    // EXACTLY as done in SpEC
    auto& local_inertial_coords =
        get<::Tags::TempI<37, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto local_three_index_constraint = three_index_constraint;
    // auto local_lapse = lapse;
    // auto local_shift = shift;
    // auto local_pi = pi;
    auto local_phi = phi;
    auto local_char_projected_rhs_dt_u_psi = char_projected_rhs_dt_u_psi;
    auto local_char_speeds = char_speeds;
    //
    auto local_constraint_gamma2 = constraint_gamma2;
    auto local_incoming_null_one_form = incoming_null_one_form;
    auto local_incoming_null_vector = incoming_null_vector;
    auto local_outgoing_null_one_form = outgoing_null_one_form;
    auto local_outgoing_null_vector = outgoing_null_vector;
    auto local_projection_Ab = projection_Ab;
    auto local_projection_ab = projection_ab;
    auto local_projection_AB = projection_AB;
    auto local_spacetime_metric = spacetime_metric;
    // auto local_inverse_spacetime_metric = inverse_spacetime_metric;
    // auto local_gauge_source = gauge_source;
    auto local_char_projected_rhs_dt_u_minus = char_projected_rhs_dt_u_minus;
    //
    auto local_constraint_char_zero_minus = constraint_char_zero_minus;
    auto local_constraint_char_zero_plus = constraint_char_zero_plus;
    // for CPBPhys
    auto local_d_phi = d_phi;
    auto local_d_pi = d_pi;
    auto local_unit_interface_normal_one_form = unit_interface_normal_one_form;
    auto local_unit_interface_normal_vector = unit_interface_normal_vector;
    auto local_spacetime_unit_normal_vector = spacetime_unit_normal_vector;
    auto local_inverse_spatial_metric = inverse_spatial_metric;
    auto local_inverse_spacetime_metric = inverse_spacetime_metric;
    auto local_extrinsic_curvature = extrinsic_curvature;

    // Setting local_spacetime_unit_normal_vector
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_spacetime_unit_normal_vector.get(0)[i] = -1.;
      local_spacetime_unit_normal_vector.get(1)[i] = -3.;
      local_spacetime_unit_normal_vector.get(2)[i] = -5.;
      local_spacetime_unit_normal_vector.get(3)[i] = -7.;
    }
    // Setting local_unit_interface_normal_one_form
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_unit_interface_normal_one_form.get(0)[i] = 1.e300;
      local_unit_interface_normal_one_form.get(1)[i] = -1.;
      local_unit_interface_normal_one_form.get(2)[i] = 1.;
      local_unit_interface_normal_one_form.get(3)[i] = 1.;
    }
    // Setting unit_interface_normal_Vector
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_unit_interface_normal_vector.get(0)[i] = 1.e300;
      local_unit_interface_normal_vector.get(1)[i] = -1.;
      local_unit_interface_normal_vector.get(2)[i] = 1.;
      local_unit_interface_normal_vector.get(3)[i] = 1.;
    }
    // Setting lapse
    // for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
    // get(local_lapse)[i] = 2.;
    //}
    // Setting shift
    // for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
    // local_shift.get(0)[i] = 1.;
    // local_shift.get(1)[i] = 2.;
    // local_shift.get(2)[i] = 3.;
    //}
    // Setting local_inverse_spatial_metric
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t j = 0; j < VolumeDim; ++j) {
        local_inverse_spatial_metric.get(0, j)[i] = 41.;
        local_inverse_spatial_metric.get(1, j)[i] = 43.;
        local_inverse_spatial_metric.get(2, j)[i] = 47.;
      }
    }
    // Setting local_spacetime_metric
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_spacetime_metric.get(0, a)[i] = 257.;
        local_spacetime_metric.get(1, a)[i] = 263.;
        local_spacetime_metric.get(2, a)[i] = 269.;
        local_spacetime_metric.get(3, a)[i] = 271.;
      }
    }
    // Setting local_inverse_spacetime_metric
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_inverse_spacetime_metric.get(0, a)[i] =
            -277.;  // needs to be < 0 for lapse
        local_inverse_spacetime_metric.get(1, a)[i] = 281.;
        local_inverse_spacetime_metric.get(2, a)[i] = 283.;
        local_inverse_spacetime_metric.get(3, a)[i] = 293.;
      }
    }
    // Setting local_extrinsic_curvature
    // ONLY ON THE +Y AXIS (Y = +0.5) -- FIXME
    //
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      if (get<0>(local_inertial_coords)[i] == 299. and
          get<1>(local_inertial_coords)[i] == 0.5 and
          get<2>(local_inertial_coords)[i] == -0.5) {
        std::array<double, 9> spec_vals = {
            {200.2198037251189, 266.7930716334918, 333.3663395418648,
             266.7930716334918, 333.3663395418648, 399.9396074502377,
             333.3663395418648, 399.9396074502377, 466.5128753586106}};
        for (size_t j = 0; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            local_extrinsic_curvature.get(j, k)[i] =
                spec_vals[j * (0 + VolumeDim) + k];
          }
        }
      } else if (get<0>(local_inertial_coords)[i] == 299.5 and
                 get<1>(local_inertial_coords)[i] == 0.5 and
                 get<2>(local_inertial_coords)[i] == -0.5) {
        std::array<double, 9> spec_vals = {
            {200.2198037251189, 266.7930716334918, 333.3663395418648,
             266.7930716334918, 333.3663395418648, 399.9396074502377,
             333.3663395418648, 399.9396074502377, 466.5128753586106}};
        for (size_t j = 0; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            local_extrinsic_curvature.get(j, k)[i] =
                spec_vals[j * (0 + VolumeDim) + k];
          }
        }
      } else if (get<0>(local_inertial_coords)[i] == 300. and
                 get<1>(local_inertial_coords)[i] == 0.5 and
                 get<2>(local_inertial_coords)[i] == -0.5) {
        std::array<double, 9> spec_vals = {
            {200.2198037251189, 266.7930716334918, 333.3663395418648,
             266.7930716334918, 333.3663395418648, 399.9396074502377,
             333.3663395418648, 399.9396074502377, 466.5128753586106}};
        for (size_t j = 0; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            local_extrinsic_curvature.get(j, k)[i] =
                spec_vals[j * (0 + VolumeDim) + k];
          }
        }
      } else if (get<0>(local_inertial_coords)[i] == 299. and
                 get<1>(local_inertial_coords)[i] == 0.5 and
                 get<2>(local_inertial_coords)[i] == 0.) {
        std::array<double, 9> spec_vals = {
            {200.2198037251189, 266.7930716334918, 333.3663395418648,
             266.7930716334918, 333.3663395418648, 399.9396074502377,
             333.3663395418648, 399.9396074502377, 466.5128753586106}};
        for (size_t j = 0; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            local_extrinsic_curvature.get(j, k)[i] =
                spec_vals[j * (0 + VolumeDim) + k];
          }
        }
      } else if (get<0>(local_inertial_coords)[i] == 299.5 and
                 get<1>(local_inertial_coords)[i] == 0.5 and
                 get<2>(local_inertial_coords)[i] == 0.) {
        std::array<double, 9> spec_vals = {
            {200.2198037251189, 266.7930716334918, 333.3663395418648,
             266.7930716334918, 333.3663395418648, 399.9396074502377,
             333.3663395418648, 399.9396074502377, 466.5128753586106}};
        for (size_t j = 0; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            local_extrinsic_curvature.get(j, k)[i] =
                spec_vals[j * (0 + VolumeDim) + k];
          }
        }
      } else if (get<0>(local_inertial_coords)[i] == 300. and
                 get<1>(local_inertial_coords)[i] == 0.5 and
                 get<2>(local_inertial_coords)[i] == 0.) {
        std::array<double, 9> spec_vals = {
            {200.2198037251189, 266.7930716334918, 333.3663395418648,
             266.7930716334918, 333.3663395418648, 399.9396074502377,
             333.3663395418648, 399.9396074502377, 466.5128753586106}};
        for (size_t j = 0; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            local_extrinsic_curvature.get(j, k)[i] =
                spec_vals[j * (0 + VolumeDim) + k];
          }
        }
      } else if (get<0>(local_inertial_coords)[i] == 299. and
                 get<1>(local_inertial_coords)[i] == 0.5 and
                 get<2>(local_inertial_coords)[i] == 0.5) {
        std::array<double, 9> spec_vals = {
            {200.2198037251189, 266.7930716334918, 333.3663395418648,
             266.7930716334918, 333.3663395418648, 399.9396074502377,
             333.3663395418648, 399.9396074502377, 466.5128753586106}};
        for (size_t j = 0; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            local_extrinsic_curvature.get(j, k)[i] =
                spec_vals[j * (0 + VolumeDim) + k];
          }
        }
      } else if (get<0>(local_inertial_coords)[i] == 299.5 and
                 get<1>(local_inertial_coords)[i] == 0.5 and
                 get<2>(local_inertial_coords)[i] == 0.5) {
        std::array<double, 9> spec_vals = {
            {200.2198037251189, 266.7930716334918, 333.3663395418648,
             266.7930716334918, 333.3663395418648, 399.9396074502377,
             333.3663395418648, 399.9396074502377, 466.5128753586106}};
        for (size_t j = 0; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            local_extrinsic_curvature.get(j, k)[i] =
                spec_vals[j * (0 + VolumeDim) + k];
          }
        }
      } else if (get<0>(local_inertial_coords)[i] == 300. and
                 get<1>(local_inertial_coords)[i] == 0.5 and
                 get<2>(local_inertial_coords)[i] == 0.5) {
        std::array<double, 9> spec_vals = {
            {200.2198037251189, 266.7930716334918, 333.3663395418648,
             266.7930716334918, 333.3663395418648, 399.9396074502377,
             333.3663395418648, 399.9396074502377, 466.5128753586106}};
        for (size_t j = 0; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            local_extrinsic_curvature.get(j, k)[i] =
                spec_vals[j * (0 + VolumeDim) + k];
          }
        }
      } else {
        // When applying BCs to various faces, only one face (i.e. the +y side)
        // will be set here, others can be set however...
        // FIXME: Disabled
        ASSERT(true,
               "Not checking the correct face, coordinates not recognized");
      }
    }

    // Setting pi AND phi
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = 0; b <= VolumeDim; ++b) {
          // local_pi.get(a, b)[i] = 1.;
          local_phi.get(0, a, b)[i] = 3.;
          local_phi.get(1, a, b)[i] = 5.;
          local_phi.get(2, a, b)[i] = 7.;
        }
      }
    }
    // Setting local_d_phi
    // initialize dPhi (with same values as for SpEC)
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = 0; b <= VolumeDim; ++b) {
          local_d_pi.get(0, a, b)[i] = 1.;
          local_d_phi.get(0, 0, a, b)[i] = 3.;
          local_d_phi.get(0, 1, a, b)[i] = 5.;
          local_d_phi.get(0, 2, a, b)[i] = 7.;
          local_d_pi.get(1, a, b)[i] = 53.;
          local_d_phi.get(1, 0, a, b)[i] = 59.;
          local_d_phi.get(1, 1, a, b)[i] = 61.;
          local_d_phi.get(1, 2, a, b)[i] = 67.;
          local_d_pi.get(2, a, b)[i] = 71.;
          local_d_phi.get(2, 0, a, b)[i] = 73.;
          local_d_phi.get(2, 1, a, b)[i] = 79.;
          local_d_phi.get(2, 2, a, b)[i] = 83.;
        }
      }
    }
    // Setting char speeds
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_char_speeds.at(0)[i] = -0.3;
      local_char_speeds.at(1)[i] = -0.1;
      local_char_speeds.at(3)[i] = -0.2;
    }
    // Setting constraint_gamma2
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      get(local_constraint_gamma2)[i] = 113.;
    }
    // Setting incoming null one_form: ui
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_incoming_null_one_form.get(0)[i] = -2.;
      local_incoming_null_one_form.get(1)[i] = 5.;
      local_incoming_null_one_form.get(2)[i] = 3.;
      local_incoming_null_one_form.get(3)[i] = 7.;
    }
    // Setting incoming null vector: uI
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_incoming_null_vector.get(0)[i] = -1.;
      local_incoming_null_vector.get(1)[i] = 13.;
      local_incoming_null_vector.get(2)[i] = 17.;
      local_incoming_null_vector.get(3)[i] = 19.;
    }
    // Setting outgoing null one_form: vi
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_outgoing_null_one_form.get(0)[i] = -1.;
      local_outgoing_null_one_form.get(1)[i] = 3.;
      local_outgoing_null_one_form.get(2)[i] = 2.;
      local_outgoing_null_one_form.get(3)[i] = 5.;
    }
    // Setting outgoing null vector: vI
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_outgoing_null_vector.get(0)[i] = -1.;
      local_outgoing_null_vector.get(1)[i] = 2.;
      local_outgoing_null_vector.get(2)[i] = 3.;
      local_outgoing_null_vector.get(3)[i] = 5.;
    }
    // Setting projection Ab
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_projection_Ab.get(0, a)[i] = 233.;
        local_projection_Ab.get(1, a)[i] = 239.;
        local_projection_Ab.get(2, a)[i] = 241.;
        local_projection_Ab.get(3, a)[i] = 251.;
      }
    }
    // Setting projection ab
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_projection_ab.get(0, a)[i] = 379.;
        local_projection_ab.get(1, a)[i] = 383.;
        local_projection_ab.get(2, a)[i] = 389.;
        local_projection_ab.get(3, a)[i] = 397.;
      }
    }
    // Setting projection AB
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_projection_AB.get(0, a)[i] = 353.;
        local_projection_AB.get(1, a)[i] = 359.;
        local_projection_AB.get(2, a)[i] = 367.;
        local_projection_AB.get(3, a)[i] = 373.;
      }
    }
    // Setting 3idxConstraint
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = 0; b <= VolumeDim; ++b) {
          local_three_index_constraint.get(0, a, b)[i] = 11. - 3.;
          local_three_index_constraint.get(1, a, b)[i] = 13. - 5.;
          local_three_index_constraint.get(2, a, b)[i] = 17. - 7.;
        }
      }
    }
    // Setting local_RhsUPsi
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_char_projected_rhs_dt_u_psi.get(0, a)[i] = 23.;
        local_char_projected_rhs_dt_u_psi.get(1, a)[i] = 29.;
        local_char_projected_rhs_dt_u_psi.get(2, a)[i] = 31.;
        local_char_projected_rhs_dt_u_psi.get(3, a)[i] = 37.;
      }
    }
    // Setting RhsUMinus
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_char_projected_rhs_dt_u_minus.get(0, a)[i] = 331.;
        local_char_projected_rhs_dt_u_minus.get(1, a)[i] = 337.;
        local_char_projected_rhs_dt_u_minus.get(2, a)[i] = 347.;
        local_char_projected_rhs_dt_u_minus.get(3, a)[i] = 349.;
      }
    }
    // Setting constraint_char_zero_plus AND constraint_char_zero_minus
    // ONLY ON THE +Y AXIS (Y = +0.5) -- FIXME
    //
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      if (get<0>(inertial_coords)[i] == 299. and
          get<1>(inertial_coords)[i] == 0.5 and
          get<2>(inertial_coords)[i] == -0.5) {
        std::array<double, 4> spec_vals = {
            {3722388974.799386, -16680127.68747905, -22991565.68745775,
             -29394198.68743645}};
        std::array<double, 4> spec_vals2 = {
            {3866802572.424386, -19802442.06247905, -19496408.06245775,
             -19257249.06243645}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 299.5 and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == -0.5) {
        std::array<double, 4> spec_vals = {
            {1718287695.133032, -35082292.16184025, -44420849.32723039,
             -53850596.45928719}};
        std::array<double, 4> spec_vals2 = {
            {1866652790.548975, -38170999.90741873, -40892085.07280888,
             -43680040.20486567}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 300. and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == -0.5) {
        std::array<double, 4> spec_vals = {
            {1718299060.415949, -35082089.27527188, -44420613.14790046,
             -53850326.98725116}};
        std::array<double, 4> spec_vals2 = {
            {1866664134.129611, -38170797.39904065, -40891849.27166924,
             -43679771.11101994}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 299. and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.) {
        std::array<double, 4> spec_vals = {
            {1718276282.501221, -35082495.89577083, -44421086.49296889,
             -53850867.05677793}};
        std::array<double, 4> spec_vals2 = {
            {1866641399.709844, -38171203.26157932, -40892321.85877737,
             -43680310.4225864}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 299.5 and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.) {
        std::array<double, 4> spec_vals = {
            {1718287685.660846, -35082292.33093269, -44420849.52407002,
             -53850596.68387395}};
        std::array<double, 4> spec_vals2 = {
            {1866652781.094877, -38171000.07619599, -40892085.26933331,
             -43680040.42913724}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 300. and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.) {
        std::array<double, 4> spec_vals = {
            {1718299050.990844, -35082089.44352208, -44420613.34375966,
             -53850327.21071925}};
        std::array<double, 4> spec_vals2 = {
            {1866664124.722503, -38170797.56697725, -40891849.46721483,
             -43679771.33417442}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 299. and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.5) {
        std::array<double, 4> spec_vals = {
            {1718276292.082241, -35082495.72473276, -44421086.29386414,
             -53850866.82960649}};
        std::array<double, 4> spec_vals2 = {
            {1866641409.272568, -38171203.09086005, -40892321.65999144,
             -43680310.19573379}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 299.5 and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.5) {
        std::array<double, 4> spec_vals = {
            {1718287695.194063, -35082292.16074976, -44420849.32596073,
             -53850596.45783833}};
        std::array<double, 4> spec_vals2 = {
            {1866652790.609889, -38170999.90633029, -40892085.07154126,
             -43680040.20341886}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 300. and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.5) {
        std::array<double, 4> spec_vals = {
            {1718299060.476573, -35082089.27418864, -44420613.14663924,
             -53850326.98581193}};
        std::array<double, 4> spec_vals2 = {
            {1866664134.190119, -38170797.39795944, -40891849.27041004,
             -43679771.10958273}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else {
        // When applying BCs to various faces, only one face (i.e. the +y side)
        // will be set here, others can be set however...
        // FIXME: Disabled
        ASSERT(true,
               "Not checking the correct face, coordinates not recognized");
      }
    }

    // debugPK
    std::fill(bc_dt_u_minus.begin(), bc_dt_u_minus.end(), 0.);
    if (debugPKon) {
      auto _ = apply_bjorhus_constraint_preserving(
          make_not_null(&bc_dt_u_minus), local_incoming_null_one_form,
          local_outgoing_null_one_form, local_incoming_null_vector,
          local_outgoing_null_vector, local_projection_ab, local_projection_Ab,
          local_projection_AB, local_constraint_char_zero_plus,
          local_constraint_char_zero_minus, local_char_projected_rhs_dt_u_minus,
          local_char_speeds, inertial_coords);
      if (debugPKon) {
        for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
          print_rank2_tensor_at_point(
              "BcDtUMinus<CP>", bc_dt_u_minus, local_inertial_coords.get(0),
              local_inertial_coords.get(1), local_inertial_coords.get(2), i);
        }
      }
    }
    if (debugPKon) {
      auto __ = apply_bjorhus_constraint_preserving_physical(
          make_not_null(&bc_dt_u_minus), local_constraint_gamma2,
          local_unit_interface_normal_one_form,
          local_unit_interface_normal_vector,
          local_spacetime_unit_normal_vector, local_projection_ab,
          local_projection_Ab, local_projection_AB,
          local_inverse_spatial_metric, local_extrinsic_curvature,
          local_spacetime_metric, local_inverse_spacetime_metric,
          local_three_index_constraint, local_char_projected_rhs_dt_u_minus,
          local_phi, local_d_phi, local_d_pi, local_char_speeds,
          inertial_coords);
      // DISPLAY results of the TEST
      if (debugPKon) {
        for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
          print_rank2_tensor_at_point("BcDtUMinus<CPBPhys>", bc_dt_u_minus,
                                      local_inertial_coords.get(0),
                                      local_inertial_coords.get(1),
                                      local_inertial_coords.get(2), i);
        }
      }
    }
    if (debugPKon) {
      auto __ = apply_gauge_sommerfeld(
          make_not_null(&bc_dt_u_minus), local_constraint_gamma2,
          local_inertial_coords, local_incoming_null_one_form,
          local_outgoing_null_one_form, local_incoming_null_vector,
          local_outgoing_null_vector, local_projection_Ab,
          local_char_projected_rhs_dt_u_psi);
      // DISPLAY results of the TEST
      if (debugPKon) {
        for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
          print_rank2_tensor_at_point(
              "BcDtUMinus<gauge_sommerfeld>", bc_dt_u_minus,
              local_inertial_coords.get(0), local_inertial_coords.get(1),
              local_inertial_coords.get(2), i);
        }
      }
    }
#endif
    std::fill(bc_dt_u_minus.begin(), bc_dt_u_minus.end(), 0.);
    // --------------------------------------- TESTS
    // Switch on prescribed boundary condition method
    switch (Method) {
      case UMinusBcMethod::Freezing:
        return apply_gauge_sommerfeld(
            make_not_null(&bc_dt_u_minus), constraint_gamma2, inertial_coords,
            incoming_null_one_form, outgoing_null_one_form,
            incoming_null_vector, outgoing_null_vector, projection_Ab,
            char_projected_rhs_dt_u_psi);
      case UMinusBcMethod::ConstraintPreservingBjorhus:
        apply_bjorhus_constraint_preserving(
            make_not_null(&bc_dt_u_minus), incoming_null_one_form,
            outgoing_null_one_form, incoming_null_vector, outgoing_null_vector,
            projection_ab, projection_Ab, projection_AB,
            constraint_char_zero_plus, constraint_char_zero_minus,
            char_projected_rhs_dt_u_minus, char_speeds, inertial_coords);
        return apply_gauge_sommerfeld(
            make_not_null(&bc_dt_u_minus), constraint_gamma2, inertial_coords,
            incoming_null_one_form, outgoing_null_one_form,
            incoming_null_vector, outgoing_null_vector, projection_Ab,
            char_projected_rhs_dt_u_psi);
      case UMinusBcMethod::ConstraintPreservingPhysicalBjorhus:
        apply_bjorhus_constraint_preserving(
            make_not_null(&bc_dt_u_minus), incoming_null_one_form,
            outgoing_null_one_form, incoming_null_vector, outgoing_null_vector,
            projection_ab, projection_Ab, projection_AB,
            constraint_char_zero_plus, constraint_char_zero_minus,
            char_projected_rhs_dt_u_minus, char_speeds, inertial_coords);
        apply_bjorhus_constraint_preserving_physical(
            make_not_null(&bc_dt_u_minus), constraint_gamma2,
            unit_interface_normal_one_form, unit_interface_normal_vector,
            spacetime_unit_normal_vector, projection_ab, projection_Ab,
            projection_AB, inverse_spatial_metric, extrinsic_curvature,
            spacetime_metric, inverse_spacetime_metric, three_index_constraint,
            char_projected_rhs_dt_u_minus, phi, d_phi, d_pi, char_speeds,
            inertial_coords);
        return apply_gauge_sommerfeld(
            make_not_null(&bc_dt_u_minus), constraint_gamma2, inertial_coords,
            incoming_null_one_form, outgoing_null_one_form,
            incoming_null_vector, outgoing_null_vector, projection_Ab,
            char_projected_rhs_dt_u_psi);
      case UMinusBcMethod::Unknown:
      default:
        ASSERT(false, "Requested BC method fo UMinus not implemented!");
    }
    // dummy return to suppress compiler warning
    return ReturnType{};
  }

 private:
  static ReturnType apply_bjorhus_constraint_preserving(
      const gsl::not_null<ReturnType*> bc_dt_u_minus,
      const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
          incoming_null_one_form,
      const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
          outgoing_null_one_form,
      const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
          incoming_null_vector,
      const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
          outgoing_null_vector,
      const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>& projection_ab,
      const tnsr::Ab<DataVector, VolumeDim, Frame::Inertial>& projection_Ab,
      const tnsr::AA<DataVector, VolumeDim, Frame::Inertial>& projection_AB,
      const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
          constraint_char_zero_plus,
      const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
          constraint_char_zero_minus,
      const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>&
          char_projected_rhs_dt_u_minus,
      const std::array<DataVector, 4>& char_speeds,
      const typename ::Tags::Coordinates<VolumeDim, Frame::Inertial>::type&
          local_inertial_coords) noexcept;
  static ReturnType apply_bjorhus_constraint_preserving_physical(
      const gsl::not_null<ReturnType*> bc_dt_u_minus,
      const Scalar<DataVector>& gamma2,
      const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
          unit_interface_normal_one_form,
      const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
          unit_interface_normal_vector,
      const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
          spacetime_unit_normal_vector,
      const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>& projection_ab,
      const tnsr::Ab<DataVector, VolumeDim, Frame::Inertial>& projection_Ab,
      const tnsr::AA<DataVector, VolumeDim, Frame::Inertial>& projection_AB,
      const tnsr::II<DataVector, VolumeDim, Frame::Inertial>&
          inverse_spatial_metric,
      const tnsr::ii<DataVector, VolumeDim, Frame::Inertial>&
          extrinsic_curvature,
      const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>& spacetime_metric,
      const tnsr::AA<DataVector, VolumeDim, Frame::Inertial>&
          inverse_spacetime_metric,
      const typename Tags::ThreeIndexConstraint<
          VolumeDim, Frame::Inertial>::type& three_index_constraint,
      const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>&
          char_projected_rhs_dt_u_minus,
      const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>& phi,
      const tnsr::ijaa<DataVector, VolumeDim, Frame::Inertial>& d_phi,
      const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>& d_pi,
      const std::array<DataVector, 4>& char_speeds,
      const typename ::Tags::Coordinates<VolumeDim, Frame::Inertial>::type&
          local_inertial_coords) noexcept;
  static ReturnType apply_gauge_sommerfeld(
      const gsl::not_null<ReturnType*> bc_dt_u_minus,
      const Scalar<DataVector>& gamma2,
      const typename ::Tags::Coordinates<VolumeDim, Frame::Inertial>::type&
          inertial_coords,
      const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
          incoming_null_one_form,
      const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
          outgoing_null_one_form,
      const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
          incoming_null_vector,
      const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
          outgoing_null_vector,
      const tnsr::Ab<DataVector, VolumeDim, Frame::Inertial>& projection_Ab,
      const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>&
          char_projected_rhs_dt_u_psi) noexcept;
};

template <typename ReturnType, size_t VolumeDim>
ReturnType
set_dt_u_minus<ReturnType, VolumeDim>::apply_bjorhus_constraint_preserving(
    const gsl::not_null<ReturnType*> bc_dt_u_minus,
    const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
        incoming_null_one_form,
    const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
        outgoing_null_one_form,
    const tnsr::A<DataVector, VolumeDim, Frame::Inertial>& incoming_null_vector,
    const tnsr::A<DataVector, VolumeDim, Frame::Inertial>& outgoing_null_vector,
    const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>& projection_ab,
    const tnsr::Ab<DataVector, VolumeDim, Frame::Inertial>& projection_Ab,
    const tnsr::AA<DataVector, VolumeDim, Frame::Inertial>& projection_AB,
    const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
        constraint_char_zero_plus,
    const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
        constraint_char_zero_minus,
    const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_u_minus,
    const std::array<DataVector, 4>& char_speeds,
    const typename ::Tags::Coordinates<VolumeDim, Frame::Inertial>::type&
        local_inertial_coords) noexcept {
  // debugPK
  if (debugPKoff) {
    for (size_t i = 0; i < get_size(get<0>(incoming_null_one_form)); ++i) {
      print_rank1_tensor_at_point(" C2PLUS<CP>", constraint_char_zero_plus,
                                  local_inertial_coords.get(0),
                                  local_inertial_coords.get(1),
                                  local_inertial_coords.get(2), i);
      print_rank1_tensor_at_point(" C2MINUS<CP>", constraint_char_zero_minus,
                                  local_inertial_coords.get(0),
                                  local_inertial_coords.get(1),
                                  local_inertial_coords.get(2), i);
    }
  }

  ASSERT(get_size(get<0, 0>(*bc_dt_u_minus)) ==
             get_size(get<0>(incoming_null_one_form)),
         "Size of input variables and temporary memory do not match.");
  const double mMu = 0.;  // hard-coded value from SpEC Bbh input file Mu = 0
  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      for (size_t c = 0; c <= VolumeDim; ++c) {
        for (size_t d = 0; d <= VolumeDim; ++d) {
          bc_dt_u_minus->get(a, b) +=
              0.25 *
              (incoming_null_vector.get(c) * incoming_null_vector.get(d) *
                   outgoing_null_one_form.get(a) *
                   outgoing_null_one_form.get(b) -
               incoming_null_vector.get(c) * projection_Ab.get(d, a) *
                   outgoing_null_one_form.get(b) -
               incoming_null_vector.get(c) * projection_Ab.get(d, b) *
                   outgoing_null_one_form.get(a) -
               incoming_null_vector.get(d) * projection_Ab.get(c, a) *
                   outgoing_null_one_form.get(b) -
               incoming_null_vector.get(d) * projection_Ab.get(c, b) *
                   outgoing_null_one_form.get(a) +
               2.0 * projection_AB.get(c, d) * projection_ab.get(a, b)) *
              char_projected_rhs_dt_u_minus.get(c, d);
        }
      }
      for (size_t c = 0; c <= VolumeDim; ++c) {
        bc_dt_u_minus->get(a, b) +=
            0.5 * char_speeds.at(3) *
            (constraint_char_zero_minus.get(c) -
             mMu * constraint_char_zero_plus.get(c)) *
            (0.5 * outgoing_null_one_form.get(a) *
                 outgoing_null_one_form.get(b) * incoming_null_vector.get(c) +
             projection_ab.get(a, b) * outgoing_null_vector.get(c) -
             projection_Ab.get(c, b) * outgoing_null_one_form.get(a) -
             projection_Ab.get(c, a) * outgoing_null_one_form.get(b));
      }
    }
  }
  return *bc_dt_u_minus;
}

template <typename ReturnType, size_t VolumeDim>
ReturnType set_dt_u_minus<ReturnType, VolumeDim>::
    apply_bjorhus_constraint_preserving_physical(
        const gsl::not_null<ReturnType*> bc_dt_u_minus,
        const Scalar<DataVector>& gamma2,
        const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
            unit_interface_normal_one_form,
        const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
            unit_interface_normal_vector,
        const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
            spacetime_unit_normal_vector,
        const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>& projection_ab,
        const tnsr::Ab<DataVector, VolumeDim, Frame::Inertial>& projection_Ab,
        const tnsr::AA<DataVector, VolumeDim, Frame::Inertial>& projection_AB,
        const tnsr::II<DataVector, VolumeDim, Frame::Inertial>&
            inverse_spatial_metric,
        const tnsr::ii<DataVector, VolumeDim, Frame::Inertial>&
            extrinsic_curvature,
        const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>&
            spacetime_metric,
        const tnsr::AA<DataVector, VolumeDim, Frame::Inertial>&
            inverse_spacetime_metric,
        const typename Tags::ThreeIndexConstraint<
            VolumeDim, Frame::Inertial>::type& three_index_constraint,
        const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>&
            char_projected_rhs_dt_u_minus,
        const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>& phi,
        const tnsr::ijaa<DataVector, VolumeDim, Frame::Inertial>& d_phi,
        const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>& d_pi,
        const std::array<DataVector, 4>& char_speeds,
        const typename ::Tags::Coordinates<VolumeDim, Frame::Inertial>::type&
            local_inertial_coords) noexcept {
  //-------------------------------------------------------------------
  // Add in physical boundary condition
  //-------------------------------------------------------------------
  ASSERT(get_size(get<0, 0>(*bc_dt_u_minus)) == get_size(get(gamma2)),
         "Size of input variables and temporary memory do not match.");
  // hard-coded value from SpEC Bbh input file Mu = MuPhys = 0
  const double mMuPhys = 0.;
  const bool mAdjustPhysUsingC4 = true;
  const bool mGamma2InPhysBc = true;

  if (debugPKon) {
    std::cout << " 0. IN apply_bjorhus_constraint_preserving_physical"
              << std::endl
              << "MuPhys = " << mMuPhys
              << ", mAdjustPhysUsingC4 = " << mAdjustPhysUsingC4
              << ", mGamma2InPhysBc = " << mGamma2InPhysBc << std::endl
              << std::flush;
  }
  // In what follows, we use the notation of Kidder, Scheel & Teukolsky
  // (2001) https://arxiv.org/pdf/gr-qc/0105031.pdf. We will refer to this
  // article as KST henceforth, and use the abbreviation in variable names
  // to disambiguate their origin.
  auto U8p = make_with_value<tnsr::ii<DataVector, VolumeDim, Frame::Inertial>>(
      unit_interface_normal_vector, 0.);
  auto U8m = make_with_value<tnsr::ii<DataVector, VolumeDim, Frame::Inertial>>(
      unit_interface_normal_vector, 0.);
  {
    // D_(k,i,j) = (1/2) \partial_k g_(ij) and its derivative
    tnsr::ijj<DataVector, VolumeDim, Frame::Inertial> kst_D(
        get_size(get(gamma2)));
    tnsr::ijkk<DataVector, VolumeDim, Frame::Inertial> d_kst_D(
        get_size(get(gamma2)));
    for (size_t i = 0; i < VolumeDim; ++i) {
      for (size_t j = i; j < VolumeDim; ++j) {
        for (size_t k = 0; k < VolumeDim; ++k) {
          kst_D.get(k, i, j) = 0.5 * phi.get(k, i + 1, j + 1);
          for (size_t l = 0; l < VolumeDim; ++l) {
            d_kst_D.get(l, k, i, j) = 0.5 * d_phi.get(l, k, i + 1, j + 1);
          }
        }
      }
    }
    if (debugPKon) {  // debugPK
      std::cout << " 1. Computed KST_D" << std::endl << std::flush;
    }

    // ComputeRicciFromD(kst_D, d_kst_D, Invg, Ricci3);
    auto ricci_3 = GeneralizedHarmonic::spatial_ricci_tensor_from_KST_vars(
        kst_D, d_kst_D, inverse_spatial_metric);
    if (debugPKoff) {  // debugPK
      std::cout << " 2.1 " << std::endl << std::flush;
      for (size_t i = 0; i < get<0, 0>(ricci_3).size(); ++i) {
        print_rank2_tensor_at_point(
            " 2.1 RICCI_3 ", ricci_3, local_inertial_coords.get(0),
            local_inertial_coords.get(1), local_inertial_coords.get(2), i, 2);
      }
    }                 // debugPK
    if (debugPKon) {  // debugPK
      std::cout << " 2. Computed RICCI_3" << std::endl << std::flush;
    }
    tnsr::ijj<DataVector, VolumeDim, Frame::Inertial> CdK(
        get_size(get(gamma2)));
    {
      const auto christoffel_second_kind =
          GeneralizedHarmonic::spatial_christoffel_second_kind_from_KST_vars(
              kst_D, inverse_spatial_metric);
      if (debugPKoff) {  // debugPK
        std::cout << " 3.1 " << std::endl << std::flush;
        for (size_t i = 0; i < get<0, 0>(ricci_3).size(); ++i) {
          print_rank3_tensor_at_point(
              " 3.1 SpatialGamma2 ", christoffel_second_kind,
              local_inertial_coords.get(0), local_inertial_coords.get(1),
              local_inertial_coords.get(2), i, 2);
        }
      }  // debugPK
      // Ordinary derivative first
      for (size_t i = 0; i < VolumeDim; ++i) {
        for (size_t j = i; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            CdK.get(k, i, j) = 0.5 * d_pi.get(k, i + 1, j + 1);
            if (debugPKoff) {  // debugPK
              std::cout << std::endl << std::flush;
              std::cout << " 3.1 i = " << i << ", j = " << j << ", k = " << k
                        << std::endl
                        << std::flush;
              std::cout << " CdK(" << k << ", " << i << ", " << j
                        << ") = " << CdK.get(k, i, j) << std::endl
                        << " d_phi(" << k << ", " << 0 << ", " << i + 1 << ", "
                        << j + 1 << ") = " << d_pi.get(k, i + 1, j + 1)
                        << std::endl
                        << std::flush;
            }  // debugPK
            for (size_t mu = 0; mu <= VolumeDim; ++mu) {
              if (debugPKoff) {  // debugPK
                std::cout << " 2.1 i = " << i << ", j = " << j << ", k = " << k
                          << ", mu = " << mu << std::endl
                          << std::flush;
                std::cout << " CdK(" << k << ", " << i << ", " << j
                          << ") = " << CdK.get(k, i, j) << std::endl
                          << " d_phi(" << k << ", " << i << ", " << j + 1
                          << ", " << mu << ") = " << d_phi.get(k, i, j + 1, mu)
                          << std::endl
                          << " d_phi(" << k << ", " << j << ", " << i + 1
                          << ", " << mu << ") = " << d_phi.get(k, j, i + 1, mu)
                          << std::endl
                          << " tI(" << mu
                          << ") = " << spacetime_unit_normal_vector.get(mu)
                          << std::endl
                          << std::flush;
              }  // debugPK
              CdK.get(k, i, j) +=
                  0.5 *
                  (d_phi.get(k, i, j + 1, mu) + d_phi.get(k, j, i + 1, mu)) *
                  spacetime_unit_normal_vector.get(mu);
              for (size_t nu = 0; nu <= VolumeDim; ++nu) {
                for (size_t a = 0; a <= VolumeDim; ++a) {
                  CdK.get(k, i, j) -=
                      0.5 * (phi.get(i, j + 1, mu) + phi.get(j, i + 1, mu)) *
                      spacetime_unit_normal_vector.get(nu) *
                      (inverse_spacetime_metric.get(a, mu) +
                       0.5 * spacetime_unit_normal_vector.get(a) *
                           spacetime_unit_normal_vector.get(mu)) *
                      phi.get(k, a, nu);
                }
              }
            }
          }
        }
      }
      if (debugPKoff) {  // debugPK
        std::cout << " 3.1 " << std::endl << std::flush;
        for (size_t i = 0; i < get<0, 0>(ricci_3).size(); ++i) {
          print_rank3_tensor_at_point(" 3.1 CdK: Ordinary derivative terms ",
                                      CdK, local_inertial_coords.get(0),
                                      local_inertial_coords.get(1),
                                      local_inertial_coords.get(2), i, 2);
        }
      }  // debugPK
      // Now add gamma terms
      for (size_t i = 0; i < VolumeDim; ++i) {
        for (size_t j = i; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            for (size_t l = 0; l < VolumeDim; ++l) {
              if (debugPKoff) {  // debugPK
                std::cout << std::endl;
                std::cout << " 3.1 i = " << i << ", j = " << j << ", k = " << k
                          << ", l = " << l << std::endl
                          << std::flush;
                std::cout << " CdK(" << k << ", " << i << ", " << j
                          << ") = " << CdK.get(k, i, j) << std::endl
                          << " christoffel_second_kind(" << l << ", " << i
                          << ", " << k
                          << ") = " << christoffel_second_kind.get(l, i, k)
                          << std::endl
                          << " christoffel_second_kind(" << l << ", " << j
                          << ", " << k
                          << ") = " << christoffel_second_kind.get(l, j, k)
                          << std::endl
                          << " extrinsic_curvature(" << l << ", " << j
                          << ") = " << extrinsic_curvature.get(l, j)
                          << std::endl
                          << " extrinsic_curvature(" << l << ", " << i
                          << ") = " << extrinsic_curvature.get(l, i)
                          << std::endl
                          << std::flush;
              }  // debugPK
              CdK.get(k, i, j) -= christoffel_second_kind.get(l, i, k) *
                                      extrinsic_curvature.get(l, j) +
                                  christoffel_second_kind.get(l, j, k) *
                                      extrinsic_curvature.get(l, i);
            }
          }
        }
      }
      if (debugPKoff) {  // debugPK
        std::cout << " 3.1 " << std::endl << std::flush;
        for (size_t i = 0; i < get<0, 0>(ricci_3).size(); ++i) {
          print_rank3_tensor_at_point(
              " 3.1 CdK: Ordinary derivative + gamma terms ", CdK,
              local_inertial_coords.get(0), local_inertial_coords.get(1),
              local_inertial_coords.get(2), i, 2);
        }
      }  // debugPK
    }
    if (debugPKoff) {  // debugPK
      std::cout << " 2.1 " << std::endl << std::flush;
      for (size_t i = 0; i < get<0, 0>(ricci_3).size(); ++i) {
        print_rank3_tensor_at_point(
            " 3.1 CdK ", CdK, local_inertial_coords.get(0),
            local_inertial_coords.get(1), local_inertial_coords.get(2), i, 2);
      }
    }  // debugPK

    if (debugPKoff) {  // debugPK
      std::cout << " 3. Computed CdK" << std::endl << std::flush;
    }
    if (mAdjustPhysUsingC4) {
      // This adds 4-index constraint terms to 3Ricci so as to cancel
      // out normal derivatives from the final expression for U8.
      // It is much easier to add them here than to recalculate U8
      // from scratch.

      // Add some 4-index constraint terms to 3Ricci.
      for (size_t i = 0; i < VolumeDim; ++i) {
        for (size_t j = i; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            for (size_t l = 0; l < VolumeDim; ++l) {
              ricci_3.get(i, j) +=
                  0.5 * inverse_spatial_metric.get(k, l) *
                  (d_kst_D.get(i, k, l, j) - d_kst_D.get(k, i, l, j) +
                   d_kst_D.get(j, k, l, i) - d_kst_D.get(k, j, l, i));
            }
          }
        }
      }

      // Add more 4-index constraint terms to 3Ricci
      // These compensate for some of the CdK terms.
      for (size_t i = 0; i < VolumeDim; ++i) {
        for (size_t j = i; j < VolumeDim; ++j) {
          for (size_t a = 0; a <= VolumeDim; ++a) {
            for (size_t k = 0; k < VolumeDim; ++k) {
              ricci_3.get(i, j) +=
                  0.5 * unit_interface_normal_vector.get(k + 1) *
                  spacetime_unit_normal_vector.get(a) *
                  (d_phi.get(i, k, j + 1, a) - d_phi.get(k, i, j + 1, a) +
                   d_phi.get(j, k, i + 1, a) - d_phi.get(k, j, i + 1, a));
            }
          }
        }
      }
    }
    if (debugPKoff) {  // debugPK
      std::cout << " 4.1 AdjustedUsingC4" << std::endl << std::flush;
      for (size_t i = 0; i < get<0, 0>(ricci_3).size(); ++i) {
        print_rank2_tensor_at_point(
            " 4.1 RICCI_3 ", ricci_3, local_inertial_coords.get(0),
            local_inertial_coords.get(1), local_inertial_coords.get(2), i, 2);
      }
    }                  // debugPK
    if (debugPKoff) {  // debugPK
      std::cout
          << " 4. AdjustedUsingC4: Added 4-idx constraint terms to RICCI_3"
          << std::endl
          << std::flush;
    }

    TempBuffer<tmpl::list<
        // spatial projection operators P_ij, P^ij, and P^i_j
        ::Tags::TempII<0, VolumeDim, Frame::Inertial, DataVector>,
        ::Tags::Tempii<1, VolumeDim, Frame::Inertial, DataVector>,
        ::Tags::TempIj<2, VolumeDim, Frame::Inertial, DataVector>>>
        local_buffer(get_size(get<0>(unit_interface_normal_vector)));
    auto& spatial_projection_IJ =
        get<::Tags::TempII<0, VolumeDim, Frame::Inertial, DataVector>>(
            local_buffer);
    auto& spatial_projection_ij =
        get<::Tags::Tempii<1, VolumeDim, Frame::Inertial, DataVector>>(
            local_buffer);
    auto& spatial_projection_Ij =
        get<::Tags::TempIj<2, VolumeDim, Frame::Inertial, DataVector>>(
            local_buffer);

    // Make spatial projection operators
    GeneralizedHarmonic::spatial_projection_tensor(
        make_not_null(&spatial_projection_IJ), inverse_spatial_metric,
        unit_interface_normal_vector);
    if (debugPKoff) {  // debugPK
      std::cout << " 3.1 " << std::endl << std::flush;
      for (size_t i = 0; i < get<0, 0>(ricci_3).size(); ++i) {
        print_rank2_tensor_at_point(" 5.1 Spatial PIJ ", spatial_projection_IJ,
                                    local_inertial_coords.get(0),
                                    local_inertial_coords.get(1),
                                    local_inertial_coords.get(2), i, 2);
      }
    }                 // debugPK
    if (debugPKon) {  // debugPK
      std::cout << " 5.1 Computed PIJ" << std::endl << std::flush;
    }

    const auto spatial_metric = [&spacetime_metric]() noexcept {
      tnsr::ii<DataVector, VolumeDim, Frame::Inertial> tmp_metric(
          get_size(get<0, 0>(spacetime_metric)));
      for (size_t j = 0; j < VolumeDim; ++j) {
        for (size_t k = j; k < VolumeDim; ++k) {
          tmp_metric.get(j, k) = spacetime_metric.get(1 + j, 1 + k);
        }
      }
      return tmp_metric;
    }
    ();
    GeneralizedHarmonic::spatial_projection_tensor(
        make_not_null(&spatial_projection_ij), spatial_metric,
        unit_interface_normal_one_form);
    if (debugPKoff) {  // debugPK
      std::cout << " 3.1 " << std::endl << std::flush;
      for (size_t i = 0; i < get<0, 0>(ricci_3).size(); ++i) {
        print_rank2_tensor_at_point(" 5.1 Spatial Pij ", spatial_projection_ij,
                                    local_inertial_coords.get(0),
                                    local_inertial_coords.get(1),
                                    local_inertial_coords.get(2), i, 2);
      }
    }                 // debugPK
    if (debugPKon) {  // debugPK
      std::cout << " 5.2 Computed Pij" << std::endl << std::flush;
    }

    GeneralizedHarmonic::spatial_projection_tensor(
        make_not_null(&spatial_projection_Ij), unit_interface_normal_vector,
        unit_interface_normal_one_form);
    if (debugPKoff) {  // debugPK
      std::cout << " 3.1 " << std::endl << std::flush;
      for (size_t i = 0; i < get<0, 0>(ricci_3).size(); ++i) {
        print_rank2_tensor_at_point(" 5.1 Spatial PIj ", spatial_projection_Ij,
                                    local_inertial_coords.get(0),
                                    local_inertial_coords.get(1),
                                    local_inertial_coords.get(2), i, 2);
      }
    }                 // debugPK
    if (debugPKon) {  // debugPK
      std::cout << " 5.3 Computed PIj" << std::endl << std::flush;
    }

    GeneralizedHarmonic::weyl_propagating(
        make_not_null(&U8p), ricci_3, extrinsic_curvature,
        inverse_spatial_metric, CdK, unit_interface_normal_one_form,
        unit_interface_normal_vector, spatial_projection_IJ,
        spatial_projection_ij, spatial_projection_Ij, 1);
    if (debugPKon) {  // debugPK
      std::cout << " 6.1 Computed U8PLUS" << std::endl << std::flush;
    }
    GeneralizedHarmonic::weyl_propagating(
        make_not_null(&U8m), ricci_3, extrinsic_curvature,
        inverse_spatial_metric, CdK, unit_interface_normal_one_form,
        unit_interface_normal_vector, spatial_projection_IJ,
        spatial_projection_ij, spatial_projection_Ij, -1);
    if (debugPKon) {  // debugPK
      std::cout << " 6.2 " << std::endl << std::flush;
      for (size_t i = 0; i < get<0, 0>(ricci_3).size(); ++i) {
        print_rank2_tensor_at_point(" 6.1 Spatial propagating WeylU8PLUS ", U8p,
                                    local_inertial_coords.get(0),
                                    local_inertial_coords.get(1),
                                    local_inertial_coords.get(2), i, 2);
        print_rank2_tensor_at_point(" 6.1 Spatial propagating WeylU8MINUS ",
                                    U8m, local_inertial_coords.get(0),
                                    local_inertial_coords.get(1),
                                    local_inertial_coords.get(2), i, 2);
      }
    }                 // debugPK
    if (debugPKon) {  // debugPK
      std::cout << " 6.2 Computed U8MINUS" << std::endl << std::flush;
    }
  }

  auto U3p = make_with_value<tnsr::aa<DataVector, VolumeDim, Frame::Inertial>>(
      unit_interface_normal_vector, 0.);
  auto U3m = make_with_value<tnsr::aa<DataVector, VolumeDim, Frame::Inertial>>(
      unit_interface_normal_vector, 0.);
  for (size_t mu = 0; mu <= VolumeDim; ++mu) {
    for (size_t nu = mu; nu <= VolumeDim; ++nu) {
      for (size_t i = 0; i < VolumeDim; ++i) {
        for (size_t j = 0; j < VolumeDim; ++j) {
          if (debugPKoff) {  // debugPK
            std::cout << " 7.1 Computing U3PLUS, U3MINUS" << std::endl
                      << std::flush;
            std::cout << " 7.1 mu = " << mu << ", nu = " << nu << ", i" << i
                      << ", j" << j << std::endl
                      << std::flush;
            std::cout << " 7.1: projection_Ab(i + 1, mu) = "
                      << projection_Ab.get(i + 1, mu) << std::endl
                      << " 7.1: projection_Ab(j + 1, nu) = "
                      << projection_Ab.get(j + 1, nu) << std::endl
                      << " 7.1: U8PLUS(i, j) = " << U8p.get(i, j) << std::endl
                      << " 7.1: U8MINUS(i, j) = " << U8m.get(i, j) << std::endl
                      << std::flush;
          }  // debugPK
          U3p.get(mu, nu) += 2.0 * projection_Ab.get(i + 1, mu) *
                             projection_Ab.get(j + 1, nu) * U8p.get(i, j);
          if (debugPKoff) {  // debugPK
            std::cout << " 7.2 U3PLUS(" << mu << ", " << nu
                      << ") = " << U3p.get(mu, nu) << std::endl
                      << std::flush;
          }  // debugPK
          U3m.get(mu, nu) += 2.0 * projection_Ab.get(i + 1, mu) *
                             projection_Ab.get(j + 1, nu) * U8m.get(i, j);
          if (debugPKoff) {  // debugPK
            std::cout << " 7.2 U3MINUS(" << mu << ", " << nu
                      << ") = " << U3m.get(mu, nu) << std::endl
                      << std::flush;
          }  // debugPK
        }
      }
    }
  }
  if (debugPKon) {  // debugPK
    std::cout << " 6.2 " << std::endl << std::flush;
    for (size_t i = 0; i < get<0, 0>(spacetime_metric).size(); ++i) {
      print_rank2_tensor_at_point(
          " 7.1 Spatial U3PLUS ", U3p, local_inertial_coords.get(0),
          local_inertial_coords.get(1), local_inertial_coords.get(2), i, 2);
      print_rank2_tensor_at_point(
          " 7.1 Spatial U3MINUS ", U3m, local_inertial_coords.get(0),
          local_inertial_coords.get(1), local_inertial_coords.get(2), i, 2);
    }
  }                 // debugPK
  if (debugPKon) {  // debugPK
    std::cout << " 7. Computed U3PLUS, U3MINUS" << std::endl << std::flush;
  }
  // Impose physical boundary condition
  if (mGamma2InPhysBc) {
    DataVector tmp(get_size(get<0>(unit_interface_normal_vector)));
    for (size_t mu = 0; mu <= VolumeDim; ++mu) {
      for (size_t nu = mu; nu <= VolumeDim; ++nu) {
        for (size_t a = 0; a <= VolumeDim; ++a) {
          for (size_t b = 0; b <= VolumeDim; ++b) {
            tmp = 0.;
            for (size_t i = 0; i < VolumeDim; ++i) {
              tmp += unit_interface_normal_vector.get(i + 1) *
                     three_index_constraint.get(i, a, b);
            }
            tmp *= get(gamma2);
            bc_dt_u_minus->get(mu, nu) +=
                (projection_Ab.get(a, mu) * projection_Ab.get(b, nu) -
                 0.5 * projection_ab.get(mu, nu) * projection_AB.get(a, b)) *
                (char_projected_rhs_dt_u_minus.get(a, b) +
                 char_speeds.at(3) *
                     (U3m.get(a, b) - tmp - mMuPhys * U3p.get(a, b)));
          }
        }
      }
    }
  } else {
    for (size_t mu = 0; mu <= VolumeDim; ++mu) {
      for (size_t nu = mu; nu <= VolumeDim; ++nu) {
        for (size_t a = 0; a <= VolumeDim; ++a) {
          for (size_t b = 0; b <= VolumeDim; ++b) {
            bc_dt_u_minus->get(mu, nu) +=
                (projection_Ab.get(a, mu) * projection_Ab.get(b, nu) -
                 0.5 * projection_ab.get(mu, nu) * projection_AB.get(a, b)) *
                (char_projected_rhs_dt_u_minus.get(a, b) +
                 char_speeds.at(3) * (U3m.get(a, b) - mMuPhys * U3p.get(a, b)));
          }
        }
      }
    }
  }
  if (debugPKon) {  // debugPK
    std::cout << " 8. Set BcDtUMinus!" << std::endl << std::flush;
  }
  return *bc_dt_u_minus;
}

template <typename ReturnType, size_t VolumeDim>
ReturnType set_dt_u_minus<ReturnType, VolumeDim>::apply_gauge_sommerfeld(
    const gsl::not_null<ReturnType*> bc_dt_u_minus,
    const Scalar<DataVector>& gamma2,
    const typename ::Tags::Coordinates<VolumeDim, Frame::Inertial>::type&
        inertial_coords,
    const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
        incoming_null_one_form,
    const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
        outgoing_null_one_form,
    const tnsr::A<DataVector, VolumeDim, Frame::Inertial>& incoming_null_vector,
    const tnsr::A<DataVector, VolumeDim, Frame::Inertial>& outgoing_null_vector,
    const tnsr::Ab<DataVector, VolumeDim, Frame::Inertial>& projection_Ab,
    const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_u_psi) noexcept {
  ASSERT(get_size(get<0, 0>(*bc_dt_u_minus)) ==
             get_size(get<0>(incoming_null_one_form)),
         "Size of input variables and temporary memory do not match.");
  // gauge_bc_coeff below is hard-coded here to its default value in SpEC
  constexpr double gauge_bc_coeff = 1.;

  DataVector inertial_radius(get_size(get<0>(inertial_coords)), 0.);
  for (size_t i = 0; i < VolumeDim; ++i) {
    inertial_radius += square(inertial_coords.get(i));
  }
  inertial_radius = sqrt(inertial_radius);

  // add in gauge BC
  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      for (size_t c = 0; c <= VolumeDim; ++c) {
        for (size_t d = 0; d <= VolumeDim; ++d) {
          bc_dt_u_minus->get(a, b) +=
              0.5 *
              (incoming_null_one_form.get(a) * projection_Ab.get(c, b) *
                   outgoing_null_vector.get(d) +
               incoming_null_one_form.get(b) * projection_Ab.get(c, a) *
                   outgoing_null_vector.get(d) -
               0.5 * incoming_null_one_form.get(a) *
                   outgoing_null_one_form.get(b) * incoming_null_vector.get(c) *
                   outgoing_null_vector.get(d) -
               0.5 * incoming_null_one_form.get(b) *
                   outgoing_null_one_form.get(a) * incoming_null_vector.get(c) *
                   outgoing_null_vector.get(d) -
               0.5 * incoming_null_one_form.get(a) *
                   incoming_null_one_form.get(b) * outgoing_null_vector.get(c) *
                   outgoing_null_vector.get(d)) *
              (get(gamma2) - gauge_bc_coeff * (1. / inertial_radius)) *
              char_projected_rhs_dt_u_psi.get(c, d);
        }
      }
    }
  }

  return *bc_dt_u_minus;
}

}  // namespace BoundaryConditions_detail
}  // namespace Actions
}  // namespace GeneralizedHarmonic
