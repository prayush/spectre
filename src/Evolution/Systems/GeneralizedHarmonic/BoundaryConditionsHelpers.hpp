// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
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

// \brief This function computes intermediate variables needed for the
// Bjorhus type boundary condition
template <size_t VolumeDim, typename TagsList, typename DbTags,
          typename VarsTagsList, typename DtVarsTagsList>
void local_variables(
    gsl::not_null<TempBuffer<TagsList>*> buffer, db::DataBox<DbTags>& box,
    const Direction<VolumeDim>& direction, const size_t& dimension,
    const db::item_type<::Tags::Mesh<VolumeDim>>& mesh,
    const Variables<VarsTagsList>& vars,
    const Variables<DtVarsTagsList>& dt_vars_volume,
    const tnsr::i<DataVector, VolumeDim, Frame::Inertial>& unit_normal_one_form,
    const db::item_type<GeneralizedHarmonic::Tags::CharacteristicSpeeds<
        VolumeDim, Frame::Inertial>>& char_speeds) noexcept {
  // I need to compute here the following quantities:
  using tags_needed_on_slice = tmpl::list<
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<VolumeDim, Frame::Inertial, DataVector>,
      gr::Tags::SpatialMetric<VolumeDim, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<VolumeDim, Frame::Inertial, DataVector>,
      gr::Tags::SpacetimeNormalOneForm<VolumeDim, Frame::Inertial, DataVector>,
      gr::Tags::SpacetimeNormalVector<VolumeDim, Frame::Inertial, DataVector>,
      gr::Tags::SpacetimeMetric<VolumeDim, Frame::Inertial, DataVector>,
      GeneralizedHarmonic::Tags::Pi<VolumeDim, Frame::Inertial>,
      GeneralizedHarmonic::Tags::Phi<VolumeDim, Frame::Inertial>,
      // derivs of Psi, Pi, and Phi.
      gr::Tags::DerivSpacetimeMetric<VolumeDim, Frame::Inertial, DataVector>,
      ::Tags::deriv<GeneralizedHarmonic::Tags::Pi<VolumeDim, Frame::Inertial>,
                    tmpl::size_t<VolumeDim>, Frame::Inertial>,
      ::Tags::deriv<GeneralizedHarmonic::Tags::Phi<VolumeDim, Frame::Inertial>,
                    tmpl::size_t<VolumeDim>, Frame::Inertial>,
      // constraint damping parameters
      GeneralizedHarmonic::Tags::ConstraintGamma0,
      GeneralizedHarmonic::Tags::ConstraintGamma1,
      GeneralizedHarmonic::Tags::ConstraintGamma2>;
  const auto vars_on_this_slice = db::data_on_slice(
      box, mesh.extents(), dimension,
      index_to_slice_at(mesh.extents(), direction), tags_needed_on_slice{});
  // 1) incoming null spacetime vector & one-form

  // 2) outgoing null spacetime vector & one-form

  // 3) spacetime projection operator

  // 4) spatial projection operator
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
  auto unit_normal_vector =
      make_with_value<tnsr::I<DataVector, SpatialDim, Frame>>(lapse, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      unit_normal_vector.get(i) +=
          inverse_spatial_metric.get(i, j) * unit_normal_one_form.get(j);
    }
  }
  for (size_t mu = 0; mu <= SpatialDim; ++mu) {
    for (size_t nu = mu; nu <= SpatialDim; ++nu) {
      bc_dt_u_psi.get(mu, nu) = dt_u_psi.get(mu, nu);
      for (size_t k = 0; k < SpatialDim; ++k) {
        bc_dt_u_psi.get(mu, nu) +=
            char_speeds[0] * unit_normal_vector.get(k) *
            (deriv_spacetime_metric.get(k, mu, nu) - phi.get(k, mu, nu));
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
  for (size_t mu = 0; mu <= SpatialDim; ++mu) {
    for (size_t nu = mu; nu <= SpatialDim; ++nu) {
      bc_dt_u_psi.get(mu, nu) = -get(lapse) * pi.get(mu, nu);
      for (size_t m = 0; m < SpatialDim; ++m) {
        bc_dt_u_psi.get(mu, nu) += shift.get(m) * phi.get(m, mu, nu);
      }
    }
  }
  return *bc_dt_u_psi;
}

}  // namespace BoundaryConditions_detail
}  // namespace Actions
}  // namespace GeneralizedHarmonic
