// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InterfaceActionHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Tags.hpp"
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

namespace ScalarWave {
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
}  // namespace BoundaryConditions_detail

/// \ingroup ActionsGroup
/// \brief Packages data on external boundaries for calculating numerical flux.
/// Computes contributions on the interior side from the volume, and imposes
/// constraint preserving boundary conditions on the exterior side.
///
/// With:
/// - Boundary<Tag> =
///   Tags::Interface<Tags::BoundaryDirections<volume_dim>, Tag>
/// - External<Tag> =
///   Tags::Interface<Tags::ExternalBoundaryDirections<volume_dim>, Tag>
///
/// Uses:
/// - ConstGlobalCache:
///   - Metavariables::normal_dot_numerical_flux
///   - Metavariables::boundary_condition
/// - DataBox:
///   - Tags::Element<volume_dim>
///   - Boundary<Tags listed in
///               Metavariables::normal_dot_numerical_flux::type::argument_tags>
///   - External<Tags listed in
///               Metavariables::normal_dot_numerical_flux::type::argument_tags>
///   - Boundary<Tags::Mesh<volume_dim - 1>>
///   - External<Tags::Mesh<volume_dim - 1>>
///   - Boundary<Tags::Magnitude<Tags::UnnormalizedFaceNormal<volume_dim>>>,
///   - External<Tags::Magnitude<Tags::UnnormalizedFaceNormal<volume_dim>>>,
///   - Boundary<Tags::BoundaryCoordinates<volume_dim>>,
///   - Metavariables::temporal_id
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///      - Tags::VariablesBoundaryData
///      - External<typename system::variables_tag>
///
/// \see ReceiveDataForFluxes
template <typename Metavariables>
struct ImposeConstraintPreservingBoundaryConditions {
 private:
  // {Psi,Phi,Pi}BcMethod and BcSelector are used to select exactly how to
  // apply the requested boundary condition depending on user input.
  // A specialized `apply_impl` struct is used that implements the
  // boundary condition calculation for the different types.
  using PsiBcMethod = BoundaryConditions_detail::PsiBcMethod;
  using PhiBcMethod = BoundaryConditions_detail::PhiBcMethod;
  using PiBcMethod = BoundaryConditions_detail::PiBcMethod;
  template <typename T, T Method>
  using BcSelector = std::integral_constant<T, Method>;

 public:
  using const_global_cache_tags =
      tmpl::list<typename Metavariables::normal_dot_numerical_flux,
                 typename Metavariables::boundary_condition_tag>;

  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Here be logic that selects from various options for
    // setting BCs on individual evolved variables
    return apply_impl<Metavariables::system::volume_dim>(
        box, cache, BcSelector<PsiBcMethod, PsiBcMethod::AnalyticBc>{},
        BcSelector<PhiBcMethod, PhiBcMethod::AnalyticBc>{},
        BcSelector<PiBcMethod, PiBcMethod::AnalyticBc>{});
  }

 private:
  /* ------------------------------------------------------------------------
   * ------------------------------------------------------------------------
   * ---------------- APPLY THE BOUNDARY CONDITION  ----------------
   * ------------------------------------------------------------------------
   * ------------------------------------------------------------------------
   */
  template <size_t VolumeDim, typename DbTags>
  static std::tuple<db::DataBox<DbTags>&&> apply_impl(
      db::DataBox<DbTags>& box,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      std::integral_constant<PsiBcMethod, PsiBcMethod::Freezing> /*meta*/,
      std::integral_constant<PhiBcMethod, PhiBcMethod::AnalyticBc> /*meta*/,
      std::integral_constant<PiBcMethod,
                             PiBcMethod::AnalyticBc> /*meta*/) noexcept {
    using system = typename Metavariables::system;

    // Apply the boundary condition
    db::mutate_apply<tmpl::list<::Tags::Interface<
                         ::Tags::BoundaryDirectionsExterior<VolumeDim>,
                         typename system::variables_tag>>,
                     tmpl::list<>>(

        [](const gsl::not_null<db::item_type<
               ::Tags::Interface<::Tags::BoundaryDirectionsExterior<VolumeDim>,
                                 typename system::variables_tag>>*>
               external_bdry_vars,
           const double time, const auto& boundary_condition,
           const auto& boundary_coords) noexcept {
          // -------------------------------
          // Loop over external boundaries
          for (auto& external_direction_and_vars : *external_bdry_vars) {
            auto& direction = external_direction_and_vars.first;
            auto& vars = external_direction_and_vars.second;

            // Get evolved variables on current boundary from AnalyticSolution
            const auto analytic_boundary_vars = boundary_condition.variables(
                boundary_coords.at(direction), time,
                typename system::variables_tag::type::tags_list{});

            // Construct CURRENT time derivs of char fields

            // Construct DESIRED time derivs of characteristic fields

            // Now reconstruct time derivs of fund fields:
            // Incoming modes should be set using DESIRED time derivs of char
            // fields, and nonincoming modes should be set using CURRENT tiem
            // derivs

            // Assign Psi
            get<ScalarWave::Psi>(vars) =
                get<ScalarWave::Psi>(analytic_boundary_vars);
            // Assign Phi
            get<ScalarWave::Phi<VolumeDim>>(vars) =
                get<ScalarWave::Phi<VolumeDim>>(analytic_boundary_vars);
            // Assign Pi
            get<ScalarWave::Psi>(vars) =
                get<ScalarWave::Psi>(analytic_boundary_vars);
          }
          // -------------------------------
        },
        make_not_null(&box), db::get<::Tags::Time>(box).value(),
        get<typename Metavariables::boundary_condition_tag>(cache),
        db::get<
            ::Tags::Interface<::Tags::BoundaryDirectionsExterior<VolumeDim>,
                              ::Tags::Coordinates<VolumeDim, Frame::Inertial>>>(
            box));

    return std::forward_as_tuple(std::move(box));
  }

  template <size_t VolumeDim, typename DbTags>
  static std::tuple<db::DataBox<DbTags>&&> apply_impl(
      db::DataBox<DbTags>& box,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      std::integral_constant<PsiBcMethod, PsiBcMethod::AnalyticBc> /*meta*/,
      std::integral_constant<PhiBcMethod, PhiBcMethod::AnalyticBc> /*meta*/,
      std::integral_constant<PiBcMethod,
                             PiBcMethod::AnalyticBc> /*meta*/) noexcept {
    using system = typename Metavariables::system;

    // Apply the boundary condition
    db::mutate_apply<tmpl::list<::Tags::Interface<
                         ::Tags::BoundaryDirectionsExterior<VolumeDim>,
                         typename system::variables_tag>>,
                     tmpl::list<>>(

        [](const gsl::not_null<db::item_type<
               ::Tags::Interface<::Tags::BoundaryDirectionsExterior<VolumeDim>,
                                 typename system::variables_tag>>*>
               external_bdry_vars,
           const double time, const auto& boundary_condition,
           const auto& boundary_coords) noexcept {
          // -------------------------------
          // Loop over external boundaries
          for (auto& external_direction_and_vars : *external_bdry_vars) {
            auto& direction = external_direction_and_vars.first;
            auto& vars = external_direction_and_vars.second;

            // Get evolved variables on current boundary from AnalyticSolution
            const auto analytic_boundary_vars = boundary_condition.variables(
                boundary_coords.at(direction), time,
                typename system::variables_tag::type::tags_list{});

            // Assign Psi
            get<ScalarWave::Psi>(vars) =
                get<ScalarWave::Psi>(analytic_boundary_vars);
            // Assign Phi
            get<ScalarWave::Phi<VolumeDim>>(vars) =
                get<ScalarWave::Phi<VolumeDim>>(analytic_boundary_vars);
            // Assign Pi
            get<ScalarWave::Psi>(vars) =
                get<ScalarWave::Psi>(analytic_boundary_vars);

            // vars.assign_subset(boundary_condition.variables(
            //     boundary_coords.at(direction), time,
            //     typename system::variables_tag::type::tags_list{}));
          }
          // -------------------------------
        },
        make_not_null(&box), db::get<::Tags::Time>(box).value(),
        get<typename Metavariables::boundary_condition_tag>(cache),
        db::get<
            ::Tags::Interface<::Tags::BoundaryDirectionsExterior<VolumeDim>,
                              ::Tags::Coordinates<VolumeDim, Frame::Inertial>>>(
            box));

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace ScalarWave
