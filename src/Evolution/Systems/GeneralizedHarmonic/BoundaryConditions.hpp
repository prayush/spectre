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
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditionsImpl.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
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
#include "Utilities/MakeWithValue.hpp"
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
///      - External<Tags::dt<typename system::variables_tag>>
///
/// \see ReceiveDataForFluxes
template <typename Metavariables>
struct ImposeConstraintPreservingBoundaryConditions {
 private:
  // {UPsi,UZero,UPlus,UMinus}BcMethod are used to select exactly
  // how to apply the requested boundary condition based on user input. A
  // specialized `apply_impl` struct is used that implements the boundary
  // condition calculation for the different types.
  using UPsiBcMethod = BoundaryConditions_detail::UPsiBcMethod;
  using UZeroBcMethod = BoundaryConditions_detail::UZeroBcMethod;
  using UPlusBcMethod = BoundaryConditions_detail::UPlusBcMethod;
  using UMinusBcMethod = BoundaryConditions_detail::UMinusBcMethod;

 public:
  using const_global_cache_tags =
      tmpl::list<typename Metavariables::normal_dot_numerical_flux,
                 typename Metavariables::boundary_condition_tag>;

 private:
  /* ------------------------------------------------------------------------
   * ------------------------------------------------------------------------
   * ---------------- APPLY BJORHUS BOUNDARY CONDITIONS  --------------------
   * ------------------------------------------------------------------------
   * ------------------------------------------------------------------------
   */
  template <size_t VolumeDim, UPsiBcMethod UPsiMethod,
            UZeroBcMethod UZeroMethod, UPlusBcMethod UPlusMethod,
            UMinusBcMethod UMinusMethod, typename DbTags>
  struct apply_impl {
    static std::tuple<db::DataBox<DbTags>&&> function_impl(
        db::DataBox<DbTags>& box,
        const Parallel::ConstGlobalCache<Metavariables>& cache) noexcept {
      // ------------------------------- (1)
      // Get information about system:
      // tags for evolved variables and their time derivatives
      using system = typename Metavariables::system;
      using variables_tag = typename system::variables_tag;
      using dt_variables_tag =
          db::add_tag_prefix<Metavariables::temporal_id::template step_prefix,
                             variables_tag>;
      constexpr const size_t number_of_independent_components =
          dt_variables_tag::type::number_of_independent_components;

      const db::item_type<::Tags::Mesh<VolumeDim>>& mesh =
          db::get<::Tags::Mesh<VolumeDim>>(box);
      const size_t volume_grid_points = mesh.extents().product();
      const auto& unit_normal_one_forms = db::get<
          ::Tags::Interface<::Tags::BoundaryDirectionsExterior<VolumeDim>,
                            ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<
                                VolumeDim, Frame::Inertial>>>>(box);
      const auto& external_bdry_vars = db::get<::Tags::Interface<
          ::Tags::BoundaryDirectionsExterior<VolumeDim>, variables_tag>>(box);
      const auto& volume_all_at_vars = db::get<dt_variables_tag>(box);
      const auto& external_bdry_char_speeds = db::get<::Tags::Interface<
          ::Tags::BoundaryDirectionsExterior<VolumeDim>,
          Tags::CharacteristicSpeeds<VolumeDim, Frame::Inertial>>>(box);

      // ------------------------------- (2)
      // Apply the boundary condition
      // Loop over external boundaries and set dt_volume_vars on them
      for (auto& external_direction_and_vars : external_bdry_vars) {
        const auto& direction = external_direction_and_vars.first;
        const size_t dimension = direction.dimension();
        const auto& unit_normal_one_form = unit_normal_one_forms.at(direction);
        const auto& vars = external_direction_and_vars.second;
        const size_t slice_grid_points =
            mesh.extents().slice_away(dimension).product();
        ASSERT(vars.number_of_grid_points() == slice_grid_points,
               "vars_on_slice has wrong number of grid points.  "
               "Expected "
                   << slice_grid_points << ", got "
                   << vars.number_of_grid_points());
        // Get dt<U> on this slice
        const auto dt_vars =
            data_on_slice(volume_all_at_vars, mesh.extents(), dimension,
                          index_to_slice_at(mesh.extents(), direction));
        const auto& char_speeds = external_bdry_char_speeds.at(direction);
        // ------------------------------- (2)
        // Create a TempTensor that stores all temporaries computed
        // here and elsewhere
        TempBuffer<BoundaryConditions_detail::all_local_vars<VolumeDim>> buffer(
            slice_grid_points);
        // ------------------------------- (2.2)
        // Compute local variables, including:
        // (A) unit normal form to interface
        // (B) 4metric, inv4metric, lapse, shift on this slice
        // (C) dampign parameter ConstraintGamma2 on this slice
        // (D) Compute projection operator on this slice
        // (E) dt<U> on this slice from `volume_all_at_vars`
        BoundaryConditions_detail::local_variables(
            make_not_null(&buffer), box, direction, dimension, mesh, vars,
            dt_vars, unit_normal_one_form, char_speeds);

        db::mutate<dt_variables_tag>(
            make_not_null(&box),
            // Function that applies bdry conditions to dt<variables>
            [
              &volume_grid_points, &slice_grid_points, &mesh, &dimension,
              &direction, &buffer, &vars, &dt_vars, &unit_normal_one_form
            ](const gsl::not_null<db::item_type<dt_variables_tag>*>
                  volume_dt_vars,
              const double /* time */, const auto& /* boundary_condition */
              ) noexcept {
              // ------------------------------- (1)
              // Preliminaries
              ASSERT(
                  volume_dt_vars->number_of_grid_points() == volume_grid_points,
                  "volume_dt_vars has wrong number of grid points.  Expected "
                      << volume_grid_points << ", got "
                      << volume_dt_vars->number_of_grid_points());
              // ------------------------------- (2)
              // Compute desired values of dt_volume_vars
              //
              // ------------------------------- (2.1)
              // Get desired values of dt<Uchar>
              // For now, we set to  (Freezing, Freezing, Freezing)
              const auto bc_dt_u_psi = BoundaryConditions_detail::set_dt_u_psi<
                  typename Tags::UPsi<VolumeDim, Frame::Inertial>::type,
                  VolumeDim>::apply(UPsiMethod, buffer, vars, dt_vars,
                                    unit_normal_one_form);
              const auto bc_dt_u_zero =
                  BoundaryConditions_detail::set_dt_u_zero<
                      typename Tags::UZero<VolumeDim, Frame::Inertial>::type,
                      VolumeDim>::apply(UZeroMethod, buffer, vars, dt_vars,
                                        unit_normal_one_form);
              const auto bc_dt_u_plus =
                  BoundaryConditions_detail::set_dt_u_plus<
                      typename Tags::UPlus<VolumeDim, Frame::Inertial>::type,
                      VolumeDim>::apply(UPlusMethod, buffer, vars, dt_vars,
                                        unit_normal_one_form);
              const auto bc_dt_u_minus =
                  BoundaryConditions_detail::set_dt_u_minus<
                      typename Tags::UMinus<VolumeDim, Frame::Inertial>::type,
                      VolumeDim>::apply(UMinusMethod, buffer, vars, dt_vars,
                                        unit_normal_one_form);
              // Convert them to desired values on dt<U>
              const auto bc_dt_all_u =
                  evolved_fields_from_characteristic_fields(
                      get<::Tags::TempScalar<26, DataVector>>(
                          buffer),  // Gamma2
                      bc_dt_u_psi, bc_dt_u_zero, bc_dt_u_plus, bc_dt_u_minus,
                      unit_normal_one_form);
              // Now store final values of dt<U> in suitable data structure
              // FIXME: How can I extract this list of dt<U> tags directly
              // from `dt_variables_tag`?
              const tuples::TaggedTuple<
                  db::add_tag_prefix<
                      Metavariables::temporal_id::template step_prefix,
                      gr::Tags::SpacetimeMetric<VolumeDim, Frame::Inertial,
                                                DataVector>>,
                  db::add_tag_prefix<
                      Metavariables::temporal_id::template step_prefix,
                      Tags::Phi<VolumeDim, Frame::Inertial>>,
                  db::add_tag_prefix<
                      Metavariables::temporal_id::template step_prefix,
                      Tags::Pi<VolumeDim, Frame::Inertial>>>
                  bc_dt_tuple(
                      std::move(get<gr::Tags::SpacetimeMetric<
                                    VolumeDim, Frame::Inertial, DataVector>>(
                          bc_dt_all_u)),
                      std::move(get<Tags::Phi<VolumeDim, Frame::Inertial>>(
                          bc_dt_all_u)),
                      std::move(get<Tags::Pi<VolumeDim, Frame::Inertial>>(
                          bc_dt_all_u)));
              const auto slice_data_ = variables_from_tagged_tuple(bc_dt_tuple);
              const auto* slice_data = slice_data_.data();

              // ------------------------------- (2.4)
              // Assign BC values of dt_volume_vars on external boundary
              // slices of volume variables
              auto* const volume_dt_data = volume_dt_vars->data();
              for (SliceIterator si(
                       mesh.extents(), dimension,
                       index_to_slice_at(mesh.extents(), direction));
                   si; ++si) {
                for (size_t i = 0; i < number_of_independent_components; ++i) {
                  // clang-tidy: do not use pointer arithmetic
                  volume_dt_data[si.volume_offset() +       // NOLINT
                                 i * volume_grid_points] =  // NOLINT
                      slice_data[si.slice_offset() +        // NOLINT
                                 i * slice_grid_points];    // NOLINT
                }
              }
            },
            db::get<::Tags::Time>(box).value(),
            get<typename Metavariables::boundary_condition_tag>(cache));
      }

      return std::forward_as_tuple(std::move(box));
    }
  };

  template <size_t VolumeDim, typename DbTags>
  struct apply_impl<VolumeDim, UPsiBcMethod::AnalyticBc,
                    UZeroBcMethod::AnalyticBc, UPlusBcMethod::AnalyticBc,
                    UMinusBcMethod::AnalyticBc, DbTags> {
    static std::tuple<db::DataBox<DbTags>&&> function_impl(
        db::DataBox<DbTags>& box,
        const Parallel::ConstGlobalCache<Metavariables>& cache) noexcept {
      using system = typename Metavariables::system;

      // Apply the boundary condition
      db::mutate_apply<tmpl::list<::Tags::Interface<
                           ::Tags::BoundaryDirectionsExterior<VolumeDim>,
                           typename system::variables_tag>>,
                       tmpl::list<>>(
          [](const gsl::not_null<db::item_type<::Tags::Interface<
                 ::Tags::BoundaryDirectionsExterior<VolumeDim>,
                 typename system::variables_tag>>*>
                 external_bdry_vars,
             const double time, const auto& boundary_condition,
             const auto& boundary_coords) noexcept {
            // Loop over external boundaries
            for (auto& external_direction_and_vars : *external_bdry_vars) {
              auto& direction = external_direction_and_vars.first;
              auto& vars = external_direction_and_vars.second;
              // Get evolved variables on current boundary from AnalyticSolution
              // and assign them to `vars`
              vars.assign_subset(boundary_condition.variables(
                  boundary_coords.at(direction), time,
                  typename system::variables_tag::type::tags_list{}));
            }
          },
          make_not_null(&box), db::get<::Tags::Time>(box).value(),
          get<typename Metavariables::boundary_condition_tag>(cache),
          db::get<::Tags::Interface<
              ::Tags::BoundaryDirectionsExterior<VolumeDim>,
              ::Tags::Coordinates<VolumeDim, Frame::Inertial>>>(box));

      contribute_data_to_mortar(make_not_null(&box), cache);
      return std::forward_as_tuple(std::move(box));
    }
  };

  /* ------------------------------------------------------------------------
   * ------------------------------------------------------------------------
   * ---------------- SEND DATA TO MORTAR FOR ANALYTIC BCs ------------------
   * ------------------------------------------------------------------------
   * ------------------------------------------------------------------------
   */
  template <typename DbTags>
  static void contribute_data_to_mortar(
      const gsl::not_null<db::DataBox<DbTags>*> box,
      const Parallel::ConstGlobalCache<Metavariables>& cache) noexcept {
    using system = typename Metavariables::system;
    constexpr size_t volume_dim = system::volume_dim;

    const auto& element = db::get<::Tags::Element<volume_dim>>(*box);
    const auto& temporal_id =
        db::get<typename Metavariables::temporal_id>(*box);
    const auto& normal_dot_numerical_flux_computer =
        get<typename Metavariables::normal_dot_numerical_flux>(cache);

    for (const auto& direction : element.external_boundaries()) {
      const auto mortar_id = std::make_pair(
          direction, ElementId<volume_dim>::external_boundary_id());

      auto interior_data = DgActions_detail::compute_local_mortar_data(
          *box, direction, normal_dot_numerical_flux_computer,
          ::Tags::BoundaryDirectionsInterior<volume_dim>{}, Metavariables{});

      auto exterior_data = DgActions_detail::compute_packaged_data(
          *box, direction, normal_dot_numerical_flux_computer,
          ::Tags::BoundaryDirectionsExterior<volume_dim>{}, Metavariables{});

      db::mutate<::Tags::VariablesBoundaryData>(
          box, [&mortar_id, &temporal_id, &interior_data, &exterior_data ](
                   const gsl::not_null<
                       db::item_type<::Tags::VariablesBoundaryData, DbTags>*>
                       mortar_data) noexcept {
            mortar_data->at(mortar_id).local_insert(temporal_id,
                                                    std::move(interior_data));
            mortar_data->at(mortar_id).remote_insert(temporal_id,
                                                     std::move(exterior_data));
          });
    }
  }

 public:
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Here be user logic that determines / selects from various options for
    // setting BCs on individual characteristic variables
    return apply_impl<Metavariables::system::volume_dim,
                      // Constraint preserving Bjorhus-type BC
                      UPsiBcMethod::ConstraintPreservingBjorhus,
                      // Freezing BC
                      UZeroBcMethod::Freezing,
                      // dont change choice for UPlus below, unless
                      // all analytic BCs are being requested
                      UPlusBcMethod::Freezing,
                      // Freezing BC
                      UMinusBcMethod::Freezing, DbTags>::function_impl(box,
                                                                       cache);
  }
};

}  // namespace Actions
}  // namespace GeneralizedHarmonic