// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
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
#include "ErrorHandling/Error.hpp"
#include "Evolution/Systems/ScalarWave/BoundaryConditionsImpl.hpp"
#include "Evolution/Systems/ScalarWave/Characteristics.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InterfaceActionHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Parallel/Abort.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
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

namespace ScalarWave {
namespace Actions {

namespace BoundaryConditions_detail {}  // namespace BoundaryConditions_detail

/// \ingroup ActionsGroup
/// \brief Computes contributions on the interior side from the volume, and
/// imposes constraint preserving boundary conditions on the exterior side.
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
            UMinusBcMethod UMinusMethod, typename DbTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename... InboxTags>
  struct apply_impl {
    static std::tuple<db::DataBox<DbTags>&&> function_impl(
        db::DataBox<DbTags>& box,
        tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
        Parallel::ConstGlobalCache<Metavariables>& cache,
        const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) noexcept {
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
          ::Tags::Interface<::Tags::BoundaryDirectionsInterior<VolumeDim>,
                            ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<
                                VolumeDim, Frame::Inertial>>>>(box);
      // const auto& external_bdry_vars = db::get<::Tags::Interface<
      //::Tags::BoundaryDirectionsInterior<VolumeDim>, variables_tag>>(box);
      const auto& volume_all_vars = db::get<variables_tag>(box);
      const auto& volume_all_dt_vars = db::get<dt_variables_tag>(box);
      const auto& external_bdry_char_speeds = db::get<
          ::Tags::Interface<::Tags::BoundaryDirectionsInterior<VolumeDim>,
                            Tags::CharacteristicSpeeds<VolumeDim>>>(box);

      // ------------------------------- (2)
      // Apply the boundary condition
      // Loop over external boundaries and set dt_volume_vars on them
      for (auto& external_direction_and_normals : unit_normal_one_forms) {
        const auto& direction = external_direction_and_normals.first;
        const size_t dimension = direction.dimension();
        const auto& unit_normal_one_form =
            external_direction_and_normals.second;
        const size_t slice_grid_points =
            mesh.extents().slice_away(dimension).product();
        // Get U on this slice
        const auto vars =
            data_on_slice(volume_all_vars, mesh.extents(), dimension,
                          index_to_slice_at(mesh.extents(), direction));
        ASSERT(vars.number_of_grid_points() == slice_grid_points,
               "vars_on_slice has wrong number of grid points.  "
               "Expected "
                   << slice_grid_points << ", got "
                   << vars.number_of_grid_points());
        // Get dt<U> on this slice
        const auto dt_vars =
            data_on_slice(volume_all_dt_vars, mesh.extents(), dimension,
                          index_to_slice_at(mesh.extents(), direction));
        // Get characteristic speeds
        const auto& char_speeds = external_bdry_char_speeds.at(direction);
        // Get constraint damping parameter: gamma2
        const auto& constraint_gamma2 =
            (db::get<::Tags::Interface<
                 ::Tags::BoundaryDirectionsInterior<VolumeDim>,
                 Tags::ConstraintGamma2>>(box))
                .at(direction);

        // ------------------------------- (2)
        // Create a TempTensor that stores all temporaries computed
        // here and elsewhere
        TempBuffer<BoundaryConditions_detail::all_local_vars<VolumeDim>> buffer(
            slice_grid_points);
        // ------------------------------- (2.2)
        // Compute local variables, including:
        // (A) unit normal form to interface
        // (B) dampign parameter ConstraintGamma2 on this slice
        // (C) dt<U> on this slice from `volume_all_dt_vars`
        BoundaryConditions_detail::local_variables(
            make_not_null(&buffer), box, direction, dimension, mesh, vars,
            dt_vars, unit_normal_one_form, char_speeds, constraint_gamma2);

        db::mutate<dt_variables_tag>(
            make_not_null(&box),
            // Function that applies bdry conditions to dt<variables>
            [
              &volume_grid_points, &slice_grid_points, &mesh, &dimension,
              &direction, &buffer, &vars, &dt_vars, &unit_normal_one_form,
              /*&inertial_coords,*/ &char_speeds, &constraint_gamma2
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
          // Get desired values of CharProjection<dt<U>>
          //
          //
#if 1
              // ---
              // debugPK : We want to compute the freezing / cp Bjorhus-type bc
              // alongwith all ingredients here:
              // 6-9. Characteristic projected time derivatives of evolved
              // fields
              // storage for DT<UChar> = CharProjection(dt<U>)
              const auto& rhs_dt_psi = get<::Tags::dt<Psi>>(dt_vars);
              const auto& rhs_dt_pi = get<::Tags::dt<Pi>>(dt_vars);
              const auto& rhs_dt_phi = get<::Tags::dt<Phi<VolumeDim>>>(dt_vars);
              // get<Tags::ConstraintGamma2>(vars_on_this_slice);
              const auto char_projected_dt_u = characteristic_fields(
                  constraint_gamma2, rhs_dt_psi, rhs_dt_pi, rhs_dt_phi,
                  unit_normal_one_form);
              auto& bc_dt_u_psi = get<Tags::UPsi>(char_projected_dt_u);
              auto& bc_dt_u_zero =
                  get<Tags::UZero<VolumeDim>>(char_projected_dt_u);
              auto& bc_dt_u_plus = get<Tags::UPlus>(char_projected_dt_u);
              auto bc_dt_u_minus = get<Tags::UMinus>(char_projected_dt_u);
              // Set BC (note that only U- is incoming in flat spacetime)
              // And the right hand side set is multiplied in
              // 1) freezing Neumann on UMinus
              // get(bc_dt_u_minus) = -get(constraint_gamma2) *
              // get(bc_dt_u_psi);

              // 2) impose penalty on U- not being zero
              constexpr double penalty_factor = -1. / 6.;
              const auto& psi = get<Psi>(vars);
              const auto& pi = get<Pi>(vars);
              const auto& phi = get<Phi<VolumeDim>>(vars);
              const auto char_projected_u = characteristic_fields(
                  constraint_gamma2, psi, pi, phi, unit_normal_one_form);
              auto& u_minus_now = get<Tags::UMinus>(char_projected_u);
              get(bc_dt_u_minus) += penalty_factor * get(u_minus_now);
#endif
#if 0
              // At all points on the interface where the char speed of any
              // (given) characteristic field is +ve, we "do nothing", and
              // when its -ve, we apply Bjorhus BCs. This is achieved through
              // `set_bc_when_char_speed_is_negative`.
              //
              // debugPK
              if (debugPK) {
                Parallel::printf("\n\nGoing to set BC for UPsi...\n");
              }
              const auto bc_dt_u_psi =
                  BoundaryConditions_detail::set_bc_when_char_speed_is_negative(
                      get<::Tags::TempScalar<6, DataVector>>(buffer),
                      BoundaryConditions_detail::set_dt_u_psi<
                          typename Tags::UPsi::type,
                          VolumeDim>::apply(UPsiMethod, buffer, vars, dt_vars,
                                            unit_normal_one_form),
                      char_speeds.at(0));
              // debugPK
              if (debugPK) {
                Parallel::printf("Going to set BC for UZero...\n");
              }
              const auto bc_dt_u_zero =
                  BoundaryConditions_detail::set_bc_when_char_speed_is_negative(
                      get<::Tags::Tempi<7, VolumeDim, Frame::Inertial,
                                        DataVector>>(buffer),
                      BoundaryConditions_detail::set_dt_u_zero<
                          typename Tags::UZero<VolumeDim>::type,
                          VolumeDim>::apply(UZeroMethod, buffer, vars, dt_vars,
                                            unit_normal_one_form),
                      char_speeds.at(1));
              // debugPK
              if (debugPK) {
                Parallel::printf("Going to set BC for U+...\n");
              }
              const auto bc_dt_u_plus =
                  BoundaryConditions_detail::set_bc_when_char_speed_is_negative(
                      get<::Tags::TempScalar<8, DataVector>>(buffer),
                      BoundaryConditions_detail::set_dt_u_plus<
                          typename Tags::UPlus::type,
                          VolumeDim>::apply(UPlusMethod, buffer, vars, dt_vars,
                                            unit_normal_one_form),
                      char_speeds.at(2));
              // copy over the value of bc_dt_u_psi finally set, accounting
              // for the char speed of UPsi. This will be used to set
              // Dt<UMinus>
              get(get<::Tags::TempScalar<6, DataVector>>(buffer)) =
                  get(bc_dt_u_psi);
              // debugPK
              if (debugPK) {
                Parallel::printf("Going to set BC for U-...\n");
              }
              const auto bc_dt_u_minus =
                  BoundaryConditions_detail::set_bc_when_char_speed_is_negative(
                      get<::Tags::TempScalar<9, DataVector>>(buffer),
                      BoundaryConditions_detail::set_dt_u_minus<
                          typename Tags::UMinus::type,
                          VolumeDim>::apply(UMinusMethod, buffer, vars, dt_vars,
                                            unit_normal_one_form),
                      char_speeds.at(3));
#endif
              // Convert them to desired values on dt<U>
              const auto bc_dt_all_u =
                  evolved_fields_from_characteristic_fields(
                      constraint_gamma2, bc_dt_u_psi, bc_dt_u_zero,
                      bc_dt_u_plus, bc_dt_u_minus, unit_normal_one_form);
              // Now store final values of dt<U> in suitable data structure
              // FIXME: How can I extract this list of dt<U> tags directly
              // from `dt_variables_tag`?
              const tuples::TaggedTuple<
                  db::add_tag_prefix<
                      Metavariables::temporal_id::template step_prefix, Psi>,
                  db::add_tag_prefix<
                      Metavariables::temporal_id::template step_prefix,
                      Phi<VolumeDim>>,
                  db::add_tag_prefix<
                      Metavariables::temporal_id::template step_prefix, Pi>>
                  bc_dt_tuple(std::move(get<Psi>(bc_dt_all_u)),
                              std::move(get<Phi<VolumeDim>>(bc_dt_all_u)),
                              std::move(get<Pi>(bc_dt_all_u)));
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
            db::get<::Tags::Time>(box),
            get<typename Metavariables::boundary_condition_tag>(cache));
      }

      return std::forward_as_tuple(std::move(box));
    }
  };

 public:
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList action_list,
      const ParallelComponent* const parallel_component) noexcept {
    // Here be user logic that determines / selects from various options for
    // setting BCs on individual characteristic variables
    return apply_impl<Metavariables::system::volume_dim,
                      // BC choice for U_\Psi
                      UPsiBcMethod::ConstraintPreservingBjorhus,
                      // BC choice for U_0
                      UZeroBcMethod::ConstraintPreservingDirichlet,
                      // BC choice for U_+
                      UPlusBcMethod::Freezing,
                      // BC choice for U_-
                      UMinusBcMethod::ConstraintPreservingBjorhus, DbTags,
                      ArrayIndex, ActionList, ParallelComponent,
                      InboxTags...>::function_impl(box, inboxes, cache,
                                                   array_index, action_list,
                                                   parallel_component);
  }
};

}  // namespace Actions
}  // namespace ScalarWave
