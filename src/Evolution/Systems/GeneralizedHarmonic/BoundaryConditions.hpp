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
///      - External<Tags::dt<typename system::variables_tag>>
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
  template <size_t VolumeDim, PsiBcMethod PsiMethod, PhiBcMethod PhiMethod,
            PiBcMethod PiMethod, typename DbTags>
  struct apply_impl {
    static std::tuple<db::DataBox<DbTags>&&> function_impl(
        db::DataBox<DbTags>& box,
        const Parallel::ConstGlobalCache<Metavariables>& cache) noexcept {
      // Get information about system: tags for evolved variables
      // and their time derivatives
      using system = typename Metavariables::system;
      using variables_tag = typename system::variables_tag;
      using dt_variables_tag =
          db::add_tag_prefix<Metavariables::temporal_id::template step_prefix,
                             variables_tag>;

      // Apply the boundary condition
      db::mutate<dt_variables_tag>(
          make_not_null(&box),
          // Function that applies bdry conditions to dt<variables>
          [&](const gsl::not_null<db::item_type<dt_variables_tag>*>
                  volume_dt_vars,
              const auto& external_bdry_vars,
              const db::item_type<::Tags::Mesh<VolumeDim>>& mesh,
              const double time, const auto& boundary_condition,
              const auto& boundary_coords, const auto& unit_normal_one_forms,
              const auto& external_bdry_char_speeds) noexcept {
            // ------------------------------- (1)
            // Get preliminary quantities
            const auto& dt_vars_const = *volume_dt_vars;
            constexpr const size_t number_of_independent_components =
                dt_variables_tag::type::number_of_independent_components;
            const size_t volume_grid_points = mesh.extents().product();
            ASSERT(
                volume_dt_vars->number_of_grid_points() == volume_grid_points,
                "volume_dt_vars has wrong number of grid points.  Expected "
                    << volume_grid_points << ", got "
                    << volume_dt_vars->number_of_grid_points());

            // ------------------------------- (2)
            // Loop over external boundaries and set dt_volume_vars on them
            for (auto& external_direction_and_vars : external_bdry_vars) {
              const auto& direction = external_direction_and_vars.first;
              const size_t dimension = direction.dimension();
              const auto& vars = external_direction_and_vars.second;
              const size_t slice_grid_points =
                  mesh.extents().slice_away(dimension).product();
              ASSERT(vars.number_of_grid_points() == slice_grid_points,
                     "vars_on_slice has wrong number of grid points.  Expected "
                         << slice_grid_points << ", got "
                         << vars.number_of_grid_points());

              // ------------------------------- (2.1)
              // Compute desired values of dt_volume_vars
              //
              // Create a TempTensor that stores all temporaries computed
              // here and elsewhere
              TempBuffer<tmpl::list<
                  ::Tags::TempI<0, VolumeDim, Frame::Inertial, DataVector>,
                  ::Tags::TempAB<1, VolumeDim, Frame::Inertial, DataVector>>>
                  buffer(slice_grid_points);
              BoundaryConditions_detail::local_variables(
                  make_not_null(&buffer), box, direction, dimension, mesh, vars,
                  dt_vars_const, unit_normal_one_forms.at(direction),
                  external_bdry_char_speeds.at(direction));

              // FIXME: Get ingredients for other BCs -
              // (A) unit normal form to interface
              // (B) 4metric, inv4metric, lapse, shift on this slice
              // (C) dampign parameter ConstraintGamma2 on this slice
              // (D) Compute projection operator on this slice
              // (E) dt<U> on this slice from `volume_dt_vars`

              // For now, we set to  (Freezing, Freezing, Freezing)
              const auto bc_dt_psi =
                  make_with_value<db::item_type<gr::Tags::SpacetimeMetric<
                      VolumeDim, Frame::Inertial, DataVector>>>(
                      get<gr::Tags::SpacetimeMetric<VolumeDim, Frame::Inertial,
                                                    DataVector>>(vars),
                      1.e-12);
              const auto bc_dt_phi = make_with_value<
                  db::item_type<Tags::Phi<VolumeDim, Frame::Inertial>>>(
                  get<Tags::Phi<VolumeDim, Frame::Inertial>>(vars), 1.e-12);
              const auto bc_dt_pi = make_with_value<
                  db::item_type<Tags::Pi<VolumeDim, Frame::Inertial>>>(
                  get<Tags::Pi<VolumeDim, Frame::Inertial>>(vars), 1.e-12);

              // Now store final values of dt<U> in suitable data structure
              // FIXME:
              // How can I extract this list of dt<U> tags directly from
              // `dt_variables_tag`?
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
                  bc_dt_tuple(std::move(bc_dt_psi), std::move(bc_dt_phi),
                              std::move(bc_dt_pi));
              const auto slice_data_ = variables_from_tagged_tuple(bc_dt_tuple);
              const auto* slice_data = slice_data_.data();

              // ------------------------------- (2.2)
              // Assign BC values of dt_volume_vars on external boundary
              // slices of volume variables
              //
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
            }
          },
          db::get<::Tags::Interface<
              ::Tags::BoundaryDirectionsExterior<VolumeDim>, variables_tag>>(
              box),
          db::get<::Tags::Mesh<VolumeDim>>(box),
          db::get<::Tags::Time>(box).value(),
          get<typename Metavariables::boundary_condition_tag>(cache),
          db::get<::Tags::Interface<
              ::Tags::BoundaryDirectionsExterior<VolumeDim>,
              ::Tags::Coordinates<VolumeDim, Frame::Inertial>>>(box),
          db::get<::Tags::Interface<
              ::Tags::BoundaryDirectionsExterior<VolumeDim>,
              ::Tags::Normalized<
                  ::Tags::UnnormalizedFaceNormal<VolumeDim, Frame::Inertial>>>>(
              box),
          db::get<
              ::Tags::Interface<::Tags::BoundaryDirectionsExterior<VolumeDim>,
                                GeneralizedHarmonic::Tags::CharacteristicSpeeds<
                                    VolumeDim, Frame::Inertial>>>(box));

      return std::forward_as_tuple(std::move(box));
    }
  };

  template <size_t VolumeDim, typename DbTags>
  struct apply_impl<VolumeDim, PsiBcMethod::AnalyticBc, PhiBcMethod::AnalyticBc,
                    PiBcMethod::AnalyticBc, DbTags> {
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
   * ---------------- SEND DATA TO MORTAR  ----------------
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
    // Here be logic that selects from various options for
    // setting BCs on individual evolved variables
    return apply_impl<Metavariables::system::volume_dim, /* **** */
                      PsiBcMethod::Freezing, PhiBcMethod::Freezing,
                      PiBcMethod::Freezing, DbTags>::function_impl(box, cache);
  }
};

namespace BoundaryConditions_detail {

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
