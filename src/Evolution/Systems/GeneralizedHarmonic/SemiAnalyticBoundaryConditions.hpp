// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InterfaceActionHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
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

namespace GeneralizedHarmonic {
namespace Actions {
namespace BoundaryConditions_detail {
template <size_t VolumeDim>
double min_characteristic_speed(
    const typename GeneralizedHarmonic::Tags::CharacteristicSpeeds<
        VolumeDim, Frame::Inertial>::type& char_speeds) noexcept {
  std::array<double, 4> min_speeds{
      {min(char_speeds.at(0)), min(char_speeds.at(1)), min(char_speeds.at(2)),
       min(char_speeds.at(3))}};
  return *std::min_element(min_speeds.begin(), min_speeds.end());
}

template <typename T, typename DataType>
T set_bc_when_char_speed_is_negative(const T& rhs_char_dt_u,
                                     const T& desired_bc_dt_u,
                                     const DataType& char_speed_u) noexcept {
  auto bc_dt_u = rhs_char_dt_u;
  auto it1 = bc_dt_u.begin();
  auto it2 = desired_bc_dt_u.begin();
  for (; it2 != desired_bc_dt_u.end(); ++it1, ++it2) {
    for (size_t i = 0; i < it1->size(); ++i) {
      if (char_speed_u[i] < 0.) {
        (*it1)[i] = (*it2)[i];
      }
    }
  }
  return bc_dt_u;
}
}  // namespace BoundaryConditions_detail
/// \ingroup ActionsGroup
/// \brief Packages data on external boundaries for calculating numerical flux.
/// Computes contributions on the interior side from the volume, and imposes
/// Dirichlet boundary conditions on the exterior side.
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
struct ImposeDirichletBoundaryConditionsForIncoming {
 private:
  // BoundaryConditionMethod and BcSelector are used to select exactly how to
  // apply the Dirichlet boundary condition depending on properties of the
  // system. An overloaded `apply_impl` method is used that implements the
  // boundary condition calculation for the different types.
  enum class BoundaryConditionMethod { AnalyticBcWhereIncomingChar, Unknown };
  template <BoundaryConditionMethod Method>
  using BcSelector = std::integral_constant<BoundaryConditionMethod, Method>;

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
    return apply_impl<Metavariables::system::volume_dim>(
        box, cache,
        BcSelector<BoundaryConditionMethod::AnalyticBcWhereIncomingChar>{});
  }

 private:
  template <typename DbTags>
  static void contribute_data_to_mortar(
      const gsl::not_null<db::DataBox<DbTags>*> box,
      const Parallel::ConstGlobalCache<Metavariables>& cache) noexcept {
    using system = typename Metavariables::system;
    constexpr size_t volume_dim = system::volume_dim;

    const auto& element = db::get<domain::Tags::Element<volume_dim>>(*box);
    const auto& temporal_id =
        db::get<typename Metavariables::temporal_id>(*box);
    const auto& normal_dot_numerical_flux_computer =
        get<typename Metavariables::normal_dot_numerical_flux>(cache);

    auto interior_data = DgActions_detail::compute_local_mortar_data(
        *box, normal_dot_numerical_flux_computer,
        domain::Tags::BoundaryDirectionsInterior<volume_dim>{},
        Metavariables{});

    auto exterior_data = DgActions_detail::compute_packaged_data(
        *box, normal_dot_numerical_flux_computer,
        domain::Tags::BoundaryDirectionsExterior<volume_dim>{},
        Metavariables{});

    for (const auto& direction : element.external_boundaries()) {
      const auto mortar_id = std::make_pair(
          direction, ElementId<volume_dim>::external_boundary_id());

      db::mutate<domain::Tags::VariablesBoundaryData>(
          box,
          [
            &mortar_id, &temporal_id, &direction, &interior_data, &exterior_data
          ](const gsl::not_null<
              db::item_type<domain::Tags::VariablesBoundaryData, DbTags>*>
                mortar_data) noexcept {
            mortar_data->at(mortar_id).local_insert(
                temporal_id, std::move(interior_data.at(direction)));
            mortar_data->at(mortar_id).remote_insert(
                temporal_id, std::move(exterior_data.at(direction)));
          });
    }
  }

  template <size_t VolumeDim, typename DbTags>
  static std::tuple<db::DataBox<DbTags>&&> apply_impl(
      db::DataBox<DbTags>& box,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      std::integral_constant<
          BoundaryConditionMethod,
          BoundaryConditionMethod::
              AnalyticBcWhereIncomingChar> /*meta*/) noexcept {
    using system = typename Metavariables::system;

    static_assert(
        system::is_in_flux_conservative_form or
            cpp17::is_same_v<typename Metavariables::initial_data_tag,
                             typename Metavariables::boundary_condition_tag>,
        "Only analytic boundary conditions, or dirichlet boundary conditions "
        "for conservative systems are implemented");

    // Apply the boundary condition
    db::mutate_apply<tmpl::list<domain::Tags::Interface<
                         domain::Tags::BoundaryDirectionsExterior<VolumeDim>,
                         typename system::variables_tag>>,
                     tmpl::list<>>(
        [&](const gsl::not_null<db::item_type<domain::Tags::Interface<
                domain::Tags::BoundaryDirectionsExterior<VolumeDim>,
                typename system::variables_tag>>*>
                external_bdry_vars,
            const double time, const auto& boundary_condition,
            const auto& boundary_coords, const auto& boundary_normals,
            const auto& boundary_inverse_spatial_metric,
            const auto& boundary_gamma2,
            const auto& boundary_evolved_char_speeds,
            const auto& boundary_evolved_char_vars) noexcept {
          for (auto& external_direction_and_vars : *external_bdry_vars) {
            auto& direction = external_direction_and_vars.first;
            auto& vars = external_direction_and_vars.second;

            // 1. Get evolved characteristic variables and speeds
            auto evolved_char_vars = boundary_evolved_char_vars.at(direction);
            auto evolved_char_speeds =
                boundary_evolved_char_speeds.at(direction);
            // For external boundaries that are within a horizon,
            // all characteristic fields are outgoing (toward the singularity)
            if (BoundaryConditions_detail::min_characteristic_speed<VolumeDim>(
                    evolved_char_speeds) >= 0.) {
              continue;
            }

            // 2. Get other prereq variables
            const auto& inverse_spatial_metric =
                boundary_inverse_spatial_metric.at(direction);
            const auto& gamma2 = boundary_gamma2.at(direction);
            const auto& unit_normal_one_form = boundary_normals.at(direction);

            // 3. Get analytic solution vars on the boundary
            auto analytic_vars = boundary_condition.variables(
                boundary_coords.at(direction), time,
                typename system::variables_tag::type::tags_list{});

            // 4. Compute characteristic variables from analytic solution vars,
            //    i.e. the boundary_condition.variables()
            auto analytic_char_vars = characteristic_fields(
                gamma2, inverse_spatial_metric,
                get<gr::Tags::SpacetimeMetric<VolumeDim, Frame::Inertial,
                                              DataVector>>(analytic_vars),
                get<Tags::Pi<VolumeDim, Frame::Inertial>>(analytic_vars),
                get<Tags::Phi<VolumeDim, Frame::Inertial>>(analytic_vars),
                unit_normal_one_form);

            // 5. For all char fields:
            //    - if given field is "incoming", replace evolved value with
            //      analytic solution's value
            get<Tags::UPsi<VolumeDim, Frame::Inertial>>(evolved_char_vars) =
                BoundaryConditions_detail::set_bc_when_char_speed_is_negative(
                    get<Tags::UPsi<VolumeDim, Frame::Inertial>>(
                        evolved_char_vars),
                    get<Tags::UPsi<VolumeDim, Frame::Inertial>>(
                        analytic_char_vars),
                    evolved_char_speeds.at(0));
            get<Tags::UZero<VolumeDim, Frame::Inertial>>(evolved_char_vars) =
                BoundaryConditions_detail::set_bc_when_char_speed_is_negative(
                    get<Tags::UZero<VolumeDim, Frame::Inertial>>(
                        evolved_char_vars),
                    get<Tags::UZero<VolumeDim, Frame::Inertial>>(
                        analytic_char_vars),
                    evolved_char_speeds.at(1));
            get<Tags::UPlus<VolumeDim, Frame::Inertial>>(evolved_char_vars) =
                BoundaryConditions_detail::set_bc_when_char_speed_is_negative(
                    get<Tags::UPlus<VolumeDim, Frame::Inertial>>(
                        evolved_char_vars),
                    get<Tags::UPlus<VolumeDim, Frame::Inertial>>(
                        analytic_char_vars),
                    evolved_char_speeds.at(2));
            get<Tags::UMinus<VolumeDim, Frame::Inertial>>(evolved_char_vars) =
                BoundaryConditions_detail::set_bc_when_char_speed_is_negative(
                    get<Tags::UMinus<VolumeDim, Frame::Inertial>>(
                        evolved_char_vars),
                    get<Tags::UMinus<VolumeDim, Frame::Inertial>>(
                        analytic_char_vars),
                    evolved_char_speeds.at(3));

            // 6. Transform back the modified char fields to evolved fields,
            auto bc_vars = evolved_fields_from_characteristic_fields(
                gamma2,
                get<Tags::UPsi<VolumeDim, Frame::Inertial>>(evolved_char_vars),
                get<Tags::UZero<VolumeDim, Frame::Inertial>>(evolved_char_vars),
                get<Tags::UPlus<VolumeDim, Frame::Inertial>>(evolved_char_vars),
                get<Tags::UMinus<VolumeDim, Frame::Inertial>>(
                    evolved_char_vars),
                unit_normal_one_form);

            // 7. assign them to `vars`
            vars.assign_subset(bc_vars);
          }
        },
        make_not_null(&box), db::get<::Tags::Time>(box),
        get<typename Metavariables::boundary_condition_tag>(cache),
        db::get<domain::Tags::Interface<
            domain::Tags::BoundaryDirectionsExterior<VolumeDim>,
            domain::Tags::Coordinates<VolumeDim, Frame::Inertial>>>(box),
        db::get<domain::Tags::Interface<
            domain::Tags::BoundaryDirectionsInterior<VolumeDim>,
            ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<
                VolumeDim, Frame::Inertial>>>>(box),
        db::get<domain::Tags::Interface<
            domain::Tags::BoundaryDirectionsInterior<VolumeDim>,
            gr::Tags::InverseSpatialMetric<VolumeDim, Frame::Inertial,
                                           DataVector>>>(box),
        db::get<domain::Tags::Interface<
            domain::Tags::BoundaryDirectionsInterior<VolumeDim>,
            Tags::ConstraintGamma2>>(box),
        db::get<domain::Tags::Interface<
            domain::Tags::BoundaryDirectionsInterior<VolumeDim>,
            Tags::CharacteristicSpeeds<VolumeDim, Frame::Inertial>>>(box),
        db::get<domain::Tags::Interface<
            domain::Tags::BoundaryDirectionsInterior<VolumeDim>,
            Tags::CharacteristicFields<VolumeDim, Frame::Inertial>>>(box));

    contribute_data_to_mortar(make_not_null(&box), cache);
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace GeneralizedHarmonic
