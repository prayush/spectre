// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "ErrorHandling/Error.hpp"
#include "Evolution/Systems/Cce/Actions/ReceiveWorldtubeData.hpp"
#include "Evolution/Systems/Cce/Components/WorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"
#include "Evolution/Systems/Cce/ReceiveTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeInterfaceManager.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace Cce {
namespace Actions {

/// \cond
template <typename BoundaryComponent, typename EvolutionComponent>
struct BoundaryComputeAndSendToEvolution;

template <typename BoundaryComponent, typename EvolutionComponent>
struct SendToEvolution;
/// \endcond

/*!
 * \ingroup ActionsGroup
 * \brief Obtains the CCE boundary data at the specified `time`, and reports it
 * to the `EvolutionComponent` via `Actions::ReceiveWorldtubeData`.
 *
 * \details This uses the `WorldtubeDataManager` to perform all of the work of
 * managing the file buffer, interpolating to the desired time point, and
 * compute the Bondi quantities on the boundary. Once readied, it sends each
 * tensor from the the full `Variables<typename
 * Metavariables::cce_boundary_communication_tags>` back to the
 * `EvolutionComponent`
 *
 * Uses:
 * - DataBox:
 *  - `Tags::H5WorldtubeBoundaryDataManager`
 *
 * \ref DataBoxGroup changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - `Tags::Variables<typename
 * Metavariables::cce_boundary_communication_tags>` (every tensor)
 */
template <typename Metavariables, typename EvolutionComponent>
struct BoundaryComputeAndSendToEvolution<H5WorldtubeBoundary<Metavariables>,
                                         EvolutionComponent> {
  template <typename ParallelComponent, typename... DbTags, typename ArrayIndex,
            Requires<tmpl2::flat_any_v<cpp17::is_same_v<
                ::Tags::Variables<
                    typename Metavariables::cce_boundary_communication_tags>,
                DbTags>...>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const TimeStepId& time) noexcept {
    if (not db::get<Tags::H5WorldtubeBoundaryDataManager>(box)
                .populate_hypersurface_boundary_data(
                    make_not_null(&box), time.substep_time().value())) {
      ERROR("Insufficient boundary data to proceed, exiting early at time " +
            std::to_string(time.substep_time().value()));
    }
    Parallel::receive_data<Cce::ReceiveTags::BoundaryData<
        typename Metavariables::cce_boundary_communication_tags>>(
        Parallel::get_parallel_component<EvolutionComponent>(cache), time,
        db::get<::Tags::Variables<
            typename Metavariables::cce_boundary_communication_tags>>(box),
        true);
  }
};

/*!
 * \ingroup ActionsGroup
 * \brief Submits a request for CCE boundary data at the specified `time` to the
 * `Cce::GHWorldtubeInterfaceManager`, and sends to the `EvolutionComponent`
 * (template argument) it is ready.
 *
 * \details This uses the `Cce::GHWorldtubeInterfaceManager` to perform all of
 * the work of managing the buffer of data sent from the GH system and
 * interpolating if necessary and supported. This dispatches then to
 * `Cce::Actions::SendToEvolution<GHWorldtubeBoundary<Metavariables>,
 * EvolutionComponent>` if the boundary data is ready, otherwise
 * simply submitting the request and waiting for data to become available via
 * `Cce::Actions::ReceiveGHWorldtubeData`, which will call
 * `Cce::Actions::SendToEvolution<GHWorldtubeBoundary<Metavariables>,
 * EvolutionComponent>` as soon as the data becomes available.
 *
 * \ref DataBoxGroup changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - `Tags::GHInterfaceManager`
 */
template <typename Metavariables, typename EvolutionComponent>
struct BoundaryComputeAndSendToEvolution<GHWorldtubeBoundary<Metavariables>,
                                         EvolutionComponent> {
  template <typename ParallelComponent, typename... DbTags, typename ArrayIndex,
            Requires<tmpl2::flat_any_v<cpp17::is_same_v<
                ::Tags::Variables<
                    typename Metavariables::cce_boundary_communication_tags>,
                DbTags>...>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const TimeStepId& time) noexcept {
    db::mutate<Tags::GHInterfaceManager>(
        make_not_null(&box),
        [&time, &
         cache ](const gsl::not_null<db::item_type<Tags::GHInterfaceManager>*>
                     interface_manager) noexcept {
          (*interface_manager)->request_gh_data(time);
          if (const auto gh_data =
                  (*interface_manager)->try_retrieve_first_ready_gh_data()) {
            Parallel::simple_action<Actions::SendToEvolution<
                GHWorldtubeBoundary<Metavariables>, EvolutionComponent>>(
                Parallel::get_parallel_component<
                    GHWorldtubeBoundary<Metavariables>>(cache),
                get<0>(*gh_data), get<1>(*gh_data), get<2>(*gh_data),
                get<3>(*gh_data));
          }
        });
  }
};

/*!
 * \ingroup ActionsGroup
 * \brief Computes Bondi boundary data from GH evolution variables and sends the
 * result to the `EvolutionComponent` (template argument).
 *
 * \details After the computation, this action will call
 * `Cce::Actions::ReceiveWorldtubeData` on the `EvolutionComponent` with each of
 * the types from `typename Metavariables::cce_boundary_communication_tags` sent
 * as arguments
 *
 * \ref DataBoxGroup changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - `Tags::Variables<typename
 *   Metavariables::cce_boundary_communication_tags>` (every tensor)
 */
template <typename Metavariables, typename EvolutionComponent>
struct SendToEvolution<GHWorldtubeBoundary<Metavariables>, EvolutionComponent> {
  template <typename ParallelComponent, typename... DbTags, typename ArrayIndex,
            Requires<tmpl2::flat_any_v<cpp17::is_same_v<
                ::Tags::Variables<
                    typename Metavariables::cce_boundary_communication_tags>,
                DbTags>...>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, const TimeStepId& time,
                    const tnsr::aa<DataVector, 3>& spacetime_metric,
                    const tnsr::iaa<DataVector, 3>& phi,
                    const tnsr::aa<DataVector, 3>& pi) noexcept {
    create_bondi_boundary_data(
        make_not_null(&box), phi, pi, spacetime_metric,
        Parallel::get<InitializationTags::ExtractionRadius>(cache),
        Parallel::get<Spectral::Swsh::Tags::LMax>(cache));
    Parallel::receive_data<Cce::ReceiveTags::BoundaryData<
      typename Metavariables::cce_boundary_communication_tags>>(
          Parallel::get_parallel_component<EvolutionComponent>(cache), time,
          db::get<::Tags::Variables<
          typename Metavariables::cce_boundary_communication_tags>>(box),
          true);
  }
};

}  // namespace Actions
}  // namespace Cce
