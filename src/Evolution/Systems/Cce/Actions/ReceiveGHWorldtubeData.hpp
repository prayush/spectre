// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Actions/ReceiveWorldtubeData.hpp"
#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"
#include "Evolution/Systems/Cce/WorldtubeInterfaceManager.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Exit.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"


namespace Cce {
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Stores the boundary data from the GH evolution in the
 * `Cce::GHWorldtubeInterfaceManager`, and sends to the `EvolutionComponent`
 * (template argument) if the data fulfills a prior request.
 *
 * \details If the new data fulfills a prior request submitted to the
 * `Cce::GHWorldtubeInterfaceManager`, this will dispatch the result to
 * `Cce::Actions::SendToEvolution<GHWorldtubeBoundary<Metavariables>,
 * EvolutionComponent>` for sending the processed boundary data to
 * the `EvolutionComponent`.
 *
 * \ref DataBoxGroup changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - `Tags::GHInterfaceManager`
 */
template <typename EvolutionComponent>
struct ReceiveGHWorldtubeData {
  template <
      typename ParallelComponent, typename... DbTags, typename Metavariables,
      typename ArrayIndex,
      Requires<tmpl::list_contains_v<tmpl::list<DbTags...>,
                                     Tags::GHInterfaceManager>>
          Requires<tmpl2::flat_any_v<cpp17::is_same_v<
              ::Tags::Variables<
                  typename Metavariables::cce_boundary_communication_tags>,
              DbTags>...>> = nullptr>
  static void apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const TimeStepId& time,
      const tnsr::aa<DataVector, 3>& spacetime_metric,
      const tnsr::iaa<DataVector, 3>& phi, const tnsr::aa<DataVector, 3>& pi,
      const tnsr::aa<DataVector, 3>& dt_spacetime_metric =
          tnsr::aa<DataVector, 3>{},
      const tnsr::iaa<DataVector, 3>& dt_phi = tnsr::iaa<DataVector, 3>{},
      const tnsr::aa<DataVector, 3>& dt_pi =
          tnsr::aa<DataVector, 3>{}) noexcept {
    db::mutate<Tags::GHInterfaceManager>(
        make_not_null(&box),
        [
          &spacetime_metric, &phi, &pi, &dt_spacetime_metric, &dt_phi, &dt_pi,
          &time, &cache
        ](const gsl::not_null<db::item_type<Tags::GHInterfaceManager>*>
              interface_manager) noexcept {
          (*interface_manager)
              ->insert_gh_data(time, spacetime_metric, phi, pi,
                               dt_spacetime_metric, dt_phi, dt_pi);
          if (const auto gh_data =
                  (*interface_manager)->try_retrieve_first_ready_gh_data()) {
            Parallel::simple_action<Actions::SendToEvolution<
                GHWorldtubeBoundary<Metavariables>, EvolutionComponent>>(
                Parallel::get_parallel_component<
                    GHWorldtubeBoundary<Metavariables>>(cache),
                get<1>(*gh_data), get<2>(*gh_data), get<3>(*gh_data),
                get<0>(*gh_data));
          }
        });
  }
};
}  // namespace Actions
}  // namespace Cce
