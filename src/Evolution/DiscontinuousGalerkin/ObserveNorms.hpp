// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/EventsAndTriggers/Event.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/ReductionActions.hpp"   // IWYU pragma: keep
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Time.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

namespace dg {
namespace Events {
template <size_t VolumeDim, typename Tensors, typename EventRegistrars>
class ObserveNorms;

namespace Registrars {
template <size_t VolumeDim, typename Tensors>
struct ObserveNorms {
  template <typename RegistrarList>
  using f = Events::ObserveNorms<VolumeDim, Tensors, RegistrarList>;
};
}  // namespace Registrars

template <size_t VolumeDim, typename Tensors,
          typename EventRegistrars =
              tmpl::list<Registrars::ObserveNorms<VolumeDim, Tensors>>>
class ObserveNorms;  // IWYU pragma: keep

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief %Observe the RMS values of tensors (averaged over all grid points
 * and all indices).
 *
 * Writes reduction quantities:
 * - `Time`
 * - `NumberOfPoints` = total number of points in the domain
 * - `Norm(*)` = RMS values in `Tensors` =
 *   \f$\operatorname{RMS}\left(\sqrt{\sum_{\text{independent components}}\left[
 *   \text{value} \right]^2}\right)\f$
 *   over all points
 *
 * \warning Currently, only one reduction observation event can be
 * triggered at a given time.  Causing multiple events to run at once
 * will produce unpredictable results.
 */
template <size_t VolumeDim, typename... Tensors, typename EventRegistrars>
class ObserveNorms<VolumeDim, tmpl::list<Tensors...>, EventRegistrars>
    : public Event<EventRegistrars> {
 private:
  using coordinates_tag = ::Tags::Coordinates<VolumeDim, Frame::Inertial>;

  template <typename Tag>
  struct LocalSquareNorm {
    using type = double;
  };

  using L2NormDatum = Parallel::ReductionDatum<double, funcl::Plus<>,
                                               funcl::Sqrt<funcl::Divides<>>,
                                               std::index_sequence<1>>;
  using ReductionData = tmpl::wrap<
      tmpl::append<
          tmpl::list<Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
                     Parallel::ReductionDatum<size_t, funcl::Plus<>>>,
          tmpl::filled_list<L2NormDatum, sizeof...(Tensors)>>,
      Parallel::ReductionData>;

 public:
  /// \cond
  explicit ObserveNorms(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveNorms);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;
  static constexpr OptionString help =
      "Observe the RMS values of tensors (averaged over all grid points\n"
      "and all indices).\n"
      "\n"
      "Writes reduction quantities:\n"
      " * Time\n"
      " * NumberOfPoints = total number of points in the domain\n"
      " * Norm(*) = RMS values in Tensors (see online help details)\n"
      "\n"
      "Warning: Currently, only one reduction observation event can be\n"
      "triggered at a given time.  Causing multiple events to run at once\n"
      "will produce unpredictable results.";

  ObserveNorms() = default;

  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<ReductionData>>;

  using argument_tags = tmpl::list<::Tags::Time, coordinates_tag, Tensors...>;

  template <typename Metavariables, typename ArrayIndex,
            typename ParallelComponent>
  void operator()(
      const Time& time,
      const db::item_type<coordinates_tag>& /*inertial_coordinates*/,
      const db::item_type<Tensors>&... tensors,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/,
      const ParallelComponent* const /*meta*/) const noexcept {
    tuples::TaggedTuple<LocalSquareNorm<Tensors>...> local_square_norms;
    const auto record_norms = [&local_square_norms](
        const auto tensor_tag_v, const auto& tensor) noexcept {
      using tensor_tag = tmpl::type_from<decltype(tensor_tag_v)>;
      double local_square_norm = 0.0;
      for (size_t i = 0; i < tensor.size(); ++i) {
        local_square_norm += alg::accumulate(square(tensor[i]), 0.0);
      }
      get<LocalSquareNorm<tensor_tag>>(local_square_norms) = local_square_norm;
      return 0;
    };
    expand_pack(record_norms(tmpl::type_<Tensors>{}, tensors)...);
    const size_t num_points = get_first_argument(tensors...).begin()->size();

    // Send data to reduction observer
    auto& local_observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();
    Parallel::simple_action<observers::Actions::ContributeReductionData>(
        local_observer,
        observers::ObservationId(
            time.value(), typename Metavariables::element_observation_type{}),
        std::string{"/element_data"},
        std::vector<std::string>{"Time", "NumberOfPoints",
                                 ("L2Norm(" + Tensors::name() + ")")...},
        ReductionData{
            time.value(), num_points,
            std::move(get<LocalSquareNorm<Tensors>>(local_square_norms))...});
  }
};

/// \cond
template <size_t VolumeDim, typename... Tensors, typename EventRegistrars>
PUP::able::PUP_ID ObserveNorms<VolumeDim, tmpl::list<Tensors...>,
                               EventRegistrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Events
}  // namespace dg
