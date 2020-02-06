// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/optional.hpp>
#include <deque>
#include <memory>
#include <tuple>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/TimeStepId.hpp"

namespace Cce {
/// \cond
class GHLockstepInterfaceManager;
/// \endcond

/*!
 * \brief Abstract base class for storage and retrieval of generalized harmonic
 * quantities communicated from a Cauchy simulation to the Cce system.
 *
 * \details The functions that are required to be overriden in the derived
 * classes are:
 * - `GHWorldtubeInterfaceManager::get_clone()`: should return a
 * `std::unique_ptr<GHWorldtubeInterfaceManager>` with cloned state.
 * - `GHWorldtubeInterfaceManager::insert_gh_data()`: should store the portions
 * of the provided generalized harmonic data that are required to provide useful
 * boundary values for the CCE evolution at requested timesteps.
 * - `GHWorldtubeInterfaceManager::request_gh_data()`: should register requests
 * from the CCE evolution for boundary data.
 * - `GHWorldtubeInterfaceManager::try_retrieve_gh_data()`: should return a
 * `std::tuple<TimeStepId, tnsr::aa<DataVector, 3>,
 *  tnsr::iaa<DataVector, 3>, tnsr::aa<DataVector, 3>>` containing the boundary
 * data associated with the least recent requested timestep if enough data has
 * been supplied via `insert_gh_data()` to determine the boundary data.
 * Otherwise, return a `boost::none` to indicate that the CCE system must
 * continue waiting for generalized harmonic input.
 */
class GHWorldtubeInterfaceManager : public PUP::able {
 public:
  using creatable_classes = tmpl::list<GHLockstepInterfaceManager>;

  WRAPPED_PUPable_abstract(GHWorldtubeInterfaceManager);

  virtual std::unique_ptr<GHWorldtubeInterfaceManager> get_clone() const
      noexcept = 0;

  virtual void insert_gh_data(
      const TimeStepId& time_id,
      const tnsr::aa<DataVector, 3>& spacetime_metric,
      const tnsr::iaa<DataVector, 3>& phi, const tnsr::aa<DataVector, 3>& pi,
      const tnsr::aa<DataVector, 3>& dt_spacetime_metric,
      const tnsr::iaa<DataVector, 3>& dt_phi,
      const tnsr::aa<DataVector, 3>& dt_pi) noexcept = 0;

  virtual void request_gh_data(const TimeStepId&) noexcept = 0;

  virtual auto try_retrieve_first_ready_gh_data() noexcept -> boost::optional<
      std::tuple<TimeStepId, tnsr::aa<DataVector, 3>, tnsr::iaa<DataVector, 3>,
                 tnsr::aa<DataVector, 3>>> = 0;
};

/*!
 * \brief Simple implementation of a `GHWorldtubeInterfaceManager` that only
 * provides boundary data on matching `TimeStepId`s
 *
 * \details This version of the interface manager demands that the CCE system
 * and the generalized harmonic system that it communicates with evolve with an
 * identical time stepper and on identical time step intervals (they evolve in
 * 'lock step'), and simply provides a buffer for the asynchronous send and
 * receive of the generalized harmonic boundary data.
 */
class GHLockstepInterfaceManager : public GHWorldtubeInterfaceManager {
 public:

  static constexpr OptionString help{
    "Pass data between GH and CCE systems on matching timesteps only."};

  using options = tmpl::list<>;

  GHLockstepInterfaceManager() = default;

  explicit GHLockstepInterfaceManager(CkMigrateMessage* /*unused*/) noexcept {}

  WRAPPED_PUPable_decl_template(GHLockstepInterfaceManager);  // NOLINT

  std::unique_ptr<GHWorldtubeInterfaceManager> get_clone() const
      noexcept override;

  /// \brief Store a provided data set in a `std::deque` for retrieval once the
  /// same timestep has been requested by the CCE system
  void insert_gh_data(
      const TimeStepId& time_id,
      const tnsr::aa<DataVector, 3>& spacetime_metric,
      const tnsr::iaa<DataVector, 3>& phi, const tnsr::aa<DataVector, 3>& pi,
      const tnsr::aa<DataVector, 3>& dt_spacetime_metric =
          tnsr::aa<DataVector, 3>{},
      const tnsr::iaa<DataVector, 3>& dt_phi = tnsr::iaa<DataVector, 3>{},
      const tnsr::aa<DataVector, 3>& dt_pi =
          tnsr::aa<DataVector, 3>{}) noexcept override;

  /// \brief Supply a request for boundary data.
  void request_gh_data(const TimeStepId& time_id) noexcept override;

  /// \brief Return a `std::tuple` of the generalized harmonic boundary data
  /// if the least recently requested `TimeStepId` from the CCE system has a
  /// corresponding data set supplied by the generalized harmonic system,
  /// otherwise returns `boost::none`.
  auto try_retrieve_first_ready_gh_data() noexcept -> boost::optional<
      std::tuple<TimeStepId, tnsr::aa<DataVector, 3>, tnsr::iaa<DataVector, 3>,
                 tnsr::aa<DataVector, 3>>> override;

  /// \brief Return the number of requests that haven't been retrieved
  size_t number_of_pending_requests() const noexcept {
    return data_requests_.size();
  }

  /// \brief Return the number of data points that haven't been retrieved
  size_t number_of_data_points() const noexcept {
    return provided_data_.size();
  }

  /// Serialization for Charm++.
  void pup(PUP::er& p) noexcept override;

 private:
  std::deque<std::tuple<TimeStepId, tnsr::aa<DataVector, 3>,
                        tnsr::iaa<DataVector, 3>, tnsr::aa<DataVector, 3>>>
      provided_data_;
  std::deque<TimeStepId> data_requests_;
};
}  // namespace Cce

