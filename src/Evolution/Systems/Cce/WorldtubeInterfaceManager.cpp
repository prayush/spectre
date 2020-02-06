// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/WorldtubeInterfaceManager.hpp"

#include <deque>
#include <memory>
#include <tuple>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Time/TimeStepId.hpp"
#include "Parallel/CharmPupable.hpp"

namespace Cce{

std::unique_ptr<GHWorldtubeInterfaceManager>
GHLockstepInterfaceManager::get_clone() const noexcept {
  auto clone = std::make_unique<GHLockstepInterfaceManager>();
  for (auto data_entry : provided_data_) {
    clone->insert_gh_data(get<0>(data_entry), get<1>(data_entry),
                          get<2>(data_entry), get<3>(data_entry));
  }
  for (auto request : data_requests_) {
    clone->request_gh_data(request);
  }
  ASSERT(clone->number_of_pending_requests() == number_of_pending_requests(),
         "Cloning of GHLockstepInterfaceManager failed.");
  ASSERT(clone->number_of_data_points() == number_of_data_points(),
         "Cloning of GHLockstepInterfaceManager failed.");
  return clone;
}

void GHLockstepInterfaceManager::insert_gh_data(
    const TimeStepId& time_id, const tnsr::aa<DataVector, 3>& spacetime_metric,
    const tnsr::iaa<DataVector, 3>& phi, const tnsr::aa<DataVector, 3>& pi,
    const tnsr::aa<DataVector, 3>& /*dt_spacetime_metric*/,
    const tnsr::iaa<DataVector, 3>& /*dt_phi*/,
    const tnsr::aa<DataVector, 3>& /*dt_pi*/) noexcept {
  provided_data_.push_back(std::make_tuple(time_id, spacetime_metric, phi, pi));
}

void GHLockstepInterfaceManager::request_gh_data(
    const TimeStepId& time_id) noexcept {
  data_requests_.push_back(time_id);
}

boost::optional<std::tuple<TimeStepId, tnsr::aa<DataVector, 3>,
                           tnsr::iaa<DataVector, 3>, tnsr::aa<DataVector, 3>>>
GHLockstepInterfaceManager::try_retrieve_first_ready_gh_data() noexcept {
  if (provided_data_.empty() or data_requests_.empty()) {
    return boost::none;
  }
  const auto lower_bound = std::lower_bound(
      provided_data_.begin(), provided_data_.end(), data_requests_.front(),
      [](const std::tuple<TimeStepId, tnsr::aa<DataVector, 3>,
                          tnsr::iaa<DataVector, 3>, tnsr::aa<DataVector, 3>>&
             data_entry,
         const TimeStepId& time_id) noexcept {
        return get<0>(data_entry) < time_id;
      });
  if (lower_bound != provided_data_.end() and
      get<0>(*lower_bound) == data_requests_.front()) {
    auto result = std::move(*lower_bound);
    provided_data_.erase(lower_bound);
    data_requests_.pop_front();
    return result;
  }
  return boost::none;
}

void GHLockstepInterfaceManager::pup(PUP::er& p) noexcept {
  p | provided_data_;
  p | data_requests_;
}

/// \cond
PUP::able::PUP_ID Cce::GHLockstepInterfaceManager::my_PUP_ID = 0;
/// \endcond
}  // namespace Cce
