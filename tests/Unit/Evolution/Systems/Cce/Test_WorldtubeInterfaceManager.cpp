// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <cstddef>
#include <memory>
#include <tuple>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/WorldtubeInterfaceManager.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Literals.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace Cce {
namespace {

template <typename Generator>
void test_gh_lockstep_interface_manager(
    const gsl::not_null<Generator*> gen) noexcept {
  UniformCustomDistribution<double> value_dist{-5.0, 5.0};
  UniformCustomDistribution<size_t> timestep_dist{1, 5};

  std::vector<std::tuple<TimeStepId, tnsr::aa<DataVector, 3>,
                         tnsr::iaa<DataVector, 3>, tnsr::aa<DataVector, 3>>>
      expected_gh_data(5);
  size_t running_total = 0;
  GHLockstepInterfaceManager interface_manager{};
  tnsr::aa<DataVector, 3> spacetime_metric{5_st};
  tnsr::iaa<DataVector, 3> phi{5_st};
  tnsr::aa<DataVector, 3> pi{5_st};
  // insert some time ids
  for(size_t i = 0; i < 5; ++i) {
    const size_t substep = running_total % 3;
    const size_t step = running_total / 3;
    // RK3-style substep
    int substep_numerator = 0;
    if(substep == 1) {
      substep_numerator = 2;
    } else if(substep == 2) {
      substep_numerator = 1;
    }
    const Time step_time{{static_cast<double>(step), step + 1.0}, {0, 1}};
    const Time substep_time{{static_cast<double>(step), step + 1.0},
                            {substep_numerator, 2}};
    const TimeStepId time_id{true, static_cast<int64_t>(running_total / 3),
                             step_time, substep, substep_time};
    fill_with_random_values(make_not_null(&spacetime_metric), gen,
                            make_not_null(&value_dist));
    fill_with_random_values(make_not_null(&phi), gen,
                            make_not_null(&value_dist));
    fill_with_random_values(make_not_null(&pi), gen,
                            make_not_null(&value_dist));
    interface_manager.insert_gh_data(time_id, spacetime_metric, phi, pi);
    expected_gh_data[i] =
        std::make_tuple(std::move(time_id), spacetime_metric, phi, pi);
    running_total += timestep_dist(*gen);
  }

  {
    INFO("Insert data then request");
    // choose a timestep to request
    interface_manager.request_gh_data(get<0>(expected_gh_data[1]));
    interface_manager.request_gh_data(get<0>(expected_gh_data[2]));
    CHECK(interface_manager.number_of_pending_requests() == 2);
    CHECK(interface_manager.number_of_data_points() == 5);
    auto retrieved_data = interface_manager.try_retrieve_first_ready_gh_data();
    CHECK(retrieved_data);
    CHECK(get<0>(*retrieved_data) == get<0>(expected_gh_data[1]));
    CHECK(get<1>(*retrieved_data) == get<1>(expected_gh_data[1]));
    CHECK(get<2>(*retrieved_data) == get<2>(expected_gh_data[1]));
    CHECK(get<3>(*retrieved_data) == get<3>(expected_gh_data[1]));

    CHECK(interface_manager.number_of_pending_requests() == 1);
    CHECK(interface_manager.number_of_data_points() == 4);
    retrieved_data = interface_manager.try_retrieve_first_ready_gh_data();
    CHECK(retrieved_data);
    CHECK(get<0>(*retrieved_data) == get<0>(expected_gh_data[2]));
    CHECK(get<1>(*retrieved_data) == get<1>(expected_gh_data[2]));
    CHECK(get<2>(*retrieved_data) == get<2>(expected_gh_data[2]));
    CHECK(get<3>(*retrieved_data) == get<3>(expected_gh_data[2]));

    CHECK(interface_manager.number_of_pending_requests() == 0);
    CHECK(interface_manager.number_of_data_points() == 3);
  }

  {
    INFO("Request then insert data");
    // check that you can request before the data is provided
    const size_t substep = running_total % 3;
    const size_t step = running_total / 3;
    // RK3-style substep
    int substep_numerator = 0;
    if (substep == 1) {
      substep_numerator = 2;
    } else if (substep == 2) {
      substep_numerator = 1;
    }
    const Time step_time{{static_cast<double>(step), step + 1.0}, {0, 1}};
    const Time substep_time{{static_cast<double>(step), step + 1.0},
                            {substep_numerator, 2}};
    const TimeStepId time_id{true, static_cast<int64_t>(running_total / 3),
                             step_time, substep, substep_time};
    fill_with_random_values(make_not_null(&spacetime_metric), gen,
                            make_not_null(&value_dist));
    fill_with_random_values(make_not_null(&phi), gen,
                            make_not_null(&value_dist));
    fill_with_random_values(make_not_null(&pi), gen,
                            make_not_null(&value_dist));
    interface_manager.request_gh_data(time_id);
    CHECK(interface_manager.number_of_pending_requests() == 1);
    CHECK(interface_manager.number_of_data_points() == 3);
    printf("first\n");
    auto retrieved_data = interface_manager.try_retrieve_first_ready_gh_data();
    printf("second\n");
    CHECK_FALSE(retrieved_data);
    printf("third\n");
    CHECK(interface_manager.number_of_pending_requests() == 1);
    printf("fourth\n");
    CHECK(interface_manager.number_of_data_points() == 3);
    interface_manager.insert_gh_data(time_id, spacetime_metric, phi, pi);

    CHECK(interface_manager.number_of_pending_requests() == 1);
    CHECK(interface_manager.number_of_data_points() == 4);
    retrieved_data = interface_manager.try_retrieve_first_ready_gh_data();
    CHECK(retrieved_data);
    CHECK(get<0>(*retrieved_data) == time_id);
    CHECK(get<1>(*retrieved_data) == spacetime_metric);
    CHECK(get<2>(*retrieved_data) == phi);
    CHECK(get<3>(*retrieved_data) == pi);
    CHECK(interface_manager.number_of_pending_requests() == 0);
    CHECK(interface_manager.number_of_data_points() == 3);
  }

  {
    INFO("Request from serialized and deserialized");
    // check that the state is preserved during serialization
    auto serialized_and_deserialized_interface_manager =
        serialize_and_deserialize(interface_manager);
    serialized_and_deserialized_interface_manager.request_gh_data(
        get<0>(expected_gh_data[3]));
    CHECK(serialized_and_deserialized_interface_manager
              .number_of_pending_requests() == 1);
    CHECK(
        serialized_and_deserialized_interface_manager.number_of_data_points() ==
        3);
    const auto retrieved_data = serialized_and_deserialized_interface_manager
                                    .try_retrieve_first_ready_gh_data();
    CHECK(retrieved_data);
    CHECK(get<0>(*retrieved_data) == get<0>(expected_gh_data[3]));
    CHECK(get<1>(*retrieved_data) == get<1>(expected_gh_data[3]));
    CHECK(get<2>(*retrieved_data) == get<2>(expected_gh_data[3]));
    CHECK(get<3>(*retrieved_data) == get<3>(expected_gh_data[3]));
    CHECK(serialized_and_deserialized_interface_manager
              .number_of_pending_requests() == 0);
    CHECK(
        serialized_and_deserialized_interface_manager.number_of_data_points() ==
        2);
  }

  {
    INFO("Request from cloned unique_ptr");
    // check that the state is preserved through cloning
    auto cloned_interface_manager = interface_manager.get_clone();
    cloned_interface_manager->request_gh_data(get<0>(expected_gh_data[3]));
    const auto retrieved_data =
        cloned_interface_manager->try_retrieve_first_ready_gh_data();
    CHECK(retrieved_data);
    CHECK(get<0>(*retrieved_data) == get<0>(expected_gh_data[3]));
    CHECK(get<1>(*retrieved_data) == get<1>(expected_gh_data[3]));
    CHECK(get<2>(*retrieved_data) == get<2>(expected_gh_data[3]));
    CHECK(get<3>(*retrieved_data) == get<3>(expected_gh_data[3]));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.WorldtubeInterfaceManager",
                  "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  test_gh_lockstep_interface_manager(make_not_null(&gen));
}
}  // namespace Cce
