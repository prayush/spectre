// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeWorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Components/WorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/Evolution/Systems/Cce/Actions/WorldtubeBoundaryMocking.hpp"
#include "tests/Unit/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

#include "Utilities/TmplDebugging.hpp"

namespace Cce {

namespace {
struct h5_metavariables {
  using cce_boundary_communication_tags =
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>;
  using component_list =
      tmpl::list<mock_h5_worldtube_boundary<h5_metavariables>>;
  enum class Phase { Initialization, Evolve, Exit };
};

struct gh_metavariables {
  using cce_boundary_communication_tags =
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>;
  using component_list =
      tmpl::list<mock_gh_worldtube_boundary<gh_metavariables>>;
  enum class Phase { Initialization, Evolve, Exit };
};

template <typename Generator>
void test_h5_initialization(const gsl::not_null<Generator*> gen) noexcept {
  using component = mock_h5_worldtube_boundary<h5_metavariables>;
  // this probably needs to be adjusted for the const global tags and option
  // tags we need.
  const size_t l_max = 8;
  ActionTesting::MockRuntimeSystem<h5_metavariables> runner{
      tuples::tagged_tuple_from_typelist<
          Parallel::get_const_global_cache_tags<h5_metavariables>>{l_max}};

  const size_t buffer_size = 8;
  const std::string filename = "InitializeWorldtubeBoundaryTest_CceR0100.h5";

  // create the test file, because on initialization the manager will need to
  // get basic data out of the file
  UniformCustomDistribution<double> value_dist{0.1, 0.5};
  // first prepare the input for the modal version
  const double mass = value_dist(*gen);
  const std::array<double, 3> spin{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  const std::array<double, 3> center{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  gr::Solutions::KerrSchild solution{mass, spin, center};

  const double extraction_radius = 100.0;
  const double frequency = 0.1 * value_dist(*gen);
  const double amplitude = 0.1 * value_dist(*gen);
  const double target_time = 50.0 * value_dist(*gen);
  TestHelpers::write_test_file(solution, filename, target_time,
                               extraction_radius, frequency, amplitude, l_max);

  ActionTesting::set_phase(make_not_null(&runner),
                          h5_metavariables::Phase::Initialization);
  ActionTesting::emplace_component<component>(
      &runner, 0,
      InitializationTags::H5WorldtubeBoundaryDataManager::create_from_options(
          l_max, filename, buffer_size,
          std::make_unique<intrp::BarycentricRationalSpanInterpolator>(3u,
                                                                       4u)));

  // this should run the initialization
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  ActionTesting::set_phase(make_not_null(&runner),
                           h5_metavariables::Phase::Evolve);
  // check that the h5 data manager copied out of the databox has the correct
  // properties that we can examine without running the other actions
  const auto& data_manager =
      ActionTesting::get_databox_tag<component,
                                     Tags::H5WorldtubeBoundaryDataManager>(
          runner, 0);
  CHECK(data_manager.get_l_max() == l_max);
  const auto time_span = data_manager.get_time_span();
  CHECK(time_span.first == 0);
  CHECK(time_span.second == 0);

  // check that the Variables is in the expected state (here we just make sure
  // it has the right size - it shouldn't have been written to yet)
  const auto& variables = ActionTesting::get_databox_tag<
      component,
      ::Tags::Variables<
          typename h5_metavariables::cce_boundary_communication_tags>>(runner,
                                                                       0);

  CHECK(get(get<Tags::BoundaryValue<Tags::BondiBeta>>(variables)).size() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}

void test_gh_initialization() noexcept {
  using component = mock_gh_worldtube_boundary<gh_metavariables>;
  // this probably needs to be adjusted for the const global tags and option
  // tags we need.
  const size_t l_max = 8;
  const double extraction_radius = 100.0;
  ActionTesting::MockRuntimeSystem<gh_metavariables> runner{
      tuples::tagged_tuple_from_typelist<
          Parallel::get_const_global_cache_tags<gh_metavariables>>{
          l_max, extraction_radius}};

  runner.set_phase(gh_metavariables::Phase::Initialization);
  ActionTesting::emplace_component<component>(
      &runner, 0,
      InitializationTags::GHInterfaceManager::create_from_options(
          std::make_unique<GHLockstepInterfaceManager>()));

  // this should run the initialization
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  runner.set_phase(gh_metavariables::Phase::Evolve);
  // check that the h5 data manager copied out of the databox has the correct
  // properties that we can examine without running the other actions
  const auto& interface_manager =
      ActionTesting::get_databox_tag<component, Tags::GHInterfaceManager>(
          runner, 0);
  CHECK(cpp17::is_same_v<decltype(interface_manager),
                         const GHWorldtubeInterfaceManager&>);

  // check that the Variables is in the expected state (here we just make sure
  // it has the right size - it shouldn't have been written to yet)
  const auto& variables = ActionTesting::get_databox_tag<
      component,
      ::Tags::Variables<
          typename gh_metavariables::cce_boundary_communication_tags>>(runner,
                                                                       0);
  CHECK(get(get<Tags::BoundaryValue<Tags::BondiBeta>>(variables)).size() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Cce.Actions.InitializeWorldtubeBoundary",
    "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  test_h5_initialization(make_not_null(&gen));
  test_gh_initialization();
}
}  // namespace Cce
