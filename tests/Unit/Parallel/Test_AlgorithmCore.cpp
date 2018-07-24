// Distributed under the MIT License.
// See LICENSE.txt for details.

// Need CATCH_CONFIG_RUNNER to avoid linking errors with Catch2
#define CATCH_CONFIG_RUNNER

#include <catch.hpp>
#include <cstddef>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "AlgorithmSingleton.hpp"
#include "ErrorHandling/Error.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmRegistration.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Unit/TestingFramework.hpp"

struct TestMetavariables;

template <class Metavariables>
struct NoOpsComponent;

struct TestAlgorithmArrayInstance {
  explicit TestAlgorithmArrayInstance(int ii) : i(ii) {}
  TestAlgorithmArrayInstance() = default;
  int i = 0;
  // clang-tidy: no non-const references
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | i;
  }
};

bool operator==(const TestAlgorithmArrayInstance& lhs,
                const TestAlgorithmArrayInstance& rhs) noexcept {
  return lhs.i == rhs.i;
}

TestAlgorithmArrayInstance& operator++(
    TestAlgorithmArrayInstance& instance) noexcept {
  instance.i++;
  return instance;
}

namespace std {
template <>
struct hash<TestAlgorithmArrayInstance> {
  size_t operator()(const TestAlgorithmArrayInstance& t) const {
    return hash<int>{}(t.i);
  }
};
}  // namespace std

/// \cond
struct ElementId {};
/// \endcond

struct CountActionsCalled : db::DataBoxTag {
  static constexpr db::DataBoxString label = "CountActionsCalled";
  using type = int;
};

struct Int0 : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Int0";
  using type = int;
};

struct Int1 : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Int1";
  using type = int;
};

struct TemporalId : db::DataBoxTag {
  static constexpr db::DataBoxString label = "TemporalId";
  using type = TestAlgorithmArrayInstance;
};

//////////////////////////////////////////////////////////////////////
// Test actions that do not add or remove from the DataBox
//////////////////////////////////////////////////////////////////////

namespace no_op_test {
struct increment_count_actions_called {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    static_assert(
        cpp17::is_same_v<ParallelComponent, NoOpsComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    db::mutate<CountActionsCalled>(
        box, [](int& count_actions_called) { count_actions_called++; });
    static int a = 0;
    return std::tuple<db::DataBox<DbTags>&&, bool>(std::move(box), ++a >= 5);
  }
};

struct no_op {
  /// [apply_iterative]
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept
  /// [apply_iterative]
  {
    static_assert(
        cpp17::is_same_v<ParallelComponent, NoOpsComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    return std::forward_as_tuple(std::move(box));
  }
};

struct initialize {
  template <typename... InboxTags, typename Metavariables, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    static_assert(
        cpp17::is_same_v<ParallelComponent, NoOpsComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    return std::make_tuple(
        db::create<tmpl::list<CountActionsCalled, Int0, Int1>>(0, 1, 100));
  }
};

struct finalize {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl2::flat_any_v<
                cpp17::is_same_v<CountActionsCalled, DbTags>...>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    static_assert(
        cpp17::is_same_v<ParallelComponent, NoOpsComponent<TestMetavariables>>,
        "The ParallelComponent is not deduced to be the right type");
    SPECTRE_PARALLEL_REQUIRE(db::get<CountActionsCalled>(box) == 5);
    SPECTRE_PARALLEL_REQUIRE(db::get<Int0>(box) == 1);
    SPECTRE_PARALLEL_REQUIRE(db::get<Int1>(box) == 100);
  }
};
}  // namespace no_op_test

template <class Metavariables>
struct NoOpsComponent {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using action_list =
      tmpl::list<no_op_test::increment_count_actions_called, no_op_test::no_op>;
  using initial_databox = db::DataBox<
      db::get_databox_list<tmpl::list<CountActionsCalled, Int0, Int1>>>;
  using const_global_cache_tag_list = tmpl::list<>;
  using options = tmpl::list<>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<NoOpsComponent>(local_cache)
        .template simple_action<no_op_test::initialize>();
  }

  static void execute_next_global_actions(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    if (next_phase == Metavariables::Phase::NoOpsStart) {
      /// [perform_algorithm]
      Parallel::get_parallel_component<NoOpsComponent>(local_cache)
          .perform_algorithm();
      /// [perform_algorithm]
    } else if (next_phase == Metavariables::Phase::NoOpsFinish) {
      Parallel::get_parallel_component<NoOpsComponent>(local_cache)
          .template simple_action<no_op_test::finalize>();
    }
  }
};

//////////////////////////////////////////////////////////////////////
// Adding and remove elements from DataBox tests
//////////////////////////////////////////////////////////////////////

namespace add_remove_test {
struct add_int_value_10 {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<CountActionsCalled>(
        box, [](int& count_actions_called) { count_actions_called++; });
    static int a = 0;
    return std::make_tuple(
        db::create_from<tmpl::list<>, tmpl::list<Int0>>(std::move(box), 10),
        ++a >= 5);
  }
};

struct increment_int0 {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<CountActionsCalled>(
        box, [](int& count_actions_called) { count_actions_called++; });
    db::mutate<Int0>(box, [](int& int0) { int0++; });
    return std::forward_as_tuple(std::move(box));
  }
};

struct remove_int0 {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    SPECTRE_PARALLEL_REQUIRE(db::get<Int0>(box) == 11);
    db::mutate<CountActionsCalled>(
        box,
        [](int& count_actions_called, const int& int0) {
          SPECTRE_PARALLEL_REQUIRE(int0 == 11);
          count_actions_called++;
        },
        db::get<Int0>(box));
    return std::make_tuple(db::create_from<tmpl::list<Int0>>(std::move(box)));
  }
};

struct test_args {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/, const double& v0,
                    std::vector<double>&& v1) noexcept {
    SPECTRE_PARALLEL_REQUIRE(v0 == 4.82937);
    SPECTRE_PARALLEL_REQUIRE(v1 == (std::vector<double>{3.2, -8.4, 7.5}));
  }
};

struct initialize {
  template <typename... InboxTags, typename Metavariables, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(
        db::create<tmpl::list<CountActionsCalled, TemporalId>>(
            0, TestAlgorithmArrayInstance{0}));
  }
};

struct finalize {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl2::flat_any_v<
                cpp17::is_same_v<CountActionsCalled, DbTags>...>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    SPECTRE_PARALLEL_REQUIRE(db::get<CountActionsCalled>(box) == 13);
  }
};
}  // namespace add_remove_test

template <class Metavariables>
struct MutateComponent {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using array_index = ElementId;  // Just to test nothing breaks
  using action_list =
      tmpl::list<add_remove_test::add_int_value_10,
                 add_remove_test::increment_int0, add_remove_test::remove_int0>;
  using initial_databox = db::DataBox<
      db::get_databox_list<tmpl::list<CountActionsCalled, TemporalId>>>;
  using const_global_cache_tag_list = tmpl::list<>;
  using options = tmpl::list<>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& cache = *(global_cache.ckLocalBranch());
    /// [simple_action_call]
    Parallel::get_parallel_component<MutateComponent>(cache)
        .template simple_action<add_remove_test::initialize>();
    /// [simple_action_call]
  }

  static void execute_next_global_actions(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    if (next_phase == Metavariables::Phase::MutateStart) {
      Parallel::get_parallel_component<MutateComponent>(local_cache)
          .perform_algorithm();
    } else if (next_phase == Metavariables::Phase::MutateFinish) {
      Parallel::get_parallel_component<MutateComponent>(local_cache)
          .template simple_action<add_remove_test::test_args>(
              std::make_tuple(4.82937, std::vector<double>{3.2, -8.4, 7.5}));
      Parallel::get_parallel_component<MutateComponent>(local_cache)
          .template simple_action<add_remove_test::finalize>();
    }
  }
};

//////////////////////////////////////////////////////////////////////
// Test receiving data
//////////////////////////////////////////////////////////////////////

namespace receive_data_test {
struct IntReceiveTag {
  using temporal_id = TestAlgorithmArrayInstance;
  using type = std::unordered_map<temporal_id, std::unordered_multiset<int>>;
};

struct add_int0_from_receive {
  using inbox_tags = tmpl::list<IntReceiveTag>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& inboxes,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<CountActionsCalled>(
        box, [](int& count_actions_called) { count_actions_called++; });
    static int a = 0;
    auto int0 = *(
        std::move(tuples::get<IntReceiveTag>(inboxes)[db::get<TemporalId>(box)])
            .begin());
    tuples::get<IntReceiveTag>(inboxes).erase(db::get<TemporalId>(box));
    return std::make_tuple(
        db::create_from<tmpl::list<>, tmpl::list<Int0>>(box, int0), ++a >= 5);
  }

  /// [is_ready_example]
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(
      const db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/) noexcept
      /// [is_ready_example]
  {
    const auto& inbox = tuples::get<IntReceiveTag>(inboxes);
    // The const_cast in this function is purely for testing purposes, this is
    // NOT an example of how to use this function.
    // clang-tidy: do not use const_cast
    db::mutate<Int1>(const_cast<db::DataBox<DbTags>&>(box),  // NOLINT
                     [](int& int1) { int1++; });
    return inbox.count(db::get<TemporalId>(box)) != 0;
  }
};

struct update_instance {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<TemporalId>(
        box, [](TestAlgorithmArrayInstance& temporal_id) { ++temporal_id; });
    return std::forward_as_tuple(std::move(box));
  }
};

struct initialize {
  template <typename... InboxTags, typename Metavariables, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(
        db::create<tmpl::list<CountActionsCalled, Int1, TemporalId>>(
            0, 0, TestAlgorithmArrayInstance{0}));
  }
};

struct finalize {
  using inbox_tags = tmpl::list<IntReceiveTag>;
  /// [requires_action]
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl2::flat_any_v<
                cpp17::is_same_v<CountActionsCalled, DbTags>...>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    const tuples::TaggedTuple<InboxTags...>& inboxes,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    /// [requires_action]
    SPECTRE_PARALLEL_REQUIRE(tuples::get<IntReceiveTag>(inboxes).empty());
    SPECTRE_PARALLEL_REQUIRE(db::get<TemporalId>(box) ==
                             TestAlgorithmArrayInstance{4});
    SPECTRE_PARALLEL_REQUIRE(db::get<CountActionsCalled>(box) == 13);
    SPECTRE_PARALLEL_REQUIRE(db::get<Int1>(box) == 10);
  }
};
}  // namespace receive_data_test

template <class Metavariables>
struct ReceiveComponent {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using array_index = ElementId;  // Just to test nothing breaks
  using action_list =
      tmpl::list<receive_data_test::add_int0_from_receive,
                 add_remove_test::increment_int0, add_remove_test::remove_int0,
                 receive_data_test::update_instance>;
  using initial_databox = db::DataBox<
      db::get_databox_list<tmpl::list<CountActionsCalled, Int1, TemporalId>>>;
  using const_global_cache_tag_list = tmpl::list<>;
  using options = tmpl::list<>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<ReceiveComponent>(local_cache)
        .template simple_action<receive_data_test::initialize>();
  }

  static void execute_next_global_actions(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    if (next_phase == Metavariables::Phase::ReceiveStart) {
      Parallel::get_parallel_component<ReceiveComponent>(local_cache)
          .perform_algorithm();
    } else if (next_phase == Metavariables::Phase::ReceiveSendData) {
      for (TestAlgorithmArrayInstance instance{0};
           not(instance == TestAlgorithmArrayInstance{5}); ++instance) {
        int dummy_int = 10;
        Parallel::get_parallel_component<ReceiveComponent>(local_cache)
            .template receive_data<receive_data_test::IntReceiveTag>(instance,
                                                                     dummy_int);
      }
    } else if (next_phase == Metavariables::Phase::ReceiveFinish) {
      Parallel::get_parallel_component<ReceiveComponent>(local_cache)
          .template simple_action<receive_data_test::finalize>();
    }
  }
};

//////////////////////////////////////////////////////////////////////
// Test out of order execution of Actions
//////////////////////////////////////////////////////////////////////

template <class Metavariables>
struct AnyOrderComponent;

namespace any_order {
struct iterate_increment_int0 {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>&
                    /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/, const ParallelComponent* const
                    /*meta*/) noexcept
      -> std::tuple<db::DataBox<tmpl::list<DbTags...>>&&, bool, size_t> {
    static_assert(cpp17::is_same_v<ParallelComponent,
                                   AnyOrderComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    db::mutate<CountActionsCalled>(
        box, [](int& count_actions_called) { count_actions_called++; });
    SPECTRE_PARALLEL_REQUIRE((db::get<CountActionsCalled>(box) - 1) / 2 ==
                             db::get<Int0>(box) - 10);

    const int max_int0_value = 25;
    if (db::get<Int0>(box) < max_int0_value) {
      return std::tuple<decltype(std::move(box)), bool, size_t>(
          std::move(box), false,
          tmpl::index_of<ActionList, ::add_remove_test::increment_int0>::value);
    }

    SPECTRE_PARALLEL_REQUIRE(db::get<Int0>(box) == max_int0_value);
    /// [out_of_order_action]
    return std::tuple<decltype(std::move(box)), bool, size_t>(
        std::move(box), true,
        tmpl::index_of<ActionList, iterate_increment_int0>::value + 1);
    /// [out_of_order_action]
  }
};

struct finalize {
  template <
      typename... DbTags, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<
          tmpl2::flat_any_v<cpp17::is_same_v<CountActionsCalled, DbTags>...> and
          tmpl2::flat_any_v<cpp17::is_same_v<Int0, DbTags>...>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>&
                    /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/, const ParallelComponent* const
                    /*meta*/) {
    static_assert(cpp17::is_same_v<ParallelComponent,
                                   AnyOrderComponent<TestMetavariables>>,
                  "The ParallelComponent is not deduced to be the right type");
    SPECTRE_PARALLEL_REQUIRE(db::get<TemporalId>(box) ==
                             db::item_type<TemporalId>{0});
    SPECTRE_PARALLEL_REQUIRE(db::get<CountActionsCalled>(box) == 31);
    SPECTRE_PARALLEL_REQUIRE(db::get<Int0>(box) == 25);
  }
};
}  // namespace any_order

template <class Metavariables>
struct AnyOrderComponent {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using array_index = ElementId;  // Just to test nothing breaks
  using action_list = tmpl::list<
      add_remove_test::add_int_value_10, add_remove_test::increment_int0,
      any_order::iterate_increment_int0, add_remove_test::remove_int0,
      receive_data_test::update_instance>;
  using initial_databox = db::DataBox<
      db::get_databox_list<tmpl::list<CountActionsCalled, TemporalId>>>;
  using const_global_cache_tag_list = tmpl::list<>;
  using options = tmpl::list<>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<AnyOrderComponent>(local_cache)
        .template simple_action<add_remove_test::initialize>();
  }

  static void execute_next_global_actions(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    if (next_phase == Metavariables::Phase::AnyOrderStart) {
      Parallel::get_parallel_component<AnyOrderComponent>(local_cache)
          .perform_algorithm();
    } else if (next_phase == Metavariables::Phase::AnyOrderFinish) {
      Parallel::get_parallel_component<AnyOrderComponent>(local_cache)
          .template simple_action<any_order::finalize>();
    }
  }
};

struct TestMetavariables {
  /// [component_list_example]
  using component_list = tmpl::list<NoOpsComponent<TestMetavariables>,
                                    MutateComponent<TestMetavariables>,
                                    ReceiveComponent<TestMetavariables>,
                                    AnyOrderComponent<TestMetavariables>>;
  /// [component_list_example]

  /// [help_string_example]
  static constexpr OptionString help =
      "An executable for testing the core functionality of the Algorithm. "
      "Actions that do not perform any operations (no-ops), invoking simple "
      "actions, mutating data in the DataBox, adding and removing items from "
      "the DataBox, receiving data from other parallel components, and "
      "out-of-order execution of Actions are all tested. All tests are run "
      "just by running the executable, no input file or command line arguments "
      "are required";
  /// [help_string_example]

  /// [determine_next_phase_example]
  enum class Phase {
    Initialization,
    NoOpsStart,
    NoOpsFinish,
    MutateStart,
    MutateFinish,
    ReceiveStart,
    ReceiveSendData,
    ReceiveFinish,
    AnyOrderStart,
    AnyOrderFinish,
    Exit
  };

  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
          TestMetavariables>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::NoOpsStart;
      case Phase::NoOpsStart:
        return Phase::NoOpsFinish;
      case Phase::NoOpsFinish:
        return Phase::MutateStart;
      case Phase::MutateStart:
        return Phase::MutateFinish;
      case Phase::MutateFinish:
        return Phase::ReceiveStart;
      case Phase::ReceiveStart:
        return Phase::ReceiveSendData;
      case Phase::ReceiveSendData:
        return Phase::ReceiveFinish;
      case Phase::ReceiveFinish:
        return Phase::AnyOrderStart;
      case Phase::AnyOrderStart:
        return Phase::AnyOrderFinish;
      case Phase::AnyOrderFinish:
        return Phase::Exit;
      case Phase::Exit:
        return Phase::Exit;
      default:
        ERROR("Unknown Phase...");
    }

    return Phase::Exit;
  }
  /// [determine_next_phase_example]
};

/// [charm_init_funcs_example]
static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
/// [charm_init_funcs_example]

/// [charm_main_example]
using charmxx_main_component = Parallel::Main<TestMetavariables>;
/// [charm_main_example]

#include "Parallel/CharmMain.cpp"
