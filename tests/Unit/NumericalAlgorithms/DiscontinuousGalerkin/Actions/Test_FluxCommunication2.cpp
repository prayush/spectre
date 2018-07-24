// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <memory>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication2.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestingFramework.hpp"

namespace {
constexpr size_t volume_dim = 2;

struct Var : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Var";
  using type = Scalar<DataVector>;
};

struct System {
  static constexpr const size_t volume_dim = ::volume_dim;
  using variables_tag = Tags::Variables<tmpl::list<Var>>;
  static constexpr const size_t number_of_independent_components =
      db::item_type<variables_tag>::number_of_independent_components;

  struct normal_dot_fluxes {
    using return_tags = tmpl::list<Tags::NormalDotFlux<Var>>;
    using argument_tags = tmpl::list<Var>;
    static void apply(
        const gsl::not_null<Scalar<DataVector>*> var_normal_dot_flux,
        const Scalar<DataVector>& var,
        const tnsr::i<DataVector, volume_dim, Frame::Inertial>&
        /*interface_unit_normal*/) noexcept {
      var_normal_dot_flux->get() = 0.25 * var.get();
    }
  };
};

using dt_variables_tag = Tags::dt<Tags::Variables<tmpl::list<Tags::dt<Var>>>>;

class NumericalFlux {
 public:
  using flux_tag =
      db::add_tag_prefix<Tags::NormalDotFlux, System::variables_tag>;
  using argument_tags = tmpl::list<flux_tag>;
  using package_tags = tmpl::list<Var>;
  using slice_tags = tmpl::list<Var>;

  void package_data(const gsl::not_null<Variables<package_tags>*> packaged_var,
                    const Scalar<DataVector>& var_flux,
                    const Scalar<DataVector>& var,
                    const tnsr::i<DataVector, 2, Frame::Inertial>&
                    /*interface_unit_normal*/) const noexcept {
    get<Var>(*packaged_var).get() = 0.5 * var.get() + var_flux.get();
  }

  void operator()(
      const gsl::not_null<Scalar<DataVector>*> normal_dot_numerical_flux,
      const Scalar<DataVector>& packaged_var_interior,
      const Scalar<DataVector>& packaged_var_exterior) const noexcept {
    normal_dot_numerical_flux->get() = 11.0 * packaged_var_interior.get() +
                                       1000. * packaged_var_exterior.get();
  }

  // clang-tidy: do not use references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

struct NumericalFluxTag {
  using type = NumericalFlux;
};

struct Metavariables;

using CharmIndexType = ElementIndex<volume_dim>;

using component = ActionTesting::MockArrayComponent<
    Metavariables, CharmIndexType, tmpl::list<NumericalFluxTag>,
    tmpl::list<dg::Actions::ComputeBoundaryFlux<Metavariables>>>;

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<component>;
  using normal_dot_numerical_flux = NumericalFluxTag;
};

using fluxes_tag = dg::Actions::ComputeBoundaryFlux<Metavariables>::FluxesTag;
using mortars_normal_dot_fluxes_tag =
    dg::Actions::DgActions_detail::mortars_tag<
        2, db::add_tag_prefix<Tags::NormalDotFlux,
                              typename System::variables_tag>>;
using mortars_packaged_data_tag = dg::Actions::DgActions_detail::mortars_tag<
    2,
    Tags::Variables<
        typename Metavariables::normal_dot_numerical_flux::type::package_tags>>;
}  // namespace

SPECTRE_TEST_CASE("Unit.DiscontinuousGalerkin.Actions.FluxCommunication2",
                  "[Unit][NumericalAlgorithms][Actions]") {
  ActionTesting::ActionRunner<Metavariables> runner{{}};

  const Slab slab(1., 3.);
  const TimeId time_id{8, slab.start(), 0};

  const Index<2> extents{{{3, 3}}};

  //      xi  Block   +- xi
  //      |   0   1   |
  // eta -+ +---+-+-+ eta
  //        |   |X| |
  //        |   +-+-+
  //        |   | | |
  //        +---+-+-+
  // We run the send action on the indicated element.
  const ElementId<2> self_id(1, {{{1, 0}, {1, 0}}});
  const ElementId<2> west_id(0);
  const ElementId<2> east_id(1, {{{1, 1}, {1, 0}}});
  const ElementId<2> south_id(1, {{{1, 0}, {1, 1}}});

  // OrientationMap from block 1 to block 0
  const OrientationMap<2> block_orientation(
      {{Direction<2>::upper_xi(), Direction<2>::upper_eta()}},
      {{Direction<2>::lower_eta(), Direction<2>::lower_xi()}});

  const CoordinateMaps::Affine xi_map{-1., 1., 3., 7.};
  const CoordinateMaps::Affine eta_map{-1., 1., -2., 4.};

  auto start_box = [&extents, &time_id, &self_id, &west_id, &east_id, &south_id,
                    &block_orientation, &xi_map, &eta_map]() {
    const Element<2> element(
        self_id, {{Direction<2>::lower_xi(), {{west_id}, block_orientation}},
                  {Direction<2>::upper_xi(), {{east_id}, {}}},
                  {Direction<2>::upper_eta(), {{south_id}, {}}}});

    auto map = ElementMap<2, Frame::Inertial>(
        self_id, make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
                     CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                                    CoordinateMaps::Affine>(
                         xi_map, eta_map)));

    Variables<tmpl::list<Var>> variables(extents.product());
    get<Var>(variables).get() = DataVector{1., 2., 3., 4., 5., 6., 7., 8., 9.};
    db::item_type<db::add_tag_prefix<Tags::dt, System::variables_tag>>
        dt_variables(extents.product(), 0.0);

    return db::create<
        db::AddTags<Tags::TimeId, Tags::Extents<2>, Tags::Element<2>,
                    Tags::ElementMap<2>, System::variables_tag,
                    db::add_tag_prefix<Tags::dt, System::variables_tag>>,
        db::AddComputeItemsTags<Tags::UnnormalizedFaceNormal<2>>>(
        time_id, extents, element, std::move(map), std::move(variables),
        std::move(dt_variables));
  }();

  auto sent_box =
      std::get<0>(runner.apply<component, dg::Actions::SendDataForFluxes>(
          start_box, CharmIndexType(self_id)));

  // Check local state
  auto local_mortar_fluxes = db::get<mortars_normal_dot_fluxes_tag>(sent_box);
  auto local_mortar_packaged_data =
      db::get<mortars_packaged_data_tag>(sent_box);
  CHECK(local_mortar_fluxes.size() == 3);
  CHECK(get<Tags::NormalDotFlux<Var>>(local_mortar_fluxes[std::make_pair(
                                          Direction<2>::lower_xi(), west_id)])
            .get() == (DataVector{0.25, 1., 1.75}));
  CHECK(get<Tags::NormalDotFlux<Var>>(local_mortar_fluxes[std::make_pair(
                                          Direction<2>::upper_xi(), east_id)])
            .get() == (DataVector{0.75, 1.5, 2.25}));
  CHECK(get<Tags::NormalDotFlux<Var>>(local_mortar_fluxes[std::make_pair(
                                          Direction<2>::upper_eta(), south_id)])
            .get() == (DataVector{1.75, 2., 2.25}));

  CHECK(local_mortar_packaged_data.size() == 3);
  CHECK(get<Var>(local_mortar_packaged_data[std::make_pair(
                     Direction<2>::lower_xi(), west_id)])
            .get() == (DataVector{0.75, 3., 5.25}));
  CHECK(get<Var>(local_mortar_packaged_data[std::make_pair(
                     Direction<2>::upper_xi(), east_id)])
            .get() == (DataVector{2.25, 4.5, 6.75}));
  CHECK(get<Var>(local_mortar_packaged_data[std::make_pair(
                     Direction<2>::upper_eta(), south_id)])
            .get() == (DataVector{5.25, 6., 6.75}));

  // Check sent data
  CHECK((runner.nonempty_inboxes<component, fluxes_tag>()) ==
        (std::unordered_set<CharmIndexType>{CharmIndexType(west_id),
                                            CharmIndexType(east_id),
                                            CharmIndexType(south_id)}));
  auto& inboxes = runner.inboxes<component>();
  const auto flux_inbox =
      [&inboxes, &time_id ](const ElementId<2>& id) noexcept {
    return tuples::get<fluxes_tag>(inboxes[CharmIndexType(id)])[time_id];
  };
  CHECK(flux_inbox(west_id).size() == 1);
  CHECK(get<Var>(flux_inbox(west_id).at({Direction<2>::lower_eta(), self_id}))
            .get() == (DataVector{5.25, 3., 0.75}));
  CHECK(flux_inbox(east_id).size() == 1);
  CHECK(get<Var>(flux_inbox(east_id).at({Direction<2>::lower_xi(), self_id}))
            .get() == (DataVector{2.25, 4.5, 6.75}));
  CHECK(flux_inbox(south_id).size() == 1);
  CHECK(get<Var>(flux_inbox(south_id).at({Direction<2>::lower_eta(), self_id}))
            .get() == (DataVector{5.25, 6., 6.75}));

  // Now check ComputeBoundaryFlux
  db::mutate<Tags::Element<2>>(sent_box, [](auto& element) {
    auto neighbors = element.neighbors();
    neighbors.erase(Direction<2>::lower_xi());
    element = Element<2>(element.id(), std::move(neighbors));
  });

  CHECK_FALSE((runner.is_ready<component,
                               dg::Actions::ComputeBoundaryFlux<Metavariables>>(
      sent_box, CharmIndexType(self_id))));

  // Send from south neighbor
  {
    const Element<2> element(south_id,
                             {{Direction<2>::lower_eta(), {{self_id}, {}}}});

    Variables<tmpl::list<Var>> variables(extents.product());
    get<Var>(variables).get() =
        DataVector{11., 12., 13., 14., 15., 16., 17., 18., 19.};

    auto map = ElementMap<2, Frame::Inertial>(
        south_id, make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
                      CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                                     CoordinateMaps::Affine>(
                          xi_map, eta_map)));

    auto box =
        db::create<db::AddTags<Tags::TimeId, Tags::Extents<2>, Tags::Element<2>,
                               System::variables_tag, Tags::ElementMap<2>>,
                   db::AddComputeItemsTags<Tags::UnnormalizedFaceNormal<2>>>(
            time_id, extents, element, std::move(variables), std::move(map));
    runner.apply<component, dg::Actions::SendDataForFluxes>(
        box, CharmIndexType(south_id));
  }
  CHECK_FALSE((runner.is_ready<component,
                               dg::Actions::ComputeBoundaryFlux<Metavariables>>(
      sent_box, CharmIndexType(self_id))));

  // Send from east neighbor
  {
    const Element<2> element(east_id,
                             {{Direction<2>::lower_xi(), {{self_id}, {}}}});

    Variables<tmpl::list<Var>> variables(extents.product());
    get<Var>(variables).get() =
        DataVector{21., 22., 23., 24., 25., 26., 27., 28., 29.};

    auto map = ElementMap<2, Frame::Inertial>(
        east_id, make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
                     CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                                    CoordinateMaps::Affine>(
                         xi_map, eta_map)));

    auto box =
        db::create<db::AddTags<Tags::TimeId, Tags::Extents<2>, Tags::Element<2>,
                               System::variables_tag, Tags::ElementMap<2>>,
                   db::AddComputeItemsTags<Tags::UnnormalizedFaceNormal<2>>>(
            time_id, extents, element, std::move(variables), std::move(map));
    runner.apply<component, dg::Actions::SendDataForFluxes>(
        box, CharmIndexType(east_id));
  }
  CHECK((runner.is_ready<component,
                         dg::Actions::ComputeBoundaryFlux<Metavariables>>(
      sent_box, CharmIndexType(self_id))));

  auto received_box = std::get<0>(
      runner.apply<component, dg::Actions::ComputeBoundaryFlux<Metavariables>>(
          sent_box, CharmIndexType(self_id)));

  CHECK(tuples::get<fluxes_tag>(
            runner.inboxes<component>().at(CharmIndexType(self_id)))
            .empty());

  const double xi_lift = -12. / (xi_map(std::array<double, 1>{{1.}}) -
                                 xi_map(std::array<double, 1>{{-1.}}))[0];
  const double eta_lift = -12. / (eta_map(std::array<double, 1>{{1.}}) -
                                  eta_map(std::array<double, 1>{{-1.}}))[0];

  const DataVector xi_boundaries{
      0.,
      0.,
      15774. /
          get(magnitude(db::get<Tags::UnnormalizedFaceNormal<2>>(start_box).at(
              Direction<2>::lower_xi())))[0],
      0.,
      0.,
      18048. /
          get(magnitude(db::get<Tags::UnnormalizedFaceNormal<2>>(start_box).at(
              Direction<2>::lower_xi())))[1],
      0.,
      0.,
      20322. /
          get(magnitude(db::get<Tags::UnnormalizedFaceNormal<2>>(start_box).at(
              Direction<2>::lower_xi())))[2]};
  const DataVector eta_boundaries{
      0.,
      0.,
      0.,
      0.,
      0.,
      0.,
      16612. / 3. /
          get(magnitude(db::get<Tags::UnnormalizedFaceNormal<2>>(start_box).at(
              Direction<2>::upper_eta())))[0],
      18128. / 3. /
          get(magnitude(db::get<Tags::UnnormalizedFaceNormal<2>>(start_box).at(
              Direction<2>::upper_eta())))[1],
      19644. / 3. /
          get(magnitude(db::get<Tags::UnnormalizedFaceNormal<2>>(start_box).at(
              Direction<2>::upper_eta())))[2]};

  CHECK_ITERABLE_APPROX(
      get<Tags::dt<Var>>(db::get<dt_variables_tag>(received_box)).get(),
      xi_lift * xi_boundaries + eta_lift * eta_boundaries);
}

SPECTRE_TEST_CASE(
    "Unit.DiscontinuousGalerkin.Actions.FluxCommunication2.NoNeighbors",
    "[Unit][NumericalAlgorithms][Actions]") {
  ActionTesting::ActionRunner<Metavariables> runner{{}};

  const Slab slab(1., 3.);
  const TimeId time_id{8, slab.start(), 0};

  const Index<2> extents{{{3, 3}}};

  const ElementId<2> self_id(1, {{{1, 0}, {1, 0}}});

  const Element<2> element(self_id, {});

  auto map = ElementMap<2, Frame::Inertial>(
      self_id, make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
                   CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                                  CoordinateMaps::Affine>(
                       {-1., 1., 3., 7.}, {-1., 1., -2., 4.})));

  Variables<tmpl::list<Var>> variables(extents.product());
  get<Var>(variables).get() = DataVector{1., 2., 3., 4., 5., 6., 7., 8., 9.};
  db::item_type<db::add_tag_prefix<Tags::dt, System::variables_tag>>
      dt_variables(extents.product(), 0.0);
  auto start_box = db::create<
      db::AddTags<Tags::TimeId, Tags::Extents<2>, Tags::Element<2>,
                  Tags::ElementMap<2>, System::variables_tag,
                  db::add_tag_prefix<Tags::dt, System::variables_tag>>,
      db::AddComputeItemsTags<Tags::UnnormalizedFaceNormal<2>>>(
      time_id, extents, element, std::move(map), std::move(variables),
      std::move(dt_variables));

  auto sent_box =
      std::get<0>(runner.apply<component, dg::Actions::SendDataForFluxes>(
          start_box, CharmIndexType(self_id)));

  CHECK(db::get<mortars_packaged_data_tag>(sent_box).empty());
  CHECK((runner.nonempty_inboxes<component, fluxes_tag>().empty()));

  CHECK((runner.is_ready<component,
                         dg::Actions::ComputeBoundaryFlux<Metavariables>>(
      sent_box, CharmIndexType(self_id))));

  auto received_box = std::get<0>(
      runner.apply<component, dg::Actions::ComputeBoundaryFlux<Metavariables>>(
          sent_box, CharmIndexType(self_id)));

  CHECK(get<Tags::dt<Var>>(db::get<dt_variables_tag>(received_box)).get() ==
        (DataVector{0., 0., 0., 0., 0., 0., 0., 0., 0.}));
}
