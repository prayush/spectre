// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "DataStructures/Index.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Direction.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "tests/Unit/TestingFramework.hpp"

SPECTRE_TEST_CASE("Unit.Domain.LogicalCoordinates", "[Domain][Unit]") {
  using Affine2d = CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                                  CoordinateMaps::Affine>;
  using Affine3d = CoordinateMaps::ProductOf3Maps<
      CoordinateMaps::Affine, CoordinateMaps::Affine, CoordinateMaps::Affine>;

  const Index<1> extents_1d(Index<1>(3));
  const Index<2> extents_2d(Index<2>(2, 3));

  /// [logical_coordinates_example]
  const Index<3> extents_3d(Index<3>(5, 3, 2));

  const CoordinateMaps::Affine x_map{-1.0, 1.0, -3.0, 7.0};
  const CoordinateMaps::Affine y_map{-1.0, 1.0, -13.0, 47.0};
  const CoordinateMaps::Affine z_map{-1.0, 1.0, -32.0, 74.0};

  const auto map_3d = make_coordinate_map<Frame::Logical, Frame::Grid>(
      Affine3d{x_map, y_map, z_map});

  const auto x_3d = map_3d(logical_coordinates(extents_3d));
  /// [logical_coordinates_example]

  const auto map_1d = make_coordinate_map<Frame::Logical, Frame::Grid>(
      CoordinateMaps::Affine{x_map});
  const auto map_2d =
      make_coordinate_map<Frame::Logical, Frame::Grid>(Affine2d{x_map, y_map});
  const auto x_1d = map_1d(logical_coordinates(extents_1d));
  const auto x_2d = map_2d(logical_coordinates(extents_2d));

  CHECK(x_1d[0][0] == -3.0);
  CHECK(x_1d[0][1] == 2.0);
  CHECK(x_1d[0][2] == 7.0);

  CHECK(x_2d[0][0] == -3.0);
  CHECK(x_2d[0][1] == 7.0);

  CHECK(x_3d[0][0] == -3.0);
  CHECK(x_3d[0][2] == 2.0);
  CHECK(x_3d[0][4] == 7.0);

  CHECK(x_2d[1][0] == -13.0);
  CHECK(x_2d[1][2] == 17.0);
  CHECK(x_2d[1][4] == 47.0);

  CHECK(x_3d[1][0] == -13.0);
  CHECK(x_3d[1][5] == 17.0);
  CHECK(x_3d[1][10] == 47.0);

  CHECK(x_3d[2][0] == -32.0);
  CHECK(x_3d[2][15] == 74.0);
}

SPECTRE_TEST_CASE("Unit.Domain.InterfaceLogicalCoordinates", "[Domain][Unit]") {
  using Affine2d = CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                                  CoordinateMaps::Affine>;
  using Affine3d = CoordinateMaps::ProductOf3Maps<
      CoordinateMaps::Affine, CoordinateMaps::Affine, CoordinateMaps::Affine>;

  const CoordinateMaps::Affine x_map{-1.0, 1.0, -3.0, 7.0};
  const CoordinateMaps::Affine y_map{-1.0, 1.0, -13.0, 47.0};
  const CoordinateMaps::Affine z_map{-1.0, 1.0, -32.0, 74.0};

  const auto map_1d = make_coordinate_map<Frame::Logical, Frame::Grid>(
      CoordinateMaps::Affine{x_map});
  const auto map_2d =
      make_coordinate_map<Frame::Logical, Frame::Grid>(Affine2d{x_map, y_map});
  const auto map_3d = make_coordinate_map<Frame::Logical, Frame::Grid>(
      Affine3d{x_map, y_map, z_map});

  const Index<0> extents_0d;

  const Index<0> extents_1d_xbdry(extents_0d);

  const auto x_1d_lb = map_1d(interface_logical_coordinates(
      extents_1d_xbdry, Direction<1>::lower_xi()));

  CHECK(x_1d_lb[0][0] == -3.0);

  const auto x_1d_ub = map_1d(interface_logical_coordinates(
      extents_1d_xbdry, Direction<1>::upper_xi()));

  CHECK(x_1d_ub[0][0] == 7.0);

  const Index<1> extents_2d_xbdry(Index<1>(3));

  const auto x_2d_lb_xi = map_2d(interface_logical_coordinates(
      extents_2d_xbdry, Direction<2>::lower_xi()));

  CHECK(x_2d_lb_xi[0][0] == -3.0);
  CHECK(x_2d_lb_xi[0][1] == -3.0);
  CHECK(x_2d_lb_xi[0][2] == -3.0);

  CHECK(x_2d_lb_xi[1][0] == -13.0);
  CHECK(x_2d_lb_xi[1][1] == 17.0);
  CHECK(x_2d_lb_xi[1][2] == 47.0);

  const auto x_2d_ub_xi = map_2d(interface_logical_coordinates(
      extents_2d_xbdry, Direction<2>::upper_xi()));

  CHECK(x_2d_ub_xi[0][0] == 7.0);
  CHECK(x_2d_ub_xi[0][1] == 7.0);
  CHECK(x_2d_ub_xi[0][2] == 7.0);

  CHECK(x_2d_ub_xi[1][0] == -13.0);
  CHECK(x_2d_ub_xi[1][1] == 17.0);
  CHECK(x_2d_ub_xi[1][2] == 47.0);

  const Index<1> extents_2d_ybdry(Index<1>(2));

  const auto x_2d_lb_eta = map_2d(interface_logical_coordinates(
      extents_2d_ybdry, Direction<2>::lower_eta()));

  CHECK(x_2d_lb_eta[0][0] == -3.0);
  CHECK(x_2d_lb_eta[0][1] == 7.0);

  CHECK(x_2d_lb_eta[1][0] == -13.0);
  CHECK(x_2d_lb_eta[1][1] == -13.0);

  const auto x_2d_ub_eta = map_2d(interface_logical_coordinates(
      extents_2d_ybdry, Direction<2>::upper_eta()));

  CHECK(x_2d_ub_eta[0][0] == -3.0);
  CHECK(x_2d_ub_eta[0][1] == 7.0);

  CHECK(x_2d_ub_eta[1][0] == 47.0);
  CHECK(x_2d_ub_eta[1][1] == 47.0);

  const Index<2> extents_3d_xbdry(Index<2>(3, 2));

  const auto x_3d_lb_xi = map_3d(interface_logical_coordinates(
      extents_3d_xbdry, Direction<3>::lower_xi()));

  CHECK(x_3d_lb_xi[0][0] == -3.0);
  CHECK(x_3d_lb_xi[0][1] == -3.0);
  CHECK(x_3d_lb_xi[0][2] == -3.0);
  CHECK(x_3d_lb_xi[0][3] == -3.0);
  CHECK(x_3d_lb_xi[0][4] == -3.0);
  CHECK(x_3d_lb_xi[0][5] == -3.0);

  CHECK(x_3d_lb_xi[1][0] == -13.0);
  CHECK(x_3d_lb_xi[1][1] == 17.0);
  CHECK(x_3d_lb_xi[1][2] == 47.0);
  CHECK(x_3d_lb_xi[1][3] == -13.0);
  CHECK(x_3d_lb_xi[1][4] == 17.0);
  CHECK(x_3d_lb_xi[1][5] == 47.0);

  CHECK(x_3d_lb_xi[2][0] == -32.0);
  CHECK(x_3d_lb_xi[2][1] == -32.0);
  CHECK(x_3d_lb_xi[2][2] == -32.0);
  CHECK(x_3d_lb_xi[2][3] == 74.0);
  CHECK(x_3d_lb_xi[2][4] == 74.0);
  CHECK(x_3d_lb_xi[2][5] == 74.0);

  const auto x_3d_ub_xi = map_3d(interface_logical_coordinates(
      extents_3d_xbdry, Direction<3>::upper_xi()));

  CHECK(x_3d_ub_xi[0][0] == 7.0);
  CHECK(x_3d_ub_xi[0][1] == 7.0);
  CHECK(x_3d_ub_xi[0][2] == 7.0);
  CHECK(x_3d_ub_xi[0][3] == 7.0);
  CHECK(x_3d_ub_xi[0][4] == 7.0);
  CHECK(x_3d_ub_xi[0][5] == 7.0);

  CHECK(x_3d_ub_xi[1][0] == -13.0);
  CHECK(x_3d_ub_xi[1][1] == 17.0);
  CHECK(x_3d_ub_xi[1][2] == 47.0);
  CHECK(x_3d_ub_xi[1][3] == -13.0);
  CHECK(x_3d_ub_xi[1][4] == 17.0);
  CHECK(x_3d_ub_xi[1][5] == 47.0);

  CHECK(x_3d_ub_xi[2][0] == -32.0);
  CHECK(x_3d_ub_xi[2][1] == -32.0);
  CHECK(x_3d_ub_xi[2][2] == -32.0);
  CHECK(x_3d_ub_xi[2][3] == 74.0);
  CHECK(x_3d_ub_xi[2][4] == 74.0);
  CHECK(x_3d_ub_xi[2][5] == 74.0);

  const Index<2> extents_3d_ybdry(Index<2>(5, 2));

  const auto x_3d_lb_eta = map_3d(interface_logical_coordinates(
      extents_3d_ybdry, Direction<3>::lower_eta()));

  CHECK(x_3d_lb_eta[0][0] == -3.0);
  CHECK(x_3d_lb_eta[0][2] == 2.0);
  CHECK(x_3d_lb_eta[0][4] == 7.0);
  CHECK(x_3d_lb_eta[0][5] == -3.0);
  CHECK(x_3d_lb_eta[0][7] == 2.0);
  CHECK(x_3d_lb_eta[0][9] == 7.0);

  CHECK(x_3d_lb_eta[1][0] == -13.0);
  CHECK(x_3d_lb_eta[1][2] == -13.0);
  CHECK(x_3d_lb_eta[1][4] == -13.0);
  CHECK(x_3d_lb_eta[1][5] == -13.0);
  CHECK(x_3d_lb_eta[1][7] == -13.0);
  CHECK(x_3d_lb_eta[1][9] == -13.0);

  CHECK(x_3d_lb_eta[2][0] == -32.0);
  CHECK(x_3d_lb_eta[2][2] == -32.0);
  CHECK(x_3d_lb_eta[2][4] == -32.0);
  CHECK(x_3d_lb_eta[2][5] == 74.0);
  CHECK(x_3d_lb_eta[2][7] == 74.0);
  CHECK(x_3d_lb_eta[2][9] == 74.0);

  const auto x_3d_ub_eta = map_3d(interface_logical_coordinates(
      extents_3d_ybdry, Direction<3>::upper_eta()));

  CHECK(x_3d_ub_eta[0][0] == -3.0);
  CHECK(x_3d_ub_eta[0][2] == 2.0);
  CHECK(x_3d_ub_eta[0][4] == 7.0);
  CHECK(x_3d_ub_eta[0][5] == -3.0);
  CHECK(x_3d_ub_eta[0][7] == 2.0);
  CHECK(x_3d_ub_eta[0][9] == 7.0);

  CHECK(x_3d_ub_eta[1][0] == 47.0);
  CHECK(x_3d_ub_eta[1][2] == 47.0);
  CHECK(x_3d_ub_eta[1][4] == 47.0);
  CHECK(x_3d_ub_eta[1][5] == 47.0);
  CHECK(x_3d_ub_eta[1][7] == 47.0);
  CHECK(x_3d_ub_eta[1][9] == 47.0);

  CHECK(x_3d_ub_eta[2][0] == -32.0);
  CHECK(x_3d_ub_eta[2][2] == -32.0);
  CHECK(x_3d_ub_eta[2][4] == -32.0);
  CHECK(x_3d_ub_eta[2][5] == 74.0);
  CHECK(x_3d_ub_eta[2][7] == 74.0);
  CHECK(x_3d_ub_eta[2][9] == 74.0);

  /// [interface_logical_coordinates_example]
  const Index<2> extents_3d_zbdry(Index<2>(5, 3));

  const auto x_3d_lb_zeta = map_3d(interface_logical_coordinates(
      extents_3d_zbdry, Direction<3>::lower_zeta()));
  /// [interface_logical_coordinates_example]

  CHECK(x_3d_lb_zeta[0][0] == -3.0);
  CHECK(x_3d_lb_zeta[0][2] == 2.0);
  CHECK(x_3d_lb_zeta[0][4] == 7.0);
  CHECK(x_3d_lb_zeta[0][5] == -3.0);
  CHECK(x_3d_lb_zeta[0][7] == 2.0);
  CHECK(x_3d_lb_zeta[0][9] == 7.0);
  CHECK(x_3d_lb_zeta[0][10] == -3.0);
  CHECK(x_3d_lb_zeta[0][12] == 2.0);
  CHECK(x_3d_lb_zeta[0][14] == 7.0);

  CHECK(x_3d_lb_zeta[1][0] == -13.0);
  CHECK(x_3d_lb_zeta[1][2] == -13.0);
  CHECK(x_3d_lb_zeta[1][4] == -13.0);
  CHECK(x_3d_lb_zeta[1][5] == 17.0);
  CHECK(x_3d_lb_zeta[1][7] == 17.0);
  CHECK(x_3d_lb_zeta[1][9] == 17.0);
  CHECK(x_3d_lb_zeta[1][10] == 47.0);
  CHECK(x_3d_lb_zeta[1][12] == 47.0);
  CHECK(x_3d_lb_zeta[1][14] == 47.0);

  CHECK(x_3d_lb_zeta[2][0] == -32.0);
  CHECK(x_3d_lb_zeta[2][2] == -32.0);
  CHECK(x_3d_lb_zeta[2][4] == -32.0);
  CHECK(x_3d_lb_zeta[2][5] == -32.0);
  CHECK(x_3d_lb_zeta[2][7] == -32.0);
  CHECK(x_3d_lb_zeta[2][9] == -32.0);
  CHECK(x_3d_lb_zeta[2][10] == -32.0);
  CHECK(x_3d_lb_zeta[2][12] == -32.0);
  CHECK(x_3d_lb_zeta[2][14] == -32.0);

  const auto x_3d_ub_zeta = map_3d(interface_logical_coordinates(
      extents_3d_zbdry, Direction<3>::upper_zeta()));

  CHECK(x_3d_ub_zeta[0][0] == -3.0);
  CHECK(x_3d_ub_zeta[0][2] == 2.0);
  CHECK(x_3d_ub_zeta[0][4] == 7.0);
  CHECK(x_3d_ub_zeta[0][5] == -3.0);
  CHECK(x_3d_ub_zeta[0][7] == 2.0);
  CHECK(x_3d_ub_zeta[0][9] == 7.0);
  CHECK(x_3d_ub_zeta[0][10] == -3.0);
  CHECK(x_3d_ub_zeta[0][12] == 2.0);
  CHECK(x_3d_ub_zeta[0][14] == 7.0);

  CHECK(x_3d_ub_zeta[1][0] == -13.0);
  CHECK(x_3d_ub_zeta[1][2] == -13.0);
  CHECK(x_3d_ub_zeta[1][4] == -13.0);
  CHECK(x_3d_ub_zeta[1][5] == 17.0);
  CHECK(x_3d_ub_zeta[1][7] == 17.0);
  CHECK(x_3d_ub_zeta[1][9] == 17.0);
  CHECK(x_3d_ub_zeta[1][10] == 47.0);
  CHECK(x_3d_ub_zeta[1][12] == 47.0);
  CHECK(x_3d_ub_zeta[1][14] == 47.0);

  CHECK(x_3d_ub_zeta[2][0] == 74.0);
  CHECK(x_3d_ub_zeta[2][2] == 74.0);
  CHECK(x_3d_ub_zeta[2][4] == 74.0);
  CHECK(x_3d_ub_zeta[2][5] == 74.0);
  CHECK(x_3d_ub_zeta[2][7] == 74.0);
  CHECK(x_3d_ub_zeta[2][9] == 74.0);
  CHECK(x_3d_ub_zeta[2][10] == 74.0);
  CHECK(x_3d_ub_zeta[2][12] == 74.0);
  CHECK(x_3d_ub_zeta[2][14] == 74.0);
}
