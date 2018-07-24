// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/DomainCreators/Brick.hpp"

#include <array>
#include <unordered_set>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BlockNeighbor.hpp"
#include "Domain/CoordinateMaps/AffineMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Domain.hpp"
#include "Options/Options.hpp"
#include "Utilities/MakeVector.hpp"

namespace DomainCreators {

Brick::Brick(
    typename LowerBound::type lower_xyz, typename UpperBound::type upper_xyz,
    typename IsPeriodicIn::type is_periodic_in_xyz,
    typename InitialRefinement::type initial_refinement_level_xyz,
    typename InitialGridPoints::type initial_number_of_grid_points_in_xyz,
    const OptionContext& /*context*/) noexcept
    // clang-tidy: trivially copyable
    : lower_xyz_(std::move(lower_xyz)),                        // NOLINT
      upper_xyz_(std::move(upper_xyz)),                        // NOLINT
      is_periodic_in_xyz_(std::move(is_periodic_in_xyz)),      // NOLINT
      initial_refinement_level_xyz_(                           // NOLINT
          std::move(initial_refinement_level_xyz)),            // NOLINT
      initial_number_of_grid_points_in_xyz_(                   // NOLINT
          std::move(initial_number_of_grid_points_in_xyz)) {}  // NOLINT

Domain<3, Frame::Inertial> Brick::create_domain() const noexcept {
  using AffineMap = CoordinateMaps::AffineMap;
  using AffineMap3D =
      CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
  std::vector<PairOfFaces> identifications{};
  if (is_periodic_in_xyz_[0]) {
    identifications.push_back({{0, 4, 2, 6}, {1, 5, 3, 7}});
  }
  if (is_periodic_in_xyz_[1]) {
    identifications.push_back({{0, 1, 4, 5}, {2, 3, 6, 7}});
  }
  if (is_periodic_in_xyz_[2]) {
    identifications.push_back({{0, 1, 2, 3}, {4, 5, 6, 7}});
  }

  return Domain<3, Frame::Inertial>{
      make_vector<std::unique_ptr<
          CoordinateMapBase<Frame::Logical, Frame::Inertial, 3>>>(
          std::make_unique<
              CoordinateMap<Frame::Logical, Frame::Inertial, AffineMap3D>>(
              AffineMap3D{AffineMap{-1., 1., lower_xyz_[0], upper_xyz_[0]},
                          AffineMap{-1., 1., lower_xyz_[1], upper_xyz_[1]},
                          AffineMap{-1., 1., lower_xyz_[2], upper_xyz_[2]}})),
      std::vector<std::array<size_t, 8>>{{{0, 1, 2, 3, 4, 5, 6, 7}}},
      identifications};
}

std::array<size_t, 3> Brick::initial_extents(const size_t block_index) const
    noexcept {
  ASSERT(0 == block_index, "index = " << block_index);
  return initial_number_of_grid_points_in_xyz_;
}

std::array<size_t, 3> Brick::initial_refinement_levels(
    const size_t block_index) const noexcept {
  ASSERT(0 == block_index, "index = " << block_index);
  return initial_refinement_level_xyz_;
}
}  // namespace DomainCreators
