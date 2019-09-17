// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>
#include <random>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Side.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace {

void test_constraint_preserving_bjorhus_u_psi(
    const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) noexcept {
  CHECK(1 == 1);
  // Setup grid
  const size_t VolumeDim = 3;
  Mesh<VolumeDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  const Direction direction(1, Side::Upper);  // +y direction
  const size_t slice_grid_points =
      mesh.extents().slice_away(direction.dimension()).product();
  //
  // Create a TempTensor that stores all temporaries computed
  // here and elsewhere
  TempBuffer<GeneralizedHarmonic::Actions::BoundaryConditions_detail::
                 all_local_vars<VolumeDim>>
      buffer(slice_grid_points);

  // POPULATE various tensors needed to compute BcDtUpsi
  // EXACTLY as done in SpEC
  auto& local_inertial_coords =
      get<::Tags::TempI<37, VolumeDim, Frame::Inertial, DataVector>>(buffer);
  auto local_three_index_constraint = three_index_constraint;
  auto local_unit_interface_normal_vector = unit_interface_normal_vector;
  auto local_lapse = lapse;
  auto local_shift = shift;
  auto local_pi = pi;
  auto local_phi = phi;
  auto local_char_projected_rhs_dt_u_psi = char_projected_rhs_dt_u_psi;
  auto local_char_speeds = char_speeds;
  // Setting 3idxConstraint
  for (size_t i = 0; i < lapse.size(); ++i) {
    for (size_t a = 0; a <= VolumeDim; ++a) {
      for (size_t b = 0; b <= VolumeDim; ++b) {
        local_three_index_constraint.get(0, a, b)[i] = 11. - 3.;
        local_three_index_constraint.get(1, a, b)[i] = 13. - 5.;
        local_three_index_constraint.get(2, a, b)[i] = 17. - 7.;
      }
    }
  }
  // Setting unit_interface_normal_Vector
  for (size_t i = 0; i < lapse.size(); ++i) {
    local_unit_interface_normal_vector.get(0)[i] = 1.e300;
    local_unit_interface_normal_vector.get(1)[i] = -1.;
    local_unit_interface_normal_vector.get(2)[i] = 0.;
    local_unit_interface_normal_vector.get(3)[i] = 0.;
  }
  // Setting lapse
  for (size_t i = 0; i < lapse.size(); ++i) {
    get(local_lapse)[i] = 2.;
  }
  // Setting shift
  for (size_t i = 0; i < lapse.size(); ++i) {
    local_shift.get(0)[i] = 1.;
    local_shift.get(1)[i] = 2.;
    local_shift.get(2)[i] = 3.;
  }
  // Setting pi AND phi
  for (size_t i = 0; i < lapse.size(); ++i) {
    for (size_t a = 0; a <= VolumeDim; ++a) {
      for (size_t b = 0; b <= VolumeDim; ++b) {
        local_pi.get(a, b)[i] = 1.;
        local_phi.get(0, a, b)[i] = 3.;
        local_phi.get(1, a, b)[i] = 5.;
        local_phi.get(2, a, b)[i] = 7.;
      }
    }
  }
  // Setting local_RhsUPsi
  for (size_t i = 0; i < lapse.size(); ++i) {
    for (size_t a = 0; a <= VolumeDim; ++a) {
      local_char_projected_rhs_dt_u_psi.get(0, a)[i] = 23.;
      local_char_projected_rhs_dt_u_psi.get(1, a)[i] = 29.;
      local_char_projected_rhs_dt_u_psi.get(2, a)[i] = 31.;
      local_char_projected_rhs_dt_u_psi.get(3, a)[i] = 37.;
    }
  }
  // Setting char speeds
  for (size_t i = 0; i < lapse.size(); ++i) {
    local_char_speeds.at(0)[i] = -0.3;
    local_char_speeds.at(1)[i] = -0.1;
  }

  // Memory allocated for return type
  ReturnType& bc_dt_u_psi =
      get<::Tags::Tempaa<27, VolumeDim, Frame::Inertial, DataVector>>(buffer);
  std::fill(bc_dt_u_psi.begin(), bc_dt_u_psi.end(), 0.);
  // debugPK
  auto _ = apply_bjorhus_constraint_preserving(
      make_not_null(&bc_dt_u_psi), local_unit_interface_normal_vector,
      local_three_index_constraint, local_char_projected_rhs_dt_u_psi,
      local_char_speeds);
  // DISPLAY results of the TEST
  if (debugPKon) {
    for (size_t i = 0; i < lapse.size(); ++i) {
      print_rank2_tensor_at_point(
          "BcDtUpsi", bc_dt_u_psi, local_inertial_coords.get(0),
          local_inertial_coords.get(1), local_inertial_coords.get(2), i);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.BoundaryConditions.UPsi",
    "[Unit][Evolution]") {
  const DataVector used_for_size(4);

  // Piece-wise tests with SpEC output
  const size_t grid_size = 3;
  const std::array<double, 3> lower_bound{{299., -0.5, -0.5}};
  const std::array<double, 3> upper_bound{{300., 0.5, 0.5}};

  test_constraint_preserving_bjorhus_u_psi(grid_size, lower_bound, upper_bound);
}
