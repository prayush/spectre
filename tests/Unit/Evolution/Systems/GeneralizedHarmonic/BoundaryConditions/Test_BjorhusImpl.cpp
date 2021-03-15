// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BjorhusImpl.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using frame = Frame::Inertial;
constexpr size_t VolumeDim = 3;

// Test boundary conditions on dt<VSpacetimeMetric>
void test_constraint_preserving_bjorhus_v_psi(
    const size_t grid_size_each_dimension) noexcept {
  // Setup grid
  Mesh<VolumeDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const Direction<VolumeDim> direction(1, Side::Upper);  // +y direction
  const size_t slice_grid_points =
      mesh.extents().slice_away(direction.dimension()).product();

  // Populate various tensors needed to compute BcDtVSpacetimeMetric
  tnsr::iaa<DataVector, VolumeDim, frame> local_three_index_constraint(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());
  tnsr::I<DataVector, VolumeDim, frame> local_unit_interface_normal_vector(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());
  tnsr::aa<DataVector, VolumeDim, frame> local_char_projected_rhs_dt_v_psi(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());
  std::array<DataVector, 4> local_char_speeds{
      DataVector(slice_grid_points,
                 std::numeric_limits<double>::signaling_NaN()),
      DataVector(slice_grid_points,
                 std::numeric_limits<double>::signaling_NaN()),
      DataVector(slice_grid_points,
                 std::numeric_limits<double>::signaling_NaN()),
      DataVector(slice_grid_points,
                 std::numeric_limits<double>::signaling_NaN())};
  {
    // Setting 3idxConstraint
    for (size_t i = 0; i < slice_grid_points; ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = 0; b <= VolumeDim; ++b) {
          local_three_index_constraint.get(0, a, b)[i] = 11. - 3.;
          local_three_index_constraint.get(1, a, b)[i] = 13. - 5.;
          local_three_index_constraint.get(2, a, b)[i] = 17. - 7.;
        }
      }
    }
    // Setting unit_interface_normal_Vector
    for (size_t i = 0; i < slice_grid_points; ++i) {
      get<0>(local_unit_interface_normal_vector)[i] = -1.;
      get<1>(local_unit_interface_normal_vector)[i] = 0.;
      get<2>(local_unit_interface_normal_vector)[i] = 0.;
    }
    // Setting local_RhsVSpacetimeMetric
    for (size_t i = 0; i < slice_grid_points; ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_char_projected_rhs_dt_v_psi.get(0, a)[i] = 23.;
        local_char_projected_rhs_dt_v_psi.get(1, a)[i] = 29.;
        local_char_projected_rhs_dt_v_psi.get(2, a)[i] = 31.;
        local_char_projected_rhs_dt_v_psi.get(3, a)[i] = 37.;
      }
    }
    // Setting char speeds
    for (size_t i = 0; i < slice_grid_points; ++i) {
      local_char_speeds.at(0)[i] = -0.3;
      local_char_speeds.at(1)[i] = -0.1;
    }
  }

  // Allocate memory for output
  tnsr::aa<DataVector, VolumeDim, frame> local_bc_dt_v_psi(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());

  // Compute rhs value
  GeneralizedHarmonic::BoundaryConditions::detail::
      set_dt_v_psi_constraint_preserving(
          make_not_null(&local_bc_dt_v_psi), local_unit_interface_normal_vector,
          local_three_index_constraint, local_char_projected_rhs_dt_v_psi,
          local_char_speeds);

  // Initialize with values from SpEC
  auto spec_bc_dt_v_psi =
      make_with_value<tnsr::aa<DataVector, VolumeDim, frame>>(
          local_bc_dt_v_psi, std::numeric_limits<double>::signaling_NaN());

  for (size_t i = 0; i < slice_grid_points; ++i) {
    get<0, 0>(spec_bc_dt_v_psi)[i] = 25.4;
    get<0, 1>(spec_bc_dt_v_psi)[i] = 25.4;
    get<0, 2>(spec_bc_dt_v_psi)[i] = 25.4;
    get<0, 3>(spec_bc_dt_v_psi)[i] = 25.4;
    get<1, 1>(spec_bc_dt_v_psi)[i] = 31.4;
    get<1, 2>(spec_bc_dt_v_psi)[i] = 31.4;
    get<1, 3>(spec_bc_dt_v_psi)[i] = 31.4;
    get<2, 2>(spec_bc_dt_v_psi)[i] = 33.4;
    get<2, 3>(spec_bc_dt_v_psi)[i] = 33.4;
    get<3, 3>(spec_bc_dt_v_psi)[i] = 39.4;
  }

  // Compare values returned by BC action vs those from SpEC
  CHECK_ITERABLE_APPROX(local_bc_dt_v_psi, spec_bc_dt_v_psi);

  // Test for another set of values
  {
    // Setting unit_interface_normal_Vector
    for (size_t i = 0; i < slice_grid_points; ++i) {
      get<0>(local_unit_interface_normal_vector)[i] = -1.;
      get<1>(local_unit_interface_normal_vector)[i] = 1.;
      get<2>(local_unit_interface_normal_vector)[i] = 1.;
    }
  }
  // Compute rhs value
  GeneralizedHarmonic::BoundaryConditions::detail::
      set_dt_v_psi_constraint_preserving(
          make_not_null(&local_bc_dt_v_psi), local_unit_interface_normal_vector,
          local_three_index_constraint, local_char_projected_rhs_dt_v_psi,
          local_char_speeds);

  // Initialize with values from SpEC
  for (size_t i = 0; i < slice_grid_points; ++i) {
    get<0, 0>(spec_bc_dt_v_psi)[i] = 20.;
    get<0, 1>(spec_bc_dt_v_psi)[i] = 20.;
    get<0, 2>(spec_bc_dt_v_psi)[i] = 20.;
    get<0, 3>(spec_bc_dt_v_psi)[i] = 20.;
    get<1, 1>(spec_bc_dt_v_psi)[i] = 26.;
    get<1, 2>(spec_bc_dt_v_psi)[i] = 26.;
    get<1, 3>(spec_bc_dt_v_psi)[i] = 26.;
    get<2, 2>(spec_bc_dt_v_psi)[i] = 28.;
    get<2, 3>(spec_bc_dt_v_psi)[i] = 28.;
    get<3, 3>(spec_bc_dt_v_psi)[i] = 34.;
  }

  // Compare values returned by BC action vs those from SpEC
  CHECK_ITERABLE_APPROX(local_bc_dt_v_psi, spec_bc_dt_v_psi);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.BCBjorhus.VPsi",
                  "[Unit][Evolution]") {
  // Piece-wise tests with SpEC output
  const size_t grid_size = 3;

  test_constraint_preserving_bjorhus_v_psi(grid_size);
}
