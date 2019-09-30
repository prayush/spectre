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
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Direction.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Side.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
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
using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
using frame = Frame::Inertial;
constexpr size_t VolumeDim = 3;

void test_constraint_preserving_bjorhus_u_psi(
    const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) noexcept {
  // Setup grid
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
  const Direction<VolumeDim> direction(1, Side::Upper);  // +y direction
  const size_t slice_grid_points =
      mesh.extents().slice_away(direction.dimension()).product();
  //
  // Create a TempTensor that stores all temporaries computed
  // here and elsewhere
  TempBuffer<GeneralizedHarmonic::Actions::BoundaryConditions_detail::
                 all_local_vars<VolumeDim>>
      buffer(slice_grid_points);
  // Also create local variables and their time derivatives, both of which we
  // will populate as needed
  db::item_type<GeneralizedHarmonic::System<VolumeDim>::variables_tag>
      local_vars(slice_grid_points, 0.);
  Variables<db::wrap_tags_in<
      ::Tags::dt, GeneralizedHarmonic::System<VolumeDim>::gradients_tags>>
      local_dt_vars(slice_grid_points, 0.);
  tnsr::i<DataVector, VolumeDim, frame> local_unit_normal_one_form(
      slice_grid_points, 0.);

  {
    // POPULATE various tensors needed to compute BcDtUpsi
    // EXACTLY as done in SpEC
    auto& local_three_index_constraint =
        get<::Tags::Tempiaa<17, VolumeDim, Frame::Inertial, DataVector>>(
            buffer);
    auto& local_unit_interface_normal_vector =
        get<::Tags::TempA<5, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_lapse = get<::Tags::TempScalar<19, DataVector>>(buffer);
    auto& local_shift =
        get<::Tags::TempI<20, VolumeDim, Frame::Inertial, DataVector>>(buffer);

    auto& local_pi =
        get<GeneralizedHarmonic::Tags::Pi<VolumeDim, frame>>(local_vars);
    auto& local_phi =
        get<GeneralizedHarmonic::Tags::Phi<VolumeDim, frame>>(local_vars);
    auto& local_psi =
        get<gr::Tags::SpacetimeMetric<VolumeDim, frame, DataVector>>(
            local_vars);

    auto& local_char_projected_rhs_dt_u_psi =
        get<::Tags::Tempaa<22, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_char_speeds_0 = get<::Tags::TempScalar<12, DataVector>>(buffer);
    auto& local_char_speeds_1 = get<::Tags::TempScalar<13, DataVector>>(buffer);
    auto& local_char_speeds_2 = get<::Tags::TempScalar<14, DataVector>>(buffer);
    auto& local_char_speeds_3 = get<::Tags::TempScalar<15, DataVector>>(buffer);

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
      local_unit_interface_normal_vector.get(0)[i] = 1.e300;
      local_unit_interface_normal_vector.get(1)[i] = -1.;
      local_unit_interface_normal_vector.get(2)[i] = 0.;
      local_unit_interface_normal_vector.get(3)[i] = 0.;
    }
    // Setting local_lapse
    for (size_t i = 0; i < slice_grid_points; ++i) {
      get(local_lapse)[i] = 2.;
    }
    // Setting shift
    for (size_t i = 0; i < slice_grid_points; ++i) {
      local_shift.get(0)[i] = 1.;
      local_shift.get(1)[i] = 2.;
      local_shift.get(2)[i] = 3.;
    }
    // Setting pi AND phi
    for (size_t i = 0; i < slice_grid_points; ++i) {
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
    for (size_t i = 0; i < slice_grid_points; ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_char_projected_rhs_dt_u_psi.get(0, a)[i] = 23.;
        local_char_projected_rhs_dt_u_psi.get(1, a)[i] = 29.;
        local_char_projected_rhs_dt_u_psi.get(2, a)[i] = 31.;
        local_char_projected_rhs_dt_u_psi.get(3, a)[i] = 37.;
      }
    }
    // Setting char speeds
    for (size_t i = 0; i < slice_grid_points; ++i) {
      get(local_char_speeds_0)[i] = -0.3;
      get(local_char_speeds_1)[i] = -0.1;
    }
    // Setting unit normal one_form
    get<1>(local_unit_normal_one_form) = 1.;
  }
  // Compute return value from Action
  auto local_bc_dt_u_psi =
      GeneralizedHarmonic::Actions::BoundaryConditions_detail::set_dt_u_psi<
          typename GeneralizedHarmonic::Tags::UPsi<VolumeDim, frame>::type,
          VolumeDim>::
          apply(GeneralizedHarmonic::Actions::BoundaryConditions_detail::
                    UPsiBcMethod::ConstraintPreservingBjorhus,
                buffer, local_vars, local_dt_vars, local_unit_normal_one_form);

  // debugPK: DISPLAY results of the TEST
  if (debugPKoff) {
    auto& local_inertial_coords =
        get<::Tags::TempI<37, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    // for (size_t i = 0; i < VolumeDim; ++i) {
    // local_inertial_coords.get(i) = x.get(i);
    //}
    for (size_t i = 0; i < slice_grid_points; ++i) {
      print_rank2_tensor_at_point(
          "BcDtUpsi", local_bc_dt_u_psi, local_inertial_coords.get(0),
          local_inertial_coords.get(1), local_inertial_coords.get(2), i);
    }
  }

  // Initialize with values from SpEC
  auto spec_bd_dt_u_psi = local_bc_dt_u_psi;
  for (size_t i = 0; i < slice_grid_points; ++i) {
    get<0, 0>(spec_bd_dt_u_psi)[i] = 25.4;
    get<0, 1>(spec_bd_dt_u_psi)[i] = 25.4;
    get<0, 2>(spec_bd_dt_u_psi)[i] = 25.4;
    get<0, 3>(spec_bd_dt_u_psi)[i] = 25.4;
    get<1, 1>(spec_bd_dt_u_psi)[i] = 31.4;
    get<1, 2>(spec_bd_dt_u_psi)[i] = 31.4;
    get<1, 3>(spec_bd_dt_u_psi)[i] = 31.4;
    get<2, 2>(spec_bd_dt_u_psi)[i] = 33.4;
    get<2, 3>(spec_bd_dt_u_psi)[i] = 33.4;
    get<3, 3>(spec_bd_dt_u_psi)[i] = 39.4;
  }

  // Compare values returned by BC action vs those from SpEC
  CHECK_ITERABLE_APPROX(local_bc_dt_u_psi, spec_bd_dt_u_psi);

  // Test for another set of values
  {
    auto& local_unit_interface_normal_vector =
        get<::Tags::TempA<5, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    // Setting unit_interface_normal_Vector
    for (size_t i = 0; i < slice_grid_points; ++i) {
      local_unit_interface_normal_vector.get(0)[i] = 1.e300;
      local_unit_interface_normal_vector.get(1)[i] = -1.;
      local_unit_interface_normal_vector.get(2)[i] = 1.;
      local_unit_interface_normal_vector.get(3)[i] = 1.;
    }
  }
  // Compute return value from Action
  local_bc_dt_u_psi =
      GeneralizedHarmonic::Actions::BoundaryConditions_detail::set_dt_u_psi<
          typename GeneralizedHarmonic::Tags::UPsi<VolumeDim, frame>::type,
          VolumeDim>::
          apply(GeneralizedHarmonic::Actions::BoundaryConditions_detail::
                    UPsiBcMethod::ConstraintPreservingBjorhus,
                buffer, local_vars, local_dt_vars, local_unit_normal_one_form);
  // Initialize with values from SpEC
  for (size_t i = 0; i < slice_grid_points; ++i) {
    get<0, 0>(spec_bd_dt_u_psi)[i] = 20.;
    get<0, 1>(spec_bd_dt_u_psi)[i] = 20.;
    get<0, 2>(spec_bd_dt_u_psi)[i] = 20.;
    get<0, 3>(spec_bd_dt_u_psi)[i] = 20.;
    get<1, 1>(spec_bd_dt_u_psi)[i] = 26.;
    get<1, 2>(spec_bd_dt_u_psi)[i] = 26.;
    get<1, 3>(spec_bd_dt_u_psi)[i] = 26.;
    get<2, 2>(spec_bd_dt_u_psi)[i] = 28.;
    get<2, 3>(spec_bd_dt_u_psi)[i] = 28.;
    get<3, 3>(spec_bd_dt_u_psi)[i] = 34.;
  }

  // Compare values returned by BC action vs those from SpEC
  CHECK_ITERABLE_APPROX(local_bc_dt_u_psi, spec_bd_dt_u_psi);
}

void test_constraint_preserving_bjorhus_u_zero(
    const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) noexcept {
  // Setup grid
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
  const Direction<VolumeDim> direction(1, Side::Upper);  // +y direction
  const size_t slice_grid_points =
      mesh.extents().slice_away(direction.dimension()).product();
  //
  // Create a TempTensor that stores all temporaries computed
  // here and elsewhere
  TempBuffer<GeneralizedHarmonic::Actions::BoundaryConditions_detail::
                 all_local_vars<VolumeDim>>
      buffer(slice_grid_points);
  // Also create local variables and their time derivatives, both of which we
  // will populate as needed
  db::item_type<GeneralizedHarmonic::System<VolumeDim>::variables_tag>
      local_vars(slice_grid_points, 0.);
  Variables<db::wrap_tags_in<
      ::Tags::dt, GeneralizedHarmonic::System<VolumeDim>::gradients_tags>>
      local_dt_vars(slice_grid_points, 0.);
  tnsr::i<DataVector, VolumeDim, frame> local_unit_normal_one_form(
      slice_grid_points, 0.);

  {
    // POPULATE various tensors needed to compute BcDtUpsi
    // EXACTLY as done in SpEC
    auto& local_four_index_constraint =
        get<::Tags::Tempiaa<18, VolumeDim, Frame::Inertial, DataVector>>(
            buffer);
    auto& local_unit_interface_normal_vector =
        get<::Tags::TempA<5, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_lapse = get<::Tags::TempScalar<19, DataVector>>(buffer);
    auto& local_shift =
        get<::Tags::TempI<20, VolumeDim, Frame::Inertial, DataVector>>(buffer);

    auto& local_pi =
        get<GeneralizedHarmonic::Tags::Pi<VolumeDim, frame>>(local_vars);
    auto& local_phi =
        get<GeneralizedHarmonic::Tags::Phi<VolumeDim, frame>>(local_vars);
    auto& local_psi =
        get<gr::Tags::SpacetimeMetric<VolumeDim, frame, DataVector>>(
            local_vars);

    auto& local_char_projected_rhs_dt_u_zero =
        get<::Tags::Tempiaa<23, VolumeDim, Frame::Inertial, DataVector>>(
            buffer);
    auto& local_char_speeds_0 = get<::Tags::TempScalar<12, DataVector>>(buffer);
    auto& local_char_speeds_1 = get<::Tags::TempScalar<13, DataVector>>(buffer);
    auto& local_char_speeds_2 = get<::Tags::TempScalar<14, DataVector>>(buffer);
    auto& local_char_speeds_3 = get<::Tags::TempScalar<15, DataVector>>(buffer);

    // Setting 4idxConstraint:
    // initialize dPhi (with same values as for SpEC) and compute C4 from it
    auto local_dphi =
        make_with_value<tnsr::ijaa<DataVector, VolumeDim, Frame::Inertial>>(
            local_lapse, 0.);
    for (size_t a = 0; a <= VolumeDim; ++a) {
      for (size_t b = 0; b <= VolumeDim; ++b) {
        for (size_t i = 0; i < slice_grid_points; ++i) {
          local_dphi.get(0, 0, a, b)[i] = 3.;
          local_dphi.get(0, 1, a, b)[i] = 5.;
          local_dphi.get(0, 2, a, b)[i] = 7.;
          local_dphi.get(1, 0, a, b)[i] = 59.;
          local_dphi.get(1, 1, a, b)[i] = 61.;
          local_dphi.get(1, 2, a, b)[i] = 67.;
          local_dphi.get(2, 0, a, b)[i] = 73.;
          local_dphi.get(2, 1, a, b)[i] = 79.;
          local_dphi.get(2, 2, a, b)[i] = 83.;
        }
      }
    }
    // C4_{iab} = LeviCivita^{ijk} dphi_{jkab}
    local_four_index_constraint =
        GeneralizedHarmonic::four_index_constraint<VolumeDim, Frame::Inertial,
                                                   DataVector>(local_dphi);

    // Setting unit_interface_normal_Vector
    for (size_t i = 0; i < slice_grid_points; ++i) {
      local_unit_interface_normal_vector.get(0)[i] = 1.e300;
      local_unit_interface_normal_vector.get(1)[i] = -1.;
      local_unit_interface_normal_vector.get(2)[i] = 1.;
      local_unit_interface_normal_vector.get(3)[i] = 1.;
    }
    // Setting lapse
    for (size_t i = 0; i < slice_grid_points; ++i) {
      get(local_lapse)[i] = 2.;
    }
    // Setting shift
    for (size_t i = 0; i < slice_grid_points; ++i) {
      local_shift.get(0)[i] = 1.;
      local_shift.get(1)[i] = 2.;
      local_shift.get(2)[i] = 3.;
    }
    // Setting pi AND phi
    for (size_t i = 0; i < slice_grid_points; ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = 0; b <= VolumeDim; ++b) {
          local_pi.get(a, b)[i] = 1.;
          local_phi.get(0, a, b)[i] = 3.;
          local_phi.get(1, a, b)[i] = 5.;
          local_phi.get(2, a, b)[i] = 7.;
        }
      }
    }
    // Setting local_RhsU0
    for (size_t i = 0; i < slice_grid_points; ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          local_char_projected_rhs_dt_u_zero.get(0, a, b)[i] = 91.;
          local_char_projected_rhs_dt_u_zero.get(1, a, b)[i] = 97.;
          local_char_projected_rhs_dt_u_zero.get(2, a, b)[i] = 101.;
        }
      }
    }
    // Setting char speeds
    for (size_t i = 0; i < slice_grid_points; ++i) {
      get(local_char_speeds_0)[i] = -0.3;
      get(local_char_speeds_1)[i] = -0.1;
    }
    // Setting unit normal one_form
    get<1>(local_unit_normal_one_form) = 1.;
  }
  // Compute return value from Action
  auto local_bc_dt_u_zero =
      GeneralizedHarmonic::Actions::BoundaryConditions_detail::set_dt_u_zero<
          typename GeneralizedHarmonic::Tags::UZero<VolumeDim, frame>::type,
          VolumeDim>::
          apply(GeneralizedHarmonic::Actions::BoundaryConditions_detail::
                    UZeroBcMethod::ConstraintPreservingBjorhus,
                buffer, local_vars, local_dt_vars, local_unit_normal_one_form);

  // Initialize with values from SpEC
  auto spec_bc_dt_u_zero = local_bc_dt_u_zero;
  for (size_t i = 0; i < slice_grid_points; ++i) {
    get<0, 0, 0>(spec_bc_dt_u_zero)[i] = 79.;
    get<0, 0, 1>(spec_bc_dt_u_zero)[i] = 79.;
    get<0, 0, 2>(spec_bc_dt_u_zero)[i] = 79.;
    get<0, 0, 3>(spec_bc_dt_u_zero)[i] = 79.;
    get<1, 0, 0>(spec_bc_dt_u_zero)[i] = 90.4;
    get<1, 0, 1>(spec_bc_dt_u_zero)[i] = 90.4;
    get<1, 0, 2>(spec_bc_dt_u_zero)[i] = 90.4;
    get<1, 0, 3>(spec_bc_dt_u_zero)[i] = 90.4;
    get<2, 0, 3>(spec_bc_dt_u_zero)[i] = 95.6;
    get<2, 0, 0>(spec_bc_dt_u_zero)[i] = 95.6;
    get<2, 0, 1>(spec_bc_dt_u_zero)[i] = 95.6;
    get<2, 0, 2>(spec_bc_dt_u_zero)[i] = 95.6;

    get<0, 1, 1>(spec_bc_dt_u_zero)[i] = 79.;
    get<0, 1, 2>(spec_bc_dt_u_zero)[i] = 79.;
    get<0, 1, 3>(spec_bc_dt_u_zero)[i] = 79.;
    get<1, 1, 1>(spec_bc_dt_u_zero)[i] = 90.4;
    get<1, 1, 2>(spec_bc_dt_u_zero)[i] = 90.4;
    get<1, 1, 3>(spec_bc_dt_u_zero)[i] = 90.4;
    get<2, 1, 1>(spec_bc_dt_u_zero)[i] = 95.6;
    get<2, 1, 2>(spec_bc_dt_u_zero)[i] = 95.6;
    get<2, 1, 3>(spec_bc_dt_u_zero)[i] = 95.6;

    get<0, 2, 2>(spec_bc_dt_u_zero)[i] = 79.;
    get<0, 2, 3>(spec_bc_dt_u_zero)[i] = 79.;
    get<1, 2, 2>(spec_bc_dt_u_zero)[i] = 90.4;
    get<1, 2, 3>(spec_bc_dt_u_zero)[i] = 90.4;
    get<2, 2, 2>(spec_bc_dt_u_zero)[i] = 95.6;
    get<2, 2, 3>(spec_bc_dt_u_zero)[i] = 95.6;

    get<0, 3, 3>(spec_bc_dt_u_zero)[i] = 79.;
    get<1, 3, 3>(spec_bc_dt_u_zero)[i] = 90.4;
    get<2, 3, 3>(spec_bc_dt_u_zero)[i] = 95.6;
  }

  // Compare values returned by BC action vs those from SpEC
  CHECK_ITERABLE_APPROX(local_bc_dt_u_zero, spec_bc_dt_u_zero);

  // Test for another set of values
  {
    auto& local_unit_interface_normal_vector =
        get<::Tags::TempA<5, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    // Setting unit_interface_normal_Vector
    for (size_t i = 0; i < slice_grid_points; ++i) {
      local_unit_interface_normal_vector.get(0)[i] = 1.e300;
      local_unit_interface_normal_vector.get(1)[i] = -1.;
      local_unit_interface_normal_vector.get(2)[i] = 0.;
      local_unit_interface_normal_vector.get(3)[i] = 0.;
    }
  }
  // Compute return value from Action
  local_bc_dt_u_zero =
      GeneralizedHarmonic::Actions::BoundaryConditions_detail::set_dt_u_zero<
          typename GeneralizedHarmonic::Tags::UZero<VolumeDim, frame>::type,
          VolumeDim>::
          apply(GeneralizedHarmonic::Actions::BoundaryConditions_detail::
                    UZeroBcMethod::ConstraintPreservingBjorhus,
                buffer, local_vars, local_dt_vars, local_unit_normal_one_form);

  // Initialize with values from SpEC
  for (size_t i = 0; i < slice_grid_points; ++i) {
    get<0, 0, 0>(spec_bc_dt_u_zero)[i] = 91.;
    get<0, 0, 1>(spec_bc_dt_u_zero)[i] = 91.;
    get<0, 0, 2>(spec_bc_dt_u_zero)[i] = 91.;
    get<0, 0, 3>(spec_bc_dt_u_zero)[i] = 91.;
    get<1, 0, 0>(spec_bc_dt_u_zero)[i] = 91.6;
    get<1, 0, 1>(spec_bc_dt_u_zero)[i] = 91.6;
    get<1, 0, 2>(spec_bc_dt_u_zero)[i] = 91.6;
    get<1, 0, 3>(spec_bc_dt_u_zero)[i] = 91.6;
    get<2, 0, 3>(spec_bc_dt_u_zero)[i] = 94.4;
    get<2, 0, 0>(spec_bc_dt_u_zero)[i] = 94.4;
    get<2, 0, 1>(spec_bc_dt_u_zero)[i] = 94.4;
    get<2, 0, 2>(spec_bc_dt_u_zero)[i] = 94.4;

    get<0, 1, 1>(spec_bc_dt_u_zero)[i] = 91.;
    get<0, 1, 2>(spec_bc_dt_u_zero)[i] = 91.;
    get<0, 1, 3>(spec_bc_dt_u_zero)[i] = 91.;
    get<1, 1, 1>(spec_bc_dt_u_zero)[i] = 91.6;
    get<1, 1, 2>(spec_bc_dt_u_zero)[i] = 91.6;
    get<1, 1, 3>(spec_bc_dt_u_zero)[i] = 91.6;
    get<2, 1, 1>(spec_bc_dt_u_zero)[i] = 94.4;
    get<2, 1, 2>(spec_bc_dt_u_zero)[i] = 94.4;
    get<2, 1, 3>(spec_bc_dt_u_zero)[i] = 94.4;

    get<0, 2, 2>(spec_bc_dt_u_zero)[i] = 91.;
    get<0, 2, 3>(spec_bc_dt_u_zero)[i] = 91.;
    get<1, 2, 2>(spec_bc_dt_u_zero)[i] = 91.6;
    get<1, 2, 3>(spec_bc_dt_u_zero)[i] = 91.6;
    get<2, 2, 2>(spec_bc_dt_u_zero)[i] = 94.4;
    get<2, 2, 3>(spec_bc_dt_u_zero)[i] = 94.4;

    get<0, 3, 3>(spec_bc_dt_u_zero)[i] = 91.;
    get<1, 3, 3>(spec_bc_dt_u_zero)[i] = 91.6;
    get<2, 3, 3>(spec_bc_dt_u_zero)[i] = 94.4;
  }

  // Compare values returned by BC action vs those from SpEC
  CHECK_ITERABLE_APPROX(local_bc_dt_u_zero, spec_bc_dt_u_zero);
}

template <GeneralizedHarmonic::Actions::BoundaryConditions_detail::
              UMinusBcMethod BcMethod>
void test_constraint_preserving_bjorhus_u_minus(
    const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) noexcept;

// This function tests the boundary condition imposed on UMinus when
// option `Freezing` is specified. Only the gauge portion of the RHS is set.
template <>
void test_constraint_preserving_bjorhus_u_minus<
    GeneralizedHarmonic::Actions::BoundaryConditions_detail::UMinusBcMethod::
        Freezing>(const size_t grid_size_each_dimension,
                  const std::array<double, 3>& lower_bound,
                  const std::array<double, 3>& upper_bound) noexcept {
  // Setup grid
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
  const Direction<VolumeDim> direction(1, Side::Upper);  // +y direction
  const size_t slice_grid_points =
      mesh.extents().slice_away(direction.dimension()).product();
  const auto inertial_coords = [&slice_grid_points, &lower_bound]() {
    tnsr::I<DataVector, VolumeDim, frame> tmp(slice_grid_points, 0.);
    // +y direction
    get<1>(tmp) = 0.5;
    for (size_t i = 0; i < VolumeDim; ++i) {
      for (size_t j = 0; j < VolumeDim; ++j) {
        get<0>(tmp)[i * VolumeDim + j] =
            lower_bound[0] + 0.5 * static_cast<double>(i);
        get<2>(tmp)[i * VolumeDim + j] =
            lower_bound[2] + 0.5 * static_cast<double>(j);
      }
    }
    return tmp;
  }();
  //
  // Create a TempTensor that stores all temporaries computed
  // here and elsewhere
  TempBuffer<GeneralizedHarmonic::Actions::BoundaryConditions_detail::
                 all_local_vars<VolumeDim>>
      buffer(slice_grid_points);
  // Also create local variables and their time derivatives, both of which we
  // will populate as needed
  db::item_type<GeneralizedHarmonic::System<VolumeDim>::variables_tag>
      local_vars(slice_grid_points, 0.);
  Variables<db::wrap_tags_in<
      ::Tags::dt, GeneralizedHarmonic::System<VolumeDim>::gradients_tags>>
      local_dt_vars(slice_grid_points, 0.);
  tnsr::i<DataVector, VolumeDim, frame> local_unit_normal_one_form(
      slice_grid_points, 0.);

  {
    // POPULATE various tensors needed to compute BcDtUpsi
    // EXACTLY as done in SpEC
    auto& local_inertial_coords =
        get<::Tags::TempI<37, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_constraint_gamma2 =
        get<::Tags::TempScalar<26, DataVector>>(buffer);
    // timelike and spacelike SPACETIME vectors, l^a and k^a
    auto& local_outgoing_null_one_form =
        get<::Tags::Tempa<0, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_incoming_null_one_form =
        get<::Tags::Tempa<1, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    // timelike and spacelike SPACETIME oneforms, l_a and k_a
    auto& local_outgoing_null_vector =
        get<::Tags::TempA<2, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_incoming_null_vector =
        get<::Tags::TempA<3, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    // projection Ab
    auto& local_projection_Ab =
        get<::Tags::TempAb<11, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    // RhsUPsi and RhsUminus
    auto& local_char_projected_rhs_dt_u_psi =
        get<::Tags::Tempaa<22, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_char_projected_rhs_dt_u_minus =
        get<::Tags::Tempaa<25, VolumeDim, Frame::Inertial, DataVector>>(buffer);

    auto& local_unit_interface_normal_vector =
        get<::Tags::TempA<5, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_lapse = get<::Tags::TempScalar<19, DataVector>>(buffer);
    auto& local_shift =
        get<::Tags::TempI<20, VolumeDim, Frame::Inertial, DataVector>>(buffer);

    auto& local_pi =
        get<GeneralizedHarmonic::Tags::Pi<VolumeDim, frame>>(local_vars);
    auto& local_phi =
        get<GeneralizedHarmonic::Tags::Phi<VolumeDim, frame>>(local_vars);
    auto& local_psi =
        get<gr::Tags::SpacetimeMetric<VolumeDim, frame, DataVector>>(
            local_vars);

    auto& local_char_projected_rhs_dt_u_zero =
        get<::Tags::Tempiaa<23, VolumeDim, Frame::Inertial, DataVector>>(
            buffer);
    auto& local_char_speeds_0 = get<::Tags::TempScalar<12, DataVector>>(buffer);
    auto& local_char_speeds_1 = get<::Tags::TempScalar<13, DataVector>>(buffer);
    auto& local_char_speeds_2 = get<::Tags::TempScalar<14, DataVector>>(buffer);
    auto& local_char_speeds_3 = get<::Tags::TempScalar<15, DataVector>>(buffer);

    // Setting coords
    for (size_t i = 0; i < VolumeDim; ++i) {
      local_inertial_coords.get(i) = inertial_coords.get(i);
    }
    // Setting constraint_gamma2
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      get(local_constraint_gamma2)[i] = 113.;
    }
    // Setting incoming null one_form: ui
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_incoming_null_one_form.get(0)[i] = -2.;
      local_incoming_null_one_form.get(1)[i] = 5.;
      local_incoming_null_one_form.get(2)[i] = 3.;
      local_incoming_null_one_form.get(3)[i] = 7.;
    }
    // Setting incoming null vector: uI
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_incoming_null_vector.get(0)[i] = -1.;
      local_incoming_null_vector.get(1)[i] = 13.;
      local_incoming_null_vector.get(2)[i] = 17.;
      local_incoming_null_vector.get(3)[i] = 19.;
    }
    // Setting outgoing null one_form: vi
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_outgoing_null_one_form.get(0)[i] = -1.;
      local_outgoing_null_one_form.get(1)[i] = 3.;
      local_outgoing_null_one_form.get(2)[i] = 2.;
      local_outgoing_null_one_form.get(3)[i] = 5.;
    }
    // Setting outgoing null vector: vI
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_outgoing_null_vector.get(0)[i] = -1.;
      local_outgoing_null_vector.get(1)[i] = 2.;
      local_outgoing_null_vector.get(2)[i] = 3.;
      local_outgoing_null_vector.get(3)[i] = 5.;
    }
    // Setting projection Ab
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_projection_Ab.get(0, a)[i] = 233.;
        local_projection_Ab.get(1, a)[i] = 239.;
        local_projection_Ab.get(2, a)[i] = 241.;
        local_projection_Ab.get(3, a)[i] = 251.;
      }
    }
    // Setting local_RhsUPsi
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_char_projected_rhs_dt_u_psi.get(0, a)[i] = 23.;
        local_char_projected_rhs_dt_u_psi.get(1, a)[i] = 29.;
        local_char_projected_rhs_dt_u_psi.get(2, a)[i] = 31.;
        local_char_projected_rhs_dt_u_psi.get(3, a)[i] = 37.;
      }
    }
    // Setting RhsUMinus
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_char_projected_rhs_dt_u_minus.get(0, a)[i] = 331.;
        local_char_projected_rhs_dt_u_minus.get(1, a)[i] = 337.;
        local_char_projected_rhs_dt_u_minus.get(2, a)[i] = 347.;
        local_char_projected_rhs_dt_u_minus.get(3, a)[i] = 349.;
      }
    }

    // Setting unit_interface_normal_Vector
    for (size_t i = 0; i < slice_grid_points; ++i) {
      local_unit_interface_normal_vector.get(0)[i] = 1.e300;
      local_unit_interface_normal_vector.get(1)[i] = -1.;
      local_unit_interface_normal_vector.get(2)[i] = 1.;
      local_unit_interface_normal_vector.get(3)[i] = 1.;
    }
    // Setting pi AND phi
    for (size_t i = 0; i < slice_grid_points; ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = 0; b <= VolumeDim; ++b) {
          local_pi.get(a, b)[i] = 1.;
          local_phi.get(0, a, b)[i] = 3.;
          local_phi.get(1, a, b)[i] = 5.;
          local_phi.get(2, a, b)[i] = 7.;
        }
      }
    }
    // Setting local_RhsU0
    for (size_t i = 0; i < slice_grid_points; ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          local_char_projected_rhs_dt_u_zero.get(0, a, b)[i] = 91.;
          local_char_projected_rhs_dt_u_zero.get(1, a, b)[i] = 97.;
          local_char_projected_rhs_dt_u_zero.get(2, a, b)[i] = 101.;
        }
      }
    }
    // Setting char speeds
    for (size_t i = 0; i < slice_grid_points; ++i) {
      get(local_char_speeds_0)[i] = -0.3;
      get(local_char_speeds_1)[i] = -0.1;
      get(local_char_speeds_3)[i] = -0.2;
    }
  }

  // Compute return value from Action
  auto local_bc_dt_u_minus =
      GeneralizedHarmonic::Actions::BoundaryConditions_detail::set_dt_u_minus<
          typename GeneralizedHarmonic::Tags::UMinus<VolumeDim, frame>::type,
          VolumeDim>::
          apply(GeneralizedHarmonic::Actions::BoundaryConditions_detail::
                    UMinusBcMethod::Freezing,
                buffer, local_vars, local_dt_vars, inertial_coords,
                local_unit_normal_one_form);

  // Initialize with values from SpEC
  auto spec_bc_dt_u_minus = local_bc_dt_u_minus;
  for (size_t i = 0; i < slice_grid_points; ++i) {
    if (get<0>(inertial_coords)[i] == 299. and
        get<1>(inertial_coords)[i] == 0.5 and
        get<2>(inertial_coords)[i] == -0.5) {
      std::array<double, 16> spec_vals = {
          {-60383491.77088407, 48998513.74318591, 17859686.38943806,
           80929560.64883879, 48998513.74318591, 132404254.8520959,
           108291107.5524412, 154536853.2719879, 17859686.38943806,
           108291107.5524412, 82283797.31617498, 133110088.4608499,
           80929560.64883879, 154536853.2719879, 133110088.4608499,
           173190849.6514583}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 299.5 and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == -0.5) {
      std::array<double, 16> spec_vals = {
          {-60383494.75455543, 48998516.16430231, 17859687.27192156,
           80929564.64773327, 48998516.16430231, 132404261.3944599,
           108291112.9033255, 154536860.9079687, 17859687.27192156,
           108291112.9033255, 82283801.38198504, 133110095.0380906,
           80929564.64773327, 154536860.9079687, 133110095.0380906,
           173190858.2091711}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 300. and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == -0.5) {
      std::array<double, 16> spec_vals = {
          {-60383497.7282813, 48998518.5773484, 17859688.15146346,
           80929568.63329825, 48998518.5773484, 132404267.9150162,
           108291118.2363736, 154536868.5184965, 17859688.15146346,
           108291118.2363736, 82283805.43424253, 133110101.5934074,
           80929568.63329825, 154536868.5184965, 133110101.5934074,
           173190866.7383583}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 299. and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.) {
      std::array<double, 16> spec_vals = {
          {-60383491.76838519, 48998513.74115818, 17859686.38869897,
           80929560.64548963, 48998513.74115818, 132404254.8466165,
           108291107.5479597, 154536853.2655927, 17859686.38869897,
           108291107.5479597, 82283797.31276976, 133110088.4553414,
           80929560.64548963, 154536853.2655927, 133110088.4553414,
           173190849.644291}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 299.5 and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.) {
      std::array<double, 16> spec_vals = {
          {-60383494.75206903, 48998516.16228471, 17859687.27118616,
           80929564.64440086, 48998516.16228471, 132404261.3890079,
           108291112.8988664, 154536860.9016054, 17859687.27118616,
           108291112.8988664, 82283801.37859686, 133110095.0326096,
           80929564.64440086, 154536860.9016054, 133110095.0326096,
           173190858.2020396}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 300. and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.) {
      std::array<double, 16> spec_vals = {
          {-60383497.72580732, 48998518.57534086, 17859688.15073173,
           80929568.62998244, 48998518.57534086, 132404267.9095915,
           108291118.2319368, 154536868.5121649, 17859688.15073173,
           108291118.2319368, 82283805.43087126, 133110101.5879537,
           80929568.62998244, 154536868.5121649, 133110101.5879537,
           173190866.7312625}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 299. and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.5) {
      std::array<double, 16> spec_vals = {
          {-60383491.77088407, 48998513.74318591, 17859686.38943806,
           80929560.64883879, 48998513.74318591, 132404254.8520959,
           108291107.5524412, 154536853.2719879, 17859686.38943806,
           108291107.5524412, 82283797.31617498, 133110088.4608499,
           80929560.64883879, 154536853.2719879, 133110088.4608499,
           173190849.6514583}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 299.5 and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.5) {
      std::array<double, 16> spec_vals = {
          {-60383494.75455543, 48998516.16430231, 17859687.27192156,
           80929564.64773327, 48998516.16430231, 132404261.3944599,
           108291112.9033255, 154536860.9079687, 17859687.27192156,
           108291112.9033255, 82283801.38198504, 133110095.0380906,
           80929564.64773327, 154536860.9079687, 133110095.0380906,
           173190858.2091711}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 300. and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.5) {
      std::array<double, 16> spec_vals = {
          {-60383497.7282813, 48998518.5773484, 17859688.15146346,
           80929568.63329825, 48998518.5773484, 132404267.9150162,
           108291118.2363736, 154536868.5184965, 17859688.15146346,
           108291118.2363736, 82283805.43424253, 133110101.5934074,
           80929568.63329825, 154536868.5184965, 133110101.5934074,
           173190866.7383583}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else {
      ASSERT(false,
             "Not checking the correct face, coordinates not recognized");
    }
  }

  // Compare values returned by BC action vs those from SpEC
  CHECK_ITERABLE_APPROX(local_bc_dt_u_minus, spec_bc_dt_u_minus);
}

// This function tests the boundary condition imposed on UMinus when
// option `ConstraintPreservingBjorhus` is specified.
template <>
void test_constraint_preserving_bjorhus_u_minus<
    GeneralizedHarmonic::Actions::BoundaryConditions_detail::UMinusBcMethod::
        ConstraintPreservingBjorhus>(
    const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) noexcept {
  // Setup grid
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
  const Direction<VolumeDim> direction(1, Side::Upper);  // +y direction
  const size_t slice_grid_points =
      mesh.extents().slice_away(direction.dimension()).product();
  const auto inertial_coords = [&slice_grid_points, &lower_bound]() {
    tnsr::I<DataVector, VolumeDim, frame> tmp(slice_grid_points, 0.);
    // +y direction
    get<1>(tmp) = 0.5;
    for (size_t i = 0; i < VolumeDim; ++i) {
      for (size_t j = 0; j < VolumeDim; ++j) {
        get<0>(tmp)[i * VolumeDim + j] =
            lower_bound[0] + 0.5 * static_cast<double>(i);
        get<2>(tmp)[i * VolumeDim + j] =
            lower_bound[2] + 0.5 * static_cast<double>(j);
      }
    }
    return tmp;
  }();
  //
  // Create a TempTensor that stores all temporaries computed
  // here and elsewhere
  TempBuffer<GeneralizedHarmonic::Actions::BoundaryConditions_detail::
                 all_local_vars<VolumeDim>>
      buffer(slice_grid_points);
  // Also create local variables and their time derivatives, both of which we
  // will populate as needed
  db::item_type<GeneralizedHarmonic::System<VolumeDim>::variables_tag>
      local_vars(slice_grid_points, 0.);
  Variables<db::wrap_tags_in<
      ::Tags::dt, GeneralizedHarmonic::System<VolumeDim>::gradients_tags>>
      local_dt_vars(slice_grid_points, 0.);
  tnsr::i<DataVector, VolumeDim, frame> local_unit_normal_one_form(
      slice_grid_points, 0.);

  {
    // POPULATE various tensors needed to compute BcDtUpsi
    // EXACTLY as done in SpEC
    auto& local_inertial_coords =
        get<::Tags::TempI<37, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_constraint_gamma2 =
        get<::Tags::TempScalar<26, DataVector>>(buffer);
    // timelike and spacelike SPACETIME vectors, l^a and k^a
    auto& local_outgoing_null_one_form =
        get<::Tags::Tempa<0, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_incoming_null_one_form =
        get<::Tags::Tempa<1, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    // timelike and spacelike SPACETIME oneforms, l_a and k_a
    auto& local_outgoing_null_vector =
        get<::Tags::TempA<2, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_incoming_null_vector =
        get<::Tags::TempA<3, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    // spacetime projection operator P_ab, P^ab, and P^a_b
    auto& local_projection_AB =
        get<::Tags::TempAA<9, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_projection_ab =
        get<::Tags::Tempaa<10, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_projection_Ab =
        get<::Tags::TempAb<11, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    // constraint characteristics
    auto& local_constraint_char_zero_minus =
        get<::Tags::Tempa<16, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_constraint_char_zero_plus =
        get<::Tags::Tempa<34, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    // RhsUPsi and RhsUminus
    auto& local_char_projected_rhs_dt_u_psi =
        get<::Tags::Tempaa<22, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_char_projected_rhs_dt_u_minus =
        get<::Tags::Tempaa<25, VolumeDim, Frame::Inertial, DataVector>>(buffer);

    auto& local_unit_interface_normal_vector =
        get<::Tags::TempA<5, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_lapse = get<::Tags::TempScalar<19, DataVector>>(buffer);
    auto& local_shift =
        get<::Tags::TempI<20, VolumeDim, Frame::Inertial, DataVector>>(buffer);

    auto& local_pi =
        get<GeneralizedHarmonic::Tags::Pi<VolumeDim, frame>>(local_vars);
    auto& local_phi =
        get<GeneralizedHarmonic::Tags::Phi<VolumeDim, frame>>(local_vars);
    auto& local_psi =
        get<gr::Tags::SpacetimeMetric<VolumeDim, frame, DataVector>>(
            local_vars);

    auto& local_char_speeds_0 = get<::Tags::TempScalar<12, DataVector>>(buffer);
    auto& local_char_speeds_1 = get<::Tags::TempScalar<13, DataVector>>(buffer);
    auto& local_char_speeds_2 = get<::Tags::TempScalar<14, DataVector>>(buffer);
    auto& local_char_speeds_3 = get<::Tags::TempScalar<15, DataVector>>(buffer);

    // Setting coords
    for (size_t i = 0; i < VolumeDim; ++i) {
      local_inertial_coords.get(i) = inertial_coords.get(i);
    }
    // Setting constraint_gamma2
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      get(local_constraint_gamma2)[i] = 113.;
    }
    // Setting incoming null one_form: ui
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_incoming_null_one_form.get(0)[i] = -2.;
      local_incoming_null_one_form.get(1)[i] = 5.;
      local_incoming_null_one_form.get(2)[i] = 3.;
      local_incoming_null_one_form.get(3)[i] = 7.;
    }
    // Setting incoming null vector: uI
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_incoming_null_vector.get(0)[i] = -1.;
      local_incoming_null_vector.get(1)[i] = 13.;
      local_incoming_null_vector.get(2)[i] = 17.;
      local_incoming_null_vector.get(3)[i] = 19.;
    }
    // Setting outgoing null one_form: vi
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_outgoing_null_one_form.get(0)[i] = -1.;
      local_outgoing_null_one_form.get(1)[i] = 3.;
      local_outgoing_null_one_form.get(2)[i] = 2.;
      local_outgoing_null_one_form.get(3)[i] = 5.;
    }
    // Setting outgoing null vector: vI
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_outgoing_null_vector.get(0)[i] = -1.;
      local_outgoing_null_vector.get(1)[i] = 2.;
      local_outgoing_null_vector.get(2)[i] = 3.;
      local_outgoing_null_vector.get(3)[i] = 5.;
    }
    // Setting projection Ab
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_projection_Ab.get(0, a)[i] = 233.;
        local_projection_Ab.get(1, a)[i] = 239.;
        local_projection_Ab.get(2, a)[i] = 241.;
        local_projection_Ab.get(3, a)[i] = 251.;
      }
    }
    // Setting projection ab
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_projection_ab.get(0, a)[i] = 379.;
        local_projection_ab.get(1, a)[i] = 383.;
        local_projection_ab.get(2, a)[i] = 389.;
        local_projection_ab.get(3, a)[i] = 397.;
      }
    }
    // Setting projection AB
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_projection_AB.get(0, a)[i] = 353.;
        local_projection_AB.get(1, a)[i] = 359.;
        local_projection_AB.get(2, a)[i] = 367.;
        local_projection_AB.get(3, a)[i] = 373.;
      }
    }
    // Setting local_RhsUPsi
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_char_projected_rhs_dt_u_psi.get(0, a)[i] = 23.;
        local_char_projected_rhs_dt_u_psi.get(1, a)[i] = 29.;
        local_char_projected_rhs_dt_u_psi.get(2, a)[i] = 31.;
        local_char_projected_rhs_dt_u_psi.get(3, a)[i] = 37.;
      }
    }
    // Setting RhsUMinus
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_char_projected_rhs_dt_u_minus.get(0, a)[i] = 331.;
        local_char_projected_rhs_dt_u_minus.get(1, a)[i] = 337.;
        local_char_projected_rhs_dt_u_minus.get(2, a)[i] = 347.;
        local_char_projected_rhs_dt_u_minus.get(3, a)[i] = 349.;
      }
    }

    // Setting unit_interface_normal_Vector
    for (size_t i = 0; i < slice_grid_points; ++i) {
      local_unit_interface_normal_vector.get(0)[i] = 1.e300;
      local_unit_interface_normal_vector.get(1)[i] = -1.;
      local_unit_interface_normal_vector.get(2)[i] = 1.;
      local_unit_interface_normal_vector.get(3)[i] = 1.;
    }
    // Setting pi AND phi
    for (size_t i = 0; i < slice_grid_points; ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = 0; b <= VolumeDim; ++b) {
          local_pi.get(a, b)[i] = 1.;
          local_phi.get(0, a, b)[i] = 3.;
          local_phi.get(1, a, b)[i] = 5.;
          local_phi.get(2, a, b)[i] = 7.;
        }
      }
    }
    // Setting char speeds
    for (size_t i = 0; i < slice_grid_points; ++i) {
      get(local_char_speeds_0)[i] = -0.3;
      get(local_char_speeds_1)[i] = -0.1;
      get(local_char_speeds_3)[i] = -0.2;
    }
    // Setting constraint_char_zero_plus AND constraint_char_zero_minus
    // ONLY ON THE +Y AXIS (Y = +0.5) -- FIXME
    for (size_t i = 0; i < slice_grid_points; ++i) {
      if (get<0>(inertial_coords)[i] == 299. and
          get<1>(inertial_coords)[i] == 0.5 and
          get<2>(inertial_coords)[i] == -0.5) {
        std::array<double, 4> spec_vals = {
            {3722388974.799386, -16680127.68747905, -22991565.68745775,
             -29394198.68743645}};
        std::array<double, 4> spec_vals2 = {
            {3866802572.424386, -19802442.06247905, -19496408.06245775,
             -19257249.06243645}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 299.5 and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == -0.5) {
        std::array<double, 4> spec_vals = {
            {1718287695.133032, -35082292.16184025, -44420849.32723039,
             -53850596.45928719}};
        std::array<double, 4> spec_vals2 = {
            {1866652790.548975, -38170999.90741873, -40892085.07280888,
             -43680040.20486567}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 300. and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == -0.5) {
        std::array<double, 4> spec_vals = {
            {1718299060.415949, -35082089.27527188, -44420613.14790046,
             -53850326.98725116}};
        std::array<double, 4> spec_vals2 = {
            {1866664134.129611, -38170797.39904065, -40891849.27166924,
             -43679771.11101994}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 299. and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.) {
        std::array<double, 4> spec_vals = {
            {1718276282.501221, -35082495.89577083, -44421086.49296889,
             -53850867.05677793}};
        std::array<double, 4> spec_vals2 = {
            {1866641399.709844, -38171203.26157932, -40892321.85877737,
             -43680310.4225864}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 299.5 and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.) {
        std::array<double, 4> spec_vals = {
            {1718287685.660846, -35082292.33093269, -44420849.52407002,
             -53850596.68387395}};
        std::array<double, 4> spec_vals2 = {
            {1866652781.094877, -38171000.07619599, -40892085.26933331,
             -43680040.42913724}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 300. and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.) {
        std::array<double, 4> spec_vals = {
            {1718299050.990844, -35082089.44352208, -44420613.34375966,
             -53850327.21071925}};
        std::array<double, 4> spec_vals2 = {
            {1866664124.722503, -38170797.56697725, -40891849.46721483,
             -43679771.33417442}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 299. and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.5) {
        std::array<double, 4> spec_vals = {
            {1718276292.082241, -35082495.72473276, -44421086.29386414,
             -53850866.82960649}};
        std::array<double, 4> spec_vals2 = {
            {1866641409.272568, -38171203.09086005, -40892321.65999144,
             -43680310.19573379}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 299.5 and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.5) {
        std::array<double, 4> spec_vals = {
            {1718287695.194063, -35082292.16074976, -44420849.32596073,
             -53850596.45783833}};
        std::array<double, 4> spec_vals2 = {
            {1866652790.609889, -38170999.90633029, -40892085.07154126,
             -43680040.20341886}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 300. and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.5) {
        std::array<double, 4> spec_vals = {
            {1718299060.476573, -35082089.27418864, -44420613.14663924,
             -53850326.98581193}};
        std::array<double, 4> spec_vals2 = {
            {1866664134.190119, -38170797.39795944, -40891849.27041004,
             -43679771.10958273}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else {
        // When applying BCs to various faces, only one face (i.e. the +y side)
        // will be set here, others can be set however...
        ASSERT(false,
               "Not checking the correct face, coordinates not recognized");
      }
    }
  }

  // Compute return value from Action
  auto local_bc_dt_u_minus =
      GeneralizedHarmonic::Actions::BoundaryConditions_detail::set_dt_u_minus<
          typename GeneralizedHarmonic::Tags::UMinus<VolumeDim, frame>::type,
          VolumeDim>::
          apply(GeneralizedHarmonic::Actions::BoundaryConditions_detail::
                    UMinusBcMethod::ConstraintPreservingBjorhus,
                buffer, local_vars, local_dt_vars, inertial_coords,
                local_unit_normal_one_form);

  // Initialize with values from SpEC
  auto spec_bc_dt_u_minus = local_bc_dt_u_minus;
  for (size_t i = 0; i < slice_grid_points; ++i) {
    if (get<0>(inertial_coords)[i] == 299. and
        get<1>(inertial_coords)[i] == 0.5 and
        get<2>(inertial_coords)[i] == -0.5) {
      std::array<double, 16> spec_vals = {
          {-22857869555.93848, 330934979077.5689, 242482973593.2169,
           507808643438.4713, 330934979077.5689, 690190606335.8783,
           600780523621.8837, 868984678067.689, 242482973593.2169,
           600780523621.8837, 514052337348.0055, 781537245156.9633,
           507808643438.4713, 868984678067.689, 781537245156.9633,
           1054439575483.625}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 299.5 and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == -0.5) {
      std::array<double, 16> spec_vals = {
          {6085210488.394159, 167380193629.8459, 127052654518.3203,
           248004925243.5957, 167380193629.8459, 332680231531.8318,
           291581334988.8124, 414851930920.4022, 127052654518.3203,
           291581334988.8124, 252051387928.7005, 374742777072.0203,
           248004925243.5957, 414851930920.4022, 374742777072.0203,
           501009779843.9042}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 300. and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == -0.5) {
      std::array<double, 16> spec_vals = {
          {6084984067.712499, 167381093626.4053, 127053272910.3827,
           248006388447.6548, 167381093626.4053, 332682261170.2852,
           291583083105.2007, 414854523601.7003, 127053272910.3827,
           291583083105.2007, 252052859833.8237, 374745093603.8436,
           248006388447.6548, 414854523601.7003, 374745093603.8436,
           501012947925.7576}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 299. and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.) {
      std::array<double, 16> spec_vals = {
          {6085437853.704128, 167379289884.4027, 127052033550.7529,
           248003455943.9015, 167379289884.4027, 332678193437.6569,
           291579579589.7132, 414849327437.366, 127052033550.7529,
           291579579589.7132, 252049909891.7733, 374740450889.0901,
           248003455943.9015, 414849327437.366, 374740450889.0901,
           501006598562.8735}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 299.5 and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.) {
      std::array<double, 16> spec_vals = {
          {6085210677.100476, 167380192879.7604, 127052654002.9328,
           248004924024.1152, 167380192879.7604, 332680229840.2671,
           291581333531.8772, 414851928759.5795, 127052654002.9328,
           291581333531.8772, 252051386701.9684, 374742775141.3494,
           248004924024.1152, 414851928759.5795, 374742775141.3494,
           501009777203.5247}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 300. and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.) {
      std::array<double, 16> spec_vals = {
          {6084984255.479748, 167381092880.0475, 127053272397.5563,
           248006387234.2355, 167381092880.0475, 332682259487.1282,
           291583081655.5068, 414854521451.6181, 127053272397.5563,
           291583081655.5068, 252052858613.1887, 374745091682.769,
           248006387234.2355, 414854521451.6181, 374745091682.769,
           501012945298.5021}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 299. and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.5) {
      std::array<double, 16> spec_vals = {
          {6085437662.827703, 167379290643.1056, 127052034072.0609,
           248003457177.3931, 167379290643.1056, 332678195148.6575,
           291579581063.3884, 414849329623.0162, 127052034072.0609,
           291579581063.3884, 252049911132.6, 374740452841.9442,
           248003457177.3931, 414849329623.0162, 374740452841.9442,
           501006601233.5914}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 299.5 and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.5) {
      std::array<double, 16> spec_vals = {
          {6085210487.177534, 167380193634.6784, 127052654521.6405,
           248004925251.4529, 167380193634.6784, 332680231542.7307,
           291581334998.1996, 414851930934.3248, 127052654521.6405,
           291581334998.1996, 252051387936.6043, 374742777084.4599,
           248004925251.4529, 414851930934.3248, 374742777084.4599,
           501009779860.9169}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 300. and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.5) {
      std::array<double, 16> spec_vals = {
          {6084984066.503965, 167381093631.2057, 127053272913.6808,
           248006388455.4596, 167381093631.2057, 332682261181.1116,
           291583083114.5252, 414854523615.53, 127053272913.6808,
           291583083114.5252, 252052859841.6748, 374745093616.2003,
           248006388455.4596, 414854523615.53, 374745093616.2003,
           501012947942.6568}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else {
      ASSERT(false,
             "Not checking the correct face, coordinates not recognized");
    }
  }

  // Compare values returned by BC action vs those from SpEC
  CHECK_ITERABLE_APPROX(local_bc_dt_u_minus, spec_bc_dt_u_minus);
}

// This function tests the boundary condition imposed on UMinus when
// option `ConstraintPreservingBjorhus` is specified.
template <>
void test_constraint_preserving_bjorhus_u_minus<
    GeneralizedHarmonic::Actions::BoundaryConditions_detail::UMinusBcMethod::
        ConstraintPreservingPhysicalBjorhus>(
    const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) noexcept {
  // Setup grid
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
  const Direction<VolumeDim> direction(1, Side::Upper);  // +y direction
  const size_t slice_grid_points =
      mesh.extents().slice_away(direction.dimension()).product();
  const auto inertial_coords = [&slice_grid_points, &lower_bound]() {
    tnsr::I<DataVector, VolumeDim, frame> tmp(slice_grid_points, 0.);
    // +y direction
    get<1>(tmp) = 0.5;
    for (size_t i = 0; i < VolumeDim; ++i) {
      for (size_t j = 0; j < VolumeDim; ++j) {
        get<0>(tmp)[i * VolumeDim + j] =
            lower_bound[0] + 0.5 * static_cast<double>(i);
        get<2>(tmp)[i * VolumeDim + j] =
            lower_bound[2] + 0.5 * static_cast<double>(j);
      }
    }
    return tmp;
  }();
  //
  // Create a TempTensor that stores all temporaries computed
  // here and elsewhere
  TempBuffer<GeneralizedHarmonic::Actions::BoundaryConditions_detail::
                 all_local_vars<VolumeDim>>
      buffer(slice_grid_points);
  // Also create local variables and their time derivatives, both of which we
  // will populate as needed
  db::item_type<GeneralizedHarmonic::System<VolumeDim>::variables_tag>
      local_vars(slice_grid_points, 0.);
  Variables<db::wrap_tags_in<
      ::Tags::dt, GeneralizedHarmonic::System<VolumeDim>::gradients_tags>>
      local_dt_vars(slice_grid_points, 0.);
  tnsr::i<DataVector, VolumeDim, frame> local_unit_normal_one_form(
      slice_grid_points, 0.);

  {
    // POPULATE various tensors needed to compute BcDtUpsi
    // EXACTLY as done in SpEC
    auto& local_inertial_coords =
        get<::Tags::TempI<37, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_constraint_gamma2 =
        get<::Tags::TempScalar<26, DataVector>>(buffer);
    auto& local_three_index_constraint =
        get<::Tags::Tempiaa<17, VolumeDim, Frame::Inertial, DataVector>>(
            buffer);
    // timelike and spacelike SPACETIME vectors, l^a and k^a
    auto& local_outgoing_null_one_form =
        get<::Tags::Tempa<0, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_incoming_null_one_form =
        get<::Tags::Tempa<1, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    // timelike and spacelike SPACETIME oneforms, l_a and k_a
    auto& local_outgoing_null_vector =
        get<::Tags::TempA<2, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_incoming_null_vector =
        get<::Tags::TempA<3, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    // spacetime projection operator P_ab, P^ab, and P^a_b
    auto& local_projection_AB =
        get<::Tags::TempAA<9, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_projection_ab =
        get<::Tags::Tempaa<10, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_projection_Ab =
        get<::Tags::TempAb<11, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    // constraint characteristics
    auto& local_constraint_char_zero_minus =
        get<::Tags::Tempa<16, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_constraint_char_zero_plus =
        get<::Tags::Tempa<34, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    // RhsUPsi and RhsUminus
    auto& local_char_projected_rhs_dt_u_psi =
        get<::Tags::Tempaa<22, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_char_projected_rhs_dt_u_minus =
        get<::Tags::Tempaa<25, VolumeDim, Frame::Inertial, DataVector>>(buffer);

    auto& local_unit_interface_normal_one_form =
        get<::Tags::Tempa<4, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_unit_interface_normal_vector =
        get<::Tags::TempA<5, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_spacetime_unit_normal_vector =
        get<::Tags::TempA<7, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_lapse = get<::Tags::TempScalar<19, DataVector>>(buffer);
    auto& local_shift =
        get<::Tags::TempI<20, VolumeDim, Frame::Inertial, DataVector>>(buffer);

    auto& local_inverse_spatial_metric =
        get<::Tags::TempII<21, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_extrinsic_curvature =
        get<::Tags::Tempii<35, VolumeDim, Frame::Inertial, DataVector>>(buffer);
    auto& local_inverse_spacetime_metric =
        get<::Tags::TempAA<36, VolumeDim, Frame::Inertial, DataVector>>(buffer);

    auto& local_pi =
        get<GeneralizedHarmonic::Tags::Pi<VolumeDim, frame>>(local_vars);
    auto& local_phi =
        get<GeneralizedHarmonic::Tags::Phi<VolumeDim, frame>>(local_vars);
    auto& local_spacetime_metric =
        get<gr::Tags::SpacetimeMetric<VolumeDim, frame, DataVector>>(
            local_vars);
    auto& local_d_pi =
        get<::Tags::Tempiaa<32, VolumeDim, Frame::Inertial, DataVector>>(
            buffer);
    auto& local_d_phi =
        get<::Tags::Tempijaa<33, VolumeDim, Frame::Inertial, DataVector>>(
            buffer);

    auto& local_char_speeds_0 = get<::Tags::TempScalar<12, DataVector>>(buffer);
    auto& local_char_speeds_1 = get<::Tags::TempScalar<13, DataVector>>(buffer);
    auto& local_char_speeds_2 = get<::Tags::TempScalar<14, DataVector>>(buffer);
    auto& local_char_speeds_3 = get<::Tags::TempScalar<15, DataVector>>(buffer);

    // Setting coords
    for (size_t i = 0; i < VolumeDim; ++i) {
      local_inertial_coords.get(i) = inertial_coords.get(i);
    }
    // Setting local_spacetime_unit_normal_vector
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_spacetime_unit_normal_vector.get(0)[i] = -1.;
      local_spacetime_unit_normal_vector.get(1)[i] = -3.;
      local_spacetime_unit_normal_vector.get(2)[i] = -5.;
      local_spacetime_unit_normal_vector.get(3)[i] = -7.;
    }
    // Setting local_unit_interface_normal_one_form
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_unit_interface_normal_one_form.get(0)[i] = 1.e300;
      local_unit_interface_normal_one_form.get(1)[i] = -1.;
      local_unit_interface_normal_one_form.get(2)[i] = 1.;
      local_unit_interface_normal_one_form.get(3)[i] = 1.;
    }
    // Setting unit_interface_normal_Vector
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_unit_interface_normal_vector.get(0)[i] = 1.e300;
      local_unit_interface_normal_vector.get(1)[i] = -1.;
      local_unit_interface_normal_vector.get(2)[i] = 1.;
      local_unit_interface_normal_vector.get(3)[i] = 1.;
    }
    // Setting lapse
    // for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
    // get(local_lapse)[i] = 2.;
    //}
    // Setting shift
    // for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
    // local_shift.get(0)[i] = 1.;
    // local_shift.get(1)[i] = 2.;
    // local_shift.get(2)[i] = 3.;
    //}
    // Setting local_inverse_spatial_metric
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t j = 0; j < VolumeDim; ++j) {
        local_inverse_spatial_metric.get(0, j)[i] = 41.;
        local_inverse_spatial_metric.get(1, j)[i] = 43.;
        local_inverse_spatial_metric.get(2, j)[i] = 47.;
      }
    }
    // Setting local_spacetime_metric
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_spacetime_metric.get(0, a)[i] = 257.;
        local_spacetime_metric.get(1, a)[i] = 263.;
        local_spacetime_metric.get(2, a)[i] = 269.;
        local_spacetime_metric.get(3, a)[i] = 271.;
      }
    }
    // Setting local_inverse_spacetime_metric
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_inverse_spacetime_metric.get(0, a)[i] =
            -277.;  // needs to be < 0 for lapse
        local_inverse_spacetime_metric.get(1, a)[i] = 281.;
        local_inverse_spacetime_metric.get(2, a)[i] = 283.;
        local_inverse_spacetime_metric.get(3, a)[i] = 293.;
      }
    }
    // Setting local_extrinsic_curvature
    // ONLY ON THE +Y AXIS (Y = +0.5) -- FIXME
    //
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      if (get<0>(local_inertial_coords)[i] == 299. and
          get<1>(local_inertial_coords)[i] == 0.5 and
          get<2>(local_inertial_coords)[i] == -0.5) {
        std::array<double, 9> spec_vals = {
            {200.2198037251189, 266.7930716334918, 333.3663395418648,
             266.7930716334918, 333.3663395418648, 399.9396074502377,
             333.3663395418648, 399.9396074502377, 466.5128753586106}};
        for (size_t j = 0; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            local_extrinsic_curvature.get(j, k)[i] =
                spec_vals[j * (0 + VolumeDim) + k];
          }
        }
      } else if (get<0>(local_inertial_coords)[i] == 299.5 and
                 get<1>(local_inertial_coords)[i] == 0.5 and
                 get<2>(local_inertial_coords)[i] == -0.5) {
        std::array<double, 9> spec_vals = {
            {200.2198037251189, 266.7930716334918, 333.3663395418648,
             266.7930716334918, 333.3663395418648, 399.9396074502377,
             333.3663395418648, 399.9396074502377, 466.5128753586106}};
        for (size_t j = 0; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            local_extrinsic_curvature.get(j, k)[i] =
                spec_vals[j * (0 + VolumeDim) + k];
          }
        }
      } else if (get<0>(local_inertial_coords)[i] == 300. and
                 get<1>(local_inertial_coords)[i] == 0.5 and
                 get<2>(local_inertial_coords)[i] == -0.5) {
        std::array<double, 9> spec_vals = {
            {200.2198037251189, 266.7930716334918, 333.3663395418648,
             266.7930716334918, 333.3663395418648, 399.9396074502377,
             333.3663395418648, 399.9396074502377, 466.5128753586106}};
        for (size_t j = 0; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            local_extrinsic_curvature.get(j, k)[i] =
                spec_vals[j * (0 + VolumeDim) + k];
          }
        }
      } else if (get<0>(local_inertial_coords)[i] == 299. and
                 get<1>(local_inertial_coords)[i] == 0.5 and
                 get<2>(local_inertial_coords)[i] == 0.) {
        std::array<double, 9> spec_vals = {
            {200.2198037251189, 266.7930716334918, 333.3663395418648,
             266.7930716334918, 333.3663395418648, 399.9396074502377,
             333.3663395418648, 399.9396074502377, 466.5128753586106}};
        for (size_t j = 0; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            local_extrinsic_curvature.get(j, k)[i] =
                spec_vals[j * (0 + VolumeDim) + k];
          }
        }
      } else if (get<0>(local_inertial_coords)[i] == 299.5 and
                 get<1>(local_inertial_coords)[i] == 0.5 and
                 get<2>(local_inertial_coords)[i] == 0.) {
        std::array<double, 9> spec_vals = {
            {200.2198037251189, 266.7930716334918, 333.3663395418648,
             266.7930716334918, 333.3663395418648, 399.9396074502377,
             333.3663395418648, 399.9396074502377, 466.5128753586106}};
        for (size_t j = 0; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            local_extrinsic_curvature.get(j, k)[i] =
                spec_vals[j * (0 + VolumeDim) + k];
          }
        }
      } else if (get<0>(local_inertial_coords)[i] == 300. and
                 get<1>(local_inertial_coords)[i] == 0.5 and
                 get<2>(local_inertial_coords)[i] == 0.) {
        std::array<double, 9> spec_vals = {
            {200.2198037251189, 266.7930716334918, 333.3663395418648,
             266.7930716334918, 333.3663395418648, 399.9396074502377,
             333.3663395418648, 399.9396074502377, 466.5128753586106}};
        for (size_t j = 0; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            local_extrinsic_curvature.get(j, k)[i] =
                spec_vals[j * (0 + VolumeDim) + k];
          }
        }
      } else if (get<0>(local_inertial_coords)[i] == 299. and
                 get<1>(local_inertial_coords)[i] == 0.5 and
                 get<2>(local_inertial_coords)[i] == 0.5) {
        std::array<double, 9> spec_vals = {
            {200.2198037251189, 266.7930716334918, 333.3663395418648,
             266.7930716334918, 333.3663395418648, 399.9396074502377,
             333.3663395418648, 399.9396074502377, 466.5128753586106}};
        for (size_t j = 0; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            local_extrinsic_curvature.get(j, k)[i] =
                spec_vals[j * (0 + VolumeDim) + k];
          }
        }
      } else if (get<0>(local_inertial_coords)[i] == 299.5 and
                 get<1>(local_inertial_coords)[i] == 0.5 and
                 get<2>(local_inertial_coords)[i] == 0.5) {
        std::array<double, 9> spec_vals = {
            {200.2198037251189, 266.7930716334918, 333.3663395418648,
             266.7930716334918, 333.3663395418648, 399.9396074502377,
             333.3663395418648, 399.9396074502377, 466.5128753586106}};
        for (size_t j = 0; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            local_extrinsic_curvature.get(j, k)[i] =
                spec_vals[j * (0 + VolumeDim) + k];
          }
        }
      } else if (get<0>(local_inertial_coords)[i] == 300. and
                 get<1>(local_inertial_coords)[i] == 0.5 and
                 get<2>(local_inertial_coords)[i] == 0.5) {
        std::array<double, 9> spec_vals = {
            {200.2198037251189, 266.7930716334918, 333.3663395418648,
             266.7930716334918, 333.3663395418648, 399.9396074502377,
             333.3663395418648, 399.9396074502377, 466.5128753586106}};
        for (size_t j = 0; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            local_extrinsic_curvature.get(j, k)[i] =
                spec_vals[j * (0 + VolumeDim) + k];
          }
        }
      } else {
        // When applying BCs to various faces, only one face (i.e. the +y side)
        // will be set here, others can be set however...
        // FIXME: Disabled
        ASSERT(true,
               "Not checking the correct face, coordinates not recognized");
      }
    }

    // Setting pi AND phi
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = 0; b <= VolumeDim; ++b) {
          // local_pi.get(a, b)[i] = 1.;
          local_phi.get(0, a, b)[i] = 3.;
          local_phi.get(1, a, b)[i] = 5.;
          local_phi.get(2, a, b)[i] = 7.;
        }
      }
    }
    // Setting local_d_phi
    // initialize dPhi (with same values as for SpEC)
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = 0; b <= VolumeDim; ++b) {
          local_d_pi.get(0, a, b)[i] = 1.;
          local_d_phi.get(0, 0, a, b)[i] = 3.;
          local_d_phi.get(0, 1, a, b)[i] = 5.;
          local_d_phi.get(0, 2, a, b)[i] = 7.;
          local_d_pi.get(1, a, b)[i] = 53.;
          local_d_phi.get(1, 0, a, b)[i] = 59.;
          local_d_phi.get(1, 1, a, b)[i] = 61.;
          local_d_phi.get(1, 2, a, b)[i] = 67.;
          local_d_pi.get(2, a, b)[i] = 71.;
          local_d_phi.get(2, 0, a, b)[i] = 73.;
          local_d_phi.get(2, 1, a, b)[i] = 79.;
          local_d_phi.get(2, 2, a, b)[i] = 83.;
        }
      }
    }
    // Setting char speeds
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      get(local_char_speeds_0)[i] = -0.3;
      get(local_char_speeds_1)[i] = -0.1;
      get(local_char_speeds_3)[i] = -0.2;
    }
    // Setting constraint_gamma2
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      get(local_constraint_gamma2)[i] = 113.;
    }
    // Setting incoming null one_form: ui
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_incoming_null_one_form.get(0)[i] = -2.;
      local_incoming_null_one_form.get(1)[i] = 5.;
      local_incoming_null_one_form.get(2)[i] = 3.;
      local_incoming_null_one_form.get(3)[i] = 7.;
    }
    // Setting incoming null vector: uI
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_incoming_null_vector.get(0)[i] = -1.;
      local_incoming_null_vector.get(1)[i] = 13.;
      local_incoming_null_vector.get(2)[i] = 17.;
      local_incoming_null_vector.get(3)[i] = 19.;
    }
    // Setting outgoing null one_form: vi
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_outgoing_null_one_form.get(0)[i] = -1.;
      local_outgoing_null_one_form.get(1)[i] = 3.;
      local_outgoing_null_one_form.get(2)[i] = 2.;
      local_outgoing_null_one_form.get(3)[i] = 5.;
    }
    // Setting outgoing null vector: vI
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_outgoing_null_vector.get(0)[i] = -1.;
      local_outgoing_null_vector.get(1)[i] = 2.;
      local_outgoing_null_vector.get(2)[i] = 3.;
      local_outgoing_null_vector.get(3)[i] = 5.;
    }
    // Setting projection Ab
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_projection_Ab.get(0, a)[i] = 233.;
        local_projection_Ab.get(1, a)[i] = 239.;
        local_projection_Ab.get(2, a)[i] = 241.;
        local_projection_Ab.get(3, a)[i] = 251.;
      }
    }
    // Setting projection ab
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_projection_ab.get(0, a)[i] = 379.;
        local_projection_ab.get(1, a)[i] = 383.;
        local_projection_ab.get(2, a)[i] = 389.;
        local_projection_ab.get(3, a)[i] = 397.;
      }
    }
    // Setting projection AB
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_projection_AB.get(0, a)[i] = 353.;
        local_projection_AB.get(1, a)[i] = 359.;
        local_projection_AB.get(2, a)[i] = 367.;
        local_projection_AB.get(3, a)[i] = 373.;
      }
    }
    // Setting 3idxConstraint
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = 0; b <= VolumeDim; ++b) {
          local_three_index_constraint.get(0, a, b)[i] = 11. - 3.;
          local_three_index_constraint.get(1, a, b)[i] = 13. - 5.;
          local_three_index_constraint.get(2, a, b)[i] = 17. - 7.;
        }
      }
    }
    // Setting local_RhsUPsi
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_char_projected_rhs_dt_u_psi.get(0, a)[i] = 23.;
        local_char_projected_rhs_dt_u_psi.get(1, a)[i] = 29.;
        local_char_projected_rhs_dt_u_psi.get(2, a)[i] = 31.;
        local_char_projected_rhs_dt_u_psi.get(3, a)[i] = 37.;
      }
    }
    // Setting RhsUMinus
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_char_projected_rhs_dt_u_minus.get(0, a)[i] = 331.;
        local_char_projected_rhs_dt_u_minus.get(1, a)[i] = 337.;
        local_char_projected_rhs_dt_u_minus.get(2, a)[i] = 347.;
        local_char_projected_rhs_dt_u_minus.get(3, a)[i] = 349.;
      }
    }
    // Setting constraint_char_zero_plus AND constraint_char_zero_minus
    // ONLY ON THE +Y AXIS (Y = +0.5) -- FIXME
    for (size_t i = 0; i < slice_grid_points; ++i) {
      if (get<0>(inertial_coords)[i] == 299. and
          get<1>(inertial_coords)[i] == 0.5 and
          get<2>(inertial_coords)[i] == -0.5) {
        std::array<double, 4> spec_vals = {
            {3722388974.799386, -16680127.68747905, -22991565.68745775,
             -29394198.68743645}};
        std::array<double, 4> spec_vals2 = {
            {3866802572.424386, -19802442.06247905, -19496408.06245775,
             -19257249.06243645}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 299.5 and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == -0.5) {
        std::array<double, 4> spec_vals = {
            {1718287695.133032, -35082292.16184025, -44420849.32723039,
             -53850596.45928719}};
        std::array<double, 4> spec_vals2 = {
            {1866652790.548975, -38170999.90741873, -40892085.07280888,
             -43680040.20486567}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 300. and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == -0.5) {
        std::array<double, 4> spec_vals = {
            {1718299060.415949, -35082089.27527188, -44420613.14790046,
             -53850326.98725116}};
        std::array<double, 4> spec_vals2 = {
            {1866664134.129611, -38170797.39904065, -40891849.27166924,
             -43679771.11101994}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 299. and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.) {
        std::array<double, 4> spec_vals = {
            {1718276282.501221, -35082495.89577083, -44421086.49296889,
             -53850867.05677793}};
        std::array<double, 4> spec_vals2 = {
            {1866641399.709844, -38171203.26157932, -40892321.85877737,
             -43680310.4225864}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 299.5 and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.) {
        std::array<double, 4> spec_vals = {
            {1718287685.660846, -35082292.33093269, -44420849.52407002,
             -53850596.68387395}};
        std::array<double, 4> spec_vals2 = {
            {1866652781.094877, -38171000.07619599, -40892085.26933331,
             -43680040.42913724}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 300. and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.) {
        std::array<double, 4> spec_vals = {
            {1718299050.990844, -35082089.44352208, -44420613.34375966,
             -53850327.21071925}};
        std::array<double, 4> spec_vals2 = {
            {1866664124.722503, -38170797.56697725, -40891849.46721483,
             -43679771.33417442}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 299. and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.5) {
        std::array<double, 4> spec_vals = {
            {1718276292.082241, -35082495.72473276, -44421086.29386414,
             -53850866.82960649}};
        std::array<double, 4> spec_vals2 = {
            {1866641409.272568, -38171203.09086005, -40892321.65999144,
             -43680310.19573379}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 299.5 and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.5) {
        std::array<double, 4> spec_vals = {
            {1718287695.194063, -35082292.16074976, -44420849.32596073,
             -53850596.45783833}};
        std::array<double, 4> spec_vals2 = {
            {1866652790.609889, -38170999.90633029, -40892085.07154126,
             -43680040.20341886}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 300. and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.5) {
        std::array<double, 4> spec_vals = {
            {1718299060.476573, -35082089.27418864, -44420613.14663924,
             -53850326.98581193}};
        std::array<double, 4> spec_vals2 = {
            {1866664134.190119, -38170797.39795944, -40891849.27041004,
             -43679771.10958273}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else {
        // When applying BCs to various faces, only one face (i.e. the +y side)
        // will be set here, others can be set however...
        ASSERT(false,
               "Not checking the correct face, coordinates not recognized");
      }
    }
  }

  // Compute return value from Action
  auto local_bc_dt_u_minus =
      GeneralizedHarmonic::Actions::BoundaryConditions_detail::set_dt_u_minus<
          typename GeneralizedHarmonic::Tags::UMinus<VolumeDim, frame>::type,
          VolumeDim>::
          apply(GeneralizedHarmonic::Actions::BoundaryConditions_detail::
                    UMinusBcMethod::ConstraintPreservingPhysicalBjorhus,
                buffer, local_vars, local_dt_vars, inertial_coords,
                local_unit_normal_one_form);

  // Initialize with values from SpEC
  auto spec_bc_dt_u_minus = local_bc_dt_u_minus;
  for (size_t i = 0; i < slice_grid_points; ++i) {
    if (get<0>(inertial_coords)[i] == 299. and
        get<1>(inertial_coords)[i] == 0.5 and
        get<2>(inertial_coords)[i] == -0.5) {
      std::array<double, 16> spec_vals = {
          {7.122948293721196e+19, 7.122948329100482e+19, 7.122948320255282e+19,
           7.122948346787851e+19, 7.122948329100482e+19, 7.639071460057165e+19,
           7.639071451116156e+19, 7.639071477936572e+19, 7.122948320255282e+19,
           7.639071451116156e+19, 8.413256084990011e+19, 8.413256111738503e+19,
           7.122948346787851e+19, 7.639071477936572e+19, 8.413256111738503e+19,
           9.445502329090969e+19}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 299.5 and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == -0.5) {
      std::array<double, 16> spec_vals = {
          {7.122948296615504e+19, 7.122948312745003e+19, 7.122948308712251e+19,
           7.122948320807479e+19, 7.122948312745003e+19, 7.639071424306128e+19,
           7.639071420196236e+19, 7.639071432523298e+19, 7.122948308712251e+19,
           7.639071420196236e+19, 8.413256058789916e+19, 8.413256071059056e+19,
           7.122948320807479e+19, 7.639071432523298e+19, 8.413256071059056e+19,
           9.445502273747989e+19}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 300. and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == -0.5) {
      std::array<double, 16> spec_vals = {
          {7.122948296615481e+19, 7.122948312745094e+19, 7.122948308712312e+19,
           7.122948320807626e+19, 7.122948312745094e+19, 7.639071424306332e+19,
           7.639071420196412e+19, 7.639071432523558e+19, 7.122948308712312e+19,
           7.639071420196412e+19, 8.413256058790063e+19, 8.413256071059287e+19,
           7.122948320807626e+19, 7.639071432523558e+19, 8.413256071059287e+19,
           9.445502273748306e+19}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 299. and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.) {
      std::array<double, 16> spec_vals = {
          {7.122948296615527e+19, 7.122948312744913e+19, 7.122948308712188e+19,
           7.122948320807332e+19, 7.122948312744913e+19, 7.639071424305924e+19,
           7.639071420196061e+19, 7.639071432523037e+19, 7.122948308712188e+19,
           7.639071420196061e+19, 8.413256058789768e+19, 8.413256071058824e+19,
           7.122948320807332e+19, 7.639071432523037e+19, 8.413256071058824e+19,
           9.445502273747671e+19}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 299.5 and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.) {
      std::array<double, 16> spec_vals = {
          {7.122948296615504e+19, 7.122948312745003e+19, 7.122948308712251e+19,
           7.122948320807479e+19, 7.122948312745003e+19, 7.639071424306128e+19,
           7.639071420196236e+19, 7.639071432523298e+19, 7.122948308712251e+19,
           7.639071420196236e+19, 8.413256058789916e+19, 8.413256071059056e+19,
           7.122948320807479e+19, 7.639071432523298e+19, 8.413256071059056e+19,
           9.445502273747989e+19}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 300. and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.) {
      std::array<double, 16> spec_vals = {
          {7.122948296615482e+19, 7.122948312745094e+19, 7.122948308712312e+19,
           7.122948320807626e+19, 7.122948312745094e+19, 7.63907142430633e+19,
           7.63907142019641e+19, 7.639071432523556e+19, 7.122948308712312e+19,
           7.63907142019641e+19, 8.413256058790063e+19, 8.413256071059287e+19,
           7.122948320807626e+19, 7.639071432523556e+19, 8.413256071059287e+19,
           9.445502273748306e+19}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 299. and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.5) {
      std::array<double, 16> spec_vals = {
          {7.122948296615527e+19, 7.122948312744913e+19, 7.122948308712188e+19,
           7.122948320807332e+19, 7.122948312744913e+19, 7.639071424305924e+19,
           7.639071420196061e+19, 7.639071432523039e+19, 7.122948308712188e+19,
           7.639071420196061e+19, 8.413256058789768e+19, 8.413256071058824e+19,
           7.122948320807332e+19, 7.639071432523039e+19, 8.413256071058824e+19,
           9.445502273747671e+19}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 299.5 and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.5) {
      std::array<double, 16> spec_vals = {
          {7.122948296615504e+19, 7.122948312745003e+19, 7.122948308712251e+19,
           7.122948320807479e+19, 7.122948312745003e+19, 7.639071424306128e+19,
           7.639071420196236e+19, 7.639071432523298e+19, 7.122948308712251e+19,
           7.639071420196236e+19, 8.413256058789916e+19, 8.413256071059056e+19,
           7.122948320807479e+19, 7.639071432523298e+19, 8.413256071059056e+19,
           9.445502273747989e+19}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 300. and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.5) {
      std::array<double, 16> spec_vals = {
          {7.122948296615481e+19, 7.122948312745094e+19, 7.122948308712312e+19,
           7.122948320807626e+19, 7.122948312745094e+19, 7.639071424306332e+19,
           7.639071420196412e+19, 7.639071432523558e+19, 7.122948308712312e+19,
           7.639071420196412e+19, 8.413256058790063e+19, 8.413256071059287e+19,
           7.122948320807626e+19, 7.639071432523558e+19, 8.413256071059287e+19,
           9.445502273748306e+19}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else {
      ASSERT(false,
             "Not checking the correct face, coordinates not recognized");
    }
  }

  // Compare values returned by BC action vs those from SpEC
  CHECK_ITERABLE_APPROX(local_bc_dt_u_minus, spec_bc_dt_u_minus);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.BoundaryConditions.UPsi",
    "[Unit][Evolution]") {
  // Piece-wise tests with SpEC output
  const size_t grid_size = 3;
  const std::array<double, 3> lower_bound{{299., -0.5, -0.5}};
  const std::array<double, 3> upper_bound{{300., 0.5, 0.5}};

  test_constraint_preserving_bjorhus_u_psi(grid_size, lower_bound, upper_bound);
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.BoundaryConditions.UZero",
    "[Unit][Evolution]") {
  // Piece-wise tests with SpEC output
  const size_t grid_size = 3;
  const std::array<double, 3> lower_bound{{299., -0.5, -0.5}};
  const std::array<double, 3> upper_bound{{300., 0.5, 0.5}};

  test_constraint_preserving_bjorhus_u_zero(grid_size, lower_bound,
                                            upper_bound);
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.BoundaryConditions.UMinus",
    "[Unit][Evolution]") {
  // Piece-wise tests with SpEC output
  const size_t grid_size = 3;
  const std::array<double, 3> lower_bound{{299., -0.5, -0.5}};
  const std::array<double, 3> upper_bound{{300., 0.5, 0.5}};

  test_constraint_preserving_bjorhus_u_minus<
      GeneralizedHarmonic::Actions::BoundaryConditions_detail::UMinusBcMethod::
          Freezing>(grid_size, lower_bound, upper_bound);
  test_constraint_preserving_bjorhus_u_minus<
      GeneralizedHarmonic::Actions::BoundaryConditions_detail::UMinusBcMethod::
          ConstraintPreservingBjorhus>(grid_size, lower_bound, upper_bound);
  test_constraint_preserving_bjorhus_u_minus<
      GeneralizedHarmonic::Actions::BoundaryConditions_detail::UMinusBcMethod::
          ConstraintPreservingPhysicalBjorhus>(grid_size, lower_bound,
                                               upper_bound);
}
