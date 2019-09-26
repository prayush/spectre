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
// option `Freezing` is specified. Only the gauge portion of the RHS is set.
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
    auto& local_char_projected_rhs_dt_u_zero =
        get<::Tags::Tempiaa<23, VolumeDim, Frame::Inertial, DataVector>>(
            buffer);
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
    // Setting constraint_char_zero_plus AND constraint_char_zero_minus
    // ONLY ON THE +Y AXIS (Y = +0.5) -- FIXME
    //
    for (size_t i = 0; i < slice_grid_points; ++i) {
      if (get<0>(inertial_coords)[i] == 299. and
          get<1>(inertial_coords)[i] == 0.5 and
          get<2>(inertial_coords)[i] == -0.5) {
        std::array<double, 4> spec_vals = {
            {116825669851.0523, -61312353.06244799, -27701645.812406,
             5420092.437635988}};
        std::array<double, 4> spec_vals2 = {
            {114778477915.6773, -238132756.937448, -325016009.687406,
             -412576643.437364}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 299.5 and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == -0.5) {
        std::array<double, 4> spec_vals = {
            {57912013103.59233, -210146467.5227862, -203382066.4200172,
             -197106629.2839151}};
        std::array<double, 4> spec_vals2 = {
            {55800195748.29776, -387039486.9857526, -500769045.8829836,
             -615175980.7468815}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 300. and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == -0.5) {
        std::array<double, 4> spec_vals = {
            {57912339733.04692, -210144827.9955755, -203380131.5377951,
             -197104399.0467367}};
        std::array<double, 4> spec_vals2 = {
            {55800522732.56735, -387037846.6557527, -500767110.1979722,
             -615173749.7069137}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 299. and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.) {
        std::array<double, 4> spec_vals = {
            {57911685113.52791, -210148113.8975043, -203384009.3832927,
             -197108868.8356919}};
        std::array<double, 4> spec_vals2 = {
            {55799867401.9404, -387041134.166613, -500770989.6524013,
             -615178221.1048005}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 299.5 and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.) {
        std::array<double, 4> spec_vals = {
            {57912012831.36965, -210146468.8892238, -203382068.0326135,
             -197106631.1426699}};
        std::array<double, 4> spec_vals2 = {
            {55800195475.77937, -387039488.3528592, -500769047.496249,
             -615175982.6063054}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 300. and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.) {
        std::array<double, 4> spec_vals = {
            {57912339462.17721, -210144829.3552071, -203380133.1423593,
             -197104400.8962334}};
        std::array<double, 4> spec_vals2 = {
            {55800522461.40339, -387037848.01605, -500767111.8032022,
             -615173751.5570762}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 299. and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.5) {
        std::array<double, 4> spec_vals = {
            {57911685388.87936, -210148112.5153456, -203384007.7521427,
             -197108866.9555508}};
        std::array<double, 4> spec_vals2 = {
            {55799867677.59096, -387041132.7837775, -500770988.0205746,
             -615178219.2239828}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 299.5 and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.5) {
        std::array<double, 4> spec_vals = {
            {57912013105.34742, -210146467.5139756, -203382066.409619,
             -197106629.2719292}};
        std::array<double, 4> spec_vals2 = {
            {55800195750.05476, -387039486.9769377, -500769045.8725812,
             -615175980.7348914}};
        for (size_t j = 0; j <= VolumeDim; ++j) {
          local_constraint_char_zero_plus.get(j)[i] = spec_vals[j];
          local_constraint_char_zero_minus.get(j)[i] = spec_vals2[j];
        }
      } else if (get<0>(inertial_coords)[i] == 300. and
                 get<1>(inertial_coords)[i] == 0.5 and
                 get<2>(inertial_coords)[i] == 0.5) {
        std::array<double, 4> spec_vals = {
            {57912339734.79037, -210144827.9868235, -203380131.527466,
             -197104399.0348305}};
        std::array<double, 4> spec_vals2 = {
            {55800522734.31269, -387037846.6469963, -500767110.1876388,
             -615173749.6950033}};
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
          {-810740692231.075, 9764923189268.199, 7121003425567.404,
           15052732370061.98, 9764923189268.199, 20492875943018.07,
           17822714727856.17, 25833172279645.71, 7121003425567.404,
           17822714727856.17, 15230095147742.09, 23220870922221.19,
           15052732370061.98, 25833172279645.71, 23220870922221.19,
           31365336904638.61}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 299.5 and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == -0.5) {
      std::array<double, 16> spec_vals = {
          {-204717553113.3578, 4832692441974.306, 3573336149876.226,
           7351374679561.159, 4832692441974.306, 9959370979624.391,
           8683814297196.055, 12510458250783.6, 3573336149876.226,
           8683814297196.055, 7449004856785.542, 11263497828910.74,
           7351374679561.159, 12510458250783.6, 11263497828910.74,
           15163376451100.38}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 300. and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == -0.5) {
      std::array<double, 16> spec_vals = {
          {-204721437273.7699, 4832719554092.487, 3573355512924.573,
           7351417289817.52, 4832719554092.487, 9959429395007.293,
           8683864917436.745, 12510532256449.63, 3573355512924.573,
           8683864917436.745, 7449047877446.927, 11263564200443.15,
           7351417289817.52, 12510532256449.63, 11263564200443.15,
           15163466522559.97}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 299. and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.) {
      std::array<double, 16> spec_vals = {
          {-204713652761.5526, 4832665216922.821, 3573316706175.753,
           7351331891809.161, 4832665216922.821, 9959312320905.646,
           8683763466092.394, 12510383936835.98, 3573316706175.753,
           8683763466092.394, 7448961656919.618, 11263431180898.95,
           7351331891809.161, 12510383936835.98, 11263431180898.95,
           15163286004433.16}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 299.5 and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.) {
      std::array<double, 16> spec_vals = {
          {-204717549876.1789, 4832692419378.265, 3573336133738.49,
           7351374644048.508, 4832692419378.265, 9959370930939.275,
           8683814255007.643, 12510458189105.08, 3573336133738.49,
           8683814255007.643, 7449004820930.844, 11263497773594.74,
           7351374644048.508, 12510458189105.08, 11263497773594.74,
           15163376376032.15}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 300. and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.) {
      std::array<double, 16> spec_vals = {
          {-204721434052.6894, 4832719531608.745, 3573355496867.036,
           7351417254481.368, 4832719531608.745, 9959429346564.145,
           8683864875458.013, 12510532195077.65, 3573355496867.036,
           8683864875458.013, 7449047841770.428, 11263564145402.07,
           7351417254481.368, 12510532195077.65, 11263564145402.07,
           15163466447864.83}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 299. and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.5) {
      std::array<double, 16> spec_vals = {
          {-204713656035.9483, 4832665239778.568, 3573316722498.961,
           7351331927729.974, 4832665239778.568, 9959312370150.318,
           8683763508765.691, 12510383999223.39, 3573316722498.961,
           8683763508765.691, 7448961693186.402, 11263431236850.72,
           7351331927729.974, 12510383999223.39, 11263431236850.72,
           15163286080364.18}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 299.5 and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.5) {
      std::array<double, 16> spec_vals = {
          {-204717553134.2295, 4832692442119.987, 3573336149980.27,
           7351374679790.118, 4832692442119.987, 9959370979938.275,
           8683814297468.053, 12510458251181.26, 3573336149980.27,
           8683814297468.053, 7449004857016.705, 11263497829267.38,
           7351374679790.118, 12510458251181.26, 11263497829267.38,
           15163376451584.36}};
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_u_minus.get(a, b)[i] = spec_vals[a * (1 + VolumeDim) + b];
        }
      }
    } else if (get<0>(inertial_coords)[i] == 300. and
               get<1>(inertial_coords)[i] == 0.5 and
               get<2>(inertial_coords)[i] == 0.5) {
      std::array<double, 16> spec_vals = {
          {-204721437294.5031, 4832719554237.202, 3573355513027.926,
           7351417290044.959, 4832719554237.202, 9959429395319.094,
           8683864917706.94, 12510532256844.65, 3573355513027.926,
           8683864917706.94, 7449047877676.557, 11263564200797.42,
           7351417290044.959, 12510532256844.65, 11263564200797.42,
           15163466523040.74}};
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
}
