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
  auto spec_bd_dt_u_zero = local_bc_dt_u_zero;
  for (size_t i = 0; i < slice_grid_points; ++i) {
    get<0, 0, 0>(spec_bd_dt_u_zero)[i] = 79.;
    get<0, 0, 1>(spec_bd_dt_u_zero)[i] = 79.;
    get<0, 0, 2>(spec_bd_dt_u_zero)[i] = 79.;
    get<0, 0, 3>(spec_bd_dt_u_zero)[i] = 79.;
    get<1, 0, 0>(spec_bd_dt_u_zero)[i] = 90.4;
    get<1, 0, 1>(spec_bd_dt_u_zero)[i] = 90.4;
    get<1, 0, 2>(spec_bd_dt_u_zero)[i] = 90.4;
    get<1, 0, 3>(spec_bd_dt_u_zero)[i] = 90.4;
    get<2, 0, 3>(spec_bd_dt_u_zero)[i] = 95.6;
    get<2, 0, 0>(spec_bd_dt_u_zero)[i] = 95.6;
    get<2, 0, 1>(spec_bd_dt_u_zero)[i] = 95.6;
    get<2, 0, 2>(spec_bd_dt_u_zero)[i] = 95.6;

    get<0, 1, 1>(spec_bd_dt_u_zero)[i] = 79.;
    get<0, 1, 2>(spec_bd_dt_u_zero)[i] = 79.;
    get<0, 1, 3>(spec_bd_dt_u_zero)[i] = 79.;
    get<1, 1, 1>(spec_bd_dt_u_zero)[i] = 90.4;
    get<1, 1, 2>(spec_bd_dt_u_zero)[i] = 90.4;
    get<1, 1, 3>(spec_bd_dt_u_zero)[i] = 90.4;
    get<2, 1, 1>(spec_bd_dt_u_zero)[i] = 95.6;
    get<2, 1, 2>(spec_bd_dt_u_zero)[i] = 95.6;
    get<2, 1, 3>(spec_bd_dt_u_zero)[i] = 95.6;

    get<0, 2, 2>(spec_bd_dt_u_zero)[i] = 79.;
    get<0, 2, 3>(spec_bd_dt_u_zero)[i] = 79.;
    get<1, 2, 2>(spec_bd_dt_u_zero)[i] = 90.4;
    get<1, 2, 3>(spec_bd_dt_u_zero)[i] = 90.4;
    get<2, 2, 2>(spec_bd_dt_u_zero)[i] = 95.6;
    get<2, 2, 3>(spec_bd_dt_u_zero)[i] = 95.6;

    get<0, 3, 3>(spec_bd_dt_u_zero)[i] = 79.;
    get<1, 3, 3>(spec_bd_dt_u_zero)[i] = 90.4;
    get<2, 3, 3>(spec_bd_dt_u_zero)[i] = 95.6;
  }

  // Compare values returned by BC action vs those from SpEC
  CHECK_ITERABLE_APPROX(local_bc_dt_u_zero, spec_bd_dt_u_zero);

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
    get<0, 0, 0>(spec_bd_dt_u_zero)[i] = 91.;
    get<0, 0, 1>(spec_bd_dt_u_zero)[i] = 91.;
    get<0, 0, 2>(spec_bd_dt_u_zero)[i] = 91.;
    get<0, 0, 3>(spec_bd_dt_u_zero)[i] = 91.;
    get<1, 0, 0>(spec_bd_dt_u_zero)[i] = 91.6;
    get<1, 0, 1>(spec_bd_dt_u_zero)[i] = 91.6;
    get<1, 0, 2>(spec_bd_dt_u_zero)[i] = 91.6;
    get<1, 0, 3>(spec_bd_dt_u_zero)[i] = 91.6;
    get<2, 0, 3>(spec_bd_dt_u_zero)[i] = 94.4;
    get<2, 0, 0>(spec_bd_dt_u_zero)[i] = 94.4;
    get<2, 0, 1>(spec_bd_dt_u_zero)[i] = 94.4;
    get<2, 0, 2>(spec_bd_dt_u_zero)[i] = 94.4;

    get<0, 1, 1>(spec_bd_dt_u_zero)[i] = 91.;
    get<0, 1, 2>(spec_bd_dt_u_zero)[i] = 91.;
    get<0, 1, 3>(spec_bd_dt_u_zero)[i] = 91.;
    get<1, 1, 1>(spec_bd_dt_u_zero)[i] = 91.6;
    get<1, 1, 2>(spec_bd_dt_u_zero)[i] = 91.6;
    get<1, 1, 3>(spec_bd_dt_u_zero)[i] = 91.6;
    get<2, 1, 1>(spec_bd_dt_u_zero)[i] = 94.4;
    get<2, 1, 2>(spec_bd_dt_u_zero)[i] = 94.4;
    get<2, 1, 3>(spec_bd_dt_u_zero)[i] = 94.4;

    get<0, 2, 2>(spec_bd_dt_u_zero)[i] = 91.;
    get<0, 2, 3>(spec_bd_dt_u_zero)[i] = 91.;
    get<1, 2, 2>(spec_bd_dt_u_zero)[i] = 91.6;
    get<1, 2, 3>(spec_bd_dt_u_zero)[i] = 91.6;
    get<2, 2, 2>(spec_bd_dt_u_zero)[i] = 94.4;
    get<2, 2, 3>(spec_bd_dt_u_zero)[i] = 94.4;

    get<0, 3, 3>(spec_bd_dt_u_zero)[i] = 91.;
    get<1, 3, 3>(spec_bd_dt_u_zero)[i] = 91.6;
    get<2, 3, 3>(spec_bd_dt_u_zero)[i] = 94.4;
  }

  // Compare values returned by BC action vs those from SpEC
  CHECK_ITERABLE_APPROX(local_bc_dt_u_zero, spec_bd_dt_u_zero);
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
