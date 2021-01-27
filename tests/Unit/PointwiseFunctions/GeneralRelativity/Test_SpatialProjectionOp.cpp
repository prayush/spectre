// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialProjectionOp.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"

namespace {
template <size_t SpatialDim, typename DataType>
void test_spatial_projection_operator(const DataType& used_for_size) {
  {
    tnsr::II<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::II<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::A<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::spatial_projection_tensor<SpatialDim, Frame::Inertial, DataType>;
    pypp::check_with_random_values<1>(f, "SpatialProjectionOp",
                                      "spatial_projection_tensor",
                                      {{{-1., 1.}}}, used_for_size);
  }

  {
    tnsr::II<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::II<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::I<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::spatial_projection_tensor<SpatialDim, Frame::Inertial, DataType>;
    pypp::check_with_random_values<1>(f, "SpatialProjectionOp",
                                      "spatial_projection_tensor",
                                      {{{-1., 1.}}}, used_for_size);
  }

  {
    tnsr::ii<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::ii<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::a<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::spatial_projection_tensor<SpatialDim, Frame::Inertial, DataType>;
    pypp::check_with_random_values<1>(f, "SpatialProjectionOp",
                                      "spatial_projection_tensor",
                                      {{{-1., 1.}}}, used_for_size);
  }

  {
    tnsr::ii<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::ii<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::i<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::spatial_projection_tensor<SpatialDim, Frame::Inertial, DataType>;
    pypp::check_with_random_values<1>(f, "SpatialProjectionOp",
                                      "spatial_projection_tensor",
                                      {{{-1., 1.}}}, used_for_size);
  }

  {
    tnsr::Ij<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::A<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::a<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::spatial_projection_tensor<SpatialDim, Frame::Inertial, DataType>;
    pypp::check_with_random_values<1>(
        f, "SpatialProjectionOp",
        "spatial_projection_tensor_mixed_from_spacetime_input", {{{-1., 1.}}},
        used_for_size);
  }

  {
    tnsr::Ij<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::I<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::i<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::spatial_projection_tensor<SpatialDim, Frame::Inertial, DataType>;
    pypp::check_with_random_values<1>(
        f, "SpatialProjectionOp",
        "spatial_projection_tensor_mixed_from_spatial_input", {{{-1., 1.}}},
        used_for_size);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.SpatialProjOp",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_spatial_projection_operator,
                                    (1, 2, 3));
}
