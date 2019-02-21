// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace {
template <size_t Dim, IndexType Index, typename DataType>
void test_christoffel(const DataType& used_for_size) {
  tnsr::abb<DataType, Dim, Frame::Inertial, Index> (*f)(
      const tnsr::abb<DataType, Dim, Frame::Inertial, Index>&) =
      &gr::christoffel_first_kind<Dim, Frame::Inertial, Index, DataType>;
  pypp::check_with_random_values<1>(f, "TestFunctions",
                                    "christoffel_first_kind", {{{-10.0, 10.0}}},
                                    used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.Christoffel.",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");
  const DataVector dv(5);
  test_christoffel<1, IndexType::Spatial>(dv);
  test_christoffel<2, IndexType::Spatial>(dv);
  test_christoffel<3, IndexType::Spatial>(dv);
  test_christoffel<1, IndexType::Spacetime>(dv);
  test_christoffel<2, IndexType::Spacetime>(dv);
  test_christoffel<3, IndexType::Spacetime>(dv);
  test_christoffel<1, IndexType::Spatial>(0.);
  test_christoffel<2, IndexType::Spatial>(0.);
  test_christoffel<3, IndexType::Spatial>(0.);
  test_christoffel<1, IndexType::Spacetime>(0.);
  test_christoffel<2, IndexType::Spacetime>(0.);
  test_christoffel<3, IndexType::Spacetime>(0.);

  // Check that compute items work correctly in the DataBox
  // First, check that the names are correct
  CHECK(gr::Tags::SpacetimeChristoffelFirstKindCompute<3, Frame::Inertial,
                                                       DataVector>::name() ==
        "SpacetimeChristoffelFirstKind");
  CHECK(gr::Tags::SpacetimeChristoffelSecondKindCompute<3, Frame::Inertial,
                                                        DataVector>::name() ==
        "SpacetimeChristoffelSecondKind");
  CHECK(
      gr::Tags::TraceSpacetimeChristoffelFirstKindCompute<3, Frame::Inertial,
                                                          DataVector>::name() ==
      "TraceSpacetimeChristoffelFirstKind");
}
