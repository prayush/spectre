// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Constraints.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace {
// Test the return-by-value four-index constraint function using random values
template <size_t Dim>
void test_one_index_constraint_random(
    const DataVector& used_for_size) noexcept {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::i<DataVector, Dim, Frame::Inertial> (*)(
          const tnsr::i<DataVector, Dim, Frame::Inertial>&,
          const tnsr::i<DataVector, Dim, Frame::Inertial>&)>(
          &CurvedScalarWave::one_index_constraint<Dim>),
      "Constraints", "one_index_constraint", {{{-10.0, 10.0}}}, used_for_size);
}

// Test the return-by-value two-index constraint function using random values
template <size_t Dim>
void test_two_index_constraint_random(
    const DataVector& used_for_size) noexcept {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::ij<DataVector, Dim, Frame::Inertial> (*)(
          const tnsr::ij<DataVector, Dim, Frame::Inertial>&)>(
          &CurvedScalarWave::two_index_constraint<Dim>),
      "Constraints", "two_index_constraint", {{{-10.0, 10.0}}}, used_for_size,
      1.0e-12);
}

// Test compute items for various constraints via insertion and retrieval
// in a databox
template <size_t Dim>
void test_constraint_compute_items(const DataVector& used_for_size) noexcept {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.0, 1.0);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  // Randomized tensors
  const auto phi = make_with_random_values<tnsr::i<DataVector, Dim>>(
      nn_generator, nn_distribution, used_for_size);
  const auto d_phi = make_with_random_values<tnsr::ij<DataVector, Dim>>(
      nn_generator, nn_distribution, used_for_size);
  const auto d_psi = make_with_random_values<tnsr::i<DataVector, Dim>>(
      nn_generator, nn_distribution, used_for_size);
  // Insert into databox
  const auto box = db::create<
      db::AddSimpleTags<CurvedScalarWave::Phi<Dim>,
                        ::Tags::deriv<CurvedScalarWave::Phi<Dim>,
                                      tmpl::size_t<Dim>, Frame::Inertial>,
                        ::Tags::deriv<CurvedScalarWave::Psi, tmpl::size_t<Dim>,
                                      Frame::Inertial>>,
      db::AddComputeTags<
          CurvedScalarWave::Tags::OneIndexConstraintCompute<Dim>,
          CurvedScalarWave::Tags::TwoIndexConstraintCompute<Dim>>>(phi, d_phi,
                                                                   d_psi);

  // Check compute tag against locally computed quantities
  CHECK(db::get<CurvedScalarWave::Tags::OneIndexConstraint<Dim>>(box) ==
        CurvedScalarWave::one_index_constraint(d_psi, phi));
  CHECK(db::get<CurvedScalarWave::Tags::TwoIndexConstraint<Dim>>(box) ==
        CurvedScalarWave::two_index_constraint(d_phi));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.CurvedScalarWave.Constraints",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/CurvedScalarWave/"};
  const auto used_for_size =
      DataVector(4, std::numeric_limits<double>::signaling_NaN());
  // Test the one-index constraint with random numbers
  test_one_index_constraint_random<1>(used_for_size);
  test_one_index_constraint_random<2>(used_for_size);
  test_one_index_constraint_random<3>(used_for_size);

  // Test the two-index constraint with random numbers
  test_two_index_constraint_random<1>(used_for_size);
  test_two_index_constraint_random<2>(used_for_size);
  test_two_index_constraint_random<3>(used_for_size);

  // Compute items
  test_constraint_compute_items<1>(used_for_size);
  test_constraint_compute_items<2>(used_for_size);
  test_constraint_compute_items<3>(used_for_size);
}
