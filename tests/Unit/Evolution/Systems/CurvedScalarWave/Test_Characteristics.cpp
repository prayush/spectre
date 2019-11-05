// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Characteristics.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace {
template <size_t Index, size_t Dim>
Scalar<DataVector> speed_with_index(
    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal) {
  return Scalar<DataVector>{
      CurvedScalarWave::CharacteristicSpeedsCompute<Dim>::function(
          gamma_1, lapse, shift, normal)[Index]};
}

template <size_t Dim>
void test_characteristic_speeds() noexcept {
  const DataVector used_for_size(5);
  pypp::check_with_random_values<1>(speed_with_index<0, Dim>, "Characteristics",
                                    "char_speed_upsi", {{{-10.0, 10.0}}},
                                    used_for_size);
  pypp::check_with_random_values<1>(speed_with_index<1, Dim>, "Characteristics",
                                    "char_speed_uzero", {{{-10.0, 10.0}}},
                                    used_for_size);
  pypp::check_with_random_values<1>(speed_with_index<3, Dim>, "Characteristics",
                                    "char_speed_uminus", {{{-10.0, 10.0}}},
                                    used_for_size);
  pypp::check_with_random_values<1>(speed_with_index<2, Dim>, "Characteristics",
                                    "char_speed_uplus", {{{-10.0, 10.0}}},
                                    used_for_size);
}
}  // namespace

namespace {
template <typename Tag, size_t Dim>
typename Tag::type field_with_tag(
    const Scalar<DataVector>& gamma_2,
    const tnsr::II<DataVector, Dim, Frame::Inertial>& inverse_spatial_metric,
    const Scalar<DataVector>& psi, const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_one_form) {
  return get<Tag>(CurvedScalarWave::CharacteristicFieldsCompute<Dim>::function(
      gamma_2, inverse_spatial_metric, psi, pi, phi, normal_one_form));
}

template <size_t Dim>
void test_characteristic_fields() noexcept {
  const DataVector used_for_size(20);
  // UPsi
  pypp::check_with_random_values<1>(
      field_with_tag<CurvedScalarWave::Tags::UPsi, Dim>, "Characteristics",
      "char_field_upsi", {{{-100., 100.}}}, used_for_size);
  // UZero
  pypp::check_with_random_values<1>(
      field_with_tag<CurvedScalarWave::Tags::UZero<Dim>, Dim>,
      "Characteristics", "char_field_uzero", {{{-100., 100.}}}, used_for_size);
  // UPlus
  pypp::check_with_random_values<1>(
      field_with_tag<CurvedScalarWave::Tags::UPlus, Dim>, "Characteristics",
      "char_field_uplus", {{{-100., 100.}}}, used_for_size);
  // UMinus
  pypp::check_with_random_values<1>(
      field_with_tag<CurvedScalarWave::Tags::UMinus, Dim>, "Characteristics",
      "char_field_uminus", {{{-100., 100.}}}, used_for_size);
}
}  // namespace

namespace {
template <typename Tag, size_t Dim>
typename Tag::type evol_field_with_tag(
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& u_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& u_zero,
    const Scalar<DataVector>& u_plus, const Scalar<DataVector>& u_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_one_form) {
  return get<Tag>(
      CurvedScalarWave::EvolvedFieldsFromCharacteristicFieldsCompute<
          Dim>::function(gamma_2, u_psi, u_zero, u_plus, u_minus,
                         normal_one_form));
}

template <size_t Dim>
void test_evolved_from_characteristic_fields() noexcept {
  const DataVector used_for_size(20);
  // Psi
  pypp::check_with_random_values<1>(
      evol_field_with_tag<CurvedScalarWave::Psi, Dim>, "Characteristics",
      "evol_field_psi", {{{-100., 100.}}}, used_for_size);
  // Pi
  pypp::check_with_random_values<1>(
      evol_field_with_tag<CurvedScalarWave::Pi, Dim>, "Characteristics",
      "evol_field_pi", {{{-100., 100.}}}, used_for_size);
  // Phi
  pypp::check_with_random_values<1>(
      evol_field_with_tag<CurvedScalarWave::Phi<Dim>, Dim>, "Characteristics",
      "evol_field_phi", {{{-100., 100.}}}, used_for_size);
}

template <size_t Dim>
void test_characteristics_compute_tags() noexcept {
  const DataVector used_for_size(20);

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.0, 1.0);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  // Randomized tensors
  const auto gamma_1 = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);
  const auto gamma_2 = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);
  const auto shift =
      make_with_random_values<tnsr::I<DataVector, Dim, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size);
  const auto inverse_spatial_metric =
      make_with_random_values<tnsr::II<DataVector, Dim, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size);
  const auto psi = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);
  const auto pi = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);
  const auto phi =
      make_with_random_values<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size);
  const auto normal =
      make_with_random_values<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size);

  // Insert into databox
  const auto box = db::create<
      db::AddSimpleTags<
          CurvedScalarWave::Tags::ConstraintGamma1,
          CurvedScalarWave::Tags::ConstraintGamma2, gr::Tags::Lapse<DataVector>,
          gr::Tags::Shift<Dim, Frame::Inertial, DataVector>,
          gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>,
          CurvedScalarWave::Psi, CurvedScalarWave::Pi,
          CurvedScalarWave::Phi<Dim>,
          ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim>>>,
      db::AddComputeTags<CurvedScalarWave::CharacteristicSpeedsCompute<Dim>,
                         CurvedScalarWave::CharacteristicFieldsCompute<Dim>>>(
      gamma_1, gamma_2, lapse, shift, inverse_spatial_metric, psi, pi, phi,
      normal);
  // Test compute tag for char speeds
  CHECK(db::get<CurvedScalarWave::Tags::CharacteristicSpeeds<Dim>>(box) ==
        CurvedScalarWave::characteristic_speeds(gamma_1, lapse, shift, normal));
  // Test compute tag for char fields
  CHECK(db::get<CurvedScalarWave::Tags::CharacteristicFields<Dim>>(box) ==
        CurvedScalarWave::characteristic_fields(gamma_2, inverse_spatial_metric,
                                                psi, pi, phi, normal));

  // more randomized tensors
  const auto u_psi = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);
  const auto u_zero =
      make_with_random_values<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size);
  const auto u_plus = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);
  const auto u_minus = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);
  // Insert into databox
  const auto box2 = db::create<
      db::AddSimpleTags<
          CurvedScalarWave::Tags::ConstraintGamma2,
          CurvedScalarWave::Tags::UPsi, CurvedScalarWave::Tags::UZero<Dim>,
          CurvedScalarWave::Tags::UPlus, CurvedScalarWave::Tags::UMinus,
          ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim>>>,
      db::AddComputeTags<
          CurvedScalarWave::EvolvedFieldsFromCharacteristicFieldsCompute<Dim>>>(
      gamma_2, u_psi, u_zero, u_plus, u_minus, normal);
  // Test compute tag for evolved fields computed from char fields
  CHECK(db::get<
            CurvedScalarWave::Tags::EvolvedFieldsFromCharacteristicFields<Dim>>(
            box2) ==
        CurvedScalarWave::evolved_fields_from_characteristic_fields(
            gamma_2, u_psi, u_zero, u_plus, u_minus, normal));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.CurvedScalarWave.Characteristics",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/CurvedScalarWave/"};

  test_characteristic_speeds<1>();
  test_characteristic_speeds<2>();
  test_characteristic_speeds<3>();

  test_characteristic_fields<1>();
  test_characteristic_fields<2>();
  test_characteristic_fields<3>();

  test_evolved_from_characteristic_fields<1>();
  test_evolved_from_characteristic_fields<2>();
  test_evolved_from_characteristic_fields<3>();

  test_characteristics_compute_tags<1>();
  test_characteristics_compute_tags<2>();
  test_characteristics_compute_tags<3>();
}
