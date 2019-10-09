// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/Characteristics.hpp"

#include <algorithm>
#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace ScalarWave {
template <size_t Dim>
void characteristic_speeds(
    const gsl::not_null<typename Tags::CharacteristicSpeeds<Dim>::type*>
        char_speeds,
    const tnsr::i<DataVector, Dim,
                  Frame::Inertial>& /*unit_normal_one_form*/) noexcept {
  (*char_speeds)[0] = 0.;   // v(UPsi)
  (*char_speeds)[1] = 0.;   // v(UZero)
  (*char_speeds)[2] = 1.;   // v(UPlus)
  (*char_speeds)[3] = -1.;  // v(UMinus)
}

template <size_t Dim>
typename Tags::CharacteristicSpeeds<Dim>::type characteristic_speeds(
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept {
  auto char_speeds =
      make_with_value<typename Tags::CharacteristicSpeeds<Dim>::type>(
          get<0>(unit_normal_one_form), 0.);
  characteristic_speeds(make_not_null(&char_speeds), unit_normal_one_form);
  return char_speeds;
}

template <size_t Dim>
void characteristic_fields(
    const gsl::not_null<typename Tags::CharacteristicFields<Dim>::type*>
        char_fields,
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& psi,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept {
  // n^j = \delta^{jk} n_k
  auto unit_normal_vector =
      make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(pi, 0.);
  for (size_t i = 0; i < Dim; ++i) {
    unit_normal_vector.get(i) = unit_normal_one_form.get(i);
  }

  // Compute phi_dot_normal = n^i \Phi_{i}
  auto phi_dot_normal = make_with_value<Scalar<DataVector>>(pi, 0.);
  for (size_t i = 0; i < Dim; ++i) {
    get(phi_dot_normal) += unit_normal_vector.get(i) * phi.get(i);
  }

  // Eq.(34) of Holst+ (2004) for UZero
  for (size_t i = 0; i < Dim; ++i) {
    get<Tags::UZero<Dim>>(*char_fields).get(i) =
        phi.get(i) - unit_normal_one_form.get(i) * get(phi_dot_normal);
  }

  // Eq.(33) of Holst+ (2004) for UPsi
  get<Tags::UPsi>(*char_fields) = psi;

  // Eq.(35) of Holst+ (2004) for UPlus and UMinus
  get(get<Tags::UPlus>(*char_fields)) =
      get(pi) + get(phi_dot_normal) - get(gamma_2) * get(psi);
  get(get<Tags::UMinus>(*char_fields)) =
      get(pi) - get(phi_dot_normal) - get(gamma_2) * get(psi);
}

template <size_t Dim>
typename Tags::CharacteristicFields<Dim>::type characteristic_fields(
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& psi,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept {
  auto char_fields =
      make_with_value<typename Tags::CharacteristicFields<Dim>::type>(
          get(gamma_2), 0.);
  characteristic_fields(make_not_null(&char_fields), gamma_2, psi, pi, phi,
                        unit_normal_one_form);
  return char_fields;
}

template <size_t Dim>
void evolved_fields_from_characteristic_fields(
    gsl::not_null<
        typename Tags::EvolvedFieldsFromCharacteristicFields<Dim>::type*>
        evolved_fields,
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& u_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& u_zero,
    const Scalar<DataVector>& u_plus, const Scalar<DataVector>& u_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept {
  // Eq.(36) of Holst+ (2005) for Psi
  get<Psi>(*evolved_fields) = u_psi;

  // Eq.(37) - (38) of Holst+ (2004) for Pi and Phi
  get<Pi>(*evolved_fields).get() =
      0.5 * (get(u_plus) + get(u_minus)) + get(gamma_2) * get(u_psi);
  for (size_t i = 0; i < Dim; ++i) {
    get<Phi<Dim>>(*evolved_fields).get(i) =
        0.5 * (get(u_plus) - get(u_minus)) * unit_normal_one_form.get(i) +
        u_zero.get(i);
  }
}

template <size_t Dim>
typename Tags::EvolvedFieldsFromCharacteristicFields<Dim>::type
evolved_fields_from_characteristic_fields(
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& u_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& u_zero,
    const Scalar<DataVector>& u_plus, const Scalar<DataVector>& u_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept {
  auto evolved_fields = make_with_value<
      typename Tags::EvolvedFieldsFromCharacteristicFields<Dim>::type>(
      get(gamma_2), 0.);
  evolved_fields_from_characteristic_fields(make_not_null(&evolved_fields),
                                            gamma_2, u_psi, u_zero, u_plus,
                                            u_minus, unit_normal_one_form);
  return evolved_fields;
}
}  // namespace ScalarWave

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void ScalarWave::characteristic_speeds(                             \
      const gsl::not_null<                                                     \
          typename ScalarWave::Tags::CharacteristicSpeeds<DIM(data)>::type*>   \
          char_speeds,                                                         \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form) noexcept;                                      \
  template ScalarWave::Tags::CharacteristicSpeeds<DIM(data)>::type             \
  ScalarWave::characteristic_speeds(                                           \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form) noexcept;                                      \
  template struct ScalarWave::CharacteristicSpeedsCompute<DIM(data)>;          \
  template void ScalarWave::characteristic_fields(                             \
      const gsl::not_null<                                                     \
          typename ScalarWave::Tags::CharacteristicFields<DIM(data)>::type*>   \
          char_fields,                                                         \
      const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& psi,        \
      const Scalar<DataVector>& pi,                                            \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& phi,              \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form) noexcept;                                      \
  template typename ScalarWave::Tags::CharacteristicFields<DIM(data)>::type    \
  ScalarWave::characteristic_fields(                                           \
      const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& psi,        \
      const Scalar<DataVector>& pi,                                            \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& phi,              \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form) noexcept;                                      \
  template struct ScalarWave::CharacteristicFieldsCompute<DIM(data)>;          \
  template void ScalarWave::evolved_fields_from_characteristic_fields(         \
      const gsl::not_null<                                                     \
          typename ScalarWave::Tags::EvolvedFieldsFromCharacteristicFields<    \
              DIM(data)>::type*>                                               \
          evolved_fields,                                                      \
      const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& u_psi,      \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& u_zero,           \
      const Scalar<DataVector>& u_plus, const Scalar<DataVector>& u_minus,     \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form) noexcept;                                      \
  template                                                                     \
      typename ScalarWave::Tags::EvolvedFieldsFromCharacteristicFields<DIM(    \
          data)>::type                                                         \
      ScalarWave::evolved_fields_from_characteristic_fields(                   \
          const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& u_psi,  \
          const tnsr::i<DataVector, DIM(data), Frame::Inertial>& u_zero,       \
          const Scalar<DataVector>& u_plus, const Scalar<DataVector>& u_minus, \
          const tnsr::i<DataVector, DIM(data), Frame::Inertial>&               \
              unit_normal_one_form) noexcept;                                  \
  template struct ScalarWave::EvolvedFieldsFromCharacteristicFieldsCompute<    \
      DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
