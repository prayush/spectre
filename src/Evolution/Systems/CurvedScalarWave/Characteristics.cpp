// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Characteristics.hpp"

#include <algorithm>
#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace CurvedScalarWave {
template <size_t Dim>
void characteristic_speeds(
    const gsl::not_null<std::array<DataVector, 4>*> char_speeds,
    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept {
  const auto shift_dot_normal = get(dot_product(shift, unit_normal_one_form));
  (*char_speeds)[0] = -(1. + get(gamma_1)) * shift_dot_normal;  // v(UPsi)
  (*char_speeds)[1] = -shift_dot_normal;                        // v(UZero)
  (*char_speeds)[2] = -shift_dot_normal + get(lapse);           // v(UPlus)
  (*char_speeds)[3] = -shift_dot_normal - get(lapse);           // v(UMinus)
}

template <size_t Dim>
std::array<DataVector, 4> characteristic_speeds(
    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept {
  auto char_speeds = make_with_value<std::array<DataVector, 4>>(
      get<0>(unit_normal_one_form), 0.);
  characteristic_speeds(make_not_null(&char_speeds), gamma_1, lapse, shift,
                        unit_normal_one_form);
  return char_speeds;
}

template <size_t Dim>
void characteristic_fields(
    const gsl::not_null<Variables<
        tmpl::list<Tags::UPsi, Tags::UZero<Dim>, Tags::UPlus, Tags::UMinus>>*>
        char_fields,
    const Scalar<DataVector>& gamma_2,
    const tnsr::II<DataVector, Dim, Frame::Inertial>& inverse_spatial_metric,
    const Scalar<DataVector>& psi, const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept {
  auto unit_normal_vector =
      raise_or_lower_index(unit_normal_one_form, inverse_spatial_metric);

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
Variables<tmpl::list<Tags::UPsi, Tags::UZero<Dim>, Tags::UPlus, Tags::UMinus>>
characteristic_fields(
    const Scalar<DataVector>& gamma_2,
    const tnsr::II<DataVector, Dim, Frame::Inertial>& inverse_spatial_metric,
    const Scalar<DataVector>& psi, const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept {
  auto char_fields = make_with_value<Variables<
      tmpl::list<Tags::UPsi, Tags::UZero<Dim>, Tags::UPlus, Tags::UMinus>>>(
      get(gamma_2), 0.);
  characteristic_fields(make_not_null(&char_fields), gamma_2,
                        inverse_spatial_metric, psi, pi, phi,
                        unit_normal_one_form);
  return char_fields;
}

template <size_t Dim>
void evolved_fields_from_characteristic_fields(
    gsl::not_null<Variables<tmpl::list<Psi, Pi, Phi<Dim>>>*> evolved_fields,
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
Variables<tmpl::list<Psi, Pi, Phi<Dim>>>
evolved_fields_from_characteristic_fields(
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& u_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& u_zero,
    const Scalar<DataVector>& u_plus, const Scalar<DataVector>& u_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept {
  auto evolved_fields =
      make_with_value<Variables<tmpl::list<Psi, Pi, Phi<Dim>>>>(get(gamma_2),
                                                                0.);
  evolved_fields_from_characteristic_fields(make_not_null(&evolved_fields),
                                            gamma_2, u_psi, u_zero, u_plus,
                                            u_minus, unit_normal_one_form);
  return evolved_fields;
}

template <size_t Dim>
double ComputeLargestCharacteristicSpeed<Dim>::apply(
    const std::array<DataVector, 4>& char_speeds) noexcept {
  std::array<double, 4> max_speeds{
      {max(abs(char_speeds.at(0))), max(abs(char_speeds.at(1))),
       max(abs(char_speeds.at(2))), max(abs(char_speeds.at(3)))}};
  return *std::max_element(max_speeds.begin(), max_speeds.end());
}
}  // namespace CurvedScalarWave

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template void CurvedScalarWave::characteristic_speeds(                      \
      const gsl::not_null<std::array<DataVector, 4>*> char_speeds,            \
      const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,     \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& shift,           \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form) noexcept;                                     \
  template std::array<DataVector, 4> CurvedScalarWave::characteristic_speeds( \
      const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,     \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& shift,           \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form) noexcept;                                     \
  template struct CurvedScalarWave::CharacteristicSpeedsCompute<DIM(data)>;   \
  template void CurvedScalarWave::characteristic_fields(                      \
      const gsl::not_null<Variables<tmpl::list<                               \
          CurvedScalarWave::Tags::UPsi,                                       \
          CurvedScalarWave::Tags::UZero<DIM(data)>,                           \
          CurvedScalarWave::Tags::UPlus, CurvedScalarWave::Tags::UMinus>>*>   \
          char_fields,                                                        \
      const Scalar<DataVector>& gamma_2,                                      \
      const tnsr::II<DataVector, DIM(data), Frame::Inertial>&                 \
          inverse_spatial_metric,                                             \
      const Scalar<DataVector>& psi, const Scalar<DataVector>& pi,            \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& phi,             \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form) noexcept;                                     \
  template Variables<tmpl::list<                                              \
      CurvedScalarWave::Tags::UPsi, CurvedScalarWave::Tags::UZero<DIM(data)>, \
      CurvedScalarWave::Tags::UPlus, CurvedScalarWave::Tags::UMinus>>         \
  CurvedScalarWave::characteristic_fields(                                    \
      const Scalar<DataVector>& gamma_2,                                      \
      const tnsr::II<DataVector, DIM(data), Frame::Inertial>&                 \
          inverse_spatial_metric,                                             \
      const Scalar<DataVector>& psi, const Scalar<DataVector>& pi,            \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& phi,             \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form) noexcept;                                     \
  template struct CurvedScalarWave::CharacteristicFieldsCompute<DIM(data)>;   \
  template void CurvedScalarWave::evolved_fields_from_characteristic_fields(  \
      const gsl::not_null<                                                    \
          Variables<tmpl::list<CurvedScalarWave::Psi, CurvedScalarWave::Pi,   \
                               CurvedScalarWave::Phi<DIM(data)>>>*>           \
          evolved_fields,                                                     \
      const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& u_psi,     \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& u_zero,          \
      const Scalar<DataVector>& u_plus, const Scalar<DataVector>& u_minus,    \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form) noexcept;                                     \
  template Variables<tmpl::list<CurvedScalarWave::Psi, CurvedScalarWave::Pi,  \
                                CurvedScalarWave::Phi<DIM(data)>>>            \
  CurvedScalarWave::evolved_fields_from_characteristic_fields(                \
      const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& u_psi,     \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& u_zero,          \
      const Scalar<DataVector>& u_plus, const Scalar<DataVector>& u_minus,    \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form) noexcept;                                     \
  template struct CurvedScalarWave::                                          \
      EvolvedFieldsFromCharacteristicFieldsCompute<DIM(data)>;                \
  template struct CurvedScalarWave::ComputeLargestCharacteristicSpeed<DIM(    \
      data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
