// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Equations.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Variables.hpp"      // IWYU pragma: keep
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/TMPL.hpp"  // IWYU pragma: keep

template <typename X, typename Symm, typename IndexList>
class Tensor;

namespace CurvedScalarWave {
namespace CurvedScalarWave_detail {
template <typename FieldTag>
db::const_item_type<FieldTag> weight_char_field(
    const db::const_item_type<FieldTag>& char_field_int,
    const DataVector& char_speed_int,
    const db::const_item_type<FieldTag>& char_field_ext,
    const DataVector& char_speed_ext) noexcept {
  const DataVector& char_speed_avg{0.5 * (char_speed_int + char_speed_ext)};
  db::const_item_type<FieldTag> weighted_char_field = char_field_int;
  auto weighted_char_field_it = weighted_char_field.begin();
  for (auto int_it = char_field_int.begin(), ext_it = char_field_ext.begin();
       int_it != char_field_int.end();
       ++int_it, ++ext_it, ++weighted_char_field_it) {
    *weighted_char_field_it *= step_function(char_speed_avg) * char_speed_avg;
    *weighted_char_field_it +=
        step_function(-char_speed_avg) * char_speed_avg * *ext_it;
  }

  return weighted_char_field;
}

template <size_t Dim>
db::const_item_type<Tags::CharacteristicFields<Dim>> weight_char_fields(
    const db::const_item_type<Tags::CharacteristicFields<Dim>>& char_fields_int,
    const db::const_item_type<Tags::CharacteristicSpeeds<Dim>>& char_speeds_int,
    const db::const_item_type<Tags::CharacteristicFields<Dim>>& char_fields_ext,
    const db::const_item_type<Tags::CharacteristicSpeeds<Dim>>&
        char_speeds_ext) noexcept {
  const auto& u_psi_int = get<Tags::UPsi>(char_fields_int);
  const auto& u_zero_int = get<Tags::UZero<Dim>>(char_fields_int);
  const auto& u_plus_int = get<Tags::UPlus>(char_fields_int);
  const auto& u_minus_int = get<Tags::UMinus>(char_fields_int);

  const DataVector& char_speed_u_psi_int{char_speeds_int[0]};
  const DataVector& char_speed_u_zero_int{char_speeds_int[1]};
  const DataVector& char_speed_u_plus_int{char_speeds_int[2]};
  const DataVector& char_speed_u_minus_int{char_speeds_int[3]};

  const auto& u_psi_ext = get<Tags::UPsi>(char_fields_ext);
  const auto& u_zero_ext = get<Tags::UZero<Dim>>(char_fields_ext);
  const auto& u_plus_ext = get<Tags::UPlus>(char_fields_ext);
  const auto& u_minus_ext = get<Tags::UMinus>(char_fields_ext);

  const DataVector& char_speed_u_psi_ext{char_speeds_ext[0]};
  const DataVector& char_speed_u_zero_ext{char_speeds_ext[1]};
  const DataVector& char_speed_u_plus_ext{char_speeds_ext[2]};
  const DataVector& char_speed_u_minus_ext{char_speeds_ext[3]};

  auto weighted_char_fields =
      make_with_value<db::const_item_type<Tags::CharacteristicFields<Dim>>>(
          char_speed_u_psi_int, 0.);

  get<Tags::UPsi>(weighted_char_fields) = weight_char_field<Tags::UPsi>(
      u_psi_int, char_speed_u_psi_int, u_psi_ext, char_speed_u_psi_ext);
  get<Tags::UZero<Dim>>(weighted_char_fields) =
      weight_char_field<Tags::UZero<Dim>>(u_zero_int, char_speed_u_zero_int,
                                          u_zero_ext, char_speed_u_zero_ext);
  get<Tags::UPlus>(weighted_char_fields) = weight_char_field<Tags::UPlus>(
      u_plus_int, char_speed_u_plus_int, u_plus_ext, char_speed_u_plus_ext);
  get<Tags::UMinus>(weighted_char_fields) = weight_char_field<Tags::UMinus>(
      u_minus_int, char_speed_u_minus_int, u_minus_ext, char_speed_u_minus_ext);

  return weighted_char_fields;
}
}  // namespace CurvedScalarWave_detail

/// \cond
template <size_t Dim>
void ComputeDuDt<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> dt_pi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> dt_phi,
    const gsl::not_null<Scalar<DataVector>*> dt_psi,
    const Scalar<DataVector>& pi, const tnsr::i<DataVector, Dim>& phi,
    const tnsr::i<DataVector, Dim>& d_psi, const tnsr::i<DataVector, Dim>& d_pi,
    const tnsr::ij<DataVector, Dim>& d_phi, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim>& shift,
    const tnsr::i<DataVector, Dim>& deriv_lapse,
    const tnsr::iJ<DataVector, Dim>& deriv_shift,
    const tnsr::II<DataVector, Dim>& upper_spatial_metric,
    const tnsr::I<DataVector, Dim>& trace_spatial_christoffel,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>& gamma1,
    const Scalar<DataVector>& gamma2) noexcept {
  dt_psi->get() = -lapse.get() * pi.get();
  for (size_t m = 0; m < Dim; ++m) {
    dt_psi->get() +=
        shift.get(m) *
        (d_psi.get(m) + gamma1.get() * (d_psi.get(m) - phi.get(m)));
  }

  dt_pi->get() = lapse.get() * pi.get() * trace_extrinsic_curvature.get();
  for (size_t m = 0; m < Dim; ++m) {
    dt_pi->get() += shift.get(m) * d_pi.get(m);
    dt_pi->get() += lapse.get() * phi.get(m) * trace_spatial_christoffel.get(m);
    dt_pi->get() += gamma1.get() * gamma2.get() * shift.get(m) *
                    (d_psi.get(m) - phi.get(m));
  }
  for (size_t m = 0; m < Dim; ++m) {
    for (size_t n = 0; n < Dim; ++n) {
      dt_pi->get() -=
          lapse.get() * upper_spatial_metric.get(m, n) * d_phi.get(m, n);
      dt_pi->get() -=
          upper_spatial_metric.get(m, n) * deriv_lapse.get(m) * phi.get(n);
    }
  }
  for (size_t k = 0; k < Dim; ++k) {
    dt_phi->get(k) =
        -lapse.get() *
            (d_pi.get(k) + gamma2.get() * (phi.get(k) - d_psi.get(k))) -
        pi.get() * deriv_lapse.get(k);
    for (size_t m = 0; m < Dim; ++m) {
      dt_phi->get(k) +=
          shift.get(m) * d_phi.get(m, k) + phi.get(m) * deriv_shift.get(k, m);
    }
  }
}

template <size_t Dim>
void ComputeNormalDotFluxes<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> pi_normal_dot_flux,
    const gsl::not_null<tnsr::i<DataVector, Dim>*> phi_normal_dot_flux,
    const gsl::not_null<Scalar<DataVector>*> psi_normal_dot_flux,
    const Scalar<DataVector>& pi, const tnsr::i<DataVector, Dim>& phi,
    const Scalar<DataVector>& psi, const Scalar<DataVector>& gamma1,
    const Scalar<DataVector>& gamma2, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim>& shift,
    const tnsr::II<DataVector, Dim>& inverse_spatial_metric,
    const tnsr::i<DataVector, Dim>& interface_unit_normal) noexcept {
  const auto shift_dot_normal = get(dot_product(shift, interface_unit_normal));
  const auto normal_dot_phi = [&]() {
    auto normal_dot_phi_ = make_with_value<Scalar<DataVector>>(psi, 0.);
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = 0; j < Dim; ++j) {
        get(normal_dot_phi_) += inverse_spatial_metric.get(i, j) *
                                interface_unit_normal.get(j) * phi.get(i);
      }
    }
    return normal_dot_phi_;
  }();

  psi_normal_dot_flux->get() =
      -(1. + get(gamma1)) * shift_dot_normal * get(psi);

  pi_normal_dot_flux->get() =
      -shift_dot_normal * get(pi) + get(lapse) * get(normal_dot_phi) -
      get(gamma1) * get(gamma2) * shift_dot_normal * get(psi);

  for (size_t i = 0; i < Dim; ++i) {
    phi_normal_dot_flux->get(i) =
        get(lapse) * (interface_unit_normal.get(i) * get(pi) -
                      get(gamma2) * interface_unit_normal.get(i) * get(psi)) -
        shift_dot_normal * phi.get(i);
  }
}

template <size_t Dim>
void UpwindFlux<Dim>::package_data(
    const gsl::not_null<Variables<package_tags>*> packaged_data,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const Scalar<DataVector>& psi, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::II<DataVector, Dim, Frame::Inertial>& inverse_spatial_metric,
    const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
    const noexcept {
  get<Psi>(*packaged_data) = psi;
  get<Pi>(*packaged_data) = pi;
  get<Phi<Dim>>(*packaged_data) = phi;
  get<gr::Tags::Lapse<DataVector>>(*packaged_data) = lapse;
  get<gr::Tags::Shift<Dim, Frame::Inertial, DataVector>>(*packaged_data) =
      shift;
  get<gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>>(
      *packaged_data) = inverse_spatial_metric;
  get<Tags::ConstraintGamma1>(*packaged_data) = gamma1;
  get<Tags::ConstraintGamma2>(*packaged_data) = gamma2;
  get<::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim, Frame::Inertial>>>(
      *packaged_data) = interface_unit_normal;
}

template <size_t Dim>
void UpwindFlux<Dim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> pi_normal_dot_numerical_flux,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        phi_normal_dot_numerical_flux,
    const gsl::not_null<Scalar<DataVector>*> psi_normal_dot_numerical_flux,
    const Scalar<DataVector>& pi_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi_int,
    const Scalar<DataVector>& psi_int, const Scalar<DataVector>& lapse_int,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift_int,
    const tnsr::II<DataVector, Dim, Frame::Inertial>&
        inverse_spatial_metric_int,
    const Scalar<DataVector>& gamma1_int, const Scalar<DataVector>& gamma2_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal_int,
    const Scalar<DataVector>& pi_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi_ext,
    const Scalar<DataVector>& psi_ext, const Scalar<DataVector>& lapse_ext,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift_ext,
    const tnsr::II<DataVector, Dim, Frame::Inertial>&
        inverse_spatial_metric_ext,
    const Scalar<DataVector>& gamma1_ext, const Scalar<DataVector>& gamma2_ext,
    const tnsr::i<DataVector, Dim,
                  Frame::Inertial>& /*interface_unit_normal_ext*/) const
    noexcept {
  const Scalar<DataVector> gamma1_avg{0.5 *
                                      (get(gamma1_int) + get(gamma1_ext))};
  const Scalar<DataVector> gamma2_avg{0.5 *
                                      (get(gamma2_int) + get(gamma2_ext))};

  const auto char_fields_int =
      characteristic_fields(gamma2_avg, inverse_spatial_metric_int, psi_int,
                            pi_int, phi_int, interface_unit_normal_int);
  const auto char_speeds_int = characteristic_speeds(
      gamma1_avg, lapse_int, shift_int, interface_unit_normal_int);
  const auto char_fields_ext =
      characteristic_fields(gamma2_avg, inverse_spatial_metric_ext, psi_ext,
                            pi_ext, phi_ext, interface_unit_normal_int);
  const auto char_speeds_ext = characteristic_speeds(
      gamma1_avg, lapse_ext, shift_ext, interface_unit_normal_int);

  const auto weighted_char_fields =
      CurvedScalarWave_detail::weight_char_fields<Dim>(
          char_fields_int, char_speeds_int, char_fields_ext, char_speeds_ext);

  const auto weighted_evolved_fields =
      evolved_fields_from_characteristic_fields(
          gamma2_avg, get<Tags::UPsi>(weighted_char_fields),
          get<Tags::UZero<Dim>>(weighted_char_fields),
          get<Tags::UPlus>(weighted_char_fields),
          get<Tags::UMinus>(weighted_char_fields), interface_unit_normal_int);

  *psi_normal_dot_numerical_flux = get<Psi>(weighted_evolved_fields);
  *pi_normal_dot_numerical_flux = get<Pi>(weighted_evolved_fields);
  *phi_normal_dot_numerical_flux = get<Phi<Dim>>(weighted_evolved_fields);
}
/// \endcond
}  // namespace CurvedScalarWave
// Generate explicit instantiations of partial_derivatives function as well as
// all other functions in Equations.cpp

#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"  // IWYU pragma: keep

template <size_t Dim>
using derivative_tags = typename CurvedScalarWave::System<Dim>::gradients_tags;

template <size_t Dim>
using variables_tags =
    typename CurvedScalarWave::System<Dim>::variables_tag::tags_list;

using derivative_frame = Frame::Inertial;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                               \
  template class CurvedScalarWave::ComputeDuDt<DIM(data)>;                   \
  template struct CurvedScalarWave::ComputeNormalDotFluxes<DIM(data)>;       \
  template struct CurvedScalarWave::UpwindFlux<DIM(data)>;                   \
  template Variables<                                                        \
      db::wrap_tags_in<::Tags::deriv, derivative_tags<DIM(data)>,            \
                       tmpl::size_t<DIM(data)>, derivative_frame>>           \
  partial_derivatives<derivative_tags<DIM(data)>, variables_tags<DIM(data)>, \
                      DIM(data), derivative_frame>(                          \
      const Variables<variables_tags<DIM(data)>>& u,                         \
      const Mesh<DIM(data)>& mesh,                                           \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,           \
                            derivative_frame>& inverse_jacobian) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
