// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave {
namespace Tags {
struct ConstraintGamma1Compute : ConstraintGamma1, db::ComputeTag {
  using argument_tags = tmpl::list<Psi>;
  static auto function(const Scalar<DataVector>& psi) noexcept {
    return make_with_value<type>(psi, 0.);
  }
  using base = ConstraintGamma1;
};

struct ConstraintGamma2Compute : ConstraintGamma2, db::ComputeTag {
  using argument_tags = tmpl::list<Psi>;
  static auto function(const Scalar<DataVector>& psi) noexcept {
    return make_with_value<type>(psi, 0.);
  }
  using base = ConstraintGamma2;
};
}  // namespace Tags
// @{
/*!
 * \ingroup CurvedScalarWave
 * \brief Compute the characteristic speeds for the scalar wave system.
 *
 * Computes the speeds as described in "Optimal constraint projection for
 * hyperbolic evolution systems" by Holst et. al \cite Holst2004wt
 * [see text following Eq.(32)]. The characteristic fields' names used here
 * are similar to the paper:
 *
 * \f{align*}
 * \mathrm{SpECTRE} && \mathrm{Holst} \\
 * u^{\psi} && Z^1 \\
 * u^0_{i} && Z^{2}_{i} \\
 * u^{\pm} && u^{1\pm}
 * \f}
 *
 * The corresponding characteristic speeds \f$v\f$ are given in the text
 * following Eq.(38) of \cite Holst2004wt :
 *
 * \f{align*}
 * v_{\psi} =& 0 \\
 * v_{0} =& 0 \\
 * v_{\pm} =& \pm 1
 * \f}
 *
 * where \f$n_k\f$ is the unit normal to the surface.
 */
template <size_t Dim>
std::array<DataVector, 4> characteristic_speeds(
    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept;

template <size_t Dim>
void characteristic_speeds(
    gsl::not_null<std::array<DataVector, 4>*> char_speeds,
    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept;

template <size_t Dim>
struct CharacteristicSpeedsCompute : Tags::CharacteristicSpeeds<Dim>,
                                     db::ComputeTag {
  using base = Tags::CharacteristicSpeeds<Dim>;
  using type = typename base::type;
  using argument_tags =
      tmpl::list<Tags::ConstraintGamma1, gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<Dim, Frame::Inertial, DataVector>,
                 ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim>>>;

  static typename Tags::CharacteristicSpeeds<Dim>::type function(
      const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          unit_normal_one_form) noexcept {
    return characteristic_speeds(gamma_1, lapse, shift, unit_normal_one_form);
  };
};
// @}

// @{
/*!
 * \ingroup CurvedScalarWave
 * \brief Computes characteristic fields from evolved fields
 *
 * \ref CharacteristicFieldsCompute and
 * \ref EvolvedFieldsFromCharacteristicFieldsCompute convert between
 * characteristic and evolved fields for the scalar-wave system.
 *
 * \ref CharacteristicFieldsCompute computes
 * characteristic fields as described in "Optimal constraint projection for
 * hyperbolic evolution systems" by Holst et. al \cite Holst2004wt .
 * Their names used here differ from this paper:
 *
 * \f{align*}
 * \mathrm{SpECTRE} && \mathrm{Holst} \\
 * u^{\psi} && Z^1 \\
 * u^0_{i} && Z^{2}_{i} \\
 * u^{\pm} && u^{1\pm}
 * \f}
 *
 * The characteristic fields \f$u\f$ are given in terms of the evolved fields by
 * Eq.(33) - (35) of \cite Holst2004wt, respectively:
 *
 * \f{align*}
 * u^{\psi} =& \psi \\
 * u^0_{i} =& (\delta^k_i - n_i n^k) \Phi_{k} := P^k_i \Phi_{k} \\
 * u^{\pm} =& \Pi \pm n^i \Phi_{i} - \gamma_2\psi
 * \f}
 *
 * where \f$\psi\f$ is the scalar field, \f$\Pi\f$ and \f$\Phi_{i}\f$ are
 * evolved fields introduced by first derivatives of \f$\psi\f$, \f$\gamma_2\f$
 * is a constraint damping parameter, and \f$n_k\f$ is the unit normal to the
 * surface.
 *
 * \ref EvolvedFieldsFromCharacteristicFieldsCompute computes evolved fields
 * \f$w\f$ in terms of the characteristic fields. This uses the inverse of
 * above relations:
 *
 * \f{align*}
 * \psi =& u^{\psi}, \\
 * \Pi =& \frac{1}{2}(u^{+} + u^{-}) + \gamma_2 u^{\psi}, \\
 * \Phi_{i} =& \frac{1}{2}(u^{+} - u^{-}) n_i + u^0_{i}.
 * \f}
 *
 * The corresponding characteristic speeds \f$v\f$ are computed by
 * \ref CharacteristicSpeedsCompute .
 */
template <size_t Dim>
Variables<tmpl::list<Tags::UPsi, Tags::UZero<Dim>, Tags::UPlus, Tags::UMinus>>
characteristic_fields(
    const Scalar<DataVector>& gamma_2,
    const tnsr::II<DataVector, Dim, Frame::Inertial>& inverse_spatial_metric,
    const Scalar<DataVector>& psi, const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept;

template <size_t Dim>
void characteristic_fields(
    gsl::not_null<Variables<
        tmpl::list<Tags::UPsi, Tags::UZero<Dim>, Tags::UPlus, Tags::UMinus>>*>
        char_fields,
    const Scalar<DataVector>& gamma_2,
    const tnsr::II<DataVector, Dim, Frame::Inertial>& inverse_spatial_metric,
    const Scalar<DataVector>& psi, const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept;

template <size_t Dim>
struct CharacteristicFieldsCompute : Tags::CharacteristicFields<Dim>,
                                     db::ComputeTag {
  using base = Tags::CharacteristicFields<Dim>;
  using type = typename base::type;
  using argument_tags = tmpl::list<
      Tags::ConstraintGamma2,
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>, Psi, Pi,
      Phi<Dim>, ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim>>>;

  static typename Tags::CharacteristicFields<Dim>::type function(
      const Scalar<DataVector>& gamma_2,
      const tnsr::II<DataVector, Dim, Frame::Inertial>& inverse_spatial_metric,
      const Scalar<DataVector>& psi, const Scalar<DataVector>& pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          unit_normal_one_form) noexcept {
    return characteristic_fields(gamma_2, inverse_spatial_metric, psi, pi, phi,
                                 unit_normal_one_form);
  };
};
// @}

// @{
/*!
 * \ingroup CurvedScalarWave
 * \brief For expressions used here to compute evolved fields from
 * characteristic ones, see \ref CharacteristicFieldsCompute.
 */
template <size_t Dim>
Variables<tmpl::list<Psi, Pi, Phi<Dim>>>
evolved_fields_from_characteristic_fields(
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& u_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& u_zero,
    const Scalar<DataVector>& u_plus, const Scalar<DataVector>& u_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept;

template <size_t Dim>
void evolved_fields_from_characteristic_fields(
    gsl::not_null<Variables<tmpl::list<Psi, Pi, Phi<Dim>>>*> evolved_fields,
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& u_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& u_zero,
    const Scalar<DataVector>& u_plus, const Scalar<DataVector>& u_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept;

template <size_t Dim>
struct EvolvedFieldsFromCharacteristicFieldsCompute
    : Tags::EvolvedFieldsFromCharacteristicFields<Dim>,
      db::ComputeTag {
  using base = Tags::EvolvedFieldsFromCharacteristicFields<Dim>;
  using type = typename base::type;
  using argument_tags =
      tmpl::list<Tags::ConstraintGamma2, Tags::UPsi, Tags::UZero<Dim>,
                 Tags::UPlus, Tags::UMinus,
                 ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim>>>;

  static typename Tags::EvolvedFieldsFromCharacteristicFields<Dim>::type
  function(const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& u_psi,
           const tnsr::i<DataVector, Dim, Frame::Inertial>& u_zero,
           const Scalar<DataVector>& u_plus, const Scalar<DataVector>& u_minus,
           const tnsr::i<DataVector, Dim, Frame::Inertial>&
               unit_normal_one_form) noexcept {
    return evolved_fields_from_characteristic_fields(
        gamma_2, u_psi, u_zero, u_plus, u_minus, unit_normal_one_form);
  };
};
// @}

/*!
 * \ingroup CurvedScalarWave
 * \brief Computes the largest magnitude of the characteristic speeds.
 */
template <size_t Dim>
struct ComputeLargestCharacteristicSpeed {
  using argument_tags = tmpl::list<Tags::CharacteristicSpeeds<Dim>>;
  static double apply(const std::array<DataVector, 4>& char_speeds) noexcept;
};
}  // namespace CurvedScalarWave
