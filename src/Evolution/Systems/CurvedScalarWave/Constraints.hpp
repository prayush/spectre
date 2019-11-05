// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines functions to calculate the scalar wave constraints

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace CurvedScalarWave {
// @{
/*!
 * \brief Computes the scalar-wave one-index constraint.
 *
 * \details Computes the scalar-wave one-index constraint,
 * \f$C_{i} = \partial_i\psi - \Phi_{i},\f$ which is
 * given by Eq. (19) of \cite Holst2004wt
 */
template <size_t Dim>
tnsr::i<DataVector, Dim, Frame::Inertial> one_index_constraint(
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi) noexcept;

template <size_t Dim>
void one_index_constraint(
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> constraint,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi) noexcept;
// @}

// @{
/*!
 * \brief Computes the scalar-wave 2-index constraint.
 *
 * \details Computes the scalar-wave 2-index FOSH constraint that
 * [Eq. (20) of \cite Holst2004wt],
 *
 * \f{eqnarray}{
 * C_{ij} &\equiv& \partial_i \Phi_j - \partial_j \Phi_i
 * \f}
 *
 * where \f$\Phi_{i} = \partial_i\psi\f$; and \f$\psi\f$ is the scalar field.
 */
template <size_t Dim>
tnsr::ij<DataVector, Dim, Frame::Inertial> two_index_constraint(
    const tnsr::ij<DataVector, Dim, Frame::Inertial>& d_phi) noexcept;

template <size_t Dim>
void two_index_constraint(
    const gsl::not_null<tnsr::ij<DataVector, Dim, Frame::Inertial>*> constraint,
    const tnsr::ij<DataVector, Dim, Frame::Inertial>& d_phi) noexcept;
// @}

namespace Tags {
/*!
 * \brief Compute item to get the one-index constraint for the scalar-wave
 * evolution system.
 *
 * \details See `one_index_constraint()`. Can be retrieved using
 * `CurvedScalarWave::Tags::OneIndexConstraint`.
 */
template <size_t Dim>
struct OneIndexConstraintCompute : OneIndexConstraint<Dim>, db::ComputeTag {
  using argument_tags =
      tmpl::list<::Tags::deriv<Psi, tmpl::size_t<Dim>, Frame::Inertial>,
                 Phi<Dim>>;
  static constexpr tnsr::i<DataVector, Dim, Frame::Inertial> (*function)(
      const tnsr::i<DataVector, Dim, Frame::Inertial>&,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&) =
      &one_index_constraint<Dim>;
  using base = OneIndexConstraint<Dim>;
};

/*!
 * \brief Compute item to get the two-index constraint for the scalar-wave
 * evolution system.
 *
 * \details See `two_index_constraint()`. Can be retrieved using
 * `CurvedScalarWave::Tags::TwoIndexConstraint`.
 */
template <size_t Dim>
struct TwoIndexConstraintCompute : TwoIndexConstraint<Dim>, db::ComputeTag {
  using argument_tags =
      tmpl::list<::Tags::deriv<Phi<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>;
  static constexpr tnsr::ij<DataVector, Dim, Frame::Inertial> (*function)(
      const tnsr::ij<DataVector, Dim, Frame::Inertial>&) =
      &two_index_constraint<Dim>;
  using base = TwoIndexConstraint<Dim>;
};

}  // namespace Tags
}  // namespace CurvedScalarWave
