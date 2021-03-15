// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ContainerHelpers.hpp"

/// \cond
class DataVector;
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace GeneralizedHarmonic::BoundaryConditions {
namespace detail {
// @{
/*!
 * \brief Computes the expression needed to set bnudary conditions on Dt<VPsi>
 *
 * \details Computes the correction to the characteristic projected
 * time-derivatives of the fundamental variables corresponding to \f$u^\psi\f$:
 *
 * \f{align}
 * \partial_t u^{\psi}_{ab} = \partial_t u^{\Psi}_{ab} + v_{\psi} n^i C_{iab}
 * \f}
 *
 * where \f$n^i\f$ is the local unit normal to the external boundary,
 * and \f$C_{iab} = \partial_i \psi_{ab} - \Phi_{iab}\f$ is the
 * three-index constraint.
 */
template <size_t VolumeDim, typename DataType>
void set_dt_v_psi_constraint_preserving(
    gsl::not_null<tnsr::aa<DataType, VolumeDim, Frame::Inertial>*> bc_dt_v_psi,
    const tnsr::I<DataType, VolumeDim, Frame::Inertial>&
        unit_interface_normal_vector,
    const tnsr::iaa<DataType, VolumeDim, Frame::Inertial>&
        three_index_constraint,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_v_psi,
    const std::array<DataType, 4>& char_speeds) noexcept;
// @}
}  // namespace detail
}  // namespace GeneralizedHarmonic::BoundaryConditions
