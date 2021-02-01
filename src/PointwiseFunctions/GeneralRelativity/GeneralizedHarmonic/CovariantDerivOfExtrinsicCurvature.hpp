// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
class DataVector;
/// \endcond

namespace GeneralizedHarmonic {
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the covariant derivative of extrinsic curvature
 *
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ijj<DataType, SpatialDim, Frame> covariant_deriv_of_extrinsic_curvature(
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_nomal_vector,
    const tnsr::Ijj<DataType, SpatialDim, Frame>&
        spatial_christoffel_second_kind,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
void covariant_deriv_of_extrinsic_curvature(
    gsl::not_null<tnsr::ijj<DataType, SpatialDim, Frame>*>
        d_extrinsic_curvature,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_nomal_vector,
    const tnsr::Ijj<DataType, SpatialDim, Frame>&
        spatial_christoffel_second_kind,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi) noexcept;
}  // namespace GeneralizedHarmonic
