// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace gr {

// @{
/// \ingroup GeneralRelativityGroup
/// Holds functions related to general relativity.
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute spatial metric from spacetime metric.
 * \details Simply pull out the spatial components.
 */
template <size_t VolumeDim, typename Frame, typename DataType>
tnsr::II<DataType, VolumeDim, Frame> spatial_projection_tensor(
    const tnsr::II<DataType, VolumeDim, Frame>& inverse_spatial_metric,
    const tnsr::A<DataType, VolumeDim, Frame>& normal_vector) noexcept;

template <size_t VolumeDim, typename Frame, typename DataType>
void spatial_projection_tensor(
    gsl::not_null<tnsr::II<DataType, VolumeDim, Frame>*> projection_tensor,
    const tnsr::II<DataType, VolumeDim, Frame>& inverse_spatial_metric,
    const tnsr::A<DataType, VolumeDim, Frame>& normal_vector) noexcept;

template <size_t VolumeDim, typename Frame, typename DataType>
tnsr::II<DataType, VolumeDim, Frame> spatial_projection_tensor(
    const tnsr::II<DataType, VolumeDim, Frame>& inverse_spatial_metric,
    const tnsr::I<DataType, VolumeDim, Frame>& normal_vector) noexcept;

template <size_t VolumeDim, typename Frame, typename DataType>
void spatial_projection_tensor(
    gsl::not_null<tnsr::II<DataType, VolumeDim, Frame>*> projection_tensor,
    const tnsr::II<DataType, VolumeDim, Frame>& inverse_spatial_metric,
    const tnsr::I<DataType, VolumeDim, Frame>& normal_vector) noexcept;
// @}

// @{
/// \ingroup GeneralRelativityGroup
/// Holds functions related to general relativity.
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute spatial metric from spacetime metric.
 * \details Simply pull out the spatial components.
 */
template <size_t VolumeDim, typename Frame, typename DataType>
tnsr::ii<DataType, VolumeDim, Frame> spatial_projection_tensor(
    const tnsr::ii<DataType, VolumeDim, Frame>& spatial_metric,
    const tnsr::a<DataType, VolumeDim, Frame>& normal_one_form) noexcept;

template <size_t VolumeDim, typename Frame, typename DataType>
void spatial_projection_tensor(
    gsl::not_null<tnsr::ii<DataType, VolumeDim, Frame>*> projection_tensor,
    const tnsr::ii<DataType, VolumeDim, Frame>& spatial_metric,
    const tnsr::a<DataType, VolumeDim, Frame>& normal_one_form) noexcept;

template <size_t VolumeDim, typename Frame, typename DataType>
tnsr::ii<DataType, VolumeDim, Frame> spatial_projection_tensor(
    const tnsr::ii<DataType, VolumeDim, Frame>& spatial_metric,
    const tnsr::i<DataType, VolumeDim, Frame>& normal_one_form) noexcept;

template <size_t VolumeDim, typename Frame, typename DataType>
void spatial_projection_tensor(
    gsl::not_null<tnsr::ii<DataType, VolumeDim, Frame>*> projection_tensor,
    const tnsr::ii<DataType, VolumeDim, Frame>& spatial_metric,
    const tnsr::i<DataType, VolumeDim, Frame>& normal_one_form) noexcept;
// @}

// @{
/// \ingroup GeneralRelativityGroup
/// Holds functions related to general relativity.
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute spatial metric from spacetime metric.
 * \details Simply pull out the spatial components.
 */
template <size_t VolumeDim, typename Frame, typename DataType>
tnsr::Ij<DataType, VolumeDim, Frame> spatial_projection_tensor(
    const tnsr::A<DataType, VolumeDim, Frame>& normal_vector,
    const tnsr::a<DataType, VolumeDim, Frame>& normal_one_form) noexcept;

template <size_t VolumeDim, typename Frame, typename DataType>
void spatial_projection_tensor(
    gsl::not_null<tnsr::Ij<DataType, VolumeDim, Frame>*> projection_tensor,
    const tnsr::A<DataType, VolumeDim, Frame>& normal_vector,
    const tnsr::a<DataType, VolumeDim, Frame>& normal_one_form) noexcept;

template <size_t VolumeDim, typename Frame, typename DataType>
tnsr::Ij<DataType, VolumeDim, Frame> spatial_projection_tensor(
    const tnsr::I<DataType, VolumeDim, Frame>& normal_vector,
    const tnsr::i<DataType, VolumeDim, Frame>& normal_one_form) noexcept;

template <size_t VolumeDim, typename Frame, typename DataType>
void spatial_projection_tensor(
    gsl::not_null<tnsr::Ij<DataType, VolumeDim, Frame>*> projection_tensor,
    const tnsr::I<DataType, VolumeDim, Frame>& normal_vector,
    const tnsr::i<DataType, VolumeDim, Frame>& normal_one_form) noexcept;
// @}
}  // namespace gr
