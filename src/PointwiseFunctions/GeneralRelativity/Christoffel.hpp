// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines functions to calculate Christoffel symbols

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

/// \cond
namespace gsl {
template <class>
class not_null;
}  // namespace gsl
/// \endcond

namespace gr {
// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes Christoffel symbol of the first kind from derivative of
 * metric
 *
 * \details Computes Christoffel symbol \f$\Gamma_{abc}\f$ as:
 * \f$ \Gamma_{cab} = \frac{1}{2} ( \partial_a g_{bc} + \partial_b g_{ac}
 *  -  \partial_c g_{ab}) \f$
 * where \f$g_{bc}\f$ is either a spatial or spacetime metric
 */
template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
void christoffel_first_kind(
    gsl::not_null<tnsr::abb<DataType, SpatialDim, Frame, Index>*> christoffel,
    const tnsr::abb<DataType, SpatialDim, Frame, Index>& d_metric) noexcept;

template <size_t SpatialDim, typename Frame, IndexType Index, typename DataType>
tnsr::abb<DataType, SpatialDim, Frame, Index> christoffel_first_kind(
    const tnsr::abb<DataType, SpatialDim, Frame, Index>& d_metric) noexcept;
// @}

namespace Tags {
/// Compute item for spacetime Christoffel symbols of the first kind
/// \f$\Gamma_{abc}\f$ computed from the first derivative of the
/// spacetime metric.
///
/// Can be retrieved using `gr::Tags::SpacetimeChristoffelFirstKind`
template <size_t SpatialDim, typename Frame, typename DataType>
struct SpacetimeChristoffelFirstKindCompute
    : SpacetimeChristoffelFirstKind<SpatialDim, Frame, DataType>,
      db::ComputeTag {
  static constexpr auto function =
      &christoffel_first_kind<SpatialDim, Frame, IndexType::Spacetime,
                              DataType>;
  using argument_tags =
      tmpl::list<::Tags::deriv<SpacetimeMetric<SpatialDim, Frame, DataType>,
                               tmpl::size_t<SpatialDim>, Frame>>;
};

/// Compute item for spacetime Christoffel symbols of the second kind
/// \f$\Gamma^a_{bc}\f$ computed from the Christoffel symbols of the
/// first kind and the inverse spacetime metric.
///
/// Can be retrieved using `gr::Tags::SpacetimeChristoffelSecondKind`
template <size_t SpatialDim, typename Frame, typename DataType>
struct SpacetimeChristoffelSecondKindCompute
    : SpacetimeChristoffelSecondKind<SpatialDim, Frame, DataType>,
      db::ComputeTag {
  static constexpr auto function =
      &raise_or_lower_first_index<DataType,
                                  SpacetimeIndex<SpatialDim, UpLo::Lo, Frame>,
                                  SpacetimeIndex<SpatialDim, UpLo::Lo, Frame>>;
  using argument_tags =
      tmpl::list<SpacetimeChristoffelFirstKind<SpatialDim, Frame, DataType>,
                 InverseSpacetimeMetric<SpatialDim, Frame, DataType>>;
};

/// Compute item for the trace of the spacetime Christoffel symbols
/// of the first kind
/// \f$\Gamma_{a} = \Gamma_{abc}g^{bc}\f$ compputed from the
/// Christoffel symbols of the first kind and the inverse spacetime metric.
///
/// Can be retrieved using `gr::Tags::TraceSpacetimeChristoffelFirstKind`
template <size_t SpatialDim, typename Frame, typename DataType>
struct TraceSpacetimeChristoffelFirstKindCompute
    : TraceSpacetimeChristoffelFirstKind<SpatialDim, Frame, DataType>,
      db::ComputeTag {
  static constexpr auto function =
      &trace_last_indices<DataType, SpacetimeIndex<SpatialDim, UpLo::Lo, Frame>,
                          SpacetimeIndex<SpatialDim, UpLo::Lo, Frame>>;
  using argument_tags =
      tmpl::list<SpacetimeChristoffelFirstKind<SpatialDim, Frame, DataType>,
                 InverseSpacetimeMetric<SpatialDim, Frame, DataType>>;
};
}  // namespace Tags
} // namespace gr
