// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines Functions for calculating spacetime tensors from 3+1 quantities

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace GeneralizedHarmonic {
/*!
 * \brief  Compute item to get spatial derivatives of the gauge source function
 * from its spacetime derivatives.
 *
 * \details Can be retrieved using
 * `::Tags::deriv<GaugeH<SpatialDim, Frame>,tmpl::size_t<SpatialDim>, Frame>`.
 */
template <size_t SpatialDim, typename Frame>
struct DerivGaugeHFromSpacetimeDerivGaugeHCompute
    : ::Tags::deriv<GaugeH<SpatialDim, Frame>, tmpl::size_t<SpatialDim>, Frame>,
      db::ComputeTag {
  using argument_tags = tmpl::list<SpacetimeDerivGaugeH<SpatialDim, Frame>>;
  static constexpr tnsr::ia<DataVector, SpatialDim, Frame> function(
      const tnsr::ab<DataVector, SpatialDim, Frame>&
          spacetime_deriv_gauge_source) {
    auto deriv_gauge_source =
        make_with_value<tnsr::ia<DataVector, SpatialDim, Frame>>(
            spacetime_deriv_gauge_source, 0.0);
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t a = 0; a < SpatialDim + 1; ++a) {
        deriv_gauge_source.get(i, a) =
            spacetime_deriv_gauge_source.get(1 + i, a);
      }
    }
    return deriv_gauge_source;
  }
  using base =
      ::Tags::deriv<GaugeH<SpatialDim, Frame>, tmpl::size_t<SpatialDim>, Frame>;
};

}  // namespace GeneralizedHarmonic
}  // namespace GeneralizedHarmonic
