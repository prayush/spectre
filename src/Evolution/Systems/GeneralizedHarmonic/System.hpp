// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TagsDeclarations.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace brigand {
template <class...>
struct list;
}  // namespace brigand

namespace Tags {
template <class>
class Variables;
}  // namespace Tags
/// \endcond

/*!
 * \ingroup EvolutionSystemsGroup
 * \brief Items related to evolving the first-order generalized harmonic system.
 */
namespace GeneralizedHarmonic {
template <size_t Dim>
struct System {
  static constexpr bool is_in_flux_conservative_form = false;
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = Dim;
  static constexpr bool is_euclidean = false;

  using variables_tag = ::Tags::Variables<tmpl::list<
      gr::Tags::SpacetimeMetric<Dim, Frame::Inertial>,
      Tags::Pi<Dim, Frame::Inertial>, Tags::Phi<Dim, Frame::Inertial>>>;
  using gradients_tags =
      tmpl::list<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial>,
                 Tags::Pi<Dim, Frame::Inertial>,
                 Tags::Phi<Dim, Frame::Inertial>>;

  using compute_time_derivative = ComputeDuDt<Dim>;
  using normal_dot_fluxes = ComputeNormalDotFluxes<3>;
  using char_speeds_tag = CharacteristicSpeedsCompute<Dim, Frame::Inertial>;
  using compute_largest_characteristic_speed =
      ComputeLargestCharacteristicSpeed<Dim, Frame::Inertial>;

  template <typename Tag>
  using magnitude_tag = ::Tags::NonEuclideanMagnitude<
      Tag, gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>>;
};
}  // namespace GeneralizedHarmonic
