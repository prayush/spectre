// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"  // for tags
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace GeneralizedHarmonic {
namespace Solutions {

/*!
 * \brief A wrapper for general-relativity analytic solutions that loads
 * the analytic solution and then adds a function that returns
 * any combination of the generalized-harmonic evolution variables,
 * specifically `gr::Tags::SpacetimeMetric`, `GeneralizedHarmonic::Tags::Pi`,
 * and `GeneralizedHarmonic::Tags::Phi`
 */
template <typename SolutionType>
class WrappedGr : public SolutionType {
 public:
  using SolutionType::SolutionType;

  static constexpr size_t volume_dim = SolutionType::volume_dim;
  using options = typename SolutionType::options;
  static constexpr OptionString help = SolutionType::help;

  using DerivLapse = ::Tags::deriv<gr::Tags::Lapse<DataVector>,
                                   tmpl::size_t<volume_dim>, Frame::Inertial>;
  using DerivShift =
      ::Tags::deriv<gr::Tags::Shift<volume_dim, Frame::Inertial, DataVector>,
                    tmpl::size_t<volume_dim>, Frame::Inertial>;
  using DerivSpatialMetric = ::Tags::deriv<
      gr::Tags::SpatialMetric<volume_dim, Frame::Inertial, DataVector>,
      tmpl::size_t<volume_dim>, Frame::Inertial>;
  using TimeDerivLapse = ::Tags::dt<gr::Tags::Lapse<DataVector>>;
  using TimeDerivShift =
      ::Tags::dt<gr::Tags::Shift<volume_dim, Frame::Inertial, DataVector>>;
  using TimeDerivSpatialMetric = ::Tags::dt<
      gr::Tags::SpatialMetric<volume_dim, Frame::Inertial, DataVector>>;

  using IntermediateVars = tuples::tagged_tuple_from_typelist<
      typename SolutionType::template tags<DataVector>>;

  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataVector, volume_dim>& x, double t,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    // Get the underlying solution's variables using the solution's tags list,
    // store in IntermediateVariables
    const IntermediateVars& intermediate_vars = SolutionType::variables(
        x, t, typename SolutionType::template tags<DataVector>{});

    return {
        get<Tags>(variables(x, t, tmpl::list<Tags>{}, intermediate_vars))...};
  }

  template <typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataVector, volume_dim>& x,
                                     double t, tmpl::list<Tag> /*meta*/) const
      noexcept {
    const IntermediateVars& intermediate_vars = SolutionType::variables(
        x, t, typename SolutionType::template tags<DataVector>{});
    return {get<Tag>(variables(x, t, tmpl::list<Tag>{}, intermediate_vars))};
  }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept { SolutionType::pup(p); }  // NOLINT

 private:
  tuples::TaggedTuple<gr::Tags::Lapse<DataVector>> variables(
      const tnsr::I<DataVector, volume_dim>& /*x*/, double /*t*/,
      tmpl::list<gr::Tags::Lapse<DataVector>> /*meta*/,
      const IntermediateVars& intermediate_vars) const noexcept;

  tuples::TaggedTuple<TimeDerivLapse> variables(
      const tnsr::I<DataVector, volume_dim>& /*x*/, double /*t*/,
      tmpl::list<TimeDerivLapse> /*meta*/,
      const IntermediateVars& intermediate_vars) const noexcept;

  tuples::TaggedTuple<DerivLapse> variables(
      const tnsr::I<DataVector, volume_dim>& /*x*/, double /*t*/,
      tmpl::list<DerivLapse> /*meta*/,
      const IntermediateVars& intermediate_vars) const noexcept;

  tuples::TaggedTuple<gr::Tags::Shift<volume_dim, Frame::Inertial, DataVector>>
  variables(const tnsr::I<DataVector, volume_dim>& /*x*/, double /*t*/,
            tmpl::list<gr::Tags::Shift<volume_dim, Frame::Inertial,
                                       DataVector>> /*meta*/,
            const IntermediateVars& intermediate_vars) const noexcept;

  tuples::TaggedTuple<TimeDerivShift> variables(
      const tnsr::I<DataVector, volume_dim>& /*x*/, double /*t*/,
      tmpl::list<TimeDerivShift> /*meta*/,
      const IntermediateVars& intermediate_vars) const noexcept;

  tuples::TaggedTuple<DerivShift> variables(
      const tnsr::I<DataVector, volume_dim>& /*x*/, double /*t*/,
      tmpl::list<DerivShift> /*meta*/,
      const IntermediateVars& intermediate_vars) const noexcept;

  tuples::TaggedTuple<
      gr::Tags::SpatialMetric<volume_dim, Frame::Inertial, DataVector>>
  variables(const tnsr::I<DataVector, volume_dim>& /*x*/, double /*t*/,
            tmpl::list<gr::Tags::SpatialMetric<volume_dim, Frame::Inertial,
                                               DataVector>> /*meta*/,
            const IntermediateVars& intermediate_vars) const noexcept;

  tuples::TaggedTuple<TimeDerivSpatialMetric> variables(
      const tnsr::I<DataVector, volume_dim>& /*x*/, double /*t*/,
      tmpl::list<TimeDerivSpatialMetric> /*meta*/,
      const IntermediateVars& intermediate_vars) const noexcept;

  tuples::TaggedTuple<DerivSpatialMetric> variables(
      const tnsr::I<DataVector, volume_dim>& /*x*/, double /*t*/,
      tmpl::list<DerivSpatialMetric> /*meta*/,
      const IntermediateVars& intermediate_vars) const noexcept;

  tuples::TaggedTuple<
      gr::Tags::InverseSpatialMetric<volume_dim, Frame::Inertial, DataVector>>
  variables(const tnsr::I<DataVector, volume_dim>& /*x*/, double /*t*/,
            tmpl::list<gr::Tags::InverseSpatialMetric<
                volume_dim, Frame::Inertial, DataVector>> /*meta*/,
            const IntermediateVars& intermediate_vars) const noexcept;

  tuples::TaggedTuple<
      gr::Tags::ExtrinsicCurvature<volume_dim, Frame::Inertial, DataVector>>
  variables(const tnsr::I<DataVector, volume_dim>& /*x*/, double /*t*/,
            tmpl::list<gr::Tags::ExtrinsicCurvature<volume_dim, Frame::Inertial,
                                                    DataVector>> /*meta*/,
            const IntermediateVars& intermediate_vars) const noexcept;

  tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DataVector>> variables(
      const tnsr::I<DataVector, volume_dim>& /*x*/, double /*t*/,
      tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataVector>> /*meta*/,
      const IntermediateVars& intermediate_vars) const noexcept;

  tuples::TaggedTuple<
      gr::Tags::SpacetimeMetric<volume_dim, Frame::Inertial, DataVector>>
  variables(const tnsr::I<DataVector, volume_dim>& /*x*/, double /*t*/,
            tmpl::list<gr::Tags::SpacetimeMetric<volume_dim, Frame::Inertial,
                                                 DataVector>> /*meta*/,
            const IntermediateVars& intermediate_vars) const noexcept;

  tuples::TaggedTuple<
      GeneralizedHarmonic::Tags::Pi<volume_dim, Frame::Inertial>>
  variables(const tnsr::I<DataVector, volume_dim>& /*x*/, double /*t*/,
            tmpl::list<GeneralizedHarmonic::Tags::Pi<volume_dim,
                                                     Frame::Inertial>> /*meta*/,
            const IntermediateVars& intermediate_vars) const noexcept;

  tuples::TaggedTuple<
      GeneralizedHarmonic::Tags::Phi<volume_dim, Frame::Inertial>>
  variables(
      const tnsr::I<DataVector, volume_dim>& /*x*/, double /*t*/,
      tmpl::list<
          GeneralizedHarmonic::Tags::Phi<volume_dim, Frame::Inertial>> /*meta*/,
      const IntermediateVars& intermediate_vars) const noexcept;
};

template <typename SolutionType>
inline constexpr bool operator==(const WrappedGr<SolutionType>& lhs,
                                 const WrappedGr<SolutionType>& rhs) noexcept {
  return dynamic_cast<const SolutionType&>(lhs) ==
         dynamic_cast<const SolutionType&>(rhs);
}

template <typename SolutionType>
inline constexpr bool operator!=(const WrappedGr<SolutionType>& lhs,
                                 const WrappedGr<SolutionType>& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace Solutions
}  // namespace GeneralizedHarmonic
