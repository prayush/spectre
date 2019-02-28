// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GeneralizedHarmonicSolution.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"     // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_forward_declare ::Tags::deriv

/// \cond
namespace GeneralizedHarmonic {
namespace Solutions {

template <typename SolutionType>
tuples::TaggedTuple<gr::Tags::Lapse<DataVector>>
GeneralizedHarmonicSolution<SolutionType>::variables(
    const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,
    tmpl::list<gr::Tags::Lapse<DataVector>> /*meta*/,
    const IntermediateVars&& intermediate_vars) const noexcept {
  return {get<gr::Tags::Lapse<DataVector>>(intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<
    typename GeneralizedHarmonicSolution<SolutionType>::TimeDerivLapse>
GeneralizedHarmonicSolution<SolutionType>::variables(
    const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,
    tmpl::list<typename GeneralizedHarmonicSolution<
        SolutionType>::TimeDerivLapse> /*meta*/,
    const IntermediateVars&& intermediate_vars) const noexcept {
  return {
      get<typename GeneralizedHarmonicSolution<SolutionType>::TimeDerivLapse>(
          intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<
    typename GeneralizedHarmonicSolution<SolutionType>::DerivLapse>
GeneralizedHarmonicSolution<SolutionType>::variables(
    const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,
    tmpl::list<typename GeneralizedHarmonicSolution<
        SolutionType>::DerivLapse> /*meta*/,
    const IntermediateVars&& intermediate_vars) const noexcept {
  return {get<typename GeneralizedHarmonicSolution<SolutionType>::DerivLapse>(
      intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<gr::Tags::Shift<3, Frame::Inertial, DataVector>>
GeneralizedHarmonicSolution<SolutionType>::variables(
    const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,
    tmpl::list<gr::Tags::Shift<3, Frame::Inertial, DataVector>> /*meta*/,
    const IntermediateVars&& intermediate_vars) const noexcept {
  return {
      get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<
    typename GeneralizedHarmonicSolution<SolutionType>::TimeDerivShift>
GeneralizedHarmonicSolution<SolutionType>::variables(
    const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,
    tmpl::list<typename GeneralizedHarmonicSolution<
        SolutionType>::TimeDerivShift> /*meta*/,
    const IntermediateVars&& intermediate_vars) const noexcept {
  return {
      get<typename GeneralizedHarmonicSolution<SolutionType>::TimeDerivShift>(
          intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<
    typename GeneralizedHarmonicSolution<SolutionType>::DerivShift>
GeneralizedHarmonicSolution<SolutionType>::variables(
    const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,
    tmpl::list<typename GeneralizedHarmonicSolution<
        SolutionType>::DerivShift> /*meta*/,
    const IntermediateVars&& intermediate_vars) const noexcept {
  return {get<DerivShift>(intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>
GeneralizedHarmonicSolution<SolutionType>::variables(
    const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,
    tmpl::list<
        gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>> /*meta*/,
    const IntermediateVars&& intermediate_vars) const noexcept {
  return {get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(
      intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<
    typename GeneralizedHarmonicSolution<SolutionType>::TimeDerivSpatialMetric>
GeneralizedHarmonicSolution<SolutionType>::variables(
    const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,
    tmpl::list<typename GeneralizedHarmonicSolution<
        SolutionType>::TimeDerivSpatialMetric> /*meta*/,
    const IntermediateVars&& intermediate_vars) const noexcept {
  return {get<typename GeneralizedHarmonicSolution<
      SolutionType>::TimeDerivSpatialMetric>(intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<
    typename GeneralizedHarmonicSolution<SolutionType>::DerivSpatialMetric>
GeneralizedHarmonicSolution<SolutionType>::variables(
    const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,
    tmpl::list<typename GeneralizedHarmonicSolution<
        SolutionType>::DerivSpatialMetric> /*meta*/,
    const IntermediateVars&& intermediate_vars) const noexcept {
  return {get<
      typename GeneralizedHarmonicSolution<SolutionType>::DerivSpatialMetric>(
      intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<
    gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>>
GeneralizedHarmonicSolution<SolutionType>::variables(
    const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,
    tmpl::list<gr::Tags::InverseSpatialMetric<3, Frame::Inertial,
                                              DataVector>> /*meta*/,
    const IntermediateVars&& intermediate_vars) const noexcept {
  return {get<gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>>(
      intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<
    gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>>
GeneralizedHarmonicSolution<SolutionType>::variables(
    const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,
    tmpl::list<
        gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>> /*meta*/,
    const IntermediateVars&& intermediate_vars) const noexcept {
  return {get<gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>>(
      intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DataVector>>
GeneralizedHarmonicSolution<SolutionType>::variables(
    const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,
    tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataVector>> /*meta*/,
    const IntermediateVars&& intermediate_vars) const noexcept {
  return {get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>>
GeneralizedHarmonicSolution<SolutionType>::variables(
    const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,
    tmpl::list<
        gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>> /*meta*/,
    const IntermediateVars&& intermediate_vars) const noexcept {
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(intermediate_vars);
  const auto& shift =
      get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(intermediate_vars);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(
          intermediate_vars);

  return {gr::spacetime_metric(lapse, shift, spatial_metric)};
}

template <typename SolutionType>
tuples::TaggedTuple<GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>
GeneralizedHarmonicSolution<SolutionType>::variables(
    const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,
    tmpl::list<GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>> /*meta*/,
    const IntermediateVars&& intermediate_vars) const noexcept {
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(intermediate_vars);
  const auto& deriv_lapse =
      get<typename GeneralizedHarmonicSolution<SolutionType>::DerivLapse>(
          intermediate_vars);

  const auto& shift =
      get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(intermediate_vars);
  const auto& deriv_shift =
      get<typename GeneralizedHarmonicSolution<SolutionType>::DerivShift>(
          intermediate_vars);

  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(
          intermediate_vars);
  const auto& deriv_spatial_metric = get<
      typename GeneralizedHarmonicSolution<SolutionType>::DerivSpatialMetric>(
      intermediate_vars);

  return {GeneralizedHarmonic::phi(lapse, deriv_lapse, shift, deriv_shift,
                                   spatial_metric, deriv_spatial_metric)};
}

template <typename SolutionType>
tuples::TaggedTuple<GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>
GeneralizedHarmonicSolution<SolutionType>::variables(
    const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,
    tmpl::list<GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>> /*meta*/,
    const IntermediateVars&& intermediate_vars) const noexcept {
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(intermediate_vars);
  const auto& dt_lapse =
      get<typename GeneralizedHarmonicSolution<SolutionType>::TimeDerivLapse>(
          intermediate_vars);
  const auto& deriv_lapse =
      get<typename GeneralizedHarmonicSolution<SolutionType>::DerivLapse>(
          intermediate_vars);

  const auto& shift =
      get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(intermediate_vars);
  const auto& dt_shift =
      get<typename GeneralizedHarmonicSolution<SolutionType>::TimeDerivShift>(
          intermediate_vars);
  const auto& deriv_shift =
      get<typename GeneralizedHarmonicSolution<SolutionType>::DerivShift>(
          intermediate_vars);

  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(
          intermediate_vars);
  const auto& dt_spatial_metric = get<typename GeneralizedHarmonicSolution<
      SolutionType>::TimeDerivSpatialMetric>(intermediate_vars);
  const auto& deriv_spatial_metric = get<
      typename GeneralizedHarmonicSolution<SolutionType>::DerivSpatialMetric>(
      intermediate_vars);

  const auto& phi =
      GeneralizedHarmonic::phi(lapse, deriv_lapse, shift, deriv_shift,
                               spatial_metric, deriv_spatial_metric);

  return {GeneralizedHarmonic::pi(lapse, dt_lapse, shift, dt_shift,
                                  spatial_metric, dt_spatial_metric, phi)};
}
}  // namespace Solutions
}  // namespace GeneralizedHarmonic

#define STYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template tuples::TaggedTuple<gr::Tags::Lapse<DataVector>>                  \
  GeneralizedHarmonic::Solutions::GeneralizedHarmonicSolution<STYPE(         \
      data)>::variables(const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,   \
                        tmpl::list<gr::Tags::Lapse<DataVector>> /*meta*/,    \
                        const IntermediateVars&& intermediate_vars)          \
      const noexcept;                                                        \
  template tuples::TaggedTuple<                                              \
      typename GeneralizedHarmonic::Solutions::GeneralizedHarmonicSolution<  \
          STYPE(data)>::TimeDerivLapse>                                      \
  GeneralizedHarmonic::Solutions::GeneralizedHarmonicSolution<STYPE(         \
      data)>::variables(const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,   \
                        tmpl::list<typename GeneralizedHarmonic::Solutions:: \
                                       GeneralizedHarmonicSolution<STYPE(    \
                                           data)>::TimeDerivLapse> /*meta*/, \
                        const IntermediateVars&& intermediate_vars)          \
      const noexcept;                                                        \
  template tuples::TaggedTuple<                                              \
      typename GeneralizedHarmonic::Solutions::GeneralizedHarmonicSolution<  \
          STYPE(data)>::DerivLapse>                                          \
  GeneralizedHarmonic::Solutions::GeneralizedHarmonicSolution<STYPE(         \
      data)>::variables(const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,   \
                        tmpl::list<typename GeneralizedHarmonic::Solutions:: \
                                       GeneralizedHarmonicSolution<STYPE(    \
                                           data)>::DerivLapse> /*meta*/,     \
                        const IntermediateVars&& intermediate_vars)          \
      const noexcept;                                                        \
  template tuples::TaggedTuple<                                              \
      gr::Tags::Shift<3, Frame::Inertial, DataVector>>                       \
  GeneralizedHarmonic::Solutions::GeneralizedHarmonicSolution<STYPE(         \
      data)>::variables(const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,   \
                        tmpl::list<gr::Tags::Shift<3, Frame::Inertial,       \
                                                   DataVector>> /*meta*/,    \
                        const IntermediateVars&& intermediate_vars)          \
      const noexcept;                                                        \
  template tuples::TaggedTuple<                                              \
      typename GeneralizedHarmonic::Solutions::GeneralizedHarmonicSolution<  \
          STYPE(data)>::TimeDerivShift>                                      \
  GeneralizedHarmonic::Solutions::GeneralizedHarmonicSolution<STYPE(         \
      data)>::variables(const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,   \
                        tmpl::list<typename GeneralizedHarmonic::Solutions:: \
                                       GeneralizedHarmonicSolution<STYPE(    \
                                           data)>::TimeDerivShift> /*meta*/, \
                        const IntermediateVars&& intermediate_vars)          \
      const noexcept;                                                        \
  template tuples::TaggedTuple<                                              \
      typename GeneralizedHarmonic::Solutions::GeneralizedHarmonicSolution<  \
          STYPE(data)>::DerivShift>                                          \
  GeneralizedHarmonic::Solutions::GeneralizedHarmonicSolution<STYPE(         \
      data)>::variables(const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,   \
                        tmpl::list<typename GeneralizedHarmonic::Solutions:: \
                                       GeneralizedHarmonicSolution<STYPE(    \
                                           data)>::DerivShift> /*meta*/,     \
                        const IntermediateVars&& intermediate_vars)          \
      const noexcept;                                                        \
  template tuples::TaggedTuple<                                              \
      gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>               \
  GeneralizedHarmonic::Solutions::GeneralizedHarmonicSolution<STYPE(         \
      data)>::variables(const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,   \
                        tmpl::list<gr::Tags::SpatialMetric<                  \
                            3, Frame::Inertial, DataVector>> /*meta*/,       \
                        const IntermediateVars&& intermediate_vars)          \
      const noexcept;                                                        \
  template tuples::TaggedTuple<                                              \
      typename GeneralizedHarmonic::Solutions::GeneralizedHarmonicSolution<  \
          STYPE(data)>::TimeDerivSpatialMetric>                              \
  GeneralizedHarmonic::Solutions::GeneralizedHarmonicSolution<STYPE(data)>:: \
      variables(const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,           \
                tmpl::list<typename GeneralizedHarmonic::Solutions::         \
                               GeneralizedHarmonicSolution<STYPE(            \
                                   data)>::TimeDerivSpatialMetric> /*meta*/, \
                const IntermediateVars&& intermediate_vars) const noexcept;  \
  template tuples::TaggedTuple<                                              \
      typename GeneralizedHarmonic::Solutions::GeneralizedHarmonicSolution<  \
          STYPE(data)>::DerivSpatialMetric>                                  \
  GeneralizedHarmonic::Solutions::GeneralizedHarmonicSolution<STYPE(data)>:: \
      variables(const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,           \
                tmpl::list<typename GeneralizedHarmonic::Solutions::         \
                               GeneralizedHarmonicSolution<STYPE(            \
                                   data)>::DerivSpatialMetric> /*meta*/,     \
                const IntermediateVars&& intermediate_vars) const noexcept;  \
  template tuples::TaggedTuple<                                              \
      gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>>        \
  GeneralizedHarmonic::Solutions::GeneralizedHarmonicSolution<STYPE(         \
      data)>::variables(const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,   \
                        tmpl::list<gr::Tags::InverseSpatialMetric<           \
                            3, Frame::Inertial, DataVector>> /*meta*/,       \
                        const IntermediateVars&& intermediate_vars)          \
      const noexcept;                                                        \
  template tuples::TaggedTuple<                                              \
      gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>>          \
  GeneralizedHarmonic::Solutions::GeneralizedHarmonicSolution<STYPE(         \
      data)>::variables(const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,   \
                        tmpl::list<gr::Tags::ExtrinsicCurvature<             \
                            3, Frame::Inertial, DataVector>> /*meta*/,       \
                        const IntermediateVars&& intermediate_vars)          \
      const noexcept;                                                        \
  template tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DataVector>>   \
  GeneralizedHarmonic::Solutions::GeneralizedHarmonicSolution<STYPE(data)>:: \
      variables(                                                             \
          const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,                 \
          tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataVector>> /*meta*/,   \
          const IntermediateVars&& intermediate_vars) const noexcept;        \
  template tuples::TaggedTuple<                                              \
      gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>>             \
  GeneralizedHarmonic::Solutions::GeneralizedHarmonicSolution<STYPE(         \
      data)>::variables(const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,   \
                        tmpl::list<gr::Tags::SpacetimeMetric<                \
                            3, Frame::Inertial, DataVector>> /*meta*/,       \
                        const IntermediateVars&& intermediate_vars)          \
      const noexcept;                                                        \
  template tuples::TaggedTuple<                                              \
      GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>                     \
  GeneralizedHarmonic::Solutions::GeneralizedHarmonicSolution<STYPE(data)>:: \
      variables(                                                             \
          const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,                 \
          tmpl::list<                                                        \
              GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>> /*meta*/,   \
          const IntermediateVars&& intermediate_vars) const noexcept;        \
  template tuples::TaggedTuple<                                              \
      GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>                    \
  GeneralizedHarmonic::Solutions::GeneralizedHarmonicSolution<STYPE(data)>:: \
      variables(                                                             \
          const tnsr::I<DataVector, 3>& /*x*/, double /*t*/,                 \
          tmpl::list<                                                        \
              GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>> /*meta*/,  \
          const IntermediateVars&& intermediate_vars) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (gr::Solutions::Minkowski<3>,
                                      gr::Solutions::KerrSchild))

#undef DIM
#undef STYPE
#undef INSTANTIATE
/// \endcond
