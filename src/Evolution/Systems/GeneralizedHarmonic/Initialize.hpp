// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>  // IWYU pragma: keep
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Initialization/DiscontinuousGalerkin.hpp"
#include "Evolution/Initialization/Domain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/Limiter.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace GeneralizedHarmonic {
namespace Actions {
template <size_t Dim>
struct InitializeConstraintsTags {
  using initialization_option_tags = tmpl::list<>;

  using Inertial = Frame::Inertial;
  using simple_tags = db::AddSimpleTags<>;
  using compute_tags = db::AddComputeTags<
      GeneralizedHarmonic::Tags::GaugeConstraintCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::FConstraintCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::TwoIndexConstraintCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::FourIndexConstraintCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::ConstraintEnergyCompute<Dim, Inertial>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(
        Initialization::merge_into_databox<InitializeConstraintsTags,
                                           simple_tags, compute_tags>(
            std::move(box)));
  }
};

template <size_t Dim>
struct InitializeGHAnd3Plus1VariablesTags {
  using initialization_option_tags = tmpl::list<>;

  using Inertial = Frame::Inertial;
  using system = GeneralizedHarmonic::System<Dim>;
  using variables_tag = typename system::variables_tag;

  // `extras_tag` can be used with `Tags::DerivCompute` to get spatial
  // derivatives of quantities that are otherwise not available within
  // a `Variables<>` container.
  using extras_tag = ::Tags::Variables<
      tmpl::list<GeneralizedHarmonic::Tags::GaugeH<Dim, Frame::Inertial>>>;

  using simple_tags = db::AddSimpleTags<
      extras_tag, ::Tags::dt<GeneralizedHarmonic::Tags::GaugeH<Dim, Inertial>>>;
  using compute_tags = db::AddComputeTags<
      gr::Tags::SpatialMetricCompute<Dim, Inertial, DataVector>,
      gr::Tags::DetAndInverseSpatialMetricCompute<Dim, Inertial, DataVector>,
      gr::Tags::ShiftCompute<Dim, Inertial, DataVector>,
      gr::Tags::LapseCompute<Dim, Inertial, DataVector>,
      gr::Tags::SqrtDetSpatialMetricCompute<Dim, Inertial, DataVector>,
      gr::Tags::SpacetimeNormalOneFormCompute<Dim, Inertial, DataVector>,
      gr::Tags::SpacetimeNormalVectorCompute<Dim, Inertial, DataVector>,
      gr::Tags::InverseSpacetimeMetricCompute<Dim, Inertial, DataVector>,
      GeneralizedHarmonic::Tags::DerivSpatialMetricCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::DerivLapseCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::DerivShiftCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::TimeDerivSpatialMetricCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::TimeDerivLapseCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::TimeDerivShiftCompute<Dim, Inertial>,
      gr::Tags::DerivativesOfSpacetimeMetricCompute<Dim, Inertial>,
      gr::Tags::DerivSpacetimeMetricCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::ThreeIndexConstraintCompute<Dim, Inertial>,
      gr::Tags::SpacetimeChristoffelFirstKindCompute<Dim, Inertial, DataVector>,
      gr::Tags::SpacetimeChristoffelSecondKindCompute<Dim, Inertial,
                                                      DataVector>,
      gr::Tags::TraceSpacetimeChristoffelFirstKindCompute<Dim, Inertial,
                                                          DataVector>,
      gr::Tags::SpatialChristoffelFirstKindCompute<Dim, Inertial, DataVector>,
      gr::Tags::SpatialChristoffelSecondKindCompute<Dim, Inertial, DataVector>,
      gr::Tags::TraceSpatialChristoffelFirstKindCompute<Dim, Inertial,
                                                        DataVector>,
      GeneralizedHarmonic::Tags::ExtrinsicCurvatureCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::TraceExtrinsicCurvatureCompute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::ConstraintGamma0Compute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::ConstraintGamma1Compute<Dim, Inertial>,
      GeneralizedHarmonic::Tags::ConstraintGamma2Compute<Dim, Inertial>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    // First add 3+1 quantities to the box
    auto _box =
        db::create_from<db::RemoveTags<>, db::AddSimpleTags<>, compute_tags>(
            std::move(box));

    // Then compute gauge related quantities
    const size_t num_grid_points =
        db::get<::Tags::Mesh<Dim>>(_box).number_of_grid_points();

    // fetch lapse, shift, spatial metric and their derivs through compute tags
    const auto& lapse = get<gr::Tags::Lapse<DataVector>>(_box);
    const auto& dt_lapse = get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(_box);
    const auto& deriv_lapse =
        get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<Dim>,
                          Inertial>>(_box);
    const auto& shift = get<gr::Tags::Shift<Dim, Inertial, DataVector>>(_box);
    const auto& dt_shift =
        get<::Tags::dt<gr::Tags::Shift<Dim, Inertial, DataVector>>>(_box);
    const auto& deriv_shift =
        get<::Tags::deriv<gr::Tags::Shift<Dim, Inertial, DataVector>,
                          tmpl::size_t<Dim>, Inertial>>(_box);
    const auto& spatial_metric =
        get<gr::Tags::SpatialMetric<Dim, Inertial, DataVector>>(_box);
    const auto& trace_extrinsic_curvature =
        get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(_box);
    const auto& trace_christoffel_last_indices = get<
        gr::Tags::TraceSpatialChristoffelFirstKind<Dim, Inertial, DataVector>>(
        _box);

    // call compute item for the gauge source function
    using ExtraVars = db::item_type<extras_tag>;
    ExtraVars extra_vars{num_grid_points};
    get<GeneralizedHarmonic::Tags::GaugeH<Dim, Inertial>>(extra_vars) =
        GeneralizedHarmonic::Tags::GaugeHImplicitFrom3p1QuantitiesCompute<
            Dim, Inertial>::function(lapse, dt_lapse, deriv_lapse, shift,
                                     dt_shift, deriv_shift, spatial_metric,
                                     trace_extrinsic_curvature,
                                     trace_christoffel_last_indices);

    // set time derivatives of GaugeH = 0
    const auto& dt_gauge_source =
        make_with_value<tnsr::a<DataVector, Dim, Inertial>>(lapse, 0.);

    // Finally, insert gauge related quantities to the box
    return std::make_tuple(
        Initialization::merge_into_databox<
            InitializeGHAnd3Plus1VariablesTags,
            // Because DerivCompute<GaugeH> operates on GaugeH wrapped in a
            // Variables container, we inserted GaugeH that way into the box
            // (through `extras_tag`), instead of adding `GaugeHCompute`
            // item here.
            simple_tags,
            db::AddComputeTags<
                ::Tags::DerivCompute<
                    extras_tag,
                    ::Tags::InverseJacobian<::Tags::ElementMap<Dim, Inertial>,
                                            ::Tags::LogicalCoordinates<Dim>>>,
                GeneralizedHarmonic::Tags::SpacetimeDerivGaugeHCompute<
                    Dim, Inertial>>>(std::move(_box), std::move(extra_vars),
                                     std::move(dt_gauge_source)));
  }
};
}  // namespace Actions
}  // namespace GeneralizedHarmonic
