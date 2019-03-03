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
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/DiscontinuousGalerkin.hpp"
#include "Evolution/Initialization/Domain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/Limiter.hpp"
#include "Evolution/Initialization/NonConservativeInterface.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace GeneralizedHarmonic {
namespace Actions {
namespace detail {
template <class T, class = cpp17::void_t<>>
struct has_analytic_solution_alias : std::false_type {};
template <class T>
struct has_analytic_solution_alias<
    T, cpp17::void_t<typename T::analytic_solution_tag>> : std::true_type {};
}  // namespace detail

// Note:  I've left the Dim and System template parameters until it is clear
// whether or not what remains is specific to this system, and what might be
// applicable to more than one system
template <size_t Dim>
struct Initialize {
  template <typename Metavariables>
  struct VariablesTags {
    using Inertial = Frame::Inertial;
    using system = typename Metavariables::system;
    using variables_tag = typename system::variables_tag;
    using damping_tag = ::Tags::Variables<
        tmpl::list<GeneralizedHarmonic::Tags::ConstraintGamma0,
                   GeneralizedHarmonic::Tags::ConstraintGamma1,
                   GeneralizedHarmonic::Tags::ConstraintGamma2>>;

    using simple_tags = db::AddSimpleTags<
        variables_tag, damping_tag,
        GeneralizedHarmonic::Tags::TimeDerivGaugeH<Dim, Inertial>>;
    using compute_tags = db::AddComputeTags<
        gr::Tags::SpatialMetricCompute<Dim, Inertial, DataVector>,
        gr::Tags::DetAndInverseSpatialMetricCompute<Dim, Inertial, DataVector>,
        gr::Tags::InverseSpatialMetricCompute<Dim, Inertial, DataVector>,
        gr::Tags::ShiftCompute<Dim, Inertial, DataVector>,
        gr::Tags::LapseCompute<Dim, Inertial, DataVector>,
        gr::Tags::SqrtDetSpatialMetricCompute<Dim, Inertial, DataVector>,
        gr::Tags::SpacetimeNormalOneFormCompute<Dim, Inertial, DataVector>,
        gr::Tags::SpacetimeNormalVectorCompute<Dim, Inertial, DataVector>,
        gr::Tags::InverseSpacetimeMetricCompute<3, Inertial, DataVector>,
        GeneralizedHarmonic::Tags::DerivSpatialMetricCompute<Dim, Inertial>,
        GeneralizedHarmonic::Tags::DerivLapseCompute<Dim, Inertial>,
        GeneralizedHarmonic::Tags::DerivShiftCompute<Dim, Inertial>,
        GeneralizedHarmonic::Tags::TimeDerivSpatialMetricCompute<Dim, Inertial>,
        GeneralizedHarmonic::Tags::TimeDerivLapseCompute<Dim, Inertial>,
        GeneralizedHarmonic::Tags::TimeDerivShiftCompute<Dim, Inertial>,
        GeneralizedHarmonic::Tags::DerivativesOfSpacetimeMetricCompute<
            Dim, Inertial>,
        gr::Tags::SpacetimeChristoffelFirstKindCompute<Dim, Inertial,
                                                       DataVector>,
        gr::Tags::SpacetimeChristoffelSecondKindCompute<Dim, Inertial,
                                                        DataVector>,
        gr::Tags::TraceSpacetimeChristoffelFirstKindCompute<Dim, Inertial,
                                                            DataVector>,
        gr::Tags::SpatialChristoffelFirstKindCompute<Dim, Inertial, DataVector>,
        gr::Tags::SpatialChristoffelSecondKindCompute<Dim, Inertial,
                                                      DataVector>,
        gr::Tags::TraceSpatialChristoffelFirstKindCompute<Dim, Inertial,
                                                          DataVector>,
        GeneralizedHarmonic::Tags::ExtrinsicCurvatureCompute<Dim, Inertial>,
        GeneralizedHarmonic::Tags::TraceExtrinsicCurvatureCompute<Dim,
                                                                  Inertial>,
        GeneralizedHarmonic::Tags::GaugeHCompute<Dim, Inertial>,
        ::Tags::deriv<GeneralizedHarmonic::Tags::GaugeH<Dim, Inertial>,
                      tmpl::size_t<Dim>, Inertial>,
        GeneralizedHarmonic::Tags::SpacetimeDerivGaugeHCompute<Dim, Inertial>>;

    /* NOT YET ADDED BUT NEEDED BY ComputeDuDt
          ::Tags::deriv<Tags::Pi<Dim>, tmpl::size_t<Dim>, Inertial>,
          ::Tags::deriv<Tags::Phi<Dim>, tmpl::size_t<Dim>, Inertial>,
          Tags::SpacetimeDerivGaugeH<Dim>,

      NEEDED BY ComputeNormalDotFluxes
          gr::Tags::SpacetimeMetric<Dim>, Tags::Pi<Dim>, Tags::Phi<Dim>,
      Tags::ConstraintGamma1, Tags::ConstraintGamma2, gr::Tags::Lapse<>,
      gr::Tags::Shift<Dim>, gr::Tags::InverseSpatialMetric<Dim>,
      ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim, Inertial>>

      NEEDED BY UpwindFlux
      Tags::UPsi<Dim, Inertial>, Tags::UZero<Dim, Inertial>,
      Tags::UPlus<Dim, Inertial>, Tags::UMinus<Dim, Inertial>,
      ::Tags::CharSpeed<Tags::UPsi<Dim, Inertial>>,
      ::Tags::CharSpeed<Tags::UZero<Dim, Inertial>>,
      ::Tags::CharSpeed<Tags::UPlus<Dim, Inertial>>,
      ::Tags::CharSpeed<Tags::UMinus<Dim, Inertial>>,
      Tags::ConstraintGamma2, ::Tags::UnitFaceNormal<Dim, Inertial>
    */

    template <typename TagsList>
    static auto initialize(
        db::DataBox<TagsList>&& box,
        const Parallel::ConstGlobalCache<Metavariables>& cache,
        const double initial_time) noexcept {
      using Vars = typename variables_tag::type;
      using DampingVars = typename damping_tag::type;

      const size_t num_grid_points =
          db::get<::Tags::Mesh<Dim>>(box).number_of_grid_points();

      const auto& inertial_coords =
          db::get<::Tags::Coordinates<Dim, Inertial>>(box);

      // Set constraint damping parameters
      // For now, hard code these; later, make these options / AnalyticData
      // The values here are the same as in SpEC standard input files for
      // evolving a single black hole.
      const DataVector r_squared =
          get(dot_product(inertial_coords, inertial_coords));
      const DataVector one = exp(0.0 * r_squared);
      const typename GeneralizedHarmonic::Tags::ConstraintGamma0::type gamma0{
          3.0 * exp(-0.5 * r_squared / 64.0) + 0.001 * one};
      const auto& gamma1 = make_with_value<
          typename GeneralizedHarmonic::Tags::ConstraintGamma1::type>(
          inertial_coords, -1.);
      const typename GeneralizedHarmonic::Tags::ConstraintGamma2::type gamma2{
          exp(-0.5 * r_squared / 64.0) + 0.001 * one};

      // Set initial data from analytic solution
      Vars vars{num_grid_points};
      DampingVars damping_vars{num_grid_points};
      make_overloader(
          [ initial_time, &inertial_coords, &gamma0, &gamma1, &gamma2 ](
              std::true_type /*is_analytic_solution*/,
              const gsl::not_null<Vars*> local_vars,
              const gsl::not_null<DampingVars*> local_damping_vars,
              const auto& local_cache) noexcept {
            using analytic_solution_tag = OptionTags::AnalyticSolutionBase;
            /*
             * It is assumed here that the analytic solution makes available the
             * following foliation-related variables (only):
             * 1. Lapse,
             * 2. Shift,
             * 3. SpatialMetric,
             * and their spatial + temporal derivatives.
             */
            const auto& solution_vars =
                Parallel::get<analytic_solution_tag>(local_cache)
                    .variables(
                        inertial_coords, initial_time,
                        typename gr::Solutions::KerrSchild::template tags<
                            DataVector>{});
            // First fetch lapse, shift, spatial metric and their derivs
            const auto& lapse = get<gr::Tags::Lapse<DataVector>>(solution_vars);
            const auto& dt_lapse =
                get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(solution_vars);
            const auto& deriv_lapse =

                get<::Tags::deriv<gr::Tags::Lapse<DataVector>,
                                  tmpl::size_t<Dim>, Inertial>>(solution_vars);

            const auto& shift =
                get<gr::Tags::Shift<Dim, Inertial, DataVector>>(solution_vars);
            const auto& dt_shift =
                get<::Tags::dt<gr::Tags::Shift<Dim, Inertial, DataVector>>>(
                    solution_vars);
            const auto& deriv_shift =
                get<::Tags::deriv<gr::Tags::Shift<Dim, Inertial, DataVector>,
                                  tmpl::size_t<Dim>, Inertial>>(solution_vars);

            const auto& spatial_metric =
                get<gr::Tags::SpatialMetric<Dim, Inertial, DataVector>>(
                    solution_vars);
            const auto& dt_spatial_metric = get<
                ::Tags::dt<gr::Tags::SpatialMetric<Dim, Inertial, DataVector>>>(
                solution_vars);
            const auto& deriv_spatial_metric = get<::Tags::deriv<
                gr::Tags::SpatialMetric<Dim, Inertial, DataVector>,
                tmpl::size_t<Dim>, Inertial>>(solution_vars);

            // Next, compute Gh evolution variables from them
            const auto& spacetime_metric =
                ::gr::spacetime_metric<Dim, Inertial, DataVector>(
                    lapse, shift, spatial_metric);
            const auto& phi =
                GeneralizedHarmonic::phi<Dim, Inertial, DataVector>(
                    lapse, deriv_lapse, shift, deriv_shift, spatial_metric,
                    deriv_spatial_metric);
            const auto& pi = GeneralizedHarmonic::pi<Dim, Inertial, DataVector>(
                lapse, dt_lapse, shift, dt_shift, spatial_metric,
                dt_spatial_metric, phi);

            const tuples::TaggedTuple<gr::Tags::SpacetimeMetric<Dim>,
                                      GeneralizedHarmonic::Tags::Phi<Dim>,
                                      GeneralizedHarmonic::Tags::Pi<Dim>>
                solution_tuple(spacetime_metric, phi, pi);

            local_vars->assign_subset(solution_tuple);

            const tuples::TaggedTuple<
                GeneralizedHarmonic::Tags::ConstraintGamma0,
                GeneralizedHarmonic::Tags::ConstraintGamma1,
                GeneralizedHarmonic::Tags::ConstraintGamma2>
                damping_tuple(gamma0, gamma1, gamma2);

            local_damping_vars->assign_subset(damping_tuple);
          },
          [&inertial_coords, &gamma0, &gamma1, &gamma2 ](
              std::false_type /*is_analytic_solution*/,
              const gsl::not_null<Vars*> local_vars,
              const gsl::not_null<DampingVars*> local_damping_vars,
              const auto& local_cache) noexcept {
            using analytic_data_tag = OptionTags::AnalyticDataBase;
            local_vars->assign_subset(
                Parallel::get<analytic_data_tag>(local_cache)
                    .variables(inertial_coords, typename Vars::tags_list{}));
            const tuples::TaggedTuple<
                GeneralizedHarmonic::Tags::ConstraintGamma0,
                GeneralizedHarmonic::Tags::ConstraintGamma1,
                GeneralizedHarmonic::Tags::ConstraintGamma2>
                damping_tuple(gamma0, gamma1, gamma2);

            local_damping_vars->assign_subset(damping_tuple);
          })(detail::has_analytic_solution_alias<Metavariables>{},
             make_not_null(&vars), make_not_null(&damping_vars), cache);

      // Set the time derivatives of GaugeH
      const auto& dt_gauge_source =
          make_with_value<tnsr::a<DataVector, Dim, Inertial>>(gamma0, 0.0);

      return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
          std::move(box), std::move(vars), std::move(damping_vars),
          std::move(dt_gauge_source));
    }
  };

  template <class Metavariables>
  using return_tag_list =
      tmpl::append<typename Initialization::Domain<Dim>::simple_tags,
                   typename VariablesTags<Metavariables>::simple_tags,
                   typename Initialization::InterfaceForNonConservativeSystem<
                       typename Metavariables::system>::simple_tags,
                   typename Initialization::Evolution<
                       typename Metavariables::system>::simple_tags,
                   typename Initialization::DiscontinuousGalerkin<
                       Metavariables>::simple_tags,
                   typename Initialization::MinMod<Dim>::simple_tags,
                   typename Initialization::Domain<Dim>::compute_tags,
                   typename VariablesTags<Metavariables>::compute_tags,
                   typename Initialization::InterfaceForNonConservativeSystem<
                       typename Metavariables::system>::compute_tags,
                   typename Initialization::Evolution<
                       typename Metavariables::system>::compute_tags,
                   typename Initialization::DiscontinuousGalerkin<
                       Metavariables>::compute_tags,
                   typename Initialization::MinMod<Dim>::compute_tags>;

  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementIndex<Dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    std::vector<std::array<size_t, Dim>> initial_extents,
                    Domain<Dim, Frame::Inertial> domain,
                    const double initial_time, const double initial_dt,
                    const double initial_slab_size) noexcept {
    using system = typename Metavariables::system;
    auto domain_box = Initialization::Domain<Dim>::initialize(
        db::DataBox<tmpl::list<>>{}, array_index, initial_extents, domain);
    auto variables_box = VariablesTags<Metavariables>::initialize(
        std::move(domain_box), cache, initial_time);
    auto domain_interface_box =
        Initialization::InterfaceForNonConservativeSystem<system>::initialize(
            std::move(variables_box));
    auto evolution_box = Initialization::Evolution<system>::initialize(
        std::move(domain_interface_box), cache, initial_time, initial_dt,
        initial_slab_size);
    auto dg_box =
        Initialization::DiscontinuousGalerkin<Metavariables>::initialize(
            std::move(evolution_box), initial_extents);
    auto limiter_box =
        Initialization::MinMod<Dim>::initialize(std::move(dg_box));
    return std::make_tuple(std::move(limiter_box));
  }
};
}  // namespace Actions
}  // namespace GeneralizedHarmonic
