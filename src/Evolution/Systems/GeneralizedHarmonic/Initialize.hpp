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
    using system = typename Metavariables::system;
    using variables_tag = typename system::variables_tag;
    using simple_tags = db::AddSimpleTags<
        variables_tag, GeneralizedHarmonic::Tags::ConstraintGamma0,
        GeneralizedHarmonic::Tags::ConstraintGamma1,
        GeneralizedHarmonic::Tags::ConstraintGamma2,
        ::Tags::Interface<::Tags::InternalDirections<Dim>,
                          GeneralizedHarmonic::Tags::ConstraintGamma1>,
        ::Tags::Interface<::Tags::InternalDirections<Dim>,
                          GeneralizedHarmonic::Tags::ConstraintGamma2>>;
    using compute_tags = db::AddComputeTags<
        gr::Tags::SpatialMetricCompute<Dim, Frame::Inertial, DataVector>,
        gr::Tags::DetAndInverseSpatialMetricCompute<Dim, Frame::Inertial,
                                                    DataVector>,
        gr::Tags::InverseSpatialMetricCompute<Dim, Frame::Inertial, DataVector>,
        gr::Tags::ShiftCompute<Dim, Frame::Inertial, DataVector>,
        gr::Tags::LapseCompute<Dim, Frame::Inertial, DataVector>,
        ::Tags::Interface<::Tags::InternalDirections<Dim>,
                          gr::Tags::Lapse<DataVector>>,
        ::Tags::Interface<::Tags::InternalDirections<Dim>,
                          gr::Tags::Shift<Dim, Frame::Inertial, DataVector>>,
        ::Tags::Interface<
            ::Tags::InternalDirections<Dim>,
            gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>>,
        ::Tags::UnitFaceNormalVectorCompute<Dim, Frame::Inertial>>;

    template <typename TagsList>
    static auto initialize(
        db::DataBox<TagsList>&& box,
        const Parallel::ConstGlobalCache<Metavariables>& cache,
        const double initial_time) noexcept {
      using Vars = typename variables_tag::type;

      const size_t num_grid_points =
          db::get<::Tags::Mesh<Dim>>(box).number_of_grid_points();

      const auto& inertial_coords =
          db::get<::Tags::Coordinates<Dim, Frame::Inertial>>(box);

      // Set constraint damping parameters
      // For now, hard code these; later, make these options / AnalyticData
      // The values here are the same as in SpEC standard input files for
      // evolving a single black hole.
      const auto& r_squared = dot_product(inertial_coords, inertial_coords);
      const auto& one = exp(r_squared - r_squared);
      const auto& gamma0 =
          3.0 * exp(-0.5 * r_squared / 64.0) + 0.001 * one;
      const auto& gamma1 = -1.0 * one;
      const auto& gamma2 =
          exp(-0.5 * r_squared / 64.0) + 0.001 * one;

      // Set initial data from analytic solution
      Vars vars{num_grid_points};

      // FIXME: for now, only support Kerr Schild
      // Later, support analytic data as well
      using solution_tag = OptionTags::AnalyticSolutionBase;
      const auto& solution_vars = Parallel::get<solution_tag>(cache).variables(
          inertial_coords, initial_time,
          typename gr::Solutions::KerrSchild::template tags<DataVector>{});

      using DerivLapse = ::Tags::deriv<gr::Tags::Lapse<DataVector>,
                                       tmpl::size_t<3>, Frame::Inertial>;
      using DerivShift =
          ::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>;
      using DerivSpatialMetric =
          ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>;

      const auto& lapse = get<gr::Tags::Lapse<DataVector>>(solution_vars);
      const auto& dt_lapse =
          get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(solution_vars);
      const auto& deriv_lapse = get<DerivLapse>(solution_vars);

      const auto& shift =
          get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(solution_vars);
      const auto& dt_shift =
          get<::Tags::dt<gr::Tags::Shift<3, Frame::Inertial, DataVector>>>(
              solution_vars);
      const auto& deriv_shift = get<DerivShift>(solution_vars);

      const auto& spatial_metric =
          get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(
              solution_vars);
      const auto& dt_spatial_metric = get<
          ::Tags::dt<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>>(
          solution_vars);
      const auto& deriv_spatial_metric = get<DerivSpatialMetric>(solution_vars);

      const auto& spacetime_metric =
          gr::spacetime_metric(lapse, shift, spatial_metric);
      const auto& phi =
          GeneralizedHarmonic::phi(lapse, deriv_lapse, shift, deriv_shift,
                                   spatial_metric, deriv_spatial_metric);
      const auto& pi =
          GeneralizedHarmonic::pi(lapse, dt_lapse, shift, dt_shift,
                                  spatial_metric, dt_spatial_metric, phi);
      const tuples::TaggedTuple<gr::Tags::SpacetimeMetric<3>,
                                GeneralizedHarmonic::Tags::Pi<3>,
                                GeneralizedHarmonic::Tags::Phi<3>>
          solution_tuple(spacetime_metric, pi, phi);

      vars.assign_subset(solution_tuple);

      return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
          std::move(box), std::move(vars), std::move(gamma0), std::move(gamma1),
          std::move(gamma2));
    }
  };

  template <class Metavariables>
  using return_tag_list =
      tmpl::append<typename Initialization::Domain<Dim>::simple_tags,
                   typename VariablesTags<Metavariables>::simple_tags,
                   typename Initialization::NonConservativeInterface<
                       typename Metavariables::system>::simple_tags,
                   typename Initialization::Evolution<
                       typename Metavariables::system>::simple_tags,
                   typename Initialization::DiscontinuousGalerkin<
                       Metavariables>::simple_tags,
                   typename Initialization::MinMod<Dim>::simple_tags,
                   typename Initialization::Domain<Dim>::compute_tags,
                   typename VariablesTags<Metavariables>::compute_tags,
                   typename Initialization::NonConservativeInterface<
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
        Initialization::NonConservativeInterface<system>::initialize(
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
