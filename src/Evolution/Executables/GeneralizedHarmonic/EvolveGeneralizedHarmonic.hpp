// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Error.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Evolution/Actions/ComputeTimeDerivative.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/Filtering.hpp"
#include "Evolution/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"  // IWYU pragma: keep
#include "Evolution/EventsAndTriggers/Event.hpp"
#include "Evolution/EventsAndTriggers/EventsAndTriggers.hpp"  // IWYU pragma: keep
#include "Evolution/Initialization/DiscontinuousGalerkin.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/Interface.hpp"
#include "Evolution/Initialization/Limiter.hpp"
#include "Evolution/Initialization/NonconservativeSystem.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"  // IWYU pragma: keep // for UpwindFlux
#include "Evolution/Systems/GeneralizedHarmonic/Initialize.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Observe.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "IO/Observer/Actions.hpp"  // IWYU pragma: keep
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/RegisterObservers.hpp"
#include "IO/Observer/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyBoundaryFluxesLocalTimeStepping.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyFluxes.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ComputeNonconservativeBoundaryFluxes.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ImposeBoundaryConditions.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/LocalLaxFriedrichs.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Time/Actions/AdvanceTime.hpp"            // IWYU pragma: keep
#include "Time/Actions/ChangeStepSize.hpp"         // IWYU pragma: keep
#include "Time/Actions/RecordTimeStepperData.hpp"  // IWYU pragma: keep
#include "Time/Actions/SelfStartActions.hpp"       // IWYU pragma: keep
#include "Time/Actions/UpdateU.hpp"                // IWYU pragma: keep
#include "Time/StepChoosers/Cfl.hpp"               // IWYU pragma: keep
#include "Time/StepChoosers/Constant.hpp"          // IWYU pragma: keep
#include "Time/StepChoosers/Increase.hpp"          // IWYU pragma: keep
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
// IWYU pragma: no_forward_declare MathFunction
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class CProxy_ConstGlobalCache;
}  // namespace Parallel
/// \endcond

struct EvolutionMetavars {
  // Customization/"input options" to simulation
  static constexpr int dim = 3;
  using Inertial = Frame::Inertial;
  using system = GeneralizedHarmonic::System<dim>;
  using temporal_id = Tags::TimeId;
  static constexpr bool local_time_stepping = false;
  using analytic_solution_tag = OptionTags::AnalyticSolution<
      GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>>;
  using boundary_condition_tag = analytic_solution_tag;
  using normal_dot_numerical_flux = OptionTags::NumericalFlux<
      // dg::NumericalFluxes::LocalLaxFriedrichs<system>>;
      GeneralizedHarmonic::UpwindFlux<dim>>;
  using events = tmpl::list<>;
  using triggers = Triggers::time_triggers;

  // A tmpl::list of tags to be added to the ConstGlobalCache by the
  // metavariables
  using const_global_cache_tag_list =
      tmpl::list<analytic_solution_tag,
                 OptionTags::TypedTimeStepper<tmpl::conditional_t<
                     local_time_stepping, LtsTimeStepper, TimeStepper>>,
                 OptionTags::EventsAndTriggers<events, triggers>>;
  using domain_creator_tag = OptionTags::DomainCreator<dim, Inertial>;

  using step_choosers = tmpl::list<StepChoosers::Registrars::Cfl<dim, Inertial>,
                                   StepChoosers::Registrars::Constant,
                                   StepChoosers::Registrars::Increase>;

  struct ObservationType {};
  using element_observation_type = ObservationType;

  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::list<GeneralizedHarmonic::Actions::Observe>>;

  using compute_rhs = tmpl::flatten<tmpl::list<
      dg::Actions::ComputeNonconservativeBoundaryFluxes<
          Tags::InternalDirections<dim>>,
      dg::Actions::SendDataForFluxes<EvolutionMetavars>,
      Actions::ComputeTimeDerivative,
      dg::Actions::ComputeNonconservativeBoundaryFluxes<
          Tags::BoundaryDirectionsInterior<dim>>,
      // dg::Actions::ImposeDirichletBoundaryConditions<EvolutionMetavars>,
      dg::Actions::ReceiveDataForFluxes<EvolutionMetavars>,
      tmpl::conditional_t<local_time_stepping, tmpl::list<>,
                          dg::Actions::ApplyFluxes>,
      Actions::RecordTimeStepperData>>;
  using update_variables = tmpl::flatten<tmpl::list<
      tmpl::conditional_t<local_time_stepping,
                          dg::Actions::ApplyBoundaryFluxesLocalTimeStepping,
                          tmpl::list<>>,
      GeneralizedHarmonic::Actions::
          ImposeConstraintPreservingBoundaryConditions<EvolutionMetavars>,
      Actions::UpdateU,
      dg::Actions::ExponentialFilter<
          0, typename system::variables_tag::type::tags_list>>>;

  enum class Phase {
    Initialization,
    InitializeTimeStepperHistory,
    RegisterWithObserver,
    Evolve,
    Exit
  };

  using initialization_actions = tmpl::list<
      dg::Actions::InitializeDomain<dim>,
      Initialization::Actions::NonconservativeSystem,
      GeneralizedHarmonic::Actions::InitializeGHAnd3Plus1VariablesTags<dim>,
      Initialization::Actions::Interface<
          system,
          Initialization::slice_tags_to_face<
              typename system::variables_tag,
              gr::Tags::SpatialMetricCompute<dim, Inertial, DataVector>,
              gr::Tags::DetAndInverseSpatialMetricCompute<dim, Inertial,
                                                          DataVector>,
              gr::Tags::ShiftCompute<dim, Inertial, DataVector>,
              gr::Tags::LapseCompute<dim, Inertial, DataVector>>,
          Initialization::slice_tags_to_exterior<
              typename system::variables_tag,
              gr::Tags::SpatialMetricCompute<dim, Inertial, DataVector>,
              gr::Tags::DetAndInverseSpatialMetricCompute<dim, Inertial,
                                                          DataVector>,
              gr::Tags::ShiftCompute<dim, Inertial, DataVector>,
              gr::Tags::LapseCompute<dim, Inertial, DataVector>>,
          Initialization::face_compute_tags<
              ::Tags::BoundaryCoordinates<dim, Inertial>,
              GeneralizedHarmonic::Tags::ConstraintGamma0Compute<dim, Inertial>,
              GeneralizedHarmonic::Tags::ConstraintGamma1Compute<dim, Inertial>,
              GeneralizedHarmonic::Tags::ConstraintGamma2Compute<dim, Inertial>,
              GeneralizedHarmonic::CharacteristicFieldsCompute<dim, Inertial>,
              GeneralizedHarmonic::CharacteristicSpeedsCompute<dim, Inertial>>,
          Initialization::exterior_compute_tags<
              GeneralizedHarmonic::Tags::ConstraintGamma0Compute<dim, Inertial>,
              GeneralizedHarmonic::Tags::ConstraintGamma1Compute<dim, Inertial>,
              GeneralizedHarmonic::Tags::ConstraintGamma2Compute<dim, Inertial>,
              GeneralizedHarmonic::CharacteristicFieldsCompute<dim, Inertial>,
              GeneralizedHarmonic::CharacteristicSpeedsCompute<dim, Inertial>>,
          false>,
      Initialization::Actions::Evolution<system>,
      GeneralizedHarmonic::Actions::InitializeConstraintsTags<dim>,
      Initialization::Actions::DiscontinuousGalerkin<EvolutionMetavars, false>,
      Initialization::Actions::Minmod<dim>,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using component_list = tmpl::list<
      observers::Observer<EvolutionMetavars>,
      observers::ObserverWriter<EvolutionMetavars>,
      DgElementArray<
          EvolutionMetavars,
          tmpl::list<
              Parallel::PhaseActions<Phase, Phase::Initialization,
                                     initialization_actions>,

              Parallel::PhaseActions<
                  Phase, Phase::InitializeTimeStepperHistory,
                  tmpl::flatten<tmpl::list<SelfStart::self_start_procedure<
                      compute_rhs, update_variables>>>>,

              Parallel::PhaseActions<
                  Phase, Phase::RegisterWithObserver,
                  tmpl::list<observers::Actions::RegisterWithObservers<
                                 observers::RegisterObservers<
                                     element_observation_type>>,
                             Parallel::Actions::TerminatePhase>>,

              Parallel::PhaseActions<
                  Phase, Phase::Evolve,
                  tmpl::flatten<tmpl::list<
                      GeneralizedHarmonic::Actions::Observe,
                      Actions::RunEventsAndTriggers,
                      tmpl::conditional_t<
                          local_time_stepping,
                          Actions::ChangeStepSize<step_choosers>, tmpl::list<>>,
                      compute_rhs, update_variables, Actions::AdvanceTime>>>>,
          Parallel::ForwardAllOptionsToDataBox<
              Initialization::option_tags<initialization_actions>>>>;

  static constexpr OptionString help{
      "Evolve a generalized harmonic analytic solution.\n\n"
      "The analytic solution is: KerrSchild\n"
      "The numerical flux is:    UpwindFlux\n"};

  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
          EvolutionMetavars>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::InitializeTimeStepperHistory;
      case Phase::InitializeTimeStepperHistory:
        return Phase::RegisterWithObserver;
      case Phase::RegisterWithObserver:
        return Phase::Evolve;
      case Phase::Evolve:
        return Phase::Exit;
      case Phase::Exit:
        ERROR(
            "Should never call determine_next_phase with the current phase "
            "being 'Exit'");
      default:
        ERROR(
            "Unknown type of phase. Did you static_cast<Phase> an integral "
            "value?");
    }
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &domain::creators::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<MathFunction<1>>,
    &Parallel::register_derived_classes_with_charm<
        Event<metavariables::events>>,
    &Parallel::register_derived_classes_with_charm<
        StepChooser<metavariables::step_choosers>>,
    &Parallel::register_derived_classes_with_charm<StepController>,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        Trigger<metavariables::triggers>>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
