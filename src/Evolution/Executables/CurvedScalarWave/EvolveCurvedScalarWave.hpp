// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsCharacteresticSpeeds.hpp"
#include "ErrorHandling/Error.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Evolution/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/DiscontinuousGalerkin.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/GrTagsForHydro.hpp"
#include "Evolution/Initialization/NonconservativeSystem.hpp"
#include "Evolution/Initialization/SetVariables.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions.hpp"
#include "Evolution/Systems/CurvedScalarWave/Characteristics.hpp"
#include "Evolution/Systems/CurvedScalarWave/Constraints.hpp"
#include "Evolution/Systems/CurvedScalarWave/Equations.hpp"
#include "Evolution/Systems/CurvedScalarWave/Initialize.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/RegisterObservers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ComputeNonconservativeBoundaryFluxes.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ImposeBoundaryConditions.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/FirstOrderScheme.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/FirstOrderSchemeLts.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/ExponentialFilter.hpp"
#include "NumericalAlgorithms/LinearOperators/FilterAction.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/CollectDataForFluxes.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/FluxCommunication.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeInterfaces.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeMortars.hpp"
#include "ParallelAlgorithms/Events/ObserveErrorNorms.hpp"
#include "ParallelAlgorithms/Events/ObserveFields.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "PointwiseFunctions/AnalyticData/CurvedWaveEquation/ScalarWaveKerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/RegularSphericalWave.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Time/Actions/AdvanceTime.hpp"                // IWYU pragma: keep
#include "Time/Actions/ChangeSlabSize.hpp"             // IWYU pragma: keep
#include "Time/Actions/ChangeStepSize.hpp"             // IWYU pragma: keep
#include "Time/Actions/RecordTimeStepperData.hpp"      // IWYU pragma: keep
#include "Time/Actions/SelfStartActions.hpp"           // IWYU pragma: keep
#include "Time/Actions/UpdateU.hpp"                    // IWYU pragma: keep
#include "Time/StepChoosers/ByBlock.hpp"               // IWYU pragma: keep
#include "Time/StepChoosers/Cfl.hpp"                   // IWYU pragma: keep
#include "Time/StepChoosers/Constant.hpp"              // IWYU pragma: keep
#include "Time/StepChoosers/Increase.hpp"              // IWYU pragma: keep
#include "Time/StepChoosers/PreventRapidIncrease.hpp"  // IWYU pragma: keep
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepChoosers/StepToTimes.hpp"
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

template <size_t Dim, typename InitialData>
struct EvolutionMetavars {
  static constexpr size_t volume_dim = Dim;
  // Customization/"input options" to simulation
  using initial_data = InitialData;
  using initial_data_tag =
      tmpl::conditional_t<evolution::is_analytic_solution_v<initial_data>,
                          Tags::AnalyticSolution<initial_data>,
                          Tags::AnalyticData<initial_data>>;

  using system = CurvedScalarWave::System<Dim>;
  using temporal_id = Tags::TimeStepId;
  static constexpr bool local_time_stepping = false;
  static constexpr bool bjorhus_external_boundary = true;
  static constexpr bool moving_mesh = true;
  using boundary_condition_tag = initial_data_tag;
  using normal_dot_numerical_flux =
      Tags::NumericalFlux<CurvedScalarWave::UpwindFlux<Dim>>;

  using step_choosers_common =
      tmpl::list<StepChoosers::Registrars::ByBlock<volume_dim>,
                 //  StepChoosers::Registrars::Cfl<volume_dim, Frame::Inertial>,
                 StepChoosers::Registrars::Constant,
                 StepChoosers::Registrars::Increase>;
  using step_choosers_for_step_only =
      tmpl::list<StepChoosers::Registrars::PreventRapidIncrease>;
  using step_choosers_for_slab_only =
      tmpl::list<StepChoosers::Registrars::StepToTimes>;
  using step_choosers = tmpl::conditional_t<
      local_time_stepping,
      tmpl::append<step_choosers_common, step_choosers_for_step_only>,
      tmpl::list<>>;
  using slab_choosers = tmpl::conditional_t<
      local_time_stepping,
      tmpl::append<step_choosers_common, step_choosers_for_slab_only>,
      tmpl::append<step_choosers_common, step_choosers_for_step_only,
                   step_choosers_for_slab_only>>;

  using time_stepper_tag = Tags::TimeStepper<
      tmpl::conditional_t<local_time_stepping, LtsTimeStepper, TimeStepper>>;
  using boundary_scheme = tmpl::conditional_t<
      local_time_stepping,
      dg::FirstOrderScheme::FirstOrderSchemeLts<
          Dim, typename system::variables_tag, normal_dot_numerical_flux,
          Tags::TimeStepId, time_stepper_tag>,
      dg::FirstOrderScheme::FirstOrderScheme<
          Dim, typename system::variables_tag, normal_dot_numerical_flux,
          Tags::TimeStepId>>;

  // public for use by the Charm++ registration code
  using observe_fields = tmpl::append<
      db::get_variables_tags_list<typename system::variables_tag>,
      db::wrap_tags_in<
          temporal_id::template step_prefix,
          db::get_variables_tags_list<typename system::variables_tag>>,
      tmpl::list<::Tags::PointwiseL2Norm<
                     CurvedScalarWave::Tags::OneIndexConstraint<volume_dim>>,
                 ::Tags::PointwiseL2Norm<
                     CurvedScalarWave::Tags::TwoIndexConstraint<volume_dim>>>>;
  using analytic_solution_fields =
      db::get_variables_tags_list<typename system::variables_tag>;

  using events = tmpl::list<dg::Events::Registrars::ObserveFields<
                                Dim, Tags::Time, observe_fields, tmpl::list<>>,
                            Events::Registrars::ChangeSlabSize<slab_choosers>>;
  using triggers = Triggers::time_triggers;

  // A tmpl::list of tags to be added to the ConstGlobalCache by the
  // metavariables
  using const_global_cache_tags =
      tmpl::list<initial_data_tag, normal_dot_numerical_flux, time_stepper_tag,
                 Tags::EventsAndTriggers<events, triggers>>;

  struct ObservationType {};
  using element_observation_type = ObservationType;

  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      typename Event<events>::creatable_classes>;

  // The scalar wave system generally does not require filtering, except
  // possibly on certain deformed domains.  Here a filter is added in 3D for
  // testing purposes.  When performing numerical experiments with the scalar
  // wave system, the user should determine whether this filter can be removed.
  static constexpr bool use_filtering = (3 == volume_dim);

  enum class Phase {
    Initialization,
    RegisterWithObserver,
    InitializeTimeStepperHistory,
    Evolve,
    Exit
  };

  using initialization_actions = tmpl::list<
      Initialization::Actions::TimeAndTimeStep<EvolutionMetavars>,
      //   dg::Actions::InitializeDomain<volume_dim>,
      evolution::dg::Initialization::Domain<volume_dim>,
      Initialization::Actions::NonconservativeSystem,
      evolution::Initialization::Actions::SetVariables<
          domain::Tags::Coordinates<Dim, Frame::Logical>>,
      Initialization::Actions::TimeStepperHistory<EvolutionMetavars>,
      CurvedScalarWave::Actions::InitializeGrVars<volume_dim>,
      tmpl::conditional_t<
          bjorhus_external_boundary,
          tmpl::list<dg::Actions::InitializeInterfaces<
              system,
              dg::Initialization::slice_tags_to_face<
                  domain::Tags::MeshVelocity<volume_dim>,
                  typename system::variables_tag,
                  CurvedScalarWave::Tags::ConstraintGamma1,
                  CurvedScalarWave::Tags::ConstraintGamma2,
                  gr::Tags::Lapse<DataVector>,
                  gr::Tags::Shift<volume_dim, Frame::Inertial, DataVector>,
                  gr::Tags::InverseSpatialMetric<volume_dim, Frame::Inertial,
                                                 DataVector>>,
              dg::Initialization::slice_tags_to_exterior<
                  domain::Tags::MeshVelocity<volume_dim>,
                  typename system::variables_tag,
                  CurvedScalarWave::Tags::ConstraintGamma1,
                  CurvedScalarWave::Tags::ConstraintGamma2,
                  gr::Tags::Lapse<DataVector>,
                  gr::Tags::Shift<volume_dim, Frame::Inertial, DataVector>,
                  gr::Tags::InverseSpatialMetric<volume_dim, Frame::Inertial,
                                                 DataVector>>,
              dg::Initialization::face_compute_tags<
                  domain::Tags::BoundaryCoordinates<volume_dim, moving_mesh>,
                  CurvedScalarWave::CharacteristicFieldsCompute<volume_dim>,
                  domain::Tags::CharSpeedCompute<
                      CurvedScalarWave::CharacteristicSpeedsCompute<volume_dim>,
                      volume_dim>>,
              dg::Initialization::exterior_compute_tags<
                  CurvedScalarWave::CharacteristicFieldsCompute<volume_dim>,
                  domain::Tags::CharSpeedCompute<
                      CurvedScalarWave::CharacteristicSpeedsCompute<volume_dim>,
                      volume_dim>>,
              !bjorhus_external_boundary, moving_mesh>>,
          tmpl::list<dg::Actions::InitializeInterfaces<
              system,
              dg::Initialization::slice_tags_to_face<
                  domain::Tags::MeshVelocity<volume_dim>,
                  typename system::variables_tag,
                  CurvedScalarWave::Tags::ConstraintGamma1,
                  CurvedScalarWave::Tags::ConstraintGamma2,
                  gr::Tags::Lapse<DataVector>,
                  gr::Tags::Shift<volume_dim, Frame::Inertial, DataVector>,
                  gr::Tags::InverseSpatialMetric<volume_dim, Frame::Inertial,
                                                 DataVector>>,
              dg::Initialization::slice_tags_to_exterior<
                  domain::Tags::MeshVelocity<volume_dim>,
                  CurvedScalarWave::Tags::ConstraintGamma1,
                  CurvedScalarWave::Tags::ConstraintGamma2,
                  gr::Tags::Lapse<DataVector>,
                  gr::Tags::Shift<volume_dim, Frame::Inertial, DataVector>,
                  gr::Tags::InverseSpatialMetric<volume_dim, Frame::Inertial,
                                                 DataVector>>,
              dg::Initialization::face_compute_tags<>,
              dg::Initialization::exterior_compute_tags<>, true, moving_mesh>>>,
      tmpl::conditional_t<
          evolution::is_analytic_solution_v<initial_data>,
          Initialization::Actions::AddComputeTags<
              tmpl::list<evolution::Tags::AnalyticCompute<
                  Dim, initial_data_tag, analytic_solution_fields>>>,
          Initialization::Actions::AddComputeTags<
              tmpl::list<evolution::Tags::AnalyticDataCompute<
                  Dim, initial_data_tag, analytic_solution_fields>>>>,
      CurvedScalarWave::Actions::InitializeConstraints<volume_dim>,
      dg::Actions::InitializeMortars<boundary_scheme,
                                     !bjorhus_external_boundary>,
      Initialization::Actions::DiscontinuousGalerkin<EvolutionMetavars>,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using step_actions = tmpl::flatten<tmpl::list<
      dg::Actions::ComputeNonconservativeBoundaryFluxes<
          domain::Tags::InternalDirections<Dim>>,
      dg::Actions::CollectDataForFluxes<
          boundary_scheme, domain::Tags::InternalDirections<volume_dim>>,
      dg::Actions::SendDataForFluxes<boundary_scheme>,
      Actions::ComputeTimeDerivative<CurvedScalarWave::ComputeDuDt<volume_dim>>,
      dg::Actions::ComputeNonconservativeBoundaryFluxes<
          domain::Tags::BoundaryDirectionsInterior<Dim>>,
      // Dirichlet boundary conditions can only be applied if there
      // is an analytic soln available
      tmpl::conditional_t<
          evolution::is_analytic_solution_v<initial_data> and
              !bjorhus_external_boundary,
          tmpl::list<
              dg::Actions::ImposeDirichletBoundaryConditions<EvolutionMetavars>,
              dg::Actions::CollectDataForFluxes<
                  boundary_scheme,
                  domain::Tags::BoundaryDirectionsInterior<volume_dim>>>,
          tmpl::list<>>,
      dg::Actions::ReceiveDataForFluxes<boundary_scheme>,
      tmpl::conditional_t<
          local_time_stepping,
          tmpl::list<tmpl::conditional_t<
                         bjorhus_external_boundary,
                         tmpl::list<CurvedScalarWave::Actions::
                                        ImposeBjorhusBoundaryConditions<
                                            EvolutionMetavars>>,
                         tmpl::list<>>,
                     Actions::RecordTimeStepperData<>,
                     Actions::MutateApply<boundary_scheme>>,
          tmpl::list<Actions::MutateApply<boundary_scheme>,
                     tmpl::conditional_t<
                         bjorhus_external_boundary,
                         tmpl::list<CurvedScalarWave::Actions::
                                        ImposeBjorhusBoundaryConditions<
                                            EvolutionMetavars>>,
                         tmpl::list<>>,
                     Actions::RecordTimeStepperData<>>>,
      Actions::UpdateU<>,
      tmpl::conditional_t<
          use_filtering,
          dg::Actions::Filter<
              Filters::Exponential<0>,
              tmpl::list<CurvedScalarWave::Pi, CurvedScalarWave::Psi,
                         CurvedScalarWave::Phi<Dim>>>,
          tmpl::list<>>>>;

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
                  SelfStart::self_start_procedure<step_actions>>,

              Parallel::PhaseActions<
                  Phase, Phase::RegisterWithObserver,
                  tmpl::list<observers::Actions::RegisterWithObservers<
                                 observers::RegisterObservers<
                                     Tags::Time, element_observation_type>>,
                             Parallel::Actions::TerminatePhase>>,

              Parallel::PhaseActions<
                  Phase, Phase::Evolve,
                  tmpl::flatten<tmpl::list<
                      Actions::RunEventsAndTriggers, Actions::ChangeSlabSize,
                      tmpl::conditional_t<
                          local_time_stepping,
                          Actions::ChangeStepSize<step_choosers>, tmpl::list<>>,
                      step_actions, Actions::AdvanceTime>>>>>>;

  static constexpr OptionString help{
      "Evolve a Curved Scalar Wave in Dim spatial dimension.\n\n"
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
    &Parallel::register_derived_classes_with_charm<
        Event<metavariables::events>>,
    &Parallel::register_derived_classes_with_charm<MathFunction<1>>,
    &Parallel::register_derived_classes_with_charm<
        StepChooser<metavariables::slab_choosers>>,
    &Parallel::register_derived_classes_with_charm<
        StepChooser<metavariables::step_choosers>>,
    &Parallel::register_derived_classes_with_charm<StepController>,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        Trigger<metavariables::triggers>>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
