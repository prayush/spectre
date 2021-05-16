// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Bjorhus.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/UpwindPenalty.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace helpers = TestHelpers::evolution::dg;

namespace {
using frame = Frame::Inertial;

template <size_t Dim>
void test() {
  MAKE_GENERATOR(gen);
  for (const std::string& bc_string :
       {"ConstraintPreserving", "ConstraintPreservingPhysical"}) {
    CAPTURE(bc_string);

    helpers::test_boundary_condition_with_python<
        GeneralizedHarmonic::BoundaryConditions::ConstraintPreservingBjorhus<
            Dim>,
        GeneralizedHarmonic::BoundaryConditions::BoundaryCondition<Dim>,
        GeneralizedHarmonic::System<Dim>,
        tmpl::list<
            GeneralizedHarmonic::BoundaryCorrections::UpwindPenalty<Dim>>>(
        make_not_null(&gen),
        "Evolution.Systems.GeneralizedHarmonic.BoundaryConditions.Bjorhus",
        tuples::TaggedTuple<
            helpers::Tags::PythonFunctionForErrorMessage<>,
            helpers::Tags::PythonFunctionName<
                ::Tags::dt<gr::Tags::SpacetimeMetric<Dim, frame, DataVector>>>,
            helpers::Tags::PythonFunctionName<
                ::Tags::dt<GeneralizedHarmonic::Tags::Pi<Dim, frame>>>,
            helpers::Tags::PythonFunctionName<
                ::Tags::dt<GeneralizedHarmonic::Tags::Phi<Dim, frame>>>>{
            "error", "dt_spacetime_metric", "dt_pi_" + bc_string,
            "dt_phi_" + bc_string},
        "ConstraintPreservingBjorhus:\n"
        "  Type: " +
            bc_string,
        Index<Dim - 1>{5}, db::DataBox<tmpl::list<>>{},
        tuples::TaggedTuple<
            helpers::Tags::Range<
                GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2>,
            helpers::Tags::Range<
                gr::Tags::SpacetimeMetric<Dim, frame, DataVector>>>{
            std::array{0.0, 1.0}, std::array{-1., 1.}},
        1.e-6);
  }
}

template <size_t Dim>
void wrap_dt_vars_corrections_ConstraintPreserving(
    const gsl::not_null<tnsr::aa<DataVector, Dim, frame>*>
        dt_spacetime_metric_correction,
    const gsl::not_null<tnsr::aa<DataVector, Dim, frame>*> dt_pi_correction,
    const gsl::not_null<tnsr::iaa<DataVector, Dim, frame>*> dt_phi_correction,
    const tnsr::I<DataVector, Dim, frame>& face_mesh_velocity,
    const tnsr::i<DataVector, Dim, frame>& normal_covector,
    const tnsr::I<DataVector, Dim, frame>& normal_vector,
    // c.f. dg_interior_evolved_variables_tags
    const tnsr::aa<DataVector, Dim, frame>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, frame>& pi,
    const tnsr::iaa<DataVector, Dim, frame>& phi,
    // c.f. dg_interior_temporary_tags
    const tnsr::I<DataVector, Dim, frame>& coords,
    const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
    const Scalar<DataVector>& lapse,
    const tnsr::a<DataVector, Dim, frame>& gauge_source,
    const tnsr::ab<DataVector, Dim, frame>& spacetime_deriv_gauge_source,
    // c.f. dg_interior_dt_vars_tags
    const tnsr::aa<DataVector, Dim, frame>& dt_spacetime_metric,
    const tnsr::aa<DataVector, Dim, frame>& dt_pi,
    const tnsr::iaa<DataVector, Dim, frame>& dt_phi,
    // c.f. dg_interior_deriv_vars_tags
    const tnsr::iaa<DataVector, Dim, frame>& d_spacetime_metric,
    const tnsr::iaa<DataVector, Dim, frame>& d_pi,
    const tnsr::ijaa<DataVector, Dim, frame>& d_phi) noexcept {
  GeneralizedHarmonic::BoundaryConditions::ConstraintPreservingBjorhus<Dim>
      bjorhus_obj{GeneralizedHarmonic::BoundaryConditions::detail::
                      ConstraintPreservingBjorhusType::ConstraintPreserving};
  bjorhus_obj.dg_time_derivative(
      dt_spacetime_metric_correction, dt_pi_correction, dt_phi_correction,
      face_mesh_velocity, normal_covector, normal_vector, spacetime_metric, pi,
      phi, coords, gamma1, gamma2, lapse, gauge_source,
      spacetime_deriv_gauge_source, dt_spacetime_metric, dt_pi, dt_phi,
      d_spacetime_metric, d_pi, d_phi);
}

template <size_t Dim>
void wrap_dt_vars_corrections_ConstraintPreservingPhysical(
    const gsl::not_null<tnsr::aa<DataVector, Dim, frame>*>
        dt_spacetime_metric_correction,
    const gsl::not_null<tnsr::aa<DataVector, Dim, frame>*> dt_pi_correction,
    const gsl::not_null<tnsr::iaa<DataVector, Dim, frame>*> dt_phi_correction,
    const tnsr::I<DataVector, Dim, frame>& face_mesh_velocity,
    const tnsr::i<DataVector, Dim, frame>& normal_covector,
    const tnsr::I<DataVector, Dim, frame>& normal_vector,
    // c.f. dg_interior_evolved_variables_tags
    const tnsr::aa<DataVector, Dim, frame>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, frame>& pi,
    const tnsr::iaa<DataVector, Dim, frame>& phi,
    // c.f. dg_interior_temporary_tags
    const tnsr::I<DataVector, Dim, frame>& coords,
    const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
    const Scalar<DataVector>& lapse,
    const tnsr::a<DataVector, Dim, frame>& gauge_source,
    const tnsr::ab<DataVector, Dim, frame>& spacetime_deriv_gauge_source,
    // c.f. dg_interior_dt_vars_tags
    const tnsr::aa<DataVector, Dim, frame>& dt_spacetime_metric,
    const tnsr::aa<DataVector, Dim, frame>& dt_pi,
    const tnsr::iaa<DataVector, Dim, frame>& dt_phi,
    // c.f. dg_interior_deriv_vars_tags
    const tnsr::iaa<DataVector, Dim, frame>& d_spacetime_metric,
    const tnsr::iaa<DataVector, Dim, frame>& d_pi,
    const tnsr::ijaa<DataVector, Dim, frame>& d_phi) noexcept {
  GeneralizedHarmonic::BoundaryConditions::ConstraintPreservingBjorhus<Dim>
      bjorhus_obj{
          GeneralizedHarmonic::BoundaryConditions::detail::
              ConstraintPreservingBjorhusType::ConstraintPreservingPhysical};
  bjorhus_obj.dg_time_derivative(
      dt_spacetime_metric_correction, dt_pi_correction, dt_phi_correction,
      face_mesh_velocity, normal_covector, normal_vector, spacetime_metric, pi,
      phi, coords, gamma1, gamma2, lapse, gauge_source,
      spacetime_deriv_gauge_source, dt_spacetime_metric, dt_pi, dt_phi,
      d_spacetime_metric, d_pi, d_phi);
}

template <size_t Dim>
void test_with_random_values(const DataVector& used_for_size) noexcept {
  pypp::check_with_random_values<1>(
      wrap_dt_vars_corrections_ConstraintPreserving<Dim>,
      "Evolution.Systems.GeneralizedHarmonic.BoundaryConditions.Bjorhus",
      {"dt_spacetime_metric", "dt_pi_ConstraintPreserving",
       "dt_phi_ConstraintPreserving"},
      {{{0., 1.}}}, used_for_size, 1.e-9);
  pypp::check_with_random_values<1>(
      wrap_dt_vars_corrections_ConstraintPreservingPhysical<Dim>,
      "Evolution.Systems.GeneralizedHarmonic.BoundaryConditions.Bjorhus",
      {"dt_spacetime_metric", "dt_pi_ConstraintPreservingPhysical",
       "dt_phi_ConstraintPreservingPhysical"},
      {{{0., 1.}}}, used_for_size, 1.e-9);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.BCBjorhus",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};

  test<1>();
  test<2>();
  test<3>();

  const DataVector used_for_size(5);

  test_with_random_values<1>(used_for_size);
  test_with_random_values<2>(used_for_size);
  test_with_random_values<3>(used_for_size);
}
