// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/RegularSphericalWaveKerrSchild.hpp"

#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Element.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Systems/CurvedScalarWave/Equations.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

// Instantiate `partial_derivatives` for the use case below --
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"  // IWYU pragma: keep

template <size_t Dim>
using VariablesTags =
    tmpl::list<CurvedScalarWave::Pi, CurvedScalarWave::Phi<Dim>>;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                              \
  template Variables<                                                     \
      db::wrap_tags_in<Tags::deriv, VariablesTags<DIM(data)>,             \
                       tmpl::size_t<DIM(data)>, Frame::Inertial>>         \
  partial_derivatives<VariablesTags<DIM(data)>, VariablesTags<DIM(data)>, \
                      DIM(data), Frame::Inertial>(                        \
      const Variables<VariablesTags<DIM(data)>>& vars,                    \
      const Mesh<DIM(data)>& mesh,                                        \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,        \
                            Frame::Inertial>& inverse_jacobian) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATION
#undef DIM

namespace CurvedScalarWave {
namespace Solutions {

RegularSphericalWaveKerrSchild::RegularSphericalWaveKerrSchild(
    double mass,
    RegularSphericalWaveKerrSchild::BlackHoleSpin::type dimensionless_spin,
    RegularSphericalWaveKerrSchild::BlackHoleCenter::type center,
    std::unique_ptr<MathFunction<1>> profile, const OptionContext& context)
    : mass_(mass),
      dimensionless_spin_(std::move(dimensionless_spin)),
      center_(std::move(center)),
      flat_space_wave_soln_(std::move(profile)),
      kerr_schild_soln_(mass_, dimensionless_spin_, center_) {}

void RegularSphericalWaveKerrSchild::pup(PUP::er& p) noexcept {
  p | mass_;
  p | dimensionless_spin_;
  p | center_;
  p | flat_space_wave_soln_;
  p | kerr_schild_soln_;
}

tuples::tagged_tuple_from_typelist<
    RegularSphericalWaveKerrSchild::scalar_wave_tags>
RegularSphericalWaveKerrSchild::variables(
    const tnsr::I<DataVector, volume_dim>& x, const double t,
    scalar_wave_tags /*meta*/) const noexcept {
  ASSERT(t == 0.,
         "`RegularSphericalWaveKerrSchild` solution should only "
         "be used to set initial data, i.e. at coordinate time t = 0.");
  const auto flat_space_scalar_wave_vars = flat_space_wave_soln_.variables(
      x, t,
      tmpl::list<ScalarWave::Pi, ScalarWave::Phi<volume_dim>,
                 ScalarWave::Psi>{});
  const auto flat_space_dt_scalar_wave_vars = flat_space_wave_soln_.variables(
      x, t,
      tmpl::list<::Tags::dt<ScalarWave::Pi>,
                 ::Tags::dt<ScalarWave::Phi<volume_dim>>,
                 ::Tags::dt<ScalarWave::Psi>>{});
  const auto kerr_schild_spacetime_variables =
      kerr_schild_soln_.variables(x, t, spacetime_tags{});
  auto result =
      make_with_value<tuples::tagged_tuple_from_typelist<scalar_wave_tags>>(x,
                                                                            0.);

  // Compute variables Psi, Pi and Phi in curved spacetime, using their
  // definitions from Eq.(14) - (16) of \cite Holst2004wt
  get<Psi>(result) = get<ScalarWave::Psi>(flat_space_scalar_wave_vars);
  get<Phi<volume_dim>>(result) =
      get<ScalarWave::Phi<volume_dim>>(flat_space_scalar_wave_vars);
  // finally, get PI
  {
    // Which dt<Psi> do we use in the definition of Pi?
    const auto shift_dot_dpsi = dot_product(
        get<gr::Tags::Shift<volume_dim, Frame::Inertial, DataVector>>(
            kerr_schild_spacetime_variables),
        get<ScalarWave::Phi<volume_dim>>(flat_space_scalar_wave_vars));

    get(get<Pi>(result)) =
        (get(shift_dot_dpsi) - get(get<::Tags::dt<ScalarWave::Psi>>(
                                   flat_space_dt_scalar_wave_vars))) /
        get(get<gr::Tags::Lapse<DataVector>>(kerr_schild_spacetime_variables));
  }

  return result;
}

tuples::tagged_tuple_from_typelist<
    RegularSphericalWaveKerrSchild::spacetime_tags>
RegularSphericalWaveKerrSchild::variables(
    const tnsr::I<DataVector, volume_dim>& x, const double t,
    spacetime_tags /*meta*/) const noexcept {
  ASSERT(t == 0.,
         "`RegularSphericalWaveKerrSchild` solution should only "
         "be used to set initial data, i.e. at coordinate time t = 0.");

  const auto kerr_schild_spacetime_variables =
      kerr_schild_soln_.variables(x, t, spacetime_tags{});
  auto result =
      make_with_value<tuples::tagged_tuple_from_typelist<spacetime_tags>>(x,
                                                                          0.);

  // Directly import spacetime variables that do not need corrections
  tmpl::for_each<spacetime_tags>(
      [&result, &kerr_schild_spacetime_variables](auto x) {
        using tag = typename decltype(x)::type;
        get<tag>(result) = get<tag>(kerr_schild_spacetime_variables);
      });

  return result;
}
}  // namespace Solutions
}  // namespace CurvedScalarWave
