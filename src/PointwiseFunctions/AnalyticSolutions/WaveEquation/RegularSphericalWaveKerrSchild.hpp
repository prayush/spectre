// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/RegularSphericalWave.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
namespace Tags {
template <typename Tag>
struct dt;
}  // namespace Tags
/// \endcond

// IWYU pragma: no_include <pup.h>

namespace CurvedScalarWave {
namespace Solutions {

/*!
 * \brief A 3D spherical wave solution to the covariant wave equation with
 * a Kerr black hole providing the spacetime background
 *
 * \details Here we obtain the flat-spacetime scalar-wave system's solution
 * from `ScalarWave::Solutions::RegularSphericalWave` and suitably augment the
 * auxiliary variable to match that of the curved scalar-wave system. We keep
 * the scalar wave's profile, as a function of coordinates, the same.
 */
class RegularSphericalWaveKerrSchild : public MarkAsAnalyticSolution {
 public:
  static constexpr size_t volume_dim = 3;

  struct WaveProfile {
    using type = std::unique_ptr<MathFunction<1>>;
    static constexpr OptionString help = {
        "The radial profile of the spherical wave."};
  };
  struct BlackHoleMass {
    using type = double;
    static constexpr OptionString help = {"Mass of the black hole"};
    static type default_value() noexcept { return 1.; }
    static type lower_bound() noexcept { return 0.; }
  };
  struct BlackHoleSpin {
    using type = std::array<double, volume_dim>;
    static constexpr OptionString help = {
        "The [x,y,z] dimensionless spin of the black hole"};
    static type default_value() noexcept { return {{0., 0., 0.}}; }
  };
  struct BlackHoleCenter {
    using type = std::array<double, volume_dim>;
    static constexpr OptionString help = {
        "The [x,y,z] center of the black hole"};
    static type default_value() noexcept { return {{0., 0., 0.}}; }
  };
  using options =
      tmpl::list<BlackHoleMass, BlackHoleSpin, BlackHoleCenter, WaveProfile>;
  static constexpr OptionString help{
      "A regular spherical wave solution of the covariant wave equation"
      " with a Kerr black hole (in Kerr-Schild coordinates) as background.\n"
      "WARNING: This solution should only be used to set initial data."};
  static std::string name() noexcept { return "RegularSphWaveKerr"; };

  explicit RegularSphericalWaveKerrSchild(
      double mass, BlackHoleSpin::type dimensionless_spin,
      BlackHoleCenter::type center, std::unique_ptr<MathFunction<1>> profile,
      const OptionContext& context = {});

  explicit RegularSphericalWaveKerrSchild(CkMigrateMessage*
                                          /*unused*/) noexcept {}

  RegularSphericalWaveKerrSchild() = default;
  RegularSphericalWaveKerrSchild(
      const RegularSphericalWaveKerrSchild& /*rhs*/) = default;
  RegularSphericalWaveKerrSchild& operator=(
      const RegularSphericalWaveKerrSchild& /*rhs*/) = default;
  RegularSphericalWaveKerrSchild(RegularSphericalWaveKerrSchild&&
                                 /*rhs*/) noexcept = default;
  RegularSphericalWaveKerrSchild& operator=(
      RegularSphericalWaveKerrSchild&& /*rhs*/) noexcept = default;
  ~RegularSphericalWaveKerrSchild() = default;

  // Tags specific to the solution
  using scalar_wave_tags =
      tmpl::list<CurvedScalarWave::Pi, CurvedScalarWave::Phi<volume_dim>,
                 CurvedScalarWave::Psi>;
  using scalar_wave_dt_tags = db::wrap_tags_in<::Tags::dt, scalar_wave_tags>;

  using DerivLapse = ::Tags::deriv<gr::Tags::Lapse<DataVector>,
                                   tmpl::size_t<volume_dim>, Frame::Inertial>;
  using DerivShift =
      ::Tags::deriv<gr::Tags::Shift<volume_dim, Frame::Inertial, DataVector>,
                    tmpl::size_t<volume_dim>, Frame::Inertial>;
  using DerivSpatialMetric = ::Tags::deriv<
      gr::Tags::SpatialMetric<volume_dim, Frame::Inertial, DataVector>,
      tmpl::size_t<volume_dim>, Frame::Inertial>;
  using spacetime_tags = tmpl::list<
      gr::Tags::Lapse<DataVector>, ::Tags::dt<gr::Tags::Lapse<DataVector>>,
      DerivLapse, gr::Tags::Shift<volume_dim, Frame::Inertial, DataVector>,
      ::Tags::dt<gr::Tags::Shift<volume_dim, Frame::Inertial, DataVector>>,
      DerivShift,
      gr::Tags::SpatialMetric<volume_dim, Frame::Inertial, DataVector>,
      ::Tags::dt<
          gr::Tags::SpatialMetric<volume_dim, Frame::Inertial, DataVector>>,
      DerivSpatialMetric, gr::Tags::SqrtDetSpatialMetric<DataVector>,
      gr::Tags::ExtrinsicCurvature<volume_dim, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<volume_dim, Frame::Inertial, DataVector>>;

  using tags = tmpl::append<spacetime_tags, scalar_wave_tags>;

  // Returns only U
  tuples::tagged_tuple_from_typelist<scalar_wave_tags> variables(
      const tnsr::I<DataVector, volume_dim>& x, double t,
      scalar_wave_tags /*meta*/) const noexcept;
  // Returns spacetime vars, and dt<spacetime vars>
  tuples::tagged_tuple_from_typelist<spacetime_tags> variables(
      const tnsr::I<DataVector, volume_dim>& x, double t,
      spacetime_tags /*meta*/) const noexcept;

  // clang-tidy: no runtime references
  void pup(PUP::er& p) noexcept;  // NOLINT

  SPECTRE_ALWAYS_INLINE double mass() const noexcept { return mass_; }
  SPECTRE_ALWAYS_INLINE const std::array<double, volume_dim>& center() const
      noexcept {
    return center_;
  }
  SPECTRE_ALWAYS_INLINE const std::array<double, volume_dim>&
  dimensionless_spin() const noexcept {
    return dimensionless_spin_;
  }

 private:
  double mass_{1.};
  std::array<double, volume_dim> dimensionless_spin_{{0., 0., 0.}},
      center_{{0., 0., 0.}};
  ScalarWave::Solutions::RegularSphericalWave flat_space_wave_soln_;
  gr::Solutions::KerrSchild kerr_schild_soln_;
};

SPECTRE_ALWAYS_INLINE bool operator==(
    const RegularSphericalWaveKerrSchild& lhs,
    const RegularSphericalWaveKerrSchild& rhs) noexcept {
  return lhs.mass() == rhs.mass() and
         lhs.dimensionless_spin() == rhs.dimensionless_spin() and
         lhs.center() == rhs.center();
}

SPECTRE_ALWAYS_INLINE bool operator!=(
    const RegularSphericalWaveKerrSchild& lhs,
    const RegularSphericalWaveKerrSchild& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace Solutions
}  // namespace CurvedScalarWave
