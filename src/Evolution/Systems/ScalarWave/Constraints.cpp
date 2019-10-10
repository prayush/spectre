// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/Constraints.hpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace ScalarWave {
template <size_t Dim>
tnsr::i<DataVector, Dim, Frame::Inertial> one_index_constraint(
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi) noexcept {
  auto constraint =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(phi, 0.);
  one_index_constraint<Dim>(make_not_null(&constraint), d_psi, phi);
  return constraint;
}

template <size_t Dim>
void one_index_constraint(
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> constraint,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi) noexcept {
  if (get_size(get<0>(*constraint)) != get_size(get<0>(phi))) {
    *constraint =
        make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(phi, 0.);
  }
  // Declare iterators for d_psi and phi outside the for loop,
  // because they are const but constraint is not
  auto d_psi_it = d_psi.begin(), phi_it = phi.begin();

  for (auto constraint_it = (*constraint).begin();
       constraint_it != (*constraint).end();
       ++constraint_it, (void)++d_psi_it, (void)++phi_it) {
    *constraint_it = *d_psi_it - *phi_it;
  }
}

template <size_t Dim>
tnsr::ij<DataVector, Dim, Frame::Inertial> two_index_constraint(
    const tnsr::ij<DataVector, Dim, Frame::Inertial>& d_phi) noexcept {
  auto constraint =
      make_with_value<tnsr::ij<DataVector, Dim, Frame::Inertial>>(d_phi, 0.);
  two_index_constraint<Dim>(make_not_null(&constraint), d_phi);
  return constraint;
}

template <size_t Dim>
void two_index_constraint(
    const gsl::not_null<tnsr::ij<DataVector, Dim, Frame::Inertial>*> constraint,
    const tnsr::ij<DataVector, Dim, Frame::Inertial>& d_phi) noexcept {
  if (get_size(get<0, 0>(*constraint)) != get_size(get<0, 0>(d_phi))) {
    *constraint =
        make_with_value<tnsr::ij<DataVector, Dim, Frame::Inertial>>(d_phi, 0.);
  }
  // Not using the antisymmetric property of the 2-index constraint, via
  // LeviCivita iterators, here as for Dim = 1 or 2 the constraint then becomes
  // a scalar.
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      constraint->get(i, j) = d_phi.get(i, j) - d_phi.get(j, i);
    }
  }
}
}  // namespace ScalarWave

// Explicit Instantiations
/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template tnsr::i<DataVector, DIM(data), Frame::Inertial>                    \
  ScalarWave::one_index_constraint(                                           \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&) noexcept;       \
  template void ScalarWave::one_index_constraint(                             \
      const gsl::not_null<tnsr::i<DataVector, DIM(data), Frame::Inertial>*>,  \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&) noexcept;       \
  template tnsr::ij<DataVector, DIM(data), Frame::Inertial>                   \
  ScalarWave::two_index_constraint(                                           \
      const tnsr::ij<DataVector, DIM(data), Frame::Inertial>&) noexcept;      \
  template void ScalarWave::two_index_constraint(                             \
      const gsl::not_null<tnsr::ij<DataVector, DIM(data), Frame::Inertial>*>, \
      const tnsr::ij<DataVector, DIM(data), Frame::Inertial>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond
