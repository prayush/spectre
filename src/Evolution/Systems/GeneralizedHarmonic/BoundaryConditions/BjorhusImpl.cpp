// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BjorhusImpl.hpp"

#include <algorithm>
#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace GeneralizedHarmonic::BoundaryConditions::detail {
template <size_t VolumeDim, typename DataType>
void set_dt_v_psi_constraint_preserving(
    const gsl::not_null<tnsr::aa<DataType, VolumeDim, Frame::Inertial>*>
        bc_dt_v_psi,
    const tnsr::I<DataType, VolumeDim, Frame::Inertial>&
        unit_interface_normal_vector,
    const tnsr::iaa<DataType, VolumeDim, Frame::Inertial>&
        three_index_constraint,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_v_psi,
    const std::array<DataType, 4>& char_speeds) noexcept {
  ASSERT(get_size(get<0, 0>(*bc_dt_v_psi)) ==
             get_size(get<0>(unit_interface_normal_vector)),
         "Size of input variables and temporary memory do not match: "
             << get_size(get<0, 0>(*bc_dt_v_psi)) << ","
             << get_size(get<0>(unit_interface_normal_vector)));

  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      bc_dt_v_psi->get(a, b) = char_projected_rhs_dt_v_psi.get(a, b);
      for (size_t i = 0; i < VolumeDim; ++i) {
        bc_dt_v_psi->get(a, b) += char_speeds.at(0) *
                                  unit_interface_normal_vector.get(i) *
                                  three_index_constraint.get(i, a, b);
      }
    }
  }
}

template <size_t VolumeDim, typename DataType>
void set_dt_v_zero_constraint_preserving(
    gsl::not_null<tnsr::iaa<DataType, VolumeDim, Frame::Inertial>*>
        bc_dt_v_zero,
    const tnsr::I<DataType, VolumeDim, Frame::Inertial>&
        unit_interface_normal_vector,
    const tnsr::iaa<DataType, VolumeDim, Frame::Inertial>&
        four_index_constraint,
    const tnsr::iaa<DataType, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_v_zero,
    const std::array<DataType, 4>& char_speeds) noexcept {
  ASSERT(get_size(get<0, 0, 0>(*bc_dt_v_zero)) ==
             get_size(get<0>(unit_interface_normal_vector)),
         "Size of input variables and temporary memory do not match: "
             << get_size(get<0, 0, 0>(*bc_dt_v_zero)) << ","
             << get_size(get<0>(unit_interface_normal_vector)));

  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      for (size_t i = 0; i < VolumeDim; ++i) {
        bc_dt_v_zero->get(i, a, b) = char_projected_rhs_dt_v_zero.get(i, a, b);
      }
      // Lets say this term is T2_{kab} := - n_l N^l n^j C_{jkab}.
      // But we store C4_{iab} = LeviCivita^{ijk} dphi_{jkab},
      // which means  C_{jkab} = LeviCivita^{ijk} C4_{iab}
      // where C4 is `four_index_constraint`.
      // therefore, T2_{iab} =  char_speed<VZero> n^j C_{jiab}
      // (since char_speed<VZero> = - n_l N^l), and therefore:
      // T2_{iab} = char_speed<VZero> n^k LeviCivita^{ijk} C4_{jab}.
      // Let LeviCivitaIterator be indexed by
      // it[0] <--> i,
      // it[1] <--> j,
      // it[2] <--> k, then
      // T2_{it[0], ab} += char_speed<VZero> n^it[2] it.sign() C4_{it[1], ab};
      for (LeviCivitaIterator<VolumeDim> it; it; ++it) {
        bc_dt_v_zero->get(it[0], a, b) +=
            it.sign() * char_speeds.at(1) *
            unit_interface_normal_vector.get(it[2]) *
            four_index_constraint.get(it[1], a, b);
      }
    }
  }
}

template <size_t VolumeDim, typename DataType>
void set_dt_v_minus_constraint_preserving(
    const gsl::not_null<tnsr::aa<DataType, VolumeDim, Frame::Inertial>*>
        bc_dt_v_minus,
    const Scalar<DataType>& gamma2,
    const tnsr::I<DataType, VolumeDim, Frame::Inertial>& inertial_coords,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>& incoming_null_one_form,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>& outgoing_null_one_form,
    const tnsr::A<DataType, VolumeDim, Frame::Inertial>& incoming_null_vector,
    const tnsr::A<DataType, VolumeDim, Frame::Inertial>& outgoing_null_vector,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>& projection_ab,
    const tnsr::Ab<DataType, VolumeDim, Frame::Inertial>& projection_Ab,
    const tnsr::AA<DataType, VolumeDim, Frame::Inertial>& projection_AB,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_v_psi,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_v_minus,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>&
        constraint_char_zero_plus,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>&
        constraint_char_zero_minus,
    const std::array<DataType, 4>& char_speeds) noexcept {
  std::fill(bc_dt_v_minus->begin(), bc_dt_v_minus->end(), 0.);
  add_constraint_preserving_terms_to_dt_v_minus(
      bc_dt_v_minus, incoming_null_one_form, outgoing_null_one_form,
      incoming_null_vector, outgoing_null_vector, projection_ab, projection_Ab,
      projection_AB, constraint_char_zero_plus, constraint_char_zero_minus,
      char_projected_rhs_dt_v_minus, char_speeds);
  add_gauge_sommerfeld_terms_to_dt_v_minus(
      bc_dt_v_minus, gamma2, inertial_coords, incoming_null_one_form,
      outgoing_null_one_form, incoming_null_vector, outgoing_null_vector,
      projection_Ab, char_projected_rhs_dt_v_psi);
}

template <size_t VolumeDim, typename DataType>
void set_dt_v_minus_constraint_preserving_physical(
    const gsl::not_null<tnsr::aa<DataType, VolumeDim, Frame::Inertial>*>
        bc_dt_v_minus,
    const Scalar<DataType>& gamma2,
    const tnsr::I<DataType, VolumeDim, Frame::Inertial>& inertial_coords,
    const tnsr::i<DataType, VolumeDim, Frame::Inertial>&
        unit_interface_normal_one_form,
    const tnsr::I<DataType, VolumeDim, Frame::Inertial>&
        unit_interface_normal_vector,
    const tnsr::A<DataType, VolumeDim, Frame::Inertial>&
        spacetime_unit_normal_vector,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>& incoming_null_one_form,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>& outgoing_null_one_form,
    const tnsr::A<DataType, VolumeDim, Frame::Inertial>& incoming_null_vector,
    const tnsr::A<DataType, VolumeDim, Frame::Inertial>& outgoing_null_vector,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>& projection_ab,
    const tnsr::Ab<DataType, VolumeDim, Frame::Inertial>& projection_Ab,
    const tnsr::AA<DataType, VolumeDim, Frame::Inertial>& projection_AB,
    const tnsr::II<DataType, VolumeDim, Frame::Inertial>&
        inverse_spatial_metric,
    const tnsr::ii<DataType, VolumeDim, Frame::Inertial>& extrinsic_curvature,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>& spacetime_metric,
    const tnsr::AA<DataType, VolumeDim, Frame::Inertial>&
        inverse_spacetime_metric,
    const tnsr::iaa<DataType, VolumeDim, Frame::Inertial>&
        three_index_constraint,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_v_psi,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_v_minus,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>&
        constraint_char_zero_plus,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>&
        constraint_char_zero_minus,
    const tnsr::iaa<DataType, VolumeDim, Frame::Inertial>& phi,
    const tnsr::ijaa<DataType, VolumeDim, Frame::Inertial>& d_phi,
    const tnsr::iaa<DataType, VolumeDim, Frame::Inertial>& d_pi,
    const std::array<DataType, 4>& char_speeds) noexcept {
  std::fill(bc_dt_v_minus->begin(), bc_dt_v_minus->end(), 0.);
  add_constraint_preserving_terms_to_dt_v_minus(
      bc_dt_v_minus, incoming_null_one_form, outgoing_null_one_form,
      incoming_null_vector, outgoing_null_vector, projection_ab, projection_Ab,
      projection_AB, constraint_char_zero_plus, constraint_char_zero_minus,
      char_projected_rhs_dt_v_minus, char_speeds);
  add_physical_dof_terms_to_dt_v_minus(
      bc_dt_v_minus, gamma2, unit_interface_normal_one_form,
      unit_interface_normal_vector, spacetime_unit_normal_vector, projection_ab,
      projection_Ab, projection_AB, inverse_spatial_metric, extrinsic_curvature,
      spacetime_metric, inverse_spacetime_metric, three_index_constraint,
      char_projected_rhs_dt_v_minus, phi, d_phi, d_pi, char_speeds);
  add_gauge_sommerfeld_terms_to_dt_v_minus(
      bc_dt_v_minus, gamma2, inertial_coords, incoming_null_one_form,
      outgoing_null_one_form, incoming_null_vector, outgoing_null_vector,
      projection_Ab, char_projected_rhs_dt_v_psi);
}
}  // namespace GeneralizedHarmonic::BoundaryConditions::detail

// Explicit Instantiations
/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                \
  template void GeneralizedHarmonic::BoundaryConditions::detail::           \
      set_dt_v_psi_constraint_preserving(                                   \
          const gsl::not_null<                                              \
              tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>*>           \
              bc_dt_v_psi,                                                  \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>&           \
              unit_interface_normal_vector,                                 \
          const tnsr::iaa<DTYPE(data), DIM(data), Frame::Inertial>&         \
              three_index_constraint,                                       \
          const tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>&          \
              char_projected_rhs_dt_v_psi,                                  \
          const std::array<DTYPE(data), 4>& char_speeds) noexcept;          \
  template void GeneralizedHarmonic::BoundaryConditions::detail::           \
      set_dt_v_minus_constraint_preserving(                                 \
          const gsl::not_null<                                              \
              tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>*>           \
              bc_dt_v_minus,                                                \
          const Scalar<DTYPE(data)>& gamma2,                                \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>&           \
              inertial_coords,                                              \
          const tnsr::a<DTYPE(data), DIM(data), Frame::Inertial>&           \
              incoming_null_one_form,                                       \
          const tnsr::a<DTYPE(data), DIM(data), Frame::Inertial>&           \
              outgoing_null_one_form,                                       \
          const tnsr::A<DTYPE(data), DIM(data), Frame::Inertial>&           \
              incoming_null_vector,                                         \
          const tnsr::A<DTYPE(data), DIM(data), Frame::Inertial>&           \
              outgoing_null_vector,                                         \
          const tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>&          \
              projection_ab,                                                \
          const tnsr::Ab<DTYPE(data), DIM(data), Frame::Inertial>&          \
              projection_Ab,                                                \
          const tnsr::AA<DTYPE(data), DIM(data), Frame::Inertial>&          \
              projection_AB,                                                \
          const tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>&          \
              char_projected_rhs_dt_v_psi,                                  \
          const tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>&          \
              char_projected_rhs_dt_v_minus,                                \
          const tnsr::a<DTYPE(data), DIM(data), Frame::Inertial>&           \
              constraint_char_zero_plus,                                    \
          const tnsr::a<DTYPE(data), DIM(data), Frame::Inertial>&           \
              constraint_char_zero_minus,                                   \
          const std::array<DTYPE(data), 4>& char_speeds) noexcept;          \
  template void GeneralizedHarmonic::BoundaryConditions::detail::           \
      set_dt_v_minus_constraint_preserving_physical(                        \
          const gsl::not_null<                                              \
              tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>*>           \
              bc_dt_v_minus,                                                \
          const Scalar<DTYPE(data)>& gamma2,                                \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>&           \
              inertial_coords,                                              \
          const tnsr::i<DTYPE(data), DIM(data), Frame::Inertial>&           \
              unit_interface_normal_one_form,                               \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>&           \
              unit_interface_normal_vector,                                 \
          const tnsr::A<DTYPE(data), DIM(data), Frame::Inertial>&           \
              spacetime_unit_normal_vector,                                 \
          const tnsr::a<DTYPE(data), DIM(data), Frame::Inertial>&           \
              incoming_null_one_form,                                       \
          const tnsr::a<DTYPE(data), DIM(data), Frame::Inertial>&           \
              outgoing_null_one_form,                                       \
          const tnsr::A<DTYPE(data), DIM(data), Frame::Inertial>&           \
              incoming_null_vector,                                         \
          const tnsr::A<DTYPE(data), DIM(data), Frame::Inertial>&           \
              outgoing_null_vector,                                         \
          const tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>&          \
              projection_ab,                                                \
          const tnsr::Ab<DTYPE(data), DIM(data), Frame::Inertial>&          \
              projection_Ab,                                                \
          const tnsr::AA<DTYPE(data), DIM(data), Frame::Inertial>&          \
              projection_AB,                                                \
          const tnsr::II<DTYPE(data), DIM(data), Frame::Inertial>&          \
              inverse_spatial_metric,                                       \
          const tnsr::ii<DTYPE(data), DIM(data), Frame::Inertial>&          \
              extrinsic_curvature,                                          \
          const tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>&          \
              spacetime_metric,                                             \
          const tnsr::AA<DTYPE(data), DIM(data), Frame::Inertial>&          \
              inverse_spacetime_metric,                                     \
          const tnsr::iaa<DTYPE(data), DIM(data), Frame::Inertial>&         \
              three_index_constraint,                                       \
          const tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>&          \
              char_projected_rhs_dt_v_psi,                                  \
          const tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>&          \
              char_projected_rhs_dt_v_minus,                                \
          const tnsr::a<DTYPE(data), DIM(data), Frame::Inertial>&           \
              constraint_char_zero_plus,                                    \
          const tnsr::a<DTYPE(data), DIM(data), Frame::Inertial>&           \
              constraint_char_zero_minus,                                   \
          const tnsr::iaa<DTYPE(data), DIM(data), Frame::Inertial>& phi,    \
          const tnsr::ijaa<DTYPE(data), DIM(data), Frame::Inertial>& d_phi, \
          const tnsr::iaa<DTYPE(data), DIM(data), Frame::Inertial>& d_pi,   \
          const std::array<DTYPE(data), 4>& char_speeds) noexcept;          \
  template void GeneralizedHarmonic::BoundaryConditions::detail::           \
      set_dt_v_zero_constraint_preserving(                                  \
          const gsl::not_null<                                              \
              tnsr::iaa<DTYPE(data), DIM(data), Frame::Inertial>*>          \
              bc_dt_v_zero,                                                 \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>&           \
              unit_interface_normal_vector,                                 \
          const tnsr::iaa<DTYPE(data), DIM(data), Frame::Inertial>&         \
              four_index_constraint,                                        \
          const tnsr::iaa<DTYPE(data), DIM(data), Frame::Inertial>&         \
              char_projected_rhs_dt_v_zero,                                 \
          const std::array<DTYPE(data), 4>& char_speeds) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (3), (DataVector))

#undef INSTANTIATE
#undef DTYPE
#undef DIM
/// \endcond
