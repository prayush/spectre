// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BjorhusImpl.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
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
}  // namespace GeneralizedHarmonic::BoundaryConditions::detail

// Explicit Instantiations
/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                        \
  template void GeneralizedHarmonic::BoundaryConditions::detail::   \
      set_dt_v_psi_constraint_preserving(                           \
          const gsl::not_null<                                      \
              tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>*>   \
              bc_dt_v_psi,                                          \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>&   \
              unit_interface_normal_vector,                         \
          const tnsr::iaa<DTYPE(data), DIM(data), Frame::Inertial>& \
              three_index_constraint,                               \
          const tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>&  \
              char_projected_rhs_dt_v_psi,                          \
          const std::array<DTYPE(data), 4>& char_speeds) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (3), (DataVector))

#undef INSTANTIATE
#undef DTYPE
#undef DIM
/// \endcond
