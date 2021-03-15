// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/CovariantDerivOfExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/ProjectionOperators.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylPropagating.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace GeneralizedHarmonic::BoundaryConditions {
namespace detail {
/*!
 * \brief Computes the expression needed to set bnudary conditions on Dt<VPsi>
 *
 * \details Computes the correction to the characteristic projected
 * time-derivatives of the fundamental variables corresponding to \f$u^\psi\f$:
 *
 * \f{align}
 * \partial_t u^{\psi}_{ab} = \partial_t u^{\Psi}_{ab} + v_{\psi} n^i C_{iab}
 * \f}
 *
 * where \f$n^i\f$ is the local unit normal to the external boundary,
 * and \f$C_{iab} = \partial_i \psi_{ab} - \Phi_{iab}\f$ is the
 * three-index constraint.
 */
template <size_t VolumeDim, typename DataType>
void set_dt_v_psi_constraint_preserving(
    gsl::not_null<tnsr::aa<DataType, VolumeDim, Frame::Inertial>*> bc_dt_v_psi,
    const tnsr::I<DataType, VolumeDim, Frame::Inertial>&
        unit_interface_normal_vector,
    const tnsr::iaa<DataType, VolumeDim, Frame::Inertial>&
        three_index_constraint,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_v_psi,
    const std::array<DataType, 4>& char_speeds) noexcept;

// @{
/*!
 * \brief Computes expressions needed to set boundary conditions on Dt<VMinus>
 *
 * \details Computes the correction to the characteristic projected
 * time-derivatives of the fundamental variables:
 *
 * \f{align}
 * \partial_t v^{\Psi}_{ab} = \partial_t v^{\Psi}_{ab} +
 *                           u_{(v^{\Psi})} n^i C_{iab}
 * \f
 *
 * where \f$n^i\f$ is the local unit normal to the external boundary,
 * and \f$C_{iab} = \partial_i \Psi_{ab} - \Phi_{iab}\f$ is the
 * three-index constraint.
 */
template <size_t VolumeDim, typename DataType>
void add_gauge_sommerfeld_terms_to_dt_v_minus(
    gsl::not_null<tnsr::aa<DataType, VolumeDim, Frame::Inertial>*>
        bc_dt_v_minus,
    const Scalar<DataType>& gamma2,
    const tnsr::I<DataType, VolumeDim, Frame::Inertial>& inertial_coords,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>& incoming_null_one_form,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>& outgoing_null_one_form,
    const tnsr::A<DataType, VolumeDim, Frame::Inertial>& incoming_null_vector,
    const tnsr::A<DataType, VolumeDim, Frame::Inertial>& outgoing_null_vector,
    const tnsr::Ab<DataType, VolumeDim, Frame::Inertial>& projection_Ab,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_v_psi) noexcept {
  ASSERT(get_size(get<0, 0>(*bc_dt_v_minus)) ==
             get_size(get<0>(incoming_null_one_form)),
         "Size of input variables and temporary memory do not match: "
             << get_size(get<0, 0>(*bc_dt_v_minus)) << ","
             << get_size(get<0>(incoming_null_one_form)));
  // gauge_bc_coeff below is hard-coded here to its default value in SpEC
  constexpr double gauge_bc_coeff = 1.;

  DataVector inertial_radius(get_size(get<0>(inertial_coords)), 0.);
  for (size_t i = 0; i < VolumeDim; ++i) {
    inertial_radius += square(inertial_coords.get(i));
  }
  inertial_radius = sqrt(inertial_radius);

  // add in gauge BC
  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      for (size_t c = 0; c <= VolumeDim; ++c) {
        for (size_t d = 0; d <= VolumeDim; ++d) {
          bc_dt_v_minus->get(a, b) +=
              0.5 *
              (incoming_null_one_form.get(a) * projection_Ab.get(c, b) *
                   outgoing_null_vector.get(d) +
               incoming_null_one_form.get(b) * projection_Ab.get(c, a) *
                   outgoing_null_vector.get(d) -
               0.5 * incoming_null_one_form.get(a) *
                   outgoing_null_one_form.get(b) * incoming_null_vector.get(c) *
                   outgoing_null_vector.get(d) -
               0.5 * incoming_null_one_form.get(b) *
                   outgoing_null_one_form.get(a) * incoming_null_vector.get(c) *
                   outgoing_null_vector.get(d) -
               0.5 * incoming_null_one_form.get(a) *
                   incoming_null_one_form.get(b) * outgoing_null_vector.get(c) *
                   outgoing_null_vector.get(d)) *
              (get(gamma2) - gauge_bc_coeff * (1. / inertial_radius)) *
              char_projected_rhs_dt_v_psi.get(c, d);
        }
      }
    }
  }
}

template <size_t VolumeDim, typename DataType>
void add_constraint_preserving_terms_to_dt_v_minus(
    const gsl::not_null<tnsr::aa<DataType, VolumeDim, Frame::Inertial>*>
        bc_dt_v_minus,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>& incoming_null_one_form,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>& outgoing_null_one_form,
    const tnsr::A<DataType, VolumeDim, Frame::Inertial>& incoming_null_vector,
    const tnsr::A<DataType, VolumeDim, Frame::Inertial>& outgoing_null_vector,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>& projection_ab,
    const tnsr::Ab<DataType, VolumeDim, Frame::Inertial>& projection_Ab,
    const tnsr::AA<DataType, VolumeDim, Frame::Inertial>& projection_AB,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>&
        constraint_char_zero_plus,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>&
        constraint_char_zero_minus,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_v_minus,
    const std::array<DataType, 4>& char_speeds) noexcept {
  ASSERT(get_size(get<0, 0>(*bc_dt_v_minus)) ==
             get_size(get<0>(incoming_null_one_form)),
         "Size of input variables and temporary memory do not match: "
             << get_size(get<0, 0>(*bc_dt_v_minus)) << ","
             << get_size(get<0>(incoming_null_one_form)));

  const double mu = 0.;  // hard-coded value from SpEC Bbh input file Mu = 0

  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      for (size_t c = 0; c <= VolumeDim; ++c) {
        for (size_t d = 0; d <= VolumeDim; ++d) {
          bc_dt_v_minus->get(a, b) +=
              0.25 *
              (incoming_null_vector.get(c) * incoming_null_vector.get(d) *
                   outgoing_null_one_form.get(a) *
                   outgoing_null_one_form.get(b) -
               incoming_null_vector.get(c) * projection_Ab.get(d, a) *
                   outgoing_null_one_form.get(b) -
               incoming_null_vector.get(c) * projection_Ab.get(d, b) *
                   outgoing_null_one_form.get(a) -
               incoming_null_vector.get(d) * projection_Ab.get(c, a) *
                   outgoing_null_one_form.get(b) -
               incoming_null_vector.get(d) * projection_Ab.get(c, b) *
                   outgoing_null_one_form.get(a) +
               2.0 * projection_AB.get(c, d) * projection_ab.get(a, b)) *
              char_projected_rhs_dt_v_minus.get(c, d);
        }
      }
      for (size_t c = 0; c <= VolumeDim; ++c) {
        bc_dt_v_minus->get(a, b) +=
            0.5 * char_speeds.at(3) *
            (constraint_char_zero_minus.get(c) -
             mu * constraint_char_zero_plus.get(c)) *
            (0.5 * outgoing_null_one_form.get(a) *
                 outgoing_null_one_form.get(b) * incoming_null_vector.get(c) +
             projection_ab.get(a, b) * outgoing_null_vector.get(c) -
             projection_Ab.get(c, b) * outgoing_null_one_form.get(a) -
             projection_Ab.get(c, a) * outgoing_null_one_form.get(b));
      }
    }
  }
}

template <size_t VolumeDim, typename DataType>
void add_physical_dof_terms_to_dt_v_minus(
    const gsl::not_null<tnsr::aa<DataType, VolumeDim, Frame::Inertial>*>
        bc_dt_v_minus,
    const Scalar<DataType>& gamma2,
    const tnsr::i<DataType, VolumeDim, Frame::Inertial>&
        unit_interface_normal_one_form,
    const tnsr::I<DataType, VolumeDim, Frame::Inertial>&
        unit_interface_normal_vector,
    const tnsr::A<DataType, VolumeDim, Frame::Inertial>&
        spacetime_unit_normal_vector,
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
        char_projected_rhs_dt_v_minus,
    const tnsr::iaa<DataType, VolumeDim, Frame::Inertial>& phi,
    const tnsr::ijaa<DataType, VolumeDim, Frame::Inertial>& d_phi,
    const tnsr::iaa<DataType, VolumeDim, Frame::Inertial>& d_pi,
    const std::array<DataType, 4>& char_speeds) noexcept {
  ASSERT(get_size(get<0, 0>(*bc_dt_v_minus)) == get_size(get(gamma2)),
         "Size of input variables and temporary memory do not match: "
             << get_size(get<0, 0>(*bc_dt_v_minus)) << ","
             << get_size(get(gamma2)));
  // hard-coded value from SpEC Bbh input file Mu = MuPhys = 0
  const double mu_phys = 0.;
  const bool adjust_phys_using_c4 = true;
  const bool gamma2_in_phys = true;

  // In what follows, we use the notation of Kidder, Scheel & Teukolsky
  // (2001) https://arxiv.org/pdf/gr-qc/0105031.pdf. We will refer to this
  // article as KST henceforth, and use the abbreviation in variable names
  // to disambiguate their origin.
  auto weyl_prop_plus =
      make_with_value<tnsr::ii<DataVector, VolumeDim, Frame::Inertial>>(
          unit_interface_normal_vector, 0.);
  auto weyl_prop_minus =
      make_with_value<tnsr::ii<DataVector, VolumeDim, Frame::Inertial>>(
          unit_interface_normal_vector, 0.);
  {
    // D_(k,i,j) = (1/2) \partial_k g_(ij) and its derivative
    tnsr::ijj<DataVector, VolumeDim, Frame::Inertial> spatial_phi(
        get_size(get(gamma2)));
    for (size_t i = 0; i < VolumeDim; ++i) {
      for (size_t j = i; j < VolumeDim; ++j) {
        for (size_t k = 0; k < VolumeDim; ++k) {
          spatial_phi.get(k, i, j) = phi.get(k, i + 1, j + 1);
        }
      }
    }

    // Compute covariant deriv of extrinsic curvature
    const auto cov_deriv_ex_curv =
        GeneralizedHarmonic::covariant_deriv_of_extrinsic_curvature(
            extrinsic_curvature, spacetime_unit_normal_vector,
            raise_or_lower_first_index(gr::christoffel_first_kind(spatial_phi),
                                       inverse_spatial_metric),
            inverse_spacetime_metric, phi, d_pi, d_phi);

    // Compute spatial Ricci tensor
    auto ricci_3 = GeneralizedHarmonic::spatial_ricci_tensor(
        phi, d_phi, inverse_spatial_metric);

    if (adjust_phys_using_c4) {
      // This adds 4-index constraint terms to 3Ricci so as to cancel
      // out normal derivatives from the final expression for U8.
      // It is much easier to add them here than to recalculate U8
      // from scratch.

      // Add some 4-index constraint terms to 3Ricci.
      for (size_t i = 0; i < VolumeDim; ++i) {
        for (size_t j = i; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            for (size_t l = 0; l < VolumeDim; ++l) {
              ricci_3.get(i, j) += 0.25 * inverse_spatial_metric.get(k, l) *
                                   (d_phi.get(i, k, 1 + l, 1 + j) -
                                    d_phi.get(k, i, 1 + l, 1 + j) +
                                    d_phi.get(j, k, 1 + l, 1 + i) -
                                    d_phi.get(k, j, 1 + l, 1 + i));
            }
          }
        }
      }

      // Add more 4-index constraint terms to 3Ricci
      // These compensate for some of the cov_deriv_ex_curv terms.
      for (size_t i = 0; i < VolumeDim; ++i) {
        for (size_t j = i; j < VolumeDim; ++j) {
          for (size_t a = 0; a <= VolumeDim; ++a) {
            for (size_t k = 0; k < VolumeDim; ++k) {
              ricci_3.get(i, j) +=
                  0.5 * unit_interface_normal_vector.get(k) *
                  spacetime_unit_normal_vector.get(a) *
                  (d_phi.get(i, k, j + 1, a) - d_phi.get(k, i, j + 1, a) +
                   d_phi.get(j, k, i + 1, a) - d_phi.get(k, j, i + 1, a));
            }
          }
        }
      }
    }

    TempBuffer<tmpl::list<
        // spatial projection operators P_ij, P^ij, and P^i_j
        ::Tags::TempII<0, VolumeDim, Frame::Inertial, DataVector>,
        ::Tags::Tempii<1, VolumeDim, Frame::Inertial, DataVector>,
        ::Tags::TempIj<2, VolumeDim, Frame::Inertial, DataVector>>>
        local_buffer(get_size(get<0>(unit_interface_normal_vector)));
    auto& spatial_projection_IJ =
        get<::Tags::TempII<0, VolumeDim, Frame::Inertial, DataVector>>(
            local_buffer);
    auto& spatial_projection_ij =
        get<::Tags::Tempii<1, VolumeDim, Frame::Inertial, DataVector>>(
            local_buffer);
    auto& spatial_projection_Ij =
        get<::Tags::TempIj<2, VolumeDim, Frame::Inertial, DataVector>>(
            local_buffer);

    // Make spatial projection operators
    gr::transverse_projection_operator(make_not_null(&spatial_projection_IJ),
                                       inverse_spatial_metric,
                                       unit_interface_normal_vector);

    const auto spatial_metric = [&spacetime_metric]() noexcept {
      tnsr::ii<DataVector, VolumeDim, Frame::Inertial> tmp_metric(
          get_size(get<0, 0>(spacetime_metric)));
      for (size_t j = 0; j < VolumeDim; ++j) {
        for (size_t k = j; k < VolumeDim; ++k) {
          tmp_metric.get(j, k) = spacetime_metric.get(1 + j, 1 + k);
        }
      }
      return tmp_metric;
    }();
    gr::transverse_projection_operator(make_not_null(&spatial_projection_ij),
                                       spatial_metric,
                                       unit_interface_normal_one_form);

    gr::transverse_projection_operator(make_not_null(&spatial_projection_Ij),
                                       unit_interface_normal_vector,
                                       unit_interface_normal_one_form);

    gr::weyl_propagating(
        make_not_null(&weyl_prop_plus), ricci_3, extrinsic_curvature,
        inverse_spatial_metric, cov_deriv_ex_curv, unit_interface_normal_vector,
        spatial_projection_IJ, spatial_projection_ij, spatial_projection_Ij, 1);
    gr::weyl_propagating(make_not_null(&weyl_prop_minus), ricci_3,
                         extrinsic_curvature, inverse_spatial_metric,
                         cov_deriv_ex_curv, unit_interface_normal_vector,
                         spatial_projection_IJ, spatial_projection_ij,
                         spatial_projection_Ij, -1);
  }

  auto U3p = make_with_value<tnsr::aa<DataVector, VolumeDim, Frame::Inertial>>(
      unit_interface_normal_vector, 0.);
  auto U3m = make_with_value<tnsr::aa<DataVector, VolumeDim, Frame::Inertial>>(
      unit_interface_normal_vector, 0.);
  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      for (size_t i = 0; i < VolumeDim; ++i) {
        for (size_t j = 0; j < VolumeDim; ++j) {
          U3p.get(a, b) += 2.0 * projection_Ab.get(i + 1, a) *
                           projection_Ab.get(j + 1, b) *
                           weyl_prop_plus.get(i, j);
          U3m.get(a, b) += 2.0 * projection_Ab.get(i + 1, a) *
                           projection_Ab.get(j + 1, b) *
                           weyl_prop_minus.get(i, j);
        }
      }
    }
  }

  // Impose physical boundary condition
  if (gamma2_in_phys) {
    DataVector tmp(get_size(get<0>(unit_interface_normal_vector)));
    for (size_t c = 0; c <= VolumeDim; ++c) {
      for (size_t d = c; d <= VolumeDim; ++d) {
        for (size_t a = 0; a <= VolumeDim; ++a) {
          for (size_t b = 0; b <= VolumeDim; ++b) {
            tmp = 0.;
            for (size_t i = 0; i < VolumeDim; ++i) {
              tmp += unit_interface_normal_vector.get(i) *
                     three_index_constraint.get(i, a, b);
            }
            tmp *= get(gamma2);
            bc_dt_v_minus->get(c, d) +=
                (projection_Ab.get(a, c) * projection_Ab.get(b, d) -
                 0.5 * projection_ab.get(c, d) * projection_AB.get(a, b)) *
                (char_projected_rhs_dt_v_minus.get(a, b) +
                 char_speeds.at(3) *
                     (U3m.get(a, b) - tmp - mu_phys * U3p.get(a, b)));
          }
        }
      }
    }
  } else {
    for (size_t c = 0; c <= VolumeDim; ++c) {
      for (size_t d = c; d <= VolumeDim; ++d) {
        for (size_t a = 0; a <= VolumeDim; ++a) {
          for (size_t b = 0; b <= VolumeDim; ++b) {
            bc_dt_v_minus->get(c, d) +=
                (projection_Ab.get(a, c) * projection_Ab.get(b, d) -
                 0.5 * projection_ab.get(c, d) * projection_AB.get(a, b)) *
                (char_projected_rhs_dt_v_minus.get(a, b) +
                 char_speeds.at(3) * (U3m.get(a, b) - mu_phys * U3p.get(a, b)));
          }
        }
      }
    }
  }
}

template <size_t VolumeDim, typename DataType>
void set_dt_v_minus_constraint_preserving(
    gsl::not_null<tnsr::aa<DataType, VolumeDim, Frame::Inertial>*>
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
    const std::array<DataType, 4>& char_speeds) noexcept;

template <size_t VolumeDim, typename DataType>
void set_dt_v_minus_constraint_preserving_physical(
    gsl::not_null<tnsr::aa<DataType, VolumeDim, Frame::Inertial>*>
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
    const std::array<DataType, 4>& char_speeds) noexcept;
// @}
}  // namespace detail
}  // namespace GeneralizedHarmonic::BoundaryConditions
