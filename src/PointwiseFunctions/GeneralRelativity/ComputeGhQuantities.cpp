// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace GeneralizedHarmonic {
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::iaa<DataType, SpatialDim, Frame> phi(
    const Scalar<DataType>& lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>&
        deriv_spatial_metric) noexcept {
  tnsr::iaa<DataType, SpatialDim, Frame> phi(
      make_with_value<DataType>(deriv_lapse, 0.));
  for (size_t k = 0; k < SpatialDim; ++k) {
    phi.get(k, 0, 0) = -2. * get(lapse) * deriv_lapse.get(k);
    for (size_t m = 0; m < SpatialDim; ++m) {
      for (size_t n = 0; n < SpatialDim; ++n) {
        phi.get(k, 0, 0) +=
            deriv_spatial_metric.get(k, m, n) * shift.get(m) * shift.get(n) +
            2. * spatial_metric.get(m, n) * shift.get(m) *
                deriv_shift.get(k, n);
      }
    }

    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t m = 0; m < SpatialDim; ++m) {
        phi.get(k, 0, i + 1) +=
            deriv_spatial_metric.get(k, m, i) * shift.get(m) +
            spatial_metric.get(m, i) * deriv_shift.get(k, m);
      }
      for (size_t j = i; j < SpatialDim; ++j) {
        phi.get(k, i + 1, j + 1) = deriv_spatial_metric.get(k, i, j);
      }
    }
  }
  return phi;
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::aa<DataType, SpatialDim, Frame> pi(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  tnsr::aa<DataType, SpatialDim, Frame> pi{
      make_with_value<DataType>(lapse, 0.)};

  get<0,0>(pi) = -2. * get(lapse) * get(dt_lapse);

  for (size_t m = 0; m < SpatialDim; ++m) {
    for (size_t n = 0; n < SpatialDim; ++n) {
      get<0,0>(pi) +=
          dt_spatial_metric.get(m, n) * shift.get(m) * shift.get(n) +
          2. * spatial_metric.get(m, n) * shift.get(m) * dt_shift.get(n);
    }
  }

  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t m = 0; m < SpatialDim; ++m) {
      pi.get(0, i + 1) += dt_spatial_metric.get(m, i) * shift.get(m) +
                          spatial_metric.get(m, i) * dt_shift.get(m);
    }
    for (size_t j = i; j < SpatialDim; ++j) {
      pi.get(i + 1, j + 1) = dt_spatial_metric.get(i, j);
    }
  }
  for (size_t mu = 0; mu < SpatialDim + 1; ++mu) {
    for (size_t nu = mu; nu < SpatialDim + 1; ++nu) {
      for (size_t i = 0; i < SpatialDim; ++i) {
        pi.get(mu, nu) -= shift.get(i) * phi.get(i, mu, nu);
      }
      pi.get(mu, nu) /= -get(lapse);
    }
  }
  return pi;
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> gauge_source(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const Scalar<DataType>& trace_extrinsic_curvature,
    const tnsr::i<DataType, SpatialDim, Frame>&
        trace_christoffel_last_indices) noexcept {
  DataType one_over_lapse = 1. / get(lapse);
  auto gauge_source_h =
      make_with_value<tnsr::a<DataType, SpatialDim, Frame>>(lapse, 0.);

  // Temporary to avoid more nested loops.
  auto temp = dt_shift;
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t k = 0; k < SpatialDim; ++k) {
      temp.get(i) -= shift.get(k) * deriv_shift.get(k, i);
    }
  }

  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t k = 0; k < SpatialDim; ++k) {
      gauge_source_h.get(i + 1) += spatial_metric.get(i, k) * temp.get(k);
    }
    gauge_source_h.get(i + 1) *= square(one_over_lapse);
  }

  for (size_t i = 0; i < SpatialDim; ++i) {
    gauge_source_h.get(i + 1) += one_over_lapse * deriv_lapse.get(i) -
                                 trace_christoffel_last_indices.get(i);
  }

  for (size_t i = 0; i < SpatialDim; ++i) {
    get<0>(gauge_source_h) +=
        shift.get(i) *
        (gauge_source_h.get(i + 1) + deriv_lapse.get(i) * one_over_lapse);
  }
  get<0>(gauge_source_h) -= one_over_lapse * get(dt_lapse) +
                            get(lapse) * get(trace_extrinsic_curvature);

  return gauge_source_h;
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> extrinsic_curvature(
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  auto ex_curv = make_with_value<tnsr::ii<DataType, SpatialDim, Frame>>(pi, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      for (size_t a = 0; a <= SpatialDim; ++a) {
        ex_curv.get(i, j) += 0.5 *
                             (phi.get(i, j + 1, a) + phi.get(j, i + 1, a)) *
                             spacetime_normal_vector.get(a);
      }
      ex_curv.get(i, j) += 0.5 * pi.get(i + 1, j + 1);
    }
  }
  return ex_curv;
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ijj<DataType, SpatialDim, Frame> deriv_spatial_metric(
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  auto deriv_spatial_metric =
      make_with_value<tnsr::ijj<DataType, SpatialDim, Frame>>(phi, 0.);
  for (size_t k = 0; k < SpatialDim; ++k) {
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = i; j < SpatialDim; ++j) {
        deriv_spatial_metric.get(k, i, j) = phi.get(k, i + 1, j + 1);
      }
    }
  }
  return deriv_spatial_metric;
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::i<DataType, SpatialDim, Frame> spatial_deriv_of_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  auto deriv_lapse =
    make_with_value<tnsr::i<DataType, SpatialDim, Frame>>(get(lapse), 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      for (size_t b = 0; b < SpatialDim + 1; ++b) {
        deriv_lapse.get(i) += phi.get(i, a, b)
          * spacetime_unit_normal.get(a) * spacetime_unit_normal.get(b);
      }
    }
    deriv_lapse.get(i) *= -0.5 * get(lapse);
  }
  return deriv_lapse;
}

template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> time_deriv_of_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept {
  auto dt_lapse = make_with_value<Scalar<DataType>>(get(lapse), 0.);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      // first term
      get(dt_lapse) += get(lapse) * pi.get(a, b)
            * spacetime_unit_normal.get(a) * spacetime_unit_normal.get(b);
      // second term
      for (size_t i = 0; i < SpatialDim; ++i) {
        get(dt_lapse) -= shift.get(i) * phi.get(i, a, b)
            * spacetime_unit_normal.get(a) * spacetime_unit_normal.get(b);
      }
    }
  }
  get(dt_lapse) *= 0.5 * get(lapse);
  return dt_lapse;
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> time_deriv_of_spatial_metric(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept {
  auto dt_spatial_metric =
    make_with_value<tnsr::ii<DataType, SpatialDim, Frame>>(get(lapse), 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      dt_spatial_metric.get(i, j) = -get(lapse) * pi.get(i + 1, j + 1);
      for (size_t k = 0; k < SpatialDim; ++k) {
        dt_spatial_metric.get(i, j) += shift.get(k) * phi.get(k, i + 1, j + 1);
      }
    }
  }
  return dt_spatial_metric;
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> spacetime_deriv_of_det_spatial_metric(
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  auto d4_det_spatial_metric = make_with_value<tnsr::a<DataType,
      SpatialDim, Frame>>(get(sqrt_det_spatial_metric), 0.);
  const auto deriv_of_g = deriv_spatial_metric(phi);
  const auto det_spatial_metric = square(get(sqrt_det_spatial_metric));
  // \f$ \partial_0 g = g g^{jk} \partial_0 g_{jk}\f$
  for (size_t j = 0; j < SpatialDim; ++j) {
    for (size_t k = 0; k < SpatialDim; ++k) {
      get<0>(d4_det_spatial_metric) += inverse_spatial_metric.get(j, k)
        * dt_spatial_metric.get(j, k);
    }
  }
  get<0>(d4_det_spatial_metric) *= det_spatial_metric;
  // \f$ \partial_i g = g g^{jk} \partial_i g_{jk}\f$
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t k = 0; k < SpatialDim; ++k) {
        d4_det_spatial_metric.get(i + 1) += inverse_spatial_metric.get(j, k)
          * deriv_of_g.get(i, j, k);
      }
    }
    d4_det_spatial_metric.get(i + 1) *= det_spatial_metric;
  }
  return d4_det_spatial_metric;
}

}  // namespace GeneralizedHarmonic

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                  \
  template tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>                     \
  GeneralizedHarmonic::phi(                                                   \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& deriv_lapse,        \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>& deriv_shift,       \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,    \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&                   \
          deriv_spatial_metric) noexcept;                                     \
  template tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>                      \
  GeneralizedHarmonic::pi(                                                    \
      const Scalar<DTYPE(data)>& lapse, const Scalar<DTYPE(data)>& dt_lapse,  \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& dt_shift,           \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,    \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& dt_spatial_metric, \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;    \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>                      \
  GeneralizedHarmonic::extrinsic_curvature(                                   \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_normal_vector,                                            \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,                \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;    \
  template tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>                     \
  GeneralizedHarmonic::deriv_spatial_metric(                                  \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;    \
  template tnsr::i<DTYPE(data), DIM(data), FRAME(data)>                       \
  GeneralizedHarmonic::spatial_deriv_of_lapse(                                \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
        spacetime_unit_normal,                                                \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;    \
  template Scalar<DTYPE(data)> GeneralizedHarmonic::time_deriv_of_lapse(      \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
        spacetime_unit_normal,                                                \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,              \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi) noexcept;      \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>                      \
  GeneralizedHarmonic::time_deriv_of_spatial_metric(                          \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,              \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi) noexcept;      \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)>                       \
  GeneralizedHarmonic::spacetime_deriv_of_det_spatial_metric(                 \
      const Scalar<DTYPE(data)>& det_spatial_metric,                          \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
        inverse_spatial_metric,                                               \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& dt_spatial_metric, \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;    \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)>                       \
  GeneralizedHarmonic::gauge_source(                                          \
      const Scalar<DTYPE(data)>& lapse, const Scalar<DTYPE(data)>& dt_lapse,  \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& deriv_lapse,        \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& dt_shift,           \
      const tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>& deriv_shift,       \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,    \
      const Scalar<DTYPE(data)>& trace_extrinsic_curvature,                   \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>&                     \
          trace_christoffel_last_indices) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
/// \endcond
