// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <charm++.h>
#include <memory>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/ElementId.hpp"
#include "Parallel/PupStlCpp11.hpp"

/*!
 * \ingroup ComputationalDomainGroup
 * \brief The CoordinateMap for the Element from the Logical frame to the
 * `TargetFrame`
 *
 * An ElementMap takes a CoordinateMap for a Block and an ElementId as input,
 * and then "prepends" the correct affine map to the CoordinateMap so that the
 * map corresponds to the coordinate map for the Element rather than the Block.
 * This allows DomainCreators to only specify the maps for the Blocks without
 * worrying about how the domain may be decomposed beyond that.
 */
template <size_t Dim, typename TargetFrame>
class ElementMap {
 public:
  /// \cond HIDDEN_SYMBOLS
  ElementMap() = default;
  /// \endcond

  ElementMap(
      ElementId<Dim> element_id,
      std::unique_ptr<CoordinateMapBase<Frame::Logical, TargetFrame, Dim>>
          block_map) noexcept;

  const CoordinateMapBase<Frame::Logical, TargetFrame, Dim>& block_map() const
      noexcept {
    return *block_map_;
  }

  const ElementId<Dim>& element_id() const noexcept { return element_id_; }

  template <typename T>
  tnsr::I<T, Dim, TargetFrame> operator()(
      tnsr::I<T, Dim, Frame::Logical> source_point) const noexcept {
    apply_affine_transformation_to_point(source_point);
    return block_map_->operator()(source_point);
  }

  template <typename T>
  tnsr::I<T, Dim, Frame::Logical> inverse(
      const tnsr::I<T, Dim, TargetFrame>& target_point) const noexcept {
    auto source_point{block_map_->inverse(target_point)};
    // Apply the affine map to the points
    for (size_t d = 0; d < Dim; ++d) {
      source_point.get(d) =
          source_point.get(d) * gsl::at(map_inverse_slope_, d) +
          gsl::at(map_inverse_offset_, d);
    }
    return source_point;
  }

  template <typename T>
  Tensor<T, tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<Dim, UpLo::Up, Frame::Logical>,
                    SpatialIndex<Dim, UpLo::Lo, TargetFrame>>>
  inv_jacobian(tnsr::I<T, Dim, Frame::Logical> source_point) const noexcept {
    apply_affine_transformation_to_point(source_point);
    auto inv_jac = block_map_->inv_jacobian(source_point);
    for (size_t d = 0; d < Dim; ++d) {
      for (size_t i = 0; i < Dim; ++i) {
        inv_jac.get(d, i) *= gsl::at(inverse_jacobian_, d);
      }
    }
    return inv_jac;
  }

  template <typename T>
  Tensor<T, tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<Dim, UpLo::Up, TargetFrame>,
                    SpatialIndex<Dim, UpLo::Lo, Frame::Logical>>>
  jacobian(tnsr::I<T, Dim, Frame::Logical> source_point) const noexcept {
    apply_affine_transformation_to_point(source_point);
    auto jac = block_map_->jacobian(source_point);
    for (size_t d = 0; d < Dim; ++d) {
      for (size_t i = 0; i < Dim; ++i) {
        jac.get(i, d) *= gsl::at(jacobian_, d);
      }
    }
    return jac;
  }

  // clang-tidy: do not use references
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  template <typename T>
  void apply_affine_transformation_to_point(
      tnsr::I<T, Dim, Frame::Logical>& source_point) const noexcept {
    for (size_t d = 0; d < Dim; ++d) {
      source_point.get(d) = source_point.get(d) * gsl::at(map_slope_, d) +
                            gsl::at(map_offset_, d);
    }
  }

  std::unique_ptr<CoordinateMapBase<Frame::Logical, TargetFrame, Dim>>
      block_map_{nullptr};
  ElementId<Dim> element_id_{};
  // map_slope_[i] = 0.5 * (segment_ids[i].endpoint(Side::Upper) -
  //                        segment_ids[i].endpoint(Side::Lower))
  std::array<double, Dim> map_slope_{
      make_array<Dim>(std::numeric_limits<double>::signaling_NaN())};
  // map_offset_[i] = 0.5 * (segment_ids[i].endpoint(Side::Upper) +
  //                         segment_ids[i].endpoint(Side::Lower))
  std::array<double, Dim> map_offset_{
      make_array<Dim>(std::numeric_limits<double>::signaling_NaN())};
  // map_inverse_slope_[i] = 1.0 / map_slope_[i]
  std::array<double, Dim> map_inverse_slope_{
      make_array<Dim>(std::numeric_limits<double>::signaling_NaN())};
  // map_inverse_offset_[i] = -map_offset_[i] / map_slope_[i]
  std::array<double, Dim> map_inverse_offset_{
      make_array<Dim>(std::numeric_limits<double>::signaling_NaN())};
  // Note: The Jacobian is diagonal
  std::array<double, Dim> jacobian_{map_slope_};
  std::array<double, Dim> inverse_jacobian_{map_inverse_slope_};
};
