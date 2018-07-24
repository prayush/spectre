// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions computing partial derivatives.

#pragma once

#include <array>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Variables.hpp"

template <size_t Dim>
class Index;

namespace Tags {
template <size_t Dim>
struct Extents;
template <class TagList>
struct Variables;

namespace Tags_detail {
template <typename T, typename S, typename U,
          typename = std::nullptr_t>
struct deriv_impl;
}  // namespace Tags_detail

/*!
 * \ingroup DataBoxTagsGroup
 * \brief Prefix for a spatial derivative
 *
 * There are three variants of this tag that change how the derivatives are
 * computed depending on which is chosen. The simplest is the non-compute
 * item version for a Tensor tag. This takes three template parameters:
 * 1. The tag to wrap
 * 2. The volume dim as a type (e.g. `tmpl::size_t<Dim>`)
 * 3. The frame of the derivative index
 *
 * The second variant of the derivative tag is a compute item that computes
 * the partial derivatives in a Frame that is not the logical frame. This
 * compute item is used by specifying the list of tags in the
 * `Tags::Variables<tmpl::list<Tags...>>`, the list of tags to be
 * differentiated, and the Tag for inverse Jacobian between the logical frame
 * and the frame in which the derivative is taken. Note that the derivative Tags
 * must be a contiguous subset from the head of the `Tags...` pack.
 *
 * The third variant of the derivative tag is a compute item that computes the
 * partial derivatives in the logical frame. In that case the template
 * parameters must be the list of Variables tags and derivative tags, with the
 * last parameter being a `std::integral_constant` of the dimension of the mesh.
 */
template <typename T, typename S = void, typename U = void>
struct deriv : Tags_detail::deriv_impl<T, S, U> {};
}  // namespace Tags

/// \ingroup NumericalAlgorithmsGroup
/// \brief Compute the partial derivatives of each variable with respect to
/// the logical coordinate.
///
/// \requires `DerivativeTags` to be the head of `VariableTags`
///
/// \return a `Variables` with a spatial tensor index appended to the front
/// of each tensor within `u` and each `Tag` wrapped with a `Tags::deriv`.
///
/// \tparam DerivativeTags the subset of `VariableTags` for which derivatives
/// are computed.
template <typename DerivativeTags, typename VariableTags, size_t Dim>
std::array<Variables<DerivativeTags>, Dim> logical_partial_derivatives(
    const Variables<VariableTags>& u, const Index<Dim>& extents) noexcept;

/// \ingroup NumericalAlgorithmsGroup
/// \brief Compute the partial derivatives of each variable with respect to
/// the coordinates of `DerivativeFrame`.
///
/// \requires `DerivativeTags` to be the head of `VariableTags`
///
/// \return a `Variables` with a spatial tensor index appended to the front
/// of each tensor within `u` and each `Tag` wrapped with a `Tags::deriv`.
///
/// \tparam DerivativeTags the subset of `VariableTags` for which derivatives
/// are computed.
template <typename DerivativeTags, typename VariableTags, size_t Dim,
          typename DerivativeFrame>
Variables<db::wrap_tags_in<Tags::deriv, DerivativeTags, tmpl::size_t<Dim>,
                           DerivativeFrame>>
partial_derivatives(
    const Variables<VariableTags>& u, const Index<Dim>& extents,
    const InverseJacobian<DataVector, Dim, Frame::Logical, DerivativeFrame>&
        inverse_jacobian) noexcept;

namespace Tags {
namespace Tags_detail {
template <typename Tag, typename VolumeDim, typename Frame>
struct deriv_impl<Tag, VolumeDim, Frame,
                  Requires<tt::is_a_v<Tensor, db::item_type<Tag>>>>
    : db::DataBoxPrefix {
  using type = TensorMetafunctions::prepend_spatial_index<
      db::item_type<Tag>, VolumeDim::value, UpLo::Lo, Frame>;
  using tag = Tag;
  static constexpr db::DataBoxString label = "deriv";
};

// Partial derivatives with derivative index in the frame related to the logical
// coordinates by InverseJacobianTag
template <typename... VariablesTags, typename... DerivativeTags,
          typename InverseJacobianTag>
struct deriv_impl<
    tmpl::list<VariablesTags...>, tmpl::list<DerivativeTags...>,
    InverseJacobianTag,
    Requires<tt::is_a_v<Tensor, db::item_type<InverseJacobianTag>>>>
    : db::ComputeItemTag {
 private:
  using derivative_frame_index =
      tmpl::back<typename db::item_type<InverseJacobianTag>::index_list>;
  using variables_tags = tmpl::list<VariablesTags...>;

 public:
  static constexpr db::DataBoxString label = "deriv";
  static constexpr auto function =
      partial_derivatives<tmpl::list<DerivativeTags...>, variables_tags,
                          derivative_frame_index::dim,
                          typename derivative_frame_index::Frame>;
  using argument_tags = tmpl::list<Tags::Variables<variables_tags>,
                                   Tags::Extents<derivative_frame_index::dim>,
                                   InverseJacobianTag>;
};

// Logical partial derivatives
template <typename... VariablesTags, typename... DerivativeTags, typename T,
          T Dim>
struct deriv_impl<tmpl::list<VariablesTags...>, tmpl::list<DerivativeTags...>,
                  std::integral_constant<T, Dim>,
                  Requires<(0 < Dim and 4 > Dim)>> : db::ComputeItemTag {
  using variables_tags = tmpl::list<VariablesTags...>;
  static constexpr db::DataBoxString label = "deriv";
  static constexpr auto function =
      logical_partial_derivatives<tmpl::list<DerivativeTags...>, variables_tags,
                                  Dim>;
  using argument_tags =
      tmpl::list<Tags::Variables<variables_tags>, Tags::Extents<Dim>>;
};
}  // namespace Tags_detail
}  // namespace Tags
