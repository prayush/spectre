// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines metafunction make_boost_variant_over

#pragma once

#include <array>
#include <boost/variant.hpp>
#include <cstddef>
#include <initializer_list>
#include <pup.h>
#include <string>
#include <utility>

#include "Utilities/PrettyType.hpp"
#include "Utilities/TypeTraits.hpp"

namespace detail {
template <typename Sequence>
struct make_boost_variant_over_impl;

template <template <typename...> class Sequence, typename... Ts>
struct make_boost_variant_over_impl<Sequence<Ts...>> {
  static_assert(not cpp17::disjunction<std::is_same<
                    std::decay_t<std::remove_pointer_t<Ts>>, void>...>::value,
                "Cannot create a boost::variant with a 'void' type.");
  using type = boost::variant<Ts...>;
};
}  // namespace detail

/*!
 * \ingroup UtilitiesGroup
 * \brief Create a boost::variant with all all the types inside the typelist
 * Sequence
 *
 * \metareturns boost::variant of all types inside `Sequence`
 */
template <typename Sequence>
using make_boost_variant_over =
    typename detail::make_boost_variant_over_impl<Sequence>::type;

namespace BoostVariant_detail {
// clang-tidy: do not use non-const references
template <class T, class... Ts>
char pup_helper(int& index, PUP::er& p, boost::variant<Ts...>& var,  // NOLINT
                const int send_index) {
  if (index == send_index) {
    if (p.isUnpacking()) {
      T t{};
      p | t;
      var = std::move(t);
    } else {
      p | boost::get<T>(var);
    }
  }
  index++;
  return '0';
}
}  // namespace BoostVariant_detail

template <class... Ts>
void pup(PUP::er& p, boost::variant<Ts...>& var) {  // NOLINT
  int index = 0;
  int send_index = var.which();
  p | send_index;
  (void)std::initializer_list<char>{
      BoostVariant_detail::pup_helper<Ts>(index, p, var, send_index)...};
}

template <typename... Ts>
inline void operator|(PUP::er& p, boost::variant<Ts...>& d) {  // NOLINT
  pup(p, d);
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Get the type name of the current state of the boost::variant
 */
template <typename... Ts>
std::string type_of_current_state(
    const boost::variant<Ts...>& variant) noexcept {
  // clang-format off
  // clang-tidy: use gsl::at (we know it'll be in bounds and want fewer
  // includes) clang-format moves the comment to the wrong line
  return std::array<std::string, sizeof...(Ts)>{  // NOLINT
      {pretty_type::get_name<Ts>()...}}[static_cast<size_t>(variant.which())];
  // clang-format on
}
