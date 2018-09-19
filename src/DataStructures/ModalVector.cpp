// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/ModalVector.hpp"

#include <algorithm>
#include <pup.h>

#include "Utilities/StdHelpers.hpp"

MAKE_EXPRESSION_VEC_DEF_CONSTRUCT_WITH_VALUE(ModalVector)

/// \cond HIDDEN_SYMBOLS
// clang-tidy: calling a base constructor other than the copy constructor.
//             We reset the base class in reset_pointer_vector after calling its
//             default constructor
MAKE_EXPRESSION_VEC_DEF_CONSTRUCT_WITH_VEC(ModalVector)
/// \endcond

MAKE_EXPRESSION_VEC_OP_PUP_CHARM(ModalVector)

std::ostream& operator<<(std::ostream& os, const ModalVector& d) {
  // This function is inside the detail namespace StdHelpers.hpp
  StdHelpers_detail::print_helper(os, d.begin(), d.end());
  return os;
}

MAKE_EXPRESSION_VECMATH_OP_DEF_COMP_SELF(ModalVector)

/// \cond
template ModalVector::ModalVector(std::initializer_list<double> list);
/// \endcond
