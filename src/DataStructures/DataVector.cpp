// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataVector.hpp"

#include <algorithm>
#include <pup.h>

#include "Utilities/StdHelpers.hpp"

MAKE_EXPRESSION_VEC_DEF_CONSTRUCT_WITH_VALUE(DataVector)

/// \cond HIDDEN_SYMBOLS
// clang-tidy: calling a base constructor other than the copy constructor.
//             We reset the base class in reset_pointer_vector after calling its
//             default constructor
MAKE_EXPRESSION_VEC_DEF_CONSTRUCT_WITH_VEC(DataVector)
/// \endcond

MAKE_EXPRESSION_VEC_OP_PUP_CHARM(DataVector)

std::ostream& operator<<(std::ostream& os, const DataVector& d) {
  // This function is inside the detail namespace StdHelpers.hpp
  StdHelpers_detail::print_helper(os, d.begin(), d.end());
  return os;
}

MAKE_EXPRESSION_VECMATH_OP_DEF_COMP_SELF(DataVector)

/// \cond
template DataVector::DataVector(std::initializer_list<double> list) noexcept;
/// \endcond
