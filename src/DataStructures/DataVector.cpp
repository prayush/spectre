// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataVector.hpp"

//~ #include <algorithm>
#include <pup.h>

//~ #include "Utilities/StdHelpers.hpp"

/// Construct a DataVector with value(s)
MAKE_EXPRESSION_VEC_DEF_CONSTRUCT_WITH_VALUE(DataVector)

/// \cond HIDDEN_SYMBOLS
/// Construct / Assign DataVector with / to DataVector reference or rvalue
// clang-tidy: calling a base constructor other than the copy constructor.
//             We reset the base class in reset_pointer_vector after calling its
//             default constructor
MAKE_EXPRESSION_VEC_DEF_CONSTRUCT_WITH_VEC(DataVector)
/// \endcond

/// Define shift and (in)equivalence operators for DataVector with itself
MAKE_EXPRESSION_VECMATH_OP_DEF_COMP_SELF(DataVector)

/// Charm++ object packing / unpacking
MAKE_EXPRESSION_VEC_OP_PUP_CHARM(DataVector)

/// \cond
template DataVector::DataVector(std::initializer_list<double> list) noexcept;
/// \endcond
