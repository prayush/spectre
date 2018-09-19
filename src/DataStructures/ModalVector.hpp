// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class ModalVector.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <ostream>
#include <type_traits>
#include <vector>

#include "DataStructures/VectorMacros.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/PointerVector.hpp" // IWYU pragma: keep
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"          // for list

/// \cond HIDDEN_SYMBOLS
namespace PUP {
class er;
}  // namespace PUP

// clang-tidy: no using declarations in header files
//             We want the std::abs to be used
using std::abs;  // NOLINT
/// \endcond

// IWYU doesn't like that we want PointerVector.hpp to expose Blaze and also
// have ModalVector.hpp to expose PointerVector.hpp without including Blaze
// directly in ModalVector.hpp
//
// IWYU pragma: no_include <blaze/math/dense/DenseVector.h>
// IWYU pragma: no_include <blaze/math/expressions/DVecDVecAddExpr.h>
// IWYU pragma: no_include <blaze/math/expressions/DVecDVecDivExpr.h>
// IWYU pragma: no_include <blaze/math/expressions/DVecDVecMultExpr.h>
// IWYU pragma: no_include <blaze/math/expressions/DVecDVecSubExpr.h>
// IWYU pragma: no_include <blaze/math/expressions/DVecMapExpr.h>
// IWYU pragma: no_include <blaze/math/expressions/DVecScalarDivExpr.h>
// IWYU pragma: no_include <blaze/math/expressions/DVecScalarMultExpr.h>
// IWYU pragma: no_include <blaze/math/expressions/DenseVector.h>
// IWYU pragma: no_include <blaze/math/expressions/Vector.h>
// IWYU pragma: no_include <blaze/math/typetraits/IsVector.h>
// IWYU pragma: no_include <blaze/math/expressions/Forward.h>
// IWYU pragma: no_include <blaze/math/AlignmentFlag.h>
// IWYU pragma: no_include <blaze/math/PaddingFlag.h>
// IWYU pragma: no_include <blaze/math/traits/AddTrait.h>
// IWYU pragma: no_include <blaze/math/traits/DivTrait.h>
// IWYU pragma: no_include <blaze/math/traits/MultTrait.h>
// IWYU pragma: no_include <blaze/math/traits/SubTrait.h>
// IWYU pragma: no_include <blaze/system/TransposeFlag.h>
// IWYU pragma: no_include <blaze/math/traits/BinaryMapTrait.h>
// IWYU pragma: no_include <blaze/math/traits/UnaryMapTrait.h>
// IWYU pragma: no_include <blaze/math/typetraits/TransposeFlag.h>
// IWYU pragma: no_include "DataStructures/DataVector.hpp"

// IWYU pragma: no_forward_declare blaze::DenseVector
// IWYU pragma: no_forward_declare blaze::UnaryMapTrait
// IWYU pragma: no_forward_declare blaze::BinaryMapTrait
// IWYU pragma: no_forward_declare blaze::IsVector
// IWYU pragma: no_forward_declare blaze::TransposeFlag

/*!
 * \ingroup DataStructuresGroup
 * \brief A class for storing spectral coefficients on a mesh.
 *
 * A ModalVector holds an array of spectral coefficients, and can be
 * either owning (the array is deleted when the ModalVector goes out of scope)
 * or non-owning, meaning it just has a pointer to an array.
 *
 * Only basic mathematical operations are supported with ModalVectors. In
 * addition to addition, subtraction, multiplication, division, there
 * are the following element-wise operations:
 *
 * - abs/fabs/magnitude
 * - max
 * - min
 *
 * In order to allow filtering, multiplication (*, *=) and division (/, /=)
 * operations with a DenseVectors (holding filters) is supported.
 *
 */
MAKE_EXPRESSION_DATA_MODAL_VECTOR_CLASSES(ModalVector)
MAKE_EXPRESSION_VECMATH_OP_COMP_SELF(ModalVector)

/// \cond
MAKE_EXPRESSION_VECMATH_OP_COMP_DV(ModalVector)
/// \endcond

// Specialize Blaze type traits to correctly handle ModalVector
MAKE_EXPRESSION_VECMATH_SPECIALIZE_BLAZE_ARITHMETIC_TRAITS(ModalVector)

// Specialize Blaze UnaryMap traits to handle ModalVector
namespace blaze {
template <typename Operator>
struct UnaryMapTrait<ModalVector, Operator> {
  // Forbid math operations in this specialization of UnaryMap traits for
  // ModalVector that are unlikely to be used on spectral coefficients
  static_assert(not tmpl::list_contains_v<tmpl::list<
                blaze::Sqrt, blaze::Cbrt,
                blaze::InvSqrt, blaze::InvCbrt,
                blaze::Acos, blaze::Acosh, blaze::Cos, blaze::Cosh,
                blaze::Asin, blaze::Asinh, blaze::Sin, blaze::Sinh,
                blaze::Atan, blaze::Atan2, blaze::Atanh,
                blaze::Tan, blaze::Tanh, blaze::Hypot,
                blaze::Exp, blaze::Exp2, blaze::Exp10,
                blaze::Log, blaze::Log2, blaze::Log10,
                blaze::Erf, blaze::Erfc, blaze::StepFunction
                >, Operator>,
                "This operation is not permitted on a ModalVector."
                "Only unary operation permitted are: max, min, abs, fabs.");
  // Selectively allow unary operations commonly used on spectral coefficients
  static_assert(tmpl::list_contains_v<tmpl::list<
                blaze::Abs,
                blaze::Max, blaze::Min,
                blaze::AddScalar<double>,
                blaze::SubScalarRhs<double>,
                blaze::SubScalarLhs<double>
                >, Operator>,
                "Only unary operation permitted on a ModalVector are:"
                " max, min, abs, fabs.");
  using Type = ModalVector;
};

// Specialize Blaze UnaryMap traits to handle ModalVector
template <typename Operator>
struct BinaryMapTrait<ModalVector, ModalVector, Operator> {
  // Forbid math operations in this specialization of BinaryMap traits for
  // ModalVector that are unlikely to be used on spectral coefficients
  static_assert(not tmpl::list_contains_v<tmpl::list<blaze::Max, blaze::Min
                >, Operator>,
                "This operation is not permitted on a ModalVector."
                "Only unary operation are permitted: abs, fabs.");
  // Selectively allow operations commonly used on spectral coefficients
  /* static_assert(not tmpl::list_contains_v<tmpl::list<blaze::Max, blaze::Min
                >, Operator>,
                "This operation is not permitted on a ModalVector."
                "Only unary operation are permitted: abs, fabs."); */
  using Type = ModalVector;
};
}  // namespace blaze

SPECTRE_ALWAYS_INLINE decltype(auto) fabs(const ModalVector& t) noexcept {
  return abs(~t);
}

SPECTRE_ALWAYS_INLINE decltype(auto) abs(const ModalVector& t) noexcept {
  return abs(~t);
}

MAKE_EXPRESSION_VECMATH_OP_ADD_ARRAYS_OF_VEC(ModalVector)
MAKE_EXPRESSION_VECMATH_OP_SUB_ARRAYS_OF_VEC(ModalVector)

/// \cond HIDDEN_SYMBOLS
MAKE_EXPRESSION_VEC_OP_ASSIGNMENT_RESTRICT_TYPE(ModalVector)
/// \endcond

MAKE_EXPRESSION_VEC_OP_MAKE_WITH_VALUE(ModalVector)
