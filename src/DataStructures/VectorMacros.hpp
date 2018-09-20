// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Common code for classes DataVector and ModalVector

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>  // for std::reference_wrapper
#include <initializer_list>
#include <limits>
#include <ostream>
#include <type_traits>
#include <vector>

#include "ErrorHandling/Assert.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/PointerVector.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/StdHelpers.hpp"

/// \cond HIDDEN_SYMBOLS
// IWYU pragma: no_forward_declare ConstantExpressions_detail::pow
namespace PUP {
class er;
}  // namespace PUP

// clang-tidy: no using declarations in header files
//             We want the std::abs to be used
using std::abs;  // NOLINT
/// \endcond

// IWYU doesn't like that we want PointerVector.hpp to expose Blaze and also
// have DataVector.hpp to expose PointerVector.hpp without including Blaze
// directly in DataVector.hpp
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

// IWYU pragma: no_forward_declare blaze::DenseVector
// IWYU pragma: no_forward_declare blaze::UnaryMapTrait
// IWYU pragma: no_forward_declare blaze::BinaryMapTrait
// IWYU pragma: no_forward_declare blaze::IsVector
// IWYU pragma: no_forward_declare blaze::TransposeFlag

/**
 * \ingroup DataStructuresGroup
 * \brief Here be code common to container classes DataVector and ModalVector.
 *
 * \details DataVector is intended to contain function values on the
 * computational domain. It holds an array of contiguous data and supports a
 * variety of mathematical operations that are applicable to nodal coefficients.
 * ModalVector, on the other hand, is intended to contain values of spectral
 * coefficients for any quantity expanded in its respective bases. It also
 * holds an array of contiguous data, but allows only limited math operations,
 * i.e. only those that are applicable to spectral coefficients, such as
 * elementwise addition, subtraction, multiplication, division and a few more.
 *
 * Therefore, both classes have significantly common structure and properties.
 * This file contains code (in the form of macros) that is used to build both
 * {Data,Modal}Vector classes.
 *
 * A VECTYPE class (below) holds an array of contiguous data. VECTYPE can be
 * owning, meaning the array is deleted when the VECTYPE goes out of scope, or
 * non-owning, meaning it just has a pointer to an array.
 */
#define MAKE_EXPRESSION_DATA_MODAL_VECTOR_CLASSES(VECTYPE)                     \
class VECTYPE                                                                  \
    : public PointerVector<double, blaze::unaligned, blaze::unpadded,          \
                           blaze::defaultTransposeFlag, VECTYPE> {             \
  /** \cond HIDDEN_SYMBOLS */                                                  \
  static constexpr void private_asserts() noexcept {                           \
    static_assert(std::is_nothrow_move_constructible<VECTYPE>::value,          \
                  "Missing move semantics");                                   \
  }                                                                            \
  /** \endcond */                                                              \
 public:                                                                       \
  using value_type = double;                                                   \
  using allocator_type = std::allocator<value_type>;                           \
  using size_type = size_t;                                                    \
  using difference_type = std::ptrdiff_t;                                      \
  using BaseType = PointerVector<double, blaze::unaligned, blaze::unpadded,    \
                                 blaze::defaultTransposeFlag, VECTYPE>;        \
  static constexpr bool transpose_flag = blaze::defaultTransposeFlag;          \
                                                                               \
  using BaseType::ElementType;                                                 \
  using TransposeType = VECTYPE;                                               \
  using CompositeType = const VECTYPE&;                                        \
                                                                               \
  using BaseType::operator[];                                                  \
  using BaseType::begin;                                                       \
  using BaseType::cbegin;                                                      \
  using BaseType::cend;                                                        \
  using BaseType::data;                                                        \
  using BaseType::end;                                                         \
  using BaseType::size;                                                        \
                                                                               \
  /** @{ */                                                                    \
  /* Upcast to `BaseType` */                                                   \
  const BaseType& operator~() const noexcept {                                 \
    return static_cast<const BaseType&>(*this);                                \
  }                                                                            \
  BaseType& operator~() noexcept { return static_cast<BaseType&>(*this); }     \
  /** @} */                                                                    \
                                                                               \
  /** Create with the given size and value. */                                 \
  /** */                                                                       \
  /** \param size number of values */                                          \
  /** \param value the value to initialize each element. */                    \
  explicit VECTYPE(                                                            \
      size_t size,                                                             \
      double value = std::numeric_limits<double>::signaling_NaN()) noexcept;   \
                                                                               \
  /** Create a non-owning VECTYPE that points to `start` */                    \
  VECTYPE(double* start, size_t size) noexcept;                                \
                                                                               \
  /** Create from an initializer list of doubles. All elements in the */       \
  /** `std::initializer_list` must have decimal points */                      \
  template <class T, Requires<cpp17::is_same_v<T, double>> = nullptr>          \
  VECTYPE(std::initializer_list<T> list) noexcept;                             \
                                                                               \
  /** Empty VECTYPE */                                                         \
  VECTYPE() noexcept = default;                                                \
  /** \cond HIDDEN_SYMBOLS */                                                  \
  ~VECTYPE() = default;                                                        \
                                                                               \
  VECTYPE(const VECTYPE& rhs);                                                 \
  VECTYPE(VECTYPE&& rhs) noexcept;                                             \
  VECTYPE& operator=(const VECTYPE& rhs);                                      \
  VECTYPE& operator=(VECTYPE&& rhs) noexcept;                                  \
                                                                               \
  /* This is a converting constructor. clang-tidy complains that it's not */   \
  /* explicit, but we want it to allow conversion.                        */   \
  /* clang-tidy: mark as explicit (we want conversion to VECTYPE)      */      \
  template <typename VT, bool VF>                                              \
  VECTYPE(const blaze::DenseVector<VT, VF>& expression) noexcept; /* NOLINT */ \
                                                                               \
  template <typename VT, bool VF>                                              \
  VECTYPE& operator=(const blaze::DenseVector<VT, VF>& expression) noexcept;   \
  /** \endcond */                                                              \
                                                                               \
  MAKE_EXPRESSION_MATH_ASSIGN_PV(+=, VECTYPE)                                  \
  MAKE_EXPRESSION_MATH_ASSIGN_PV(-=, VECTYPE)                                  \
  MAKE_EXPRESSION_MATH_ASSIGN_PV(*=, VECTYPE)                                  \
  MAKE_EXPRESSION_MATH_ASSIGN_PV(/=, VECTYPE)                                  \
                                                                               \
  VECTYPE& operator=(const double& rhs) noexcept {                             \
    ~*this = rhs;                                                              \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  /** @{ */                                                                    \
  /** Set the VECTYPE to be a reference to another VECTYPE object */           \
  void set_data_ref(gsl::not_null<VECTYPE*> rhs) noexcept {                    \
    set_data_ref(rhs->data(), rhs->size());                                    \
  }                                                                            \
  void set_data_ref(double* start, size_t size) noexcept {                     \
    owned_data_ = decltype(owned_data_){};                                     \
    (~*this).reset(start, size);                                               \
    owning_ = false;                                                           \
  }                                                                            \
  /** @} */                                                                    \
                                                                               \
  /** Returns true if the class owns the data */                               \
  bool is_owning() const noexcept { return owning_; }                          \
                                                                               \
  /** Serialization for Charm++ */                                             \
  /* clang-tidy: google-runtime-references */                                  \
  void pup(PUP::er& p) noexcept;  /* NOLINT */                                 \
                                                                               \
 private:                                                                      \
  SPECTRE_ALWAYS_INLINE void reset_pointer_vector() noexcept {                 \
    reset(owned_data_.data(), owned_data_.size());                             \
  }                                                                            \
                                                                               \
  /** \cond HIDDEN_SYMBOLS */                                                  \
  std::vector<double, allocator_type> owned_data_;                             \
  bool owning_{true};                                                          \
  /** \endcond */                                                              \
};


/**
 * Declare left-shift, equivalence, and inequivalence operations for VECTYPE
 * with itself
 */
#define MAKE_EXPRESSION_VECMATH_OP_COMP_SELF(VECTYPE)             \
/**Output operator for VECTYPE */                                 \
std::ostream& operator<<(std::ostream& os, const VECTYPE& d);     \
                                                                  \
/** Equivalence operator for VECTYPE */                           \
bool operator==(const VECTYPE& lhs, const VECTYPE& rhs) noexcept; \
                                                                  \
/** Inequivalence operator for VECTYPE */                         \
bool operator!=(const VECTYPE& lhs, const VECTYPE& rhs) noexcept;


/**
 * Define equivalence, and inequivalence operations for VECTYPE
 * with blaze::DenseVector<VT, VF>
 */
/// \cond
#define MAKE_EXPRESSION_VECMATH_OP_COMP_DV(VECTYPE)               \
/* Used for comparing VECTYPE to an expression */                 \
template <typename VT, bool VF>                                   \
bool operator==(const VECTYPE& lhs,                               \
                const blaze::DenseVector<VT, VF>& rhs) noexcept { \
  return lhs == VECTYPE(rhs);                                     \
}                                                                 \
                                                                  \
template <typename VT, bool VF>                                   \
bool operator!=(const VECTYPE& lhs,                               \
                const blaze::DenseVector<VT, VF>& rhs) noexcept { \
  return not(lhs == rhs);                                         \
}                                                                 \
                                                                  \
template <typename VT, bool VF>                                   \
bool operator==(const blaze::DenseVector<VT, VF>& lhs,            \
                const VECTYPE& rhs) noexcept {                    \
  return VECTYPE(lhs) == rhs;                                     \
}                                                                 \
                                                                  \
template <typename VT, bool VF>                                   \
bool operator!=(const blaze::DenseVector<VT, VF>& lhs,            \
                const VECTYPE& rhs) noexcept {                    \
  return not(lhs == rhs);                                         \
}
/// \endcond

/**
 * Specialize the Blaze type traits (Add,Sub,Mult,Div) to handle VECTYPE
 * correctly.
 */
#define MAKE_EXPRESSION_VECMATH_SPECIALIZE_BLAZE_ARITHMETIC_TRAITS(TYPE) \
namespace blaze {                                                        \
template <>                                                              \
struct IsVector<TYPE> : std::true_type {};                               \
                                                                         \
template <>                                                              \
struct TransposeFlag<TYPE> : BoolConstant<                               \
                TYPE::transpose_flag> {};                                \
                                                                         \
template <>                                                              \
struct AddTrait<TYPE, TYPE> {                                            \
  using Type = TYPE;                                                     \
};                                                                       \
                                                                         \
template <>                                                              \
struct AddTrait<TYPE, double> {                                          \
  using Type = TYPE;                                                     \
};                                                                       \
                                                                         \
template <>                                                              \
struct AddTrait<double, TYPE> {                                          \
  using Type = TYPE;                                                     \
};                                                                       \
                                                                         \
template <>                                                              \
struct SubTrait<TYPE, TYPE> {                                            \
  using Type = TYPE;                                                     \
};                                                                       \
                                                                         \
template <>                                                              \
struct SubTrait<TYPE, double> {                                          \
  using Type = TYPE;                                                     \
};                                                                       \
                                                                         \
template <>                                                              \
struct SubTrait<double, TYPE> {                                          \
  using Type = TYPE;                                                     \
};                                                                       \
                                                                         \
template <>                                                              \
struct MultTrait<TYPE, TYPE> {                                           \
  using Type = TYPE;                                                     \
};                                                                       \
                                                                         \
template <>                                                              \
struct MultTrait<TYPE, double> {                                         \
  using Type = TYPE;                                                     \
};                                                                       \
                                                                         \
template <>                                                              \
struct MultTrait<double, TYPE> {                                         \
  using Type = TYPE;                                                     \
};                                                                       \
                                                                         \
template <>                                                              \
struct DivTrait<TYPE, TYPE> {                                            \
  using Type = TYPE;                                                     \
};                                                                       \
                                                                         \
template <>                                                              \
struct DivTrait<TYPE, double> {                                          \
  using Type = TYPE;                                                     \
};                                                                       \
} /* namespace blaze*/


/**
 * Specialize the Blaze Map traits to correctly handle VECTYPE
 */
#define MAKE_EXPRESSION_VECMATH_SPECIALIZE_BLAZE_MAP_TRAITS(VECTYPE) \
namespace blaze {                                                    \
template <typename Operator>                                         \
struct UnaryMapTrait<VECTYPE, Operator> {                            \
  using Type = VECTYPE;                                              \
};                                                                   \
                                                                     \
template <typename Operator>                                         \
struct BinaryMapTrait<VECTYPE, VECTYPE, Operator> {                  \
  using Type = VECTYPE;                                              \
};                                                                   \
}  /* namespace blaze */


/**
 * Define + and += operations for std::arrays of VECTYPE's
 */
#define MAKE_EXPRESSION_VECMATH_OP_ADD_ARRAYS_OF_VEC(VECTYPE)                \
template <typename T, size_t Dim>                                            \
std::array<VECTYPE, Dim> operator+(                                          \
    const std::array<T, Dim>& lhs,                                           \
    const std::array<VECTYPE, Dim>& rhs) noexcept {                          \
  std::array<VECTYPE, Dim> result;                                           \
  for (size_t i = 0; i < Dim; i++) {                                         \
    gsl::at(result, i) = gsl::at(lhs, i) + gsl::at(rhs, i);                  \
  }                                                                          \
  return result;                                                             \
}                                                                            \
template <typename U, size_t Dim>                                            \
std::array<VECTYPE, Dim> operator+(const std::array<VECTYPE, Dim>& lhs,      \
                                   const std::array<U, Dim>& rhs) noexcept { \
  return rhs + lhs;                                                          \
}                                                                            \
template <size_t Dim>                                                        \
std::array<VECTYPE, Dim> operator+(                                          \
    const std::array<VECTYPE, Dim>& lhs,                                     \
    const std::array<VECTYPE, Dim>& rhs) noexcept {                          \
  std::array<VECTYPE, Dim> result;                                           \
  for (size_t i = 0; i < Dim; i++) {                                         \
    gsl::at(result, i) = gsl::at(lhs, i) + gsl::at(rhs, i);                  \
  }                                                                          \
  return result;                                                             \
}                                                                            \
template <size_t Dim>                                                        \
std::array<VECTYPE, Dim>& operator+=(                                        \
    std::array<VECTYPE, Dim>& lhs,                                           \
    const std::array<VECTYPE, Dim>& rhs) noexcept {                          \
  for (size_t i = 0; i < Dim; i++) {                                         \
    gsl::at(lhs, i) += gsl::at(rhs, i);                                      \
  }                                                                          \
  return lhs;                                                                \
}

/**
 * Define - and -= operations for std::arrays of VECTYPE's
 */
#define MAKE_EXPRESSION_VECMATH_OP_SUB_ARRAYS_OF_VEC(VECTYPE)                \
template <typename T, size_t Dim>                                            \
std::array<VECTYPE, Dim> operator-(                                          \
    const std::array<T, Dim>& lhs,                                           \
    const std::array<VECTYPE, Dim>& rhs) noexcept {                          \
  std::array<VECTYPE, Dim> result;                                           \
  for (size_t i = 0; i < Dim; i++) {                                         \
    gsl::at(result, i) = gsl::at(lhs, i) - gsl::at(rhs, i);                  \
  }                                                                          \
  return result;                                                             \
}                                                                            \
template <typename U, size_t Dim>                                            \
std::array<VECTYPE, Dim> operator-(const std::array<VECTYPE, Dim>& lhs,      \
                                   const std::array<U, Dim>& rhs) noexcept { \
  std::array<VECTYPE, Dim> result;                                           \
  for (size_t i = 0; i < Dim; i++) {                                         \
    gsl::at(result, i) = gsl::at(lhs, i) - gsl::at(rhs, i);                  \
  }                                                                          \
  return result;                                                             \
}                                                                            \
template <size_t Dim>                                                        \
std::array<VECTYPE, Dim> operator-(                                          \
    const std::array<VECTYPE, Dim>& lhs,                                     \
    const std::array<VECTYPE, Dim>& rhs) noexcept {                          \
  std::array<VECTYPE, Dim> result;                                           \
  for (size_t i = 0; i < Dim; i++) {                                         \
    gsl::at(result, i) = gsl::at(lhs, i) - gsl::at(rhs, i);                  \
  }                                                                          \
  return result;                                                             \
}                                                                            \
template <size_t Dim>                                                        \
std::array<VECTYPE, Dim>& operator-=(                                        \
    std::array<VECTYPE, Dim>& lhs,                                           \
    const std::array<VECTYPE, Dim>& rhs) noexcept {                          \
  for (size_t i = 0; i < Dim; i++) {                                         \
    gsl::at(lhs, i) -= gsl::at(rhs, i);                                      \
  }                                                                          \
  return lhs;                                                                \
}


/**
 * Forbid assignment of blaze::DenseVector<VT,VF>'s to VECTYPE, if the result
 * type VT::ResultType is not VECTYPE
 */
#define MAKE_EXPRESSION_VEC_OP_ASSIGNMENT_RESTRICT_TYPE(VECTYPE)               \
template <typename VT, bool VF>                                                \
VECTYPE::VECTYPE(const blaze::DenseVector<VT, VF>& expression) noexcept        \
    : owned_data_((~expression).size()) {                                      \
  static_assert(cpp17::is_same_v<typename VT::ResultType, VECTYPE>,            \
              "You are attempting to assign the result of an expression that " \
              "is not a " #VECTYPE " to a " #VECTYPE ".");                     \
  reset_pointer_vector();                                                      \
  ~*this = expression;                                                         \
}                                                                              \
                                                                               \
template <typename VT, bool VF>                                                \
VECTYPE& VECTYPE::operator=(                                                   \
    const blaze::DenseVector<VT, VF>& expression) noexcept {                   \
  static_assert(cpp17::is_same_v<typename VT::ResultType, VECTYPE>,            \
              "You are attempting to assign the result of an expression that " \
              "is not a " #VECTYPE " to a " #VECTYPE ".");                     \
  if (owning_ and (~expression).size() != size()) {                            \
    owned_data_.resize((~expression).size());                                  \
    reset_pointer_vector();                                                    \
  } else if (not owning_) {                                                    \
    ASSERT((~expression).size() == size(), "Must copy into same size, not "    \
                                               << (~expression).size()         \
                                               << " into " << size());         \
  }                                                                            \
  ~*this = expression;                                                         \
  return *this;                                                                \
}


#define MAKE_EXPRESSION_VEC_OP_MAKE_WITH_VALUE(VECTYPE)                     \
namespace MakeWithValueImpls {                                              \
/** \brief Returns a VECTYPE the same size as `input`, with each element */ \
/** equal to `value`. */                                                    \
template <>                                                                 \
SPECTRE_ALWAYS_INLINE VECTYPE                                               \
MakeWithValueImpl<VECTYPE, VECTYPE>::apply(const VECTYPE& input,            \
                                           const double value) {            \
  return VECTYPE(input.size(), value);                                      \
}                                                                           \
}  /* namespace MakeWithValueImpls*/



/**                 Function definitions               */

/**
 * Construct VECTYPE with value(s)
 */
#define MAKE_EXPRESSION_VEC_DEF_CONSTRUCT_WITH_VALUE(VECTYPE)    \
VECTYPE::VECTYPE(const size_t size, const double value) noexcept \
    : owned_data_(size, value) {                                 \
  reset_pointer_vector();                                        \
}                                                                \
                                                                 \
VECTYPE::VECTYPE(double* start, size_t size) noexcept            \
    : BaseType(start, size), owned_data_(0), owning_(false) {}   \
                                                                 \
template <class T, Requires<cpp17::is_same_v<T, double>>>        \
VECTYPE::VECTYPE(std::initializer_list<T> list) noexcept         \
    : owned_data_(std::move(list)) {                             \
  reset_pointer_vector();                                        \
}

/**
 * Construct / Assign VECTYPE with / to VECTYPE reference or rvalue
 */
// clang-tidy: calling a base constructor other than the copy constructor.
//             We reset the base class in reset_pointer_vector after calling its
//             default constructor
#define MAKE_EXPRESSION_VEC_DEF_CONSTRUCT_WITH_VEC(VECTYPE)                  \
VECTYPE::VECTYPE(const VECTYPE& rhs) : BaseType{} {  /** NOLINT */           \
  if (rhs.is_owning()) {                                                     \
    owned_data_ = rhs.owned_data_;                                           \
  } else {                                                                   \
    owned_data_.assign(rhs.begin(), rhs.end());                              \
  }                                                                          \
  reset_pointer_vector();                                                    \
}                                                                            \
                                                                             \
VECTYPE& VECTYPE::operator=(const VECTYPE& rhs) {                            \
  if (this != &rhs) {                                                        \
    if (owning_) {                                                           \
      if (rhs.is_owning()) {                                                 \
        owned_data_ = rhs.owned_data_;                                       \
      } else {                                                               \
        owned_data_.assign(rhs.begin(), rhs.end());                          \
      }                                                                      \
      reset_pointer_vector();                                                \
    } else {                                                                 \
      ASSERT(rhs.size() == size(), "Must copy into same size, not "          \
                                       << rhs.size() << " into " << size()); \
      std::copy(rhs.begin(), rhs.end(), begin());                            \
    }                                                                        \
  }                                                                          \
  return *this;                                                              \
}                                                                            \
/** NOLINTNEXTLINE(misc-macro-parentheses) */                                \
VECTYPE::VECTYPE(VECTYPE&& rhs) noexcept {                                   \
  owned_data_ = std::move(rhs.owned_data_);                                  \
  ~*this = ~rhs;  /* PointerVector is trivially copyable */                  \
  owning_ = rhs.owning_;                                                     \
                                                                             \
  rhs.owning_ = true;                                                        \
  rhs.reset();                                                               \
}                                                                            \
/** NOLINTNEXTLINE(misc-macro-parentheses) */                                \
VECTYPE& VECTYPE::operator=(VECTYPE&& rhs) noexcept {                        \
  if (this != &rhs) {                                                        \
    if (owning_) {                                                           \
      owned_data_ = std::move(rhs.owned_data_);                              \
      ~*this = ~rhs;  /* PointerVector is trivially copyable */              \
      owning_ = rhs.owning_;                                                 \
    } else {                                                                 \
      ASSERT(rhs.size() == size(), "Must copy into same size, not "          \
                                       << rhs.size() << " into " << size()); \
      std::copy(rhs.begin(), rhs.end(), begin());                            \
    }                                                                        \
    rhs.owning_ = true;                                                      \
    rhs.reset();                                                             \
  }                                                                          \
  return *this;                                                              \
}


/**
 * Charm++ packing / unpacking of object
 */
#define MAKE_EXPRESSION_VEC_OP_PUP_CHARM(VECTYPE)       \
void VECTYPE::pup(PUP::er& p) noexcept {  /** NOLINT */ \
  auto my_size = size();                                \
  p | my_size;                                          \
  if (my_size > 0) {                                    \
    if (p.isUnpacking()) {                              \
      owning_ = true;                                   \
      owned_data_.resize(my_size);                      \
      reset_pointer_vector();                           \
    }                                                   \
    PUParray(p, data(), size());                        \
  }                                                     \
}


/**
 * Define left-shift, equivalence, and inequivalence operations for VECTYPE
 * with itself
 */
#define MAKE_EXPRESSION_VECMATH_OP_DEF_COMP_SELF(VECTYPE)           \
/** Left-shift operator for VECTYPE */                              \
std::ostream& operator<<(std::ostream& os, const VECTYPE& d) {      \
  /* This function is inside the detail namespace StdHelpers.hpp */ \
  StdHelpers_detail::print_helper(os, d.begin(), d.end());          \
  return os;                                                        \
}                                                                   \
                                                                    \
/** Equivalence operator for VECTYPE */                             \
bool operator==(const VECTYPE& lhs, const VECTYPE& rhs) noexcept {  \
  return lhs.size() == rhs.size() and                               \
         std::equal(lhs.begin(), lhs.end(), rhs.begin());           \
}                                                                   \
                                                                    \
/** Inequivalence operator for VECTYPE */                           \
bool operator!=(const VECTYPE& lhs, const VECTYPE& rhs) noexcept {  \
  return not(lhs == rhs);                                           \
}
