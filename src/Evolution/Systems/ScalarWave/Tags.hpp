// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for scalar wave system

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

class DataVector;

namespace ScalarWave {
struct Psi : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Psi"; }
};

struct Pi : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Pi"; }
};

template <size_t Dim>
struct Phi : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
  static std::string name() noexcept { return "Phi"; }
};

namespace Tags {
// @{
/// \brief Tags corresponding to the characteristic fields of the flat-spacetime
/// scalar-wave system.
///
/// \details For details on how these are defined and computed, \see
/// CharacteristicSpeedsCompute
struct UPsi : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "UPsi"; }
};
template <size_t Dim, typename Frame = Frame::Inertial>
struct UZero : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame>;
  static std::string name() noexcept { return "UZero"; }
};
struct UPlus : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "UPlus"; }
};
struct UMinus : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "UMinus"; }
};
// @}

template <size_t Dim>
struct CharacteristicSpeeds : db::SimpleTag {
  using type = std::array<DataVector, 4>;
  static std::string name() noexcept { return "CharacteristicSpeeds"; }
};

template <size_t Dim, typename Frame = Frame::Inertial>
struct CharacteristicFields : db::SimpleTag {
  using type = Variables<tmpl::list<UPsi, UZero<Dim, Frame>, UPlus, UMinus>>;
  static std::string name() noexcept { return "CharacteristicFields"; }
};

template <size_t Dim, typename Frame>
struct EvolvedFieldsFromCharacteristicFields : db::SimpleTag {
  using type = Variables<tmpl::list<Psi, Pi, Phi<Dim>>>;
  static std::string name() noexcept {
    return "EvolvedFieldsFromCharacteristicFields";
  }
};
}  // namespace Tags
}  // namespace ScalarWave
