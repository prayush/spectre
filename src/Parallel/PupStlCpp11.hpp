// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// PUP routines for new C+11 STL containers

#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <pup_stl.h>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "Utilities/Requires.hpp"

namespace PUP {

// @{
/// \ingroup Parallel
/// Serialization of std::array for Charm++
template <typename T, std::size_t N,
          Requires<not std::is_arithmetic<T>::value> = nullptr>
inline void pup(PUP::er& p, std::array<T, N>& a) {  // NOLINT
  std::for_each(a.begin(), a.end(), [&p](auto& t) { p | t; });
}

template <typename T, std::size_t N,
          Requires<std::is_arithmetic<T>::value> = nullptr>
inline void pup(PUP::er& p, std::array<T, N>& a) {  // NOLINT
  PUParray(p, a.data(), N);
}
// @}

/// \ingroup Parallel
/// Serialization of std::array for Charm++
template <typename T, std::size_t N>
inline void operator|(er& p, std::array<T, N>& a) {  // NOLINT
  pup(p, a);
}

/// \ingroup Parallel
/// Serialization of std::unordered_map for Charm++
/// \warning This does not work with custom hash functions that have state
template <typename K, typename V, typename H>
inline void pup(PUP::er& p, std::unordered_map<K, V, H>& m) {  // NOLINT
  size_t number_elem = PUP_stl_container_size(p, m);

  if (p.isUnpacking()) {
    for (size_t i = 0; i < number_elem; ++i) {
      std::pair<K, V> kv;
      p | kv;
      m.emplace(std::move(kv));
    }
  } else {
    for (auto& kv : m) {
      p | kv;
    }
  }
}

/// \ingroup Parallel
/// Serialization of std::unordered_map for Charm++
/// \warning This does not work with custom hash functions that have state
template <typename K, typename V, typename H>
inline void operator|(er& p, std::unordered_map<K, V, H>& m) {  // NOLINT
  pup(p, m);
}

/// \ingroup Parallel
/// Serialization of std::unordered_set for Charm++
template <typename T>
inline void pup(PUP::er& p, std::unordered_set<T>& s) {  // NOLINT
  size_t number_elem = PUP_stl_container_size(p, s);

  if (p.isUnpacking()) {
    for (size_t i = 0; i < number_elem; ++i) {
      T element;
      p | element;
      s.emplace(std::move(element));
    }
  } else {
    // This intenionally is not a reference because at least with stdlibc++ the
    // reference code does not compile because it turns the dereferenced
    // iterator into a value
    for (T e : s) {
      p | e;
    }
  }
}

/// \ingroup Parallel
/// Serialization of std::unordered_set for Charm++
template <class T>
inline void operator|(er& p, std::unordered_set<T>& s) {  // NOLINT
  pup(p, s);
}

/// \ingroup Parallel
/// Serialization of enum for Charm++
///
/// \note This requires a change to Charm++ to work
template <typename T, Requires<std::is_enum<T>::value> = nullptr>
inline void operator|(PUP::er& p, T& s) {  // NOLINT
  pup_bytes(&p, static_cast<void*>(&s), sizeof(T));
}

template <size_t N = 0, typename... Args,
          Requires<0 == sizeof...(Args)> = nullptr>
void pup_tuple_impl(PUP::er& /* p */, std::tuple<Args...>& /* t */) {  // NOLINT
}

template <size_t N = 0, typename... Args,
          Requires<(0 < sizeof...(Args) and 0 == N)> = nullptr>
void pup_tuple_impl(PUP::er& p, std::tuple<Args...>& t) {  // NOLINT
  p | std::get<N>(t);
}

template <size_t N, typename... Args,
          Requires<(sizeof...(Args) > 0 and N > 0)> = nullptr>
void pup_tuple_impl(PUP::er& p, std::tuple<Args...>& t) {  // NOLINT
  p | std::get<N>(t);
  pup_tuple_impl<N - 1>(p, t);
}

/// \ingroup Parallel
/// Serialization of std::tuple for Charm++
template <typename... Args>
inline void pup(PUP::er& p, std::tuple<Args...>& t) {  // NOLINT
  if (p.isUnpacking()) {
    t = std::tuple<Args...>{};
  }
  pup_tuple_impl<sizeof...(Args) - 1>(p, t);
}

/// \ingroup Parallel
/// Serialization of std::tuple for Charm++
template <typename... Args>
inline void operator|(PUP::er& p, std::tuple<Args...>& t) {  // NOLINT
  pup(p, t);
}

// @{
/// \ingroup Parallel
/// Serialization of a unique_ptr for Charm++
template <typename T,
          Requires<not std::is_base_of<PUP::able, T>::value> = nullptr>
inline void pup(PUP::er& p, std::unique_ptr<T>& t) {  // NOLINT
  bool is_nullptr = nullptr == t;
  p | is_nullptr;
  if (not is_nullptr) {
    T* t1;
    if (p.isUnpacking()) {
      t1 = new T;
    } else {
      t1 = t.get();
    }
    p | *t1;
    if (p.isUnpacking()) {
      t.reset(t1);
    }
  }
}

template <typename T, Requires<std::is_base_of<PUP::able, T>::value> = nullptr>
inline void pup(PUP::er& p, std::unique_ptr<T>& t) {  // NOLINT
  T* t1 = nullptr;
  if (p.isUnpacking()) {
    p | t1;
    t = std::unique_ptr<T>(t1);
  } else {
    t1 = t.get();
    p | t1;
  }
}
// @}

/// \ingroup Parallel
/// Serialization of a unique_ptr for Charm++
template <typename T>
inline void operator|(PUP::er& p, std::unique_ptr<T>& t) {  // NOLINT
  pup(p, t);
}
}  // namespace PUP
