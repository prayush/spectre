// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Orientation.hpp"

#include <ostream>

#include "Domain/SegmentId.hpp"

template <size_t VolumeDim>
Orientation<VolumeDim>::Orientation() {
  for (size_t j = 0; j < VolumeDim; j++) {
    gsl::at(mapped_directions_, j) = Direction<VolumeDim>(j, Side::Upper);
  }
}
template <size_t VolumeDim>
Orientation<VolumeDim>::Orientation(
    std::array<Direction<VolumeDim>, VolumeDim> mapped_directions)
    : mapped_directions_(std::move(mapped_directions)) {
  for (size_t j = 0; j < VolumeDim; j++) {
    if (gsl::at(mapped_directions_, j).dimension() != j or
        gsl::at(mapped_directions_, j).side() != Side::Upper) {
      is_aligned_ = false;
    }
  }
}

template <size_t VolumeDim>
Orientation<VolumeDim>::Orientation(
    const std::array<Direction<VolumeDim>, VolumeDim>& directions_in_host,
    const std::array<Direction<VolumeDim>, VolumeDim>& directions_in_neighbor) {
  for (size_t j = 0; j < VolumeDim; j++) {
    gsl::at(mapped_directions_, gsl::at(directions_in_host, j).dimension()) =
        (gsl::at(directions_in_host, j).side() == Side::Upper
             ? gsl::at(directions_in_neighbor, j)
             : gsl::at(directions_in_neighbor, j).opposite());
    if (gsl::at(directions_in_host, j) != gsl::at(directions_in_neighbor, j)) {
      is_aligned_ = false;
    }
  }
}

template <size_t VolumeDim>
std::array<SegmentId, VolumeDim> Orientation<VolumeDim>::mapped(
    const std::array<SegmentId, VolumeDim>& segmentIds) const {
  std::array<SegmentId, VolumeDim> result = segmentIds;
  for (size_t d = 0; d < VolumeDim; d++) {
    gsl::at(result, gsl::at(mapped_directions_, d).dimension()) =
        gsl::at(mapped_directions_, d).side() == Side::Upper
            ? gsl::at(segmentIds, d)
            : gsl::at(segmentIds, d).id_if_flipped();
  }
  return result;
}

template <size_t VolumeDim>
void Orientation<VolumeDim>::pup(PUP::er& p) {
  p | mapped_directions_;
  p | is_aligned_;
}

template <>
std::ostream& operator<<(std::ostream& os, const Orientation<1>& orientation) {
  os << "(" << orientation.mapped(Direction<1>::upper_xi()) << ")";
  return os;
}

template <>
std::ostream& operator<<(std::ostream& os, const Orientation<2>& orientation) {
  os << "(" << orientation.mapped(Direction<2>::upper_xi()) << ", "
     << orientation.mapped(Direction<2>::upper_eta()) << ")";
  return os;
}

template <>
std::ostream& operator<<(std::ostream& os, const Orientation<3>& orientation) {
  os << "(" << orientation.mapped(Direction<3>::upper_xi()) << ", "
     << orientation.mapped(Direction<3>::upper_eta()) << ", "
     << orientation.mapped(Direction<3>::upper_zeta()) << ")";
  return os;
}

template class Orientation<1>;
template class Orientation<2>;
template class Orientation<3>;

