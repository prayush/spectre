// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/ElementId.hpp"

#include <boost/functional/hash.hpp>
#include <limits>

#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdHelpers.hpp"

template <size_t VolumeDim>
ElementId<VolumeDim>::ElementId(const size_t block_id)
    : block_id_{block_id},
      segment_ids_(make_array<VolumeDim>(SegmentId(0, 0))) {}

template <size_t VolumeDim>
ElementId<VolumeDim>::ElementId(const size_t block_id,
                                std::array<SegmentId, VolumeDim> segment_ids)
    : block_id_{block_id}, segment_ids_(std::move(segment_ids)) {}

template <size_t VolumeDim>
ElementId<VolumeDim> ElementId<VolumeDim>::id_of_child(const size_t dim,
                                                       const Side side) const {
  std::array<SegmentId, VolumeDim> new_segment_ids = segment_ids_;
  gsl::at(new_segment_ids, dim) =
      gsl::at(new_segment_ids, dim).id_of_child(side);
  return ElementId<VolumeDim>(block_id_, new_segment_ids);
}

template <size_t VolumeDim>
ElementId<VolumeDim> ElementId<VolumeDim>::id_of_parent(
    const size_t dim) const {
  std::array<SegmentId, VolumeDim> new_segment_ids = segment_ids_;
  gsl::at(new_segment_ids, dim) = gsl::at(new_segment_ids, dim).id_of_parent();
  return ElementId<VolumeDim>(block_id_, new_segment_ids);
}

template <size_t VolumeDim>
void ElementId<VolumeDim>::pup(PUP::er& p) {
  p | block_id_;
  p | segment_ids_;
}

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os, const ElementId<VolumeDim>& id) {
  os << "[B" << id.block_id() << ',' << id.segment_ids() << ']';
  return os;
}

// LCOV_EXCL_START
namespace std {
template <size_t VolumeDim>
size_t hash<ElementId<VolumeDim>>::operator()(
    const ElementId<VolumeDim>& c) const {
  size_t h = 0;
  boost::hash_combine(h, c.block_id());
  boost::hash_combine(h, c.segment_ids());
  return h;
}
}  // namespace std
// LCOV_EXCL_STOP

template class ElementId<1>;
template class ElementId<2>;
template class ElementId<3>;

template std::ostream& operator<<(std::ostream&, const ElementId<1>&);
template std::ostream& operator<<(std::ostream&, const ElementId<2>&);
template std::ostream& operator<<(std::ostream&, const ElementId<3>&);

namespace std {
template struct hash<ElementId<1>>;
template struct hash<ElementId<2>>;
template struct hash<ElementId<3>>;
}  // namespace std
