// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/Header.hpp"

#include <chrono>
#include <regex>
#include <sstream>

#include "IO/H5/CheckH5.hpp"
#include "IO/H5/Helpers.hpp"
#include "Informer/InfoFromBuild.hpp"

namespace h5 {
/// \cond HIDDEN_SYMOLS
Header::Header(const bool exists, detail::OpenGroup&& group,
               const hid_t location, const std::string& name)
    : group_(std::move(group)) {
  if (exists) {
    header_info_ = h5::detail::read_strings_from_attribute(
        location, name + extension())[0];
  } else {
    std::vector<std::string> header_info{[]() {
      std::stringstream ss;
      std::time_t current_time = std::chrono::system_clock::to_time_t(
          std::chrono::system_clock::now());
      ss << "#\n# File created on " << std::ctime(&current_time) << "# ";
      auto build_info = info_from_build();
      ss << std::regex_replace(build_info, std::regex{"\n"}, "\n# ");
      return ss.str();
    }()};
    detail::write_strings_to_attribute(location, name + extension(),
                                       header_info);
    header_info_ = header_info[0];
  }
}
/// \endcond
}  // namespace h5
