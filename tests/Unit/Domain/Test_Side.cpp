// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "Domain/Side.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Unit/TestingFramework.hpp"

SPECTRE_TEST_CASE("Unit.Domain.Side", "[Domain][Unit]") {
  Side side_lower = Side::Lower;
  CHECK(opposite(side_lower) == Side::Upper);
  CHECK(opposite(opposite(side_lower)) == Side::Lower);
  CHECK(get_output(side_lower) == "Lower");
  CHECK(get_output(Side::Upper) == "Upper");
}
