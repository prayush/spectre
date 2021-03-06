# Distributed under the MIT License.
# See LICENSE.txt for details.

if(NOT CLANG_FORMAT_ROOT)
  # Need to set to empty to avoid warnings with --warn-uninitialized
  set(CLANG_FORMAT_ROOT "")
  set(CLANG_FORMAT_ROOT $ENV{CLANG_FORMAT_ROOT})
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  string(
      REGEX MATCH "^[0-9]+.[0-9]+" LLVM_VERSION
      "${CMAKE_CXX_COMPILER_VERSION}"
  )
  find_program(
      CLANG_FORMAT_BIN
      NAMES "clang-format-${LLVM_VERSION}" "clang-format"
      HINTS ${CLANG_FORMAT_ROOT}
  )
else()
  find_program(
      CLANG_FORMAT_BIN
      NAMES
      "clang-format-4.0"
      "clang-format-3.9"
      "clang-format-3.8"
      "clang-format"
  )
endif()

if(CLANG_FORMAT_BIN AND EXISTS ${CMAKE_SOURCE_DIR}/.git AND Git_FOUND)
  set(CLANG_FORMAT_FOUND 1)
  get_filename_component(CLANG_FORMAT_NAME ${CLANG_FORMAT_BIN} NAME)
  add_custom_target(
      git-clang-format
      COMMAND
      export PATH=${CMAKE_SOURCE_DIR}/tools:$$PATH
      && cd ${CMAKE_SOURCE_DIR}
      && ${GIT_EXECUTABLE} ${CLANG_FORMAT_NAME} -f
  )
else()
  set(CLANG_FORMAT_FOUND 0)
endif()
