// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function pypp::call<R,Args...>

#pragma once

#include <Python.h>
#include <boost/range/combine.hpp>
#include <stdexcept>
#include <string>

#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/Pypp/PyppFundamentals.hpp"

/// \ingroup TestingFrameworkGroup
/// Contains all functions for calling python from C++
namespace pypp {
namespace detail {

template <typename R, typename = std::nullptr_t>
struct CallImpl {
  template <typename... Args>
  static R call(const std::string& module_name,
                const std::string& function_name, const Args&... t) {
    PyObject* module = PyImport_ImportModule(module_name.c_str());
    if (module == nullptr) {
      PyErr_Print();
      throw std::runtime_error{std::string("Could not find python module.\n") +
          module_name};
    }
    PyObject *func = PyObject_GetAttrString(module, function_name.c_str());
    if (func == nullptr or not PyCallable_Check(func)) {
      if (PyErr_Occurred()) {
        PyErr_Print();
      }
      throw std::runtime_error{"Could not find python function in module.\n"};
    }
    PyObject *args = pypp::make_py_tuple(t...);
    PyObject *value = PyObject_CallObject(func, args);
    Py_DECREF(args);  // NOLINT
    if (value == nullptr) {
      Py_DECREF(func);    // NOLINT
      Py_DECREF(module);  // NOLINT
      PyErr_Print();
      throw std::runtime_error{"Function returned null"};
    }
    auto ret = from_py_object<R>(value);
    Py_DECREF(value);   // NOLINT
    Py_DECREF(func);    // NOLINT
    Py_DECREF(module);  // NOLINT
    return ret;
  }
};

template <typename R>
struct CallImpl<R, Requires<tt::is_a_v<Tensor, R> and
                            cpp17::is_same_v<typename R::type, DataVector>>> {
  template <typename... Args>
  static R call(const std::string& module_name,
                const std::string& function_name, const Args&... t) {
    static_assert(sizeof...(Args) > 0,
                  "Call to python which returns a Tensor of DataVectors must "
                  "pass at least one Tensor of DataVectors.");

    PyObject* module = PyImport_ImportModule(module_name.c_str());
    if (module == nullptr) {
      PyErr_Print();
      throw std::runtime_error{std::string("Could not find python module.\n") +
                               module_name};
    }
    PyObject* func = PyObject_GetAttrString(module, function_name.c_str());
    if (func == nullptr or not PyCallable_Check(func)) {
      if (PyErr_Occurred()) {
        PyErr_Print();
      }
      throw std::runtime_error{"Could not find python function in module.\n"};
    }

    const auto slice_tensor_of_datavectors_to_tensor_of_doubles = [](
        const auto& tnsr_dv, const size_t slice_idx) noexcept {
      Tensor<double, typename std::decay_t<decltype(tnsr_dv)>::symmetry,
             typename std::decay_t<decltype(tnsr_dv)>::index_list>
          tnsr_double{};
      ASSERT(slice_idx < tnsr_dv.begin()->size(),
             "Trying to slice DataVector of size "
                 << tnsr_dv.begin()->size() << "with slice_idx " << slice_idx);
      for (decltype(auto) double_and_datavector_components :
           boost::combine(tnsr_double, tnsr_dv)) {
        boost::get<0>(double_and_datavector_components) =
            boost::get<1>(double_and_datavector_components)[slice_idx];
      }
      return tnsr_double;
    };

    const auto put_tensor_of_doubles_into_tensor_of_datavector = [](
        auto& tnsr_dv, const auto& tnsr_double,
        const size_t slice_idx) noexcept {
      ASSERT(slice_idx < tnsr_dv.begin()->size(),
             "Trying to slice DataVector of size "
                 << tnsr_dv.begin()->size() << "with slice_idx " << slice_idx);
      for (decltype(auto) datavector_and_double_components :
           boost::combine(tnsr_dv, tnsr_double)) {
        boost::get<0>(datavector_and_double_components)[slice_idx] =
            boost::get<1>(datavector_and_double_components);
      }
    };

    // It's a GCC bug that rest_of_args needs to be given a name in order to
    // compile, and as such it needs to be used to prevent a compiler warning.
    const size_t npts = [](const auto& first_tensor,
                           const auto&... rest_of_args) noexcept {
      (void)std::make_tuple(rest_of_args...);
      return first_tensor.begin()->size();
    }(t...);

    auto return_tensor = make_with_value<R>(
        DataVector{npts, 0.}, std::numeric_limits<double>::signaling_NaN());

    for (size_t s = 0; s < npts; ++s) {
      PyObject* args = pypp::make_py_tuple(
          slice_tensor_of_datavectors_to_tensor_of_doubles(t, s)...);
      PyObject* value = PyObject_CallObject(func, args);
      Py_DECREF(args);  // NOLINT
      if (value == nullptr) {
        Py_DECREF(func);    // NOLINT
        Py_DECREF(module);  // NOLINT
        PyErr_Print();
        throw std::runtime_error{"Function returned null"};
      }
      const auto ret = from_py_object<
          Tensor<double, typename R::symmetry, typename R::index_list>>(value);
      Py_DECREF(value);  // NOLINT
      put_tensor_of_doubles_into_tensor_of_datavector(return_tensor, ret, s);
    }
    Py_DECREF(func);    // NOLINT
    Py_DECREF(module);  // NOLINT
    return return_tensor;
  }
};
}  // namespace detail

/// Calls a Python function from a module/file with given parameters
///
/// \param module_name name of module the function is in
/// \param function_name name of Python function in module
/// \param t the arguments to be passed to the Python function
/// \return the object returned by the Python function converted to a C++ type
///
/// Custom classes can be converted between Python and C++ by overloading the
/// `pypp::ToPythonObject<T>` and `pypp::FromPythonObject<T>` structs for your
/// own types. This tells C++ how to deconstruct the Python object into
/// fundamental types and reconstruct the C++ object and vice-versa.
///
/// \note In order to setup the python interpreter and add the local directories
/// to the path, a SetupLocalPythonEnvironment object needs to be constructed
/// in the local scope.
///
/// \example
/// The following example calls the function `test_numeric` from the module
/// `pypp_py_tests` which multiplies two integers.
/// \snippet Test_Pypp.cpp pypp_int_test
/// Alternatively, this examples calls `test_vector` from `pypp_py_tests` which
/// converts two vectors to python lists and multiplies them pointwise.
/// \snippet Test_Pypp.cpp pypp_vector_test
///
/// Pypp can also be used to take a function that performs manipulations of
/// NumPy arrays and apply it to either a Tensor of doubles or a Tensor of
/// DataVectors. This is useful for testing functions which act on Tensors
/// pointwise. For example, let's say we wanted to call the NumPy function which
/// performs \f$ v_i = A B^a C_{ia} + D^{ab} E_{iab} \f$, which is implemented
/// in python as
///
/// \code{.py} def test_einsum(scalar, t_A, t_ia, t_AA, t_iaa):
///    return scalar * np.einsum("a,ia->i", t_A, t_ia) +
///           np.einsum("ab, iab->i", t_AA, t_iaa)
/// \endcode
///
/// where \f$ v_i \f$ is the return tensor and
/// \f$ A, B^a, C_{ia},D^{ab}, E_{iab} \f$ are the input tensors respectively.
/// We call this function through C++ as:
/// \snippet Test_Pypp.cpp einsum_example
/// for type `T` either a `double` or `DataVector`.
//
/// \note In order to return a
/// `Tensor<DataVector...>` from `pypp::call`, at least one
/// `Tensor<DataVector...>` must be taken as an argument, as the size of the
/// returned tensor needs to be deduced.
template<typename R, typename... Args>
R call(const std::string &module_name, const std::string &function_name,
       const Args &... t) {
  return detail::CallImpl<R>::call(module_name, function_name, t...);
}
}  // namespace pypp
