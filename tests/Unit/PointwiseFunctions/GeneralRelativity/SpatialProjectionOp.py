# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def spatial_projection_tensor(spatial_metric_or_its_inverse,
                              normal_vector_or_one_form):
    dim_offset = np.shape(normal_vector_or_one_form)[0] - np.shape(
        spatial_metric_or_its_inverse)[1]
    if dim_offset != 0 and dim_offset != 1:
        raise RuntimeError("Incompatible inputs passed")
    return (spatial_metric_or_its_inverse -
            np.einsum('i,j->ij', normal_vector_or_one_form[dim_offset:],
                      normal_vector_or_one_form[dim_offset:]))


def spatial_projection_tensor_mixed_from_spacetime_input(
        normal_vector, normal_one_form):
    return (np.eye(np.shape(normal_vector)[0] - 1) -
            np.einsum('i,j->ij', normal_vector[1:], normal_one_form[1:]))


def spatial_projection_tensor_mixed_from_spatial_input(normal_vector,
                                                       normal_one_form):
    return (np.eye(np.shape(normal_vector)[0]) -
            np.einsum('i,j->ij', normal_vector, normal_one_form))
