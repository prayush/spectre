# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

# Test functions for characteristic speeds

def char_speed_upsi(gamma1, lapse, shift, unit_normal):
    return - (1. + gamma1) * np.dot(shift, unit_normal)

def char_speed_uzero(gamma1, lapse, shift, unit_normal):
    return - np.dot(shift, unit_normal)

def char_speed_uplus(gamma1, lapse, shift, unit_normal):
    return - np.dot(shift, unit_normal) + lapse

def char_speed_uminus(gamma1, lapse, shift, unit_normal):
    return - np.dot(shift, unit_normal) - lapse

# End test functions for characteristic speeds


# Test functions for characteristic fields

def char_field_upsi(gamma2, inverse_spatial_metric, psi,
                    pi, phi, normal_one_form):
    return psi

def char_field_uzero(gamma2, inverse_spatial_metric, psi,
                     pi, phi, normal_one_form):
    normal_vector =\
        np.einsum('ij,j', inverse_spatial_metric, normal_one_form)
    projection_tensor = np.identity(len(normal_vector)) -\
        np.einsum('i,j', normal_one_form, normal_vector)
    return np.einsum('ij,j->i', projection_tensor, phi)

def char_field_uplus(gamma2, inverse_spatial_metric, psi,
                     pi, phi, normal_one_form):
    normal_vector =\
        np.einsum('ij,j', inverse_spatial_metric, normal_one_form)
    phi_dot_normal = np.einsum('i,i->', normal_vector, phi)
    return pi + 1*phi_dot_normal - (gamma2 * psi)

def char_field_uminus(gamma2, inverse_spatial_metric, psi,
                      pi, phi, normal_one_form):
    normal_vector =\
        np.einsum('ij,j', inverse_spatial_metric, normal_one_form)
    phi_dot_normal = np.einsum('i,i->', normal_vector, phi)
    return pi - phi_dot_normal - (gamma2 * psi)


def evol_field_psi(gamma2, upsi, uzero, uplus, uminus, normal_one_form):
    return upsi

def evol_field_pi(gamma2, upsi, uzero, uplus, uminus, normal_one_form):
    return 0.5 * (uplus + uminus) + gamma2 * upsi

def evol_field_phi(gamma2, upsi, uzero, uplus, uminus, normal_one_form):
    udiff = 0.5 * (uplus - uminus)
    return (normal_one_form * udiff) + uzero

# End test functions for characteristic fields
