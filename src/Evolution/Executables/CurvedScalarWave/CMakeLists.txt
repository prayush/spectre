# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
    CoordinateMaps
    CurvedScalarWave
    DiscontinuousGalerkin
    Domain
    DomainCreators
    GeneralRelativity
    GeneralRelativitySolutions
    IO
    Informer
    LinearOperators
    MathFunctions
    Time
    Utilities
    WaveEquation
    ${SPECTRE_LIBRARIES}
  )

add_spectre_parallel_executable(
  EvolveCurvedScalarWave3D
  EvolveCurvedScalarWave
  Evolution/Executables/CurvedScalarWave
  EvolutionMetavars<3>
  "${LIBS_TO_LINK}"
)
