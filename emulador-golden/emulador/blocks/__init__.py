"""Registro de bloques. Al sumar un bloque nuevo, agregarlo acá."""
from .awgn import AWGN
from .chromatic_dispersion import ChromaticDispersion
from .isi import ISI
from .phase_noise import PhaseNoise
from .phase_offset import PhaseOffset
from .pmd import PMD
from .timing_error import TimingError

# nombre del bloque -> clase (es el nombre que usa la línea de comandos)
REGISTRY = {
    "isi": ISI,
    "awgn": AWGN,
    "chromatic_dispersion": ChromaticDispersion,
    "pmd": PMD,
    "phase_offset": PhaseOffset,
    "phase_noise": PhaseNoise,
    "timing_error": TimingError,
}
