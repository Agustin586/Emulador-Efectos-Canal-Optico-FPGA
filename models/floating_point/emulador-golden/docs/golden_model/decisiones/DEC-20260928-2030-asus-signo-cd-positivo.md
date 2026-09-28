# DEC-20260928-2030-asus: Convención de signo positiva de la CD
- Estado: aceptada
- Fecha: 2026-09-28
- Máquina: asus
- Bloque: CD
- Afecta al equipo: sí

## Contexto
El signo de la función de transferencia de la CD depende de la convención de la onda (e^{-jβz} o e^{+jβz}) y de la transformada de Fourier. Las dos son físicamente correctas, pero el emulador y el compensador (tesis de CDC) tienen que usar la misma. Si no, el compensador duplica la dispersión en lugar de anularla. El esqueleto `chromatic_dispersion.py` tiene hoy el signo opuesto en su docstring.

## Decisión
H_CD(f) = exp(+j·π·D·λ²·z·f²/c), equivalente a exp(−j·β(ω)·z) con β₂ = −D·λ²/(2πc) y la transformada de `numpy.fft` (X[k] = Σ x[n]·e^{−j2πkn/L}). Con D > 0, las frecuencias altas llegan antes: el retardo de grupo tiene pendiente −Dλ²z/c.

## Alternativas consideradas
- Signo negativo, exp(−jπDλ²z f²/c): el del docstring actual. Descartado para coincidir con Jiang et al. (2026), ec. (1).

## Consecuencias (código, tests, notebooks, hardware)
- Código: `cd_transfer` en `emulador/cd/physics.py`; hay que corregir el docstring del esqueleto.
- Tests: el test de signo (fase = +πDλ²z f²/c a baja frecuencia) y el de pendiente del retardo de grupo fijan la convención; si alguien la cambia, fallan.
- Equipo: **comunicar a la tesis de CDC** que su filtro tiene que usar H_CD^* (el conjugado). Actualizar "Convenciones abiertas" del README por PR.

## Referencias
- referencia_guia_ISI_CD.md §3 y B.1; PDF Fundamentos §4.4 (nota sobre la convención de signo), Tabla 12.
- J. Jiang, C. Xie y M. Chen, J. Lightwave Technol. 44(12), 2026, ec. (1).
