# DEC-20260928-2031-asus: Se descarta el factor 1/2 de la guía original del proyecto
- Estado: aceptada
- Fecha: 2026-09-28
- Máquina: asus
- Bloque: CD
- Afecta al equipo: sí

## Contexto
La guía original del proyecto escribe H_CD(ω) = exp(−j·Dλ²z/(4πc)·ω²/2). Frente a la forma de la literatura tiene un factor 1/2 adicional: el 1/2 del desarrollo de Taylor ya está incluido en el denominador 4π. Implementada así, emularía la mitad de la dispersión indicada por z.

## Decisión
No se usa el factor 1/2. La ecuación válida es H_CD(ω) = exp[−j(β₂ω²/2 + β₃ω³/6)z], que con β₃ = 0 se reduce a exp(+jπDλ²z f²/c).

## Alternativas consideradas
- Mantener el 1/2 por fidelidad a la guía: descartado, porque subestima la dispersión a la mitad.

## Consecuencias (código, tests, notebooks, hardware)
- Tests: el test de pendiente del retardo de grupo (−Dλ²z/c, error relativo < 0,1 %) detecta el factor 1/2.
- Equipo: proponer la corrección de la guía del proyecto a la dirección de la tesis. El README del paquete ya lo menciona en "Convenciones abiertas".

## Referencias
- referencia_guia_ISI_CD.md §0 y B.1; PDF Fundamentos §4.4 (nota sobre la expresión de la guía del proyecto) y §6.4.
