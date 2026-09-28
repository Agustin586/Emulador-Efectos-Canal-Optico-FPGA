# DEC-20260928-2033-asus: CD por muestreo en frecuencia con overlap-save
- Estado: aceptada
- Fecha: 2026-09-28
- Máquina: asus
- Bloque: CD
- Afecta al equipo: no

## Contexto
La respuesta al impulso de la CD es un chirp infinito y no causal. Hay tres métodos: FIR truncado (Savory), bloques en frecuencia (OLS/OLA) y FFT completa. El FIR truncado tiene rizado y un costo proporcional a R por muestra.

## Decisión
- **Modelo principal:** muestreo en frecuencia con overlap-save.
  - FFT de L = 2N; H[k] = H_CD(2π f_k) con f_k = `fftfreq(L, 1/fs)`.
  - Filtro con retardo de N/2: H̄[k] = H[k]·e^{−j2πk(N/2)/L}.
  - Por bloque: N previas + N nuevas → FFT → ×H̄ → IFFT → se descartan las primeras N.
  - La salida queda retrasada N/2 muestras.
- **Regla N ≥ R**, con R = 2⌊|D|λ²z/(2cT_s²)⌋ + 1.
- **N automático** = max(N_min, menor potencia de 2 ≥ R), con N_min = 128 (a z = 0, R = 1 y N/2 no sería entero) y N_max de `HardwareParams` (provisorio: 2048).
- **Referencia ideal:** FFT completa con relleno de ≥ R ceros a cada lado.

## Alternativas consideradas
- FIR truncado de Savory: queda como perfil "CD corta" del bloque ISI y como curva de comparación.
- FIR por mínimos cuadrados: mejora opcional futura.
- Overlap-add: equivalente, no se usa.

## Consecuencias (código, tests, notebooks, hardware)
- Código: `emulador/cd/{physics,reference,overlap_save}.py`; bloque con `delay_samples = N/2`.
- Tests:
  - NRMSE del OLS contra la referencia < 10⁻² con N ≥ R;
  - valores de R exactos (Tabla 6 del PDF);
  - retardo declarado = N/2.
- Hardware: FFT y IFFT de 2N puntos; tasa efectiva P·f_clk/2.

## Referencias
- referencia_guia_ISI_CD.md B.2–B.4; PDF Fundamentos §4.7, Ecs. 26–29, Fig. 9; Jiang et al. (2026), ec. (7).
