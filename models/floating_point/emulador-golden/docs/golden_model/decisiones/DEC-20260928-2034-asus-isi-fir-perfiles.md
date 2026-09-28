# DEC-20260928-2034-asus: ISI como FIR complejo causal con perfiles diseñados en frecuencia
- Estado: aceptada
- Fecha: 2026-09-28
- Máquina: asus
- Bloque: ISI
- Afecta al equipo: no

## Contexto
El bloque de ISI genérica tiene que poder reproducir componentes limitados en banda, ecos y respuestas medidas, con una estructura directamente implementable en hardware.

## Decisión
- **Estructura:** FIR complejo causal de K ≤ 32 coeficientes, definidos a f_s, con energía normalizada (Σ|h|² = 1). El cursor k₀ = argmax|h|.
- **Diseño de coeficientes, siempre en frecuencia:**
  - gaussiano: centrado antes de truncar;
  - Bessel y Butterworth: desde el prototipo analógico (`analog=True`, Bessel con `norm="mag"`), con pre-retardo d = 3 (K = 16) o d = 6 (K = 32);
  - **nunca** con la transformación bilineal.
- **Perfil principal de ensayo:** Bessel, m = 4, B = 0,6 R_s, K = 16. El valor **por defecto** del bloque sigue siendo el bypass, porque así lo exige el contrato.
- **Regla de K:** K = 32 si B ≤ 0,4 R_s y m ≥ 4; K = 16 en los demás casos.
- **Perfiles disponibles:** taps arbitrarios, eco, gaussiano (validación), Bessel, Butterworth (estrés), medido y CD corta.

## Alternativas consideradas
- Bilineal (`signal.bessel(m, Wn)` digital): a 2 sps, el Bessel pierde la fase lineal (sobrepico del 15 %, errores de 0,53 dB y 0,78 muestras). Descartada.
- Muestrear la h(t) continua del gaussiano: aliasing y −3 dB corrido. Descartado.
- Normalizar el cursor (h[k₀] = 1): no se usa, porque cambiaría la potencia y la SNR.

## Consecuencias (código, tests, notebooks, hardware)
- Código: `emulador/isi/{fir,profiles}.py`; `emulador/blocks/isi.py`.
- Tests:
  - −3,0 ± 0,3 dB en B;
  - energía capturada ≥ 98,5 %;
  - fidelidad < 0,5 dB y < 0,4 muestras (m ≥ 3);
  - simetría del gaussiano;
  - regresión del perfil principal.
- Hardware: solo cambia el vector de coeficientes que se carga; los perfiles no tienen costo.

## Referencias
- referencia_guia_ISI_CD.md Parte A; PDF Fundamentos §3.5–3.7, Tablas 2–4 y 19.
