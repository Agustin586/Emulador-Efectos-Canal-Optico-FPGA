# DEC-20260928-2032-asus: Término β₃ incluido, con S = 0 por defecto
- Estado: aceptada
- Fecha: 2026-09-28
- Máquina: asus
- Bloque: CD
- Afecta al equipo: no

## Contexto
A 32 GBd, β₃ es despreciable. A partir de 64–100 GBd deja ISI residual después de un compensador que solo invierte β₂, y esa es justamente la ISI que tienen que absorber los ecualizadores adaptativos del Proyecto Marco.

## Decisión
- Se incluye β₃ = (λ²/(2πc))²·(S + 2D/λ) como parámetro opcional, a través de la pendiente de dispersión S (`S_ps_nm2_km`), con S = 0 por defecto.
- Con S = 0 el modelo es idéntico al de solo β₂ (β₃ = 0 exacto).

## Alternativas consideradas
- Solo β₂: descartado, porque el emulador evaluaría la DSP contra un canal más benigno que el real.

## Consecuencias (código, tests, notebooks, hardware)
- Código: un término más en la fase de H[k]. No cambia el camino de datos ni el costo en hardware.
- Tests:
  - parábola del retardo de grupo (error relativo < 1 %);
  - compatibilidad con S = 0 (exacta);
  - reversibilidad con (−D, −S);
  - residuo de la Tabla 17 del PDF (±5 % relativo).
- Notebooks: figura del residuo de β₃ en `04_cd_overlap_save.ipynb`.

## Referencias
- referencia_guia_ISI_CD.md B.1 y B.6; PDF Fundamentos §6.2, Tablas 16–18.
