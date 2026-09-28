# Emulador-Efectos-Canal-Optico-FPGA

Emulador de efectos de canal óptico para FPGA (Tesis 4.3.13, Proyecto Marco UNS / Fundación Fulgor).

## Estructura

- **[models/floating_point/emulador-golden/](models/floating_point/emulador-golden/)** — Modelos de referencia en **punto flotante (Python)** de cada deterioro del canal óptico (ISI, AWGN, dispersión cromática, PMD, offset de fase, ruido de fase, error de timing). Sirven como "matemática exacta" contra la que luego se valida el hardware. Es la parte activa del proyecto hoy.
- **[models/fixed_point/emulador-real/](models/fixed_point/emulador-real/)** — Modelos en **punto fijo** de esos mismos bloques, portados desde el golden model una vez validado cada uno. Todavía no arrancó.
- **[rtl/](rtl/)** — Código **Verilog/RTL** para síntesis en FPGA, implementando los bloques ya validados en punto fijo. Todavía no arrancó.
- **[verification/](verification/)** — Verificación del RTL (testbenches, comparación contra los golden models). Todavía no arrancó.

El flujo de trabajo es en cascada: golden model (Python, punto flotante) → modelo en punto fijo → RTL → verificación contra el golden model correspondiente.

Para instrucciones de puesta en marcha, estructura interna y estado de avance, ver el README de cada carpeta.
