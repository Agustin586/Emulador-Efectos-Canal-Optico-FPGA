# Emulador-Efectos-Canal-Optico-FPGA

Emulador de efectos de canal óptico para FPGA (Tesis 4.3.13, Proyecto Marco UNS / Fundación Fulgor).

El proyecto se organiza en dos etapas, cada una en su propia carpeta:

- **[emulador-golden/](emulador-golden/)** — Modelos de referencia en **punto flotante (Python)** de cada deterioro del canal óptico (ISI, AWGN, dispersión cromática, PMD, offset de fase, ruido de fase, error de timing). Sirven como "matemática exacta" contra la que luego se valida el hardware. Es la parte activa del proyecto hoy.
- **[emulador-real/](emulador-real/)** — Implementación en **punto fijo / RTL (FPGA)** de esos mismos bloques. Todavía no arrancó; se completará una vez cerrados los golden models.

Para instrucciones de puesta en marcha, estructura y estado de avance, ver el README de cada carpeta.
