# RTL

Código **Verilog** de cada bloque del canal óptico, listo para síntesis en FPGA.

**Estado actual: sin empezar.** Cada bloque llega acá después de pasar por [`../models/floating_point/emulador-golden`](../models/floating_point/emulador-golden/) (modelo de referencia en punto flotante) y [`../models/fixed_point/emulador-real`](../models/fixed_point/emulador-real/) (modelo en punto fijo).

Cuando arranque, esta carpeta va a documentar:

- Estructura de los módulos RTL por bloque.
- Cómo simular y sintetizar (herramienta, target de FPGA).
- Convenciones de nombres de señales/módulos.
