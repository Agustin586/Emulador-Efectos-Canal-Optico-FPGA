# Implementación en punto fijo / RTL

Acá va a vivir la implementación en **punto fijo** de cada bloque del canal óptico, pensada para síntesis en FPGA (Verilog/RTL), junto con su verificación.

**Estado actual: sin empezar.** El trabajo de referencia se está haciendo primero en [`../emulador-golden`](../emulador-golden/), donde cada efecto se modela y valida en Python con punto flotante. Recién cuando un golden model esté cerrado, se porta acá a punto fijo.

Cuando arranque, esta carpeta va a documentar:

- Formato de punto fijo usado por bloque (ancho de palabra, bits fraccionarios).
- Estructura del RTL y cómo simular/sintetizar.
- Cómo se compara cada bloque en punto fijo contra su golden model equivalente (métrica de error, EVM, etc.).
