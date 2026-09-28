# Decisiones del golden model de ISI y CD

Registro de las decisiones técnicas del golden model de ISI genérica y dispersión cromática (rama `wanda_dev`). Cada decisión es un archivo; la tabla de abajo **se genera**, no se edita a mano.

## Reglas

- **Un archivo por decisión**, llamado `DEC-AAAAMMDD-HHMM-<maquina>-<titulo-corto>.md`.
  - `<maquina>` es el nombre corto de la tabla de máquinas de [ESTADO.md](../ESTADO.md): `asus` o `lenovo`.
  - `<titulo-corto>` va en minúsculas, sin tildes, con guiones.
  - El ID es `DEC-AAAAMMDD-HHMM-<maquina>` y tiene que ser único. Si se crean varias en el mismo minuto, se suma un minuto a cada una.
  - Como el nombre incluye la máquina, dos computadoras nunca crean el mismo archivo.
- **Las decisiones no se editan en paralelo.** Para cambiar una decisión "aceptada" se crea una nueva, y en la anterior solo se cambia el estado a `reemplazada por DEC-...`.
- **Estados posibles:**
  - `propuesta`: todavía no confirmada por Wanda, o confirmada pero pendiente de acuerdo del equipo (se aclara en el texto).
  - `aceptada`.
  - `reemplazada por DEC-...`.
  - `descartada`.
- Una decisión no confirmada por Wanda **no se aplica al código**.
- Si dice "Afecta al equipo: sí", se comunica además al equipo, por PR o por el canal del equipo.
- Después de crear o cambiar una decisión, regenerar la tabla desde `models/floating_point/emulador-golden/`:

  ```bash
  python -m scripts.indice_decisiones           # reescribe la tabla
  python -m scripts.indice_decisiones --check   # valida formato y que la tabla esté al día
  ```

## Formato

```markdown
# DEC-AAAAMMDD-HHMM-<maquina>: <título>
- Estado: propuesta | aceptada | reemplazada por DEC-... | descartada
- Fecha: AAAA-MM-DD
- Máquina: <maquina>
- Bloque: ISI | CD | común
- Afecta al equipo: sí | no
## Contexto
## Decisión
## Alternativas consideradas
## Consecuencias (código, tests, notebooks, hardware)
## Referencias
```

## Índice

<!-- INDICE:INICIO (generado por scripts/indice_decisiones.py; no editar a mano) -->
| ID | Título | Estado | Fecha | Máquina | Bloque | Afecta al equipo |
|---|---|---|---|---|---|---|
| [DEC-20260928-2029-asus](DEC-20260928-2029-asus-escenario-referencia.md) | Escenario de referencia configurable (32 GBd, 2 sps, 64 GSa/s) | aceptada | 2026-09-28 | asus | común | sí |
| [DEC-20260928-2030-asus](DEC-20260928-2030-asus-signo-cd-positivo.md) | Convención de signo positiva de la CD | aceptada | 2026-09-28 | asus | CD | sí |
| [DEC-20260928-2031-asus](DEC-20260928-2031-asus-sin-factor-medio.md) | Se descarta el factor 1/2 de la guía original del proyecto | aceptada | 2026-09-28 | asus | CD | sí |
| [DEC-20260928-2032-asus](DEC-20260928-2032-asus-beta3-incluido.md) | Término β₃ incluido, con S = 0 por defecto | aceptada | 2026-09-28 | asus | CD | no |
| [DEC-20260928-2033-asus](DEC-20260928-2033-asus-cd-overlap-save.md) | CD por muestreo en frecuencia con overlap-save | aceptada | 2026-09-28 | asus | CD | no |
| [DEC-20260928-2034-asus](DEC-20260928-2034-asus-isi-fir-perfiles.md) | ISI como FIR complejo causal con perfiles diseñados en frecuencia | aceptada | 2026-09-28 | asus | ISI | no |
| [DEC-20260928-2035-asus](DEC-20260928-2035-asus-hardware-rfsoc.md) | Hardware RFSoC Zynq UltraScale+ y FFT radix-2² con P parametrizable | aceptada | 2026-09-28 | asus | común | sí |
| [DEC-20260928-2036-asus](DEC-20260928-2036-asus-retardo-salida-alineada.md) | Manejo del retardo en ISI y CD (salida alineada + salida de hardware) | aceptada | 2026-09-28 | asus | común | sí |
| [DEC-20260928-2037-asus](DEC-20260928-2037-asus-extension-block.md) | Extender common.Block con delay_samples, coefficients() y process_hw() | propuesta | 2026-09-28 | asus | común | sí |
| [DEC-20260928-2038-asus](DEC-20260928-2038-asus-estimulos-modulo-nuevo.md) | Estímulos en un módulo nuevo emulador/stimuli/ | propuesta | 2026-09-28 | asus | común | sí |
| [DEC-20260928-2039-asus](DEC-20260928-2039-asus-notebooks-extra.md) | Carpeta notebooks/ y extra "notebooks" en pyproject.toml | propuesta | 2026-09-28 | asus | común | sí |
| [DEC-20260928-2040-asus](DEC-20260928-2040-asus-entorno-venv.md) | Entorno con venv por máquina (Docker más adelante) | aceptada | 2026-09-28 | asus | común | no |
| [DEC-20260928-2041-asus](DEC-20260928-2041-asus-adaptacion-estructura.md) | Adaptación de la estructura de la referencia al repositorio | aceptada | 2026-09-28 | asus | común | sí |
| [DEC-20260928-2042-asus](DEC-20260928-2042-asus-ubicacion-modelo-bit-a-bit.md) | Ubicación del modelo bit a bit (punto fijo) | propuesta | 2026-09-28 | asus | común | sí |
<!-- INDICE:FIN -->
