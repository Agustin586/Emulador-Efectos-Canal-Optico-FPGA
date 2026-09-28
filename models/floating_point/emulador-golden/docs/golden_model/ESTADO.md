# Estado del golden model de ISI y CD

Lo escribe **solo la sesión que cierra**. Si al hacer `git pull --rebase` hay un conflicto en este archivo:
- se conservan las líneas de **Historial** de las dos máquinas, ordenadas de la más nueva a la más vieja;
- en **Última sesión** queda la más reciente.

Guía: [GUIA_ISI_CD.md](GUIA_ISI_CD.md) · Decisiones: [decisiones/](decisiones/README.md)

## Máquinas

| Nombre | Hostname | Sistema operativo | Entorno | Notas |
|---|---|---|---|---|
| asus | `wanda-ASUS-Vivobook-16-M1607KA-M1607KA` | Ubuntu 24.04.5 LTS | venv en `models/floating_point/emulador-golden/.venv` | Python 3.12.3, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, pytest 9.1.1. `pytest`: 9 passed, 12 skipped (2026-09-28) |
| lenovo | `wanda-IdeaPad-5-15ALC05` | a completar en la primera sesión | venv (mismo esquema) | Completar Python/numpy/scipy en P0 |

Docker queda para más adelante (DEC-20260928-2040-asus).

## Última sesión

- **Fecha:** 2026-09-28
- **Máquina:** asus
- **Qué se hizo:**
  - Relevamiento del repositorio (ramas, estructura, convenciones, conflictos con la referencia).
  - Guía de trabajo `GUIA_ISI_CD.md`, con los pasos detallados.
  - Mecanismo de coordinación: `CLAUDE.md` en la raíz, `decisiones/` con 14 decisiones y el script `scripts/indice_decisiones.py`, y este archivo.
  - P0 en ASUS: venv creado, `pip install -e ".[dev]"`, `pytest` con 9 passed y 12 skipped (bloques sin implementar).
- **Paso de la guía:** P0 completo en ASUS. Falta P0 en LENOVO.
- **Próximos pasos:**
  1. Push de `wanda_dev` (sin push, la LENOVO no ve nada).
  2. P0 en LENOVO: crear `.venv`, `pip install -e ".[dev]"`, correr `pytest` (esperado: 9 passed, 12 skipped) y completar su fila en la tabla de máquinas.
  3. P1 (`emulador/config.py`).
- **Archivos modificados:**
  - `CLAUDE.md` (nuevo, raíz);
  - `models/floating_point/emulador-golden/docs/golden_model/{GUIA_ISI_CD.md, ESTADO.md, decisiones/*}` (nuevos);
  - `models/floating_point/emulador-golden/scripts/indice_decisiones.py` (nuevo);
  - `referencia_guia_ISI_CD.md` y el PDF de fundamentos (se agregan al repo).

## Historial

Una línea por sesión, las más nuevas arriba: `AAAA-MM-DD · máquina · paso(s) · resumen`.

- 2026-09-28 · asus · P0 (asus) · Relevamiento, guía GUIA_ISI_CD.md, protocolo de coordinación y 14 decisiones iniciales.

## Preguntas abiertas

- [ ] Ubicación del modelo bit a bit: golden o `models/fixed_point/emulador-real/` (DEC-20260928-2042-asus, **propuesta sin confirmar**).
- [ ] Acuerdo del equipo para DEC-2037 (extensión de `Block`), DEC-2038 (`stimuli/`) y DEC-2039 (extra `notebooks`).
- [ ] Comunicar a la tesis de CDC la convención de signo positiva (DEC-2030) y proponer la corrección del factor 1/2 (DEC-2031).
- [ ] Modelo de RFSoC, P, f_clk y N_max (paso C1).
- [ ] Criterios a confirmar:
  - tolerancia de la Tabla 4 (±0,5 dB, paso A5);
  - criterio de la "CD corta" contra la referencia (paso A4);
  - EVM back-to-back < 0,5 % (paso P3);
  - convención Qm.n (paso C2).
- [ ] Sistema operativo de la LENOVO.
