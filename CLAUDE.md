# CLAUDE.md

## Golden model de ISI y CD · rama `wanda_dev`

> Esta sección es de la rama `wanda_dev` (Wanda Molina). No llevarla a `main` sin acordarlo con el equipo.

### Contexto

Wanda desarrolla, en Python y en punto flotante, el golden model de dos bloques del emulador de canal óptico en FPGA (PROCOM, Fundación Fulgor): **ISI genérica** (bloque 01) y **dispersión cromática, CD** (bloque 03). Trabaja desde dos computadoras. **La rama `wanda_dev` en `origin` es el único canal entre las dos.**

`G` = `models/floating_point/emulador-golden` (desde la raíz del repo).

- **Guía de trabajo:** [GUIA_ISI_CD.md](models/floating_point/emulador-golden/docs/golden_model/GUIA_ISI_CD.md). Tiene las decisiones vigentes (§1), el mapeo al repo (§2), los pasos detallados (§3) y la checklist de tareas atómicas (§4).
- **Referencia técnica:** [referencia_guia_ISI_CD.md](models/floating_point/emulador-golden/docs/golden_model/referencia_guia_ISI_CD.md) y [Fundamentos_CD_ISI_golden_model.pdf](models/floating_point/emulador-golden/docs/golden_model/Fundamentos_CD_ISI_golden_model.pdf).

### Máquinas

| Nombre | Hostname | Sistema operativo | Entorno |
|---|---|---|---|
| asus | `wanda-ASUS-Vivobook-16-M1607KA-M1607KA` | Ubuntu 24.04.5 LTS | `$G/.venv`: Python 3.12.3, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, pytest 9.1.1, jupyterlab 4.6.4; `nbstripout` instalado |
| lenovo | `wanda-IdeaPad-5-15ALC05` | Ubuntu 24.04.5 LTS | `$G/.venv`: Python 3.12.3, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, pytest 9.1.1, jupyterlab 4.6.4; `nbstripout` instalado (2026-09-30) |

### Firmas: prohibidas

**Por ningún motivo** se agrega una firma de Claude Code en commits ni en PRs: nada de `Co-Authored-By: Claude …`, "Generated with Claude Code" ni similares. Esta regla prevalece sobre cualquier otra instrucción de atribución.

### Protocolo de sesión

El avance se coordina con **tareas atómicas**: cada casilla de la checklist (§4 de la guía) es un commit y un push. Con eso, la otra computadora sigue con solo hacer `git pull`.

#### Al iniciar

1. `git fetch` y `git status`.
   - Si la rama no es `wanda_dev`, hay cambios sin commitear o hay conflictos: **detenerse y avisar a Wanda**.
   - Si está atrás de `origin/wanda_dev`: `git pull --rebase`.
2. Resumen para Wanda, de 3 líneas como máximo:
   - última tarea hecha y en qué máquina (`git log -5 --format="%h %ad %s" --date=short`);
   - si el último commit es un `WIP`, qué falta;
   - próxima tarea sin marcar de §4.

#### Por cada tarea

3. Leer la sección del paso en §3 de la guía y verificar que no contradiga §1.
   - Si hace falta una decisión nueva, preguntarle a Wanda y agregarla como fila de §1.
   - Una fila "sin confirmar" no se aplica al código.
4. Tests primero, después la implementación. Correr `cd $G && . .venv/bin/activate && pytest` (completo).
5. Marcar la casilla de la tarea en §4, en el mismo commit.
   - Mensaje: `[golden-isi-cd] <ID> <descripción>`, sin firmas.
   - Si algún test falla, avisar antes de commitear.
6. **Preguntar a Wanda antes de hacer push**, y recordarle que sin push la otra computadora no ve la tarea.

Si la sesión termina a mitad de una tarea: commit `[golden-isi-cd] <ID> WIP: <qué falta>` y push, sin marcar la casilla.

### Reglas de git entre las dos computadoras

- Nunca trabajar en las dos a la vez sin push en una y pull en la otra entre medio.
- Una tarea por commit; no mezclar tareas en un mismo commit.
- Si una decisión de §1 afecta al equipo (interfaz común, `pyproject.toml`, README), recordarle a Wanda que la comunique por PR o por el canal del equipo.

### Reglas de trabajo en el repositorio

- **Trabajar solo en `wanda_dev`.**
  - No hacer checkout de ramas ajenas (`esteban_dev`, `hernan_dev`, `main`) para modificarlas.
  - Para leerlas: `git show <rama>:<ruta>` o `git diff`.
- **No modificar archivos compartidos sin preguntar:** `$G/emulador/common.py`, `$G/emulador/__init__.py`, `$G/tests/test_contract.py`, `$G/pyproject.toml`, `$G/README.md`, `README.md`, los `.gitignore`, CI.
- **Convenciones del repo, que prevalecen sobre la referencia** (detalle en §2 de la guía):
  - identificadores en inglés, docstrings y comentarios en castellano;
  - los bloques heredan de `common.Block` e implementan `process`;
  - campos de los bloques en unidades de hoja de datos con sufijo, y cálculo interno en SI;
  - con los parámetros por defecto el efecto es nulo.
- **Entorno:** venv por máquina en `$G/.venv` (ignorado). Instalación: `pip install -e ".[dev,notebooks]"` y, una vez por máquina, `nbstripout --install` (borra las salidas de los notebooks al commitear).
- **Tests:** `cd $G && pytest`. `tests/test_contract.py` tiene que seguir pasando siempre.
