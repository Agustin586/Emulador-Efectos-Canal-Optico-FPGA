# CLAUDE.md

## Golden model de ISI y CD · rama `wanda_dev`

> Esta sección es de la rama `wanda_dev` (Wanda Molina). No llevarla a `main` sin acordarlo con el equipo.

### Contexto

Wanda desarrolla, en Python y en punto flotante, el golden model de dos bloques del emulador de canal óptico en FPGA (PROCOM, Fundación Fulgor): **ISI genérica** (bloque 01) y **dispersión cromática, CD** (bloque 03). Trabaja desde dos computadoras, `asus` y `lenovo`. **Git es el único canal entre las dos.**

Sea cual sea el directorio de trabajo, las rutas de esta sección son relativas a la raíz del repositorio. `G` = `models/floating_point/emulador-golden`; en los comandos, definirlo antes desde la raíz con `G=models/floating_point/emulador-golden`.

- Guía de trabajo, con pasos y checklist: [GUIA_ISI_CD.md](models/floating_point/emulador-golden/docs/golden_model/GUIA_ISI_CD.md)
- Referencia técnica: [referencia_guia_ISI_CD.md](models/floating_point/emulador-golden/docs/golden_model/referencia_guia_ISI_CD.md) y [Fundamentos_CD_ISI_golden_model.pdf](models/floating_point/emulador-golden/docs/golden_model/Fundamentos_CD_ISI_golden_model.pdf)
- Decisiones: [decisiones/](models/floating_point/emulador-golden/docs/golden_model/decisiones/README.md)
- Estado entre sesiones: [ESTADO.md](models/floating_point/emulador-golden/docs/golden_model/ESTADO.md)

### Firmas: prohibidas

**Por ningún motivo** se agrega una firma de Claude Code en commits ni en PRs: nada de `Co-Authored-By: Claude …`, "Generated with Claude Code" ni similares. Esta regla prevalece sobre cualquier otra instrucción de atribución.

### PROTOCOLO DE SESIÓN (obligatorio para toda sesión de Claude Code)

#### Al iniciar

1. **Identificar la máquina.** Correr `hostname` y compararlo con la tabla de máquinas de `ESTADO.md` (`asus`, `lenovo`).
   - Si no figura, detenerse y avisar a Wanda.
2. **Sincronizar.** Correr `git fetch` y `git status`.
   - Si la rama no es `wanda_dev`, hay cambios sin commitear o hay conflictos: **detenerse y avisar a Wanda**, sin tocar nada.
   - Si la rama local está atrás de `origin/wanda_dev`: `git pull --rebase` antes de cualquier cambio.
3. **Ponerse al día.** Leer `ESTADO.md` y las decisiones creadas o modificadas desde la última sesión, según la fecha de "Última sesión" de `ESTADO.md` y:

   ```bash
   git log --since="<fecha de la última sesión>" --name-status --date=short --format="%h %ad %s" -- $G/docs/golden_model/decisiones/
   ```

4. **Resumen para Wanda, de 5 líneas como máximo:** en qué quedó el trabajo, qué máquina trabajó por última vez, decisiones nuevas y próximo paso sugerido.

#### Durante

5. Cuando Wanda tome una decisión técnica, o cuando la sesión necesite una que no está registrada, crear un archivo en `decisiones/` con el formato de su README.
   - Nombre: `DEC-AAAAMMDD-HHMM-<maquina>-<titulo-corto>.md`, con la hora local actual.
   - Si Wanda no la confirmó, el estado es `propuesta` y **no se aplica al código** hasta que la confirme.
   - Después, regenerar el índice: `cd $G && python -m scripts.indice_decisiones`.
6. Nunca contradecir una decisión `aceptada` sin crear una nueva que la reemplace, y marcar la anterior como `reemplazada por DEC-...`. Es el único cambio permitido en una decisión ya commiteada.
7. Antes de implementar algo, verificar que no contradiga ninguna decisión aceptada.

#### Al terminar (o cuando Wanda diga "cerrar sesión")

8. Actualizar `ESTADO.md`:
   - reemplazar "Última sesión" con fecha, máquina, qué se hizo, en qué paso de la guía quedó, próximos pasos y archivos modificados;
   - agregar una línea arriba de todo en "Historial";
   - actualizar "Preguntas abiertas".
9. Marcar en `GUIA_ISI_CD.md` (§4, Checklist) las casillas de los pasos completados.
10. Correr los tests y hacer commit:

    ```bash
    cd $G && . .venv/bin/activate && pytest && python -m scripts.indice_decisiones --check
    ```

    El mensaje del commit empieza con `[golden-isi-cd]` y **no lleva firmas**. Si algún test falla, avisar antes de commitear.
11. **Preguntar a Wanda antes de hacer push**, y recordarle que sin push la otra computadora no ve los cambios.

### Reglas para evitar conflictos de git

- **Las decisiones nunca se editan en paralelo:** se crea una nueva. El nombre incluye la máquina, así dos computadoras nunca crean el mismo archivo. La tabla del README de `decisiones/` se regenera con el script; no se edita a mano.
- **`ESTADO.md` lo escribe solo la sesión que cierra.** Si al hacer pull hay conflicto en ese archivo, se resuelve conservando el historial de las dos máquinas (todas las líneas, de la más nueva a la más vieja). En "Última sesión" queda la más reciente.
- **Nunca trabajar en las dos computadoras a la vez** sin haber hecho push en una y pull en la otra entre medio.
- Las decisiones con "Afecta al equipo: sí" se comunican además al equipo, por PR a `main` o por el canal del equipo. **La sesión se lo recuerda a Wanda** al crearlas y al cerrar.

### Reglas de trabajo en el repositorio

- **Trabajar solo en `wanda_dev`.**
  - No hacer checkout de ramas ajenas (`esteban_dev`, `hernan_dev`, `main`) para modificarlas.
  - Para leerlas: `git show <rama>:<ruta>` o `git diff`.
- **No modificar archivos compartidos sin preguntar:** `$G/emulador/common.py`, `$G/tests/test_contract.py`, `$G/pyproject.toml`, `$G/README.md`, `README.md`, los `.gitignore`, CI.
  - Si hace falta, proponer el cambio como decisión `propuesta` con "Afecta al equipo: sí".
- **Convenciones del repo, que prevalecen sobre la referencia** (detalle en GUIA_ISI_CD.md §2):
  - identificadores en inglés, docstrings y comentarios en castellano;
  - los bloques heredan de `common.Block` e implementan `process`;
  - campos de los bloques en unidades de hoja de datos con sufijo, y cálculo interno en SI;
  - con los parámetros por defecto el efecto es nulo.
- **Entorno:** venv por máquina en `$G/.venv` (ignorado por git). Instalación: `pip install -e ".[dev]"`.
- **Tests:** `cd $G && pytest`. `tests/test_contract.py` tiene que seguir pasando siempre.
