# quant_crosscheck · test cruzado de cuantización (C3.0)

Simulador HALCON mínimo que comprueba que `emulador.quantization` (`quantize`, `to_int`) del golden y `ac_fixed` de HALCON dan **los mismos enteros** para las mismas entradas, empates exactos incluidos. Es el requisito de la decisión 33 antes de exportar vectores. Las decisiones están en `models/floating_point/emulador-golden/docs/golden_model/GUIA_ISI_CD.md`, §1.

## Qué prueba

| Caso | Camino | Formato de salida |
|---|---|---|
| double | double → `ac_fixed<W, I>`, la misma asignación que hace `SET` | `<14,1>`, `<20,2>`, `<18,1>`, `<14,2>`, con `AC_RND_CONV`, `AC_RND` y `AC_TRN`, y con `AC_SAT` o `AC_WRAP` |
| a) producto | `<20,2>` × `<18,1>` exacto en `<38,3>`; suma o resta de dos productos exacta en `<39,4>` | `<20,2>`, `AC_RND`, `AC_SAT` |
| b) mariposa | a ± b exacto en `<21,3>`, desplazamiento de s ∈ {0, 1} bits | `<20,2>`, `AC_RND_CONV`, `AC_SAT` |
| c) negación (×(−j)) | −x exacto en `<21,3>`; −(−2) tiene que saturar a 2 − q | `<20,2>`, `AC_RND_CONV`, `AC_SAT` |

Los anchos de los tipos intermedios están fijados con `static_assert` en `src/root/root.cpp`.

**Desplazamiento de la mariposa.** No usa el operador `>>` de `ac_fixed`. Ese operador conserva el tipo del operando y descarta los s bits menos significativos, o sea que **trunca** (`lib/core/inc/ac_fixed.h`, sección "Shift (result constrained by left operand)"). El desplazamiento se hace como cambio exacto de escala: los mismos 21 bits se reinterpretan como `<21, 3 − s>` (`ScaleDown`, con `set_slc`) y después hay **un** redondeo a `<20,2>`. El simulador también escribe el resultado con `>> 1`, y el test comprueba que coincide con `AC_TRN`.

**Archivos de datos.** Formatos en la cabecera de `src/root/root.cpp`. Un código es el entero con signo k del valor k·2^−(W−I). Se lee y se escribe con `slc`/`set_slc`, sin pasar por double. Los doubles se escriben con `repr` de Python, que es de ida y vuelta exacta.

## Cómo correrlo

Requisitos:
- Docker;
- la imagen `hawk_workspace`: `cd <halcon>/tutorials/lib/tools/docker && docker build -t hawk_workspace .`;
- una copia de HALCON con los submódulos de `lib/` inicializados.

```bash
# desde el golden: genera las entradas, corre el simulador y compara
cd models/floating_point/emulador-golden && pytest tests/test_quant_crosscheck.py

# a mano, con *_in.txt ya escritos en <dir>
models/fixed_point/emulador-real/quant_crosscheck/run_crosscheck.sh <dir>
```

Variables de entorno:
- `HALCON_DIR`: por defecto `/home/wanda/halcon/tutorials`;
- `HAWK_IMAGE`: por defecto `hawk_workspace`.

Si falta alguno de los requisitos, el test se saltea.

`run_crosscheck.sh` **no escribe en el repo ni en HALCON**:
- monta `<halcon>/tutorials/lib` de solo lectura en `/app/lib`;
- monta este proyecto de solo lectura y lo copia a `/tmp/work/quant_crosscheck` dentro del contenedor. Hace falta porque `main.py` se reescribe a sí mismo y la compilación crea `build/` y `run/`;
- en `/tmp/work`, `lib` apunta a `/app/lib`. Así `core_dir = "../lib/core/"` de `main.py` funciona sin cambios, y el `PYTHONPATH` de la imagen (`/app/lib/tools/python`) encuentra `halcon`;
- corre con el usuario del host;
- verifica que existan las salidas, porque `simulator_handler` usa `os.system` y no propaga los errores.

Compilar y correr tarda unos 10 s.

## Versiones

- HALCON (`/home/wanda/halcon/tutorials`):
  - `lib/core` d3a04c3 (devel);
  - `lib/tools` e5b7e02 (devel);
  - `lib/modules` a662c3d (rama `issue_1-5`).
- Contenedor: Python, compiladores y `pip freeze` en [container_environment.txt](container_environment.txt). El dockerfile no fija las versiones de pip.

`CMakeLists.txt` y `src/main.cpp` son copias sin cambios de `tutorials/tp2`. `main.py` y `src/root/` están adaptados de tp2 (MIT, Fundación Fulgor).
