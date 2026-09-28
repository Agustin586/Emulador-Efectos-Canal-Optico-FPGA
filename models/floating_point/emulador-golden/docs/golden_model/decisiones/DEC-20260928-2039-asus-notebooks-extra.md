# DEC-20260928-2039-asus: Carpeta notebooks/ y extra "notebooks" en pyproject.toml
- Estado: propuesta
- Fecha: 2026-09-28
- Máquina: asus
- Bloque: común
- Afecta al equipo: sí

## Contexto
El repo no tiene notebooks ni dependencias de Jupyter. La referencia documenta cada etapa en un notebook (00 a 06). `pyproject.toml` es compartido.

## Decisión
- Notebooks en `models/floating_point/emulador-golden/notebooks/`, con los nombres de la referencia (§7).
- Agregar a `pyproject.toml` un extra separado: `notebooks = ["jupyterlab>=4", "ipykernel", "jupytext>=1.16", "nbstripout", "nbconvert"]`. No se toca el extra `dev`.
- Cada máquina corre `nbstripout --install` (configuración local de git, no versionada).

Confirmación de Wanda: sí (2026-09-28). Pendiente: acuerdo del equipo, por el cambio en `pyproject.toml`.

## Alternativas consideradas
- Meter Jupyter en `dev`: obliga a todo el equipo a instalarlo.
- Instalar Jupyter a mano sin declararlo: el entorno no queda reproducible entre ASUS y LENOVO.

## Consecuencias (código, tests, notebooks, hardware)
- Instalación: `pip install -e ".[dev,notebooks]"`.
- Notebooks: salidas borradas al commitear; figuras en `docs/golden_model/figuras/`.
- Pendiente: decidir si además se usan pares `jupytext` (formato percent).

## Referencias
- referencia_guia_ISI_CD.md §5 y §7; GUIA_ISI_CD.md §5.
