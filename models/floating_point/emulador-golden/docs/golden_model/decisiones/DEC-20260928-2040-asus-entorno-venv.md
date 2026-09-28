# DEC-20260928-2040-asus: Entorno con venv por máquina (Docker más adelante)
- Estado: aceptada
- Fecha: 2026-09-28
- Máquina: asus
- Bloque: común
- Afecta al equipo: no

## Contexto
El README habla de un contenedor, pero el repo no tiene Dockerfile. El Python del sistema de la ASUS no tiene scipy. Se trabaja desde dos máquinas (ASUS y LENOVO).

## Decisión
- Por ahora, un venv por máquina en `models/floating_point/emulador-golden/.venv/` (ya ignorado por `.gitignore`), instalado con `pip install -e ".[dev]"` (más `notebooks` cuando se acepte DEC-20260928-2039-asus).
- Las versiones de Python, numpy y scipy de cada máquina se anotan en la tabla de máquinas de ESTADO.md.
- Docker se evaluará más adelante, en una decisión nueva.

## Alternativas consideradas
- Docker ya: no hay Dockerfile acordado con el equipo.
- conda: no es la convención del repo.

## Consecuencias (código, tests, notebooks, hardware)
- Tests: se corren con el venv activo (`. .venv/bin/activate && pytest`).
- Riesgo: diferencias de versiones entre máquinas; se mitiga registrándolas.

## Referencias
- README.md del paquete ("Puesta en marcha"); GUIA_ISI_CD.md paso P0.
