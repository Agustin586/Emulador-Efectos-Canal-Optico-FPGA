"""Regenera la tabla de decisiones de docs/golden_model/decisiones/README.md.

La tabla no se edita a mano: se arma a partir de los archivos DEC-*.md, así dos
computadoras que crean decisiones distintas nunca editan la misma línea.

Uso (desde models/floating_point/emulador-golden/):
    python -m scripts.indice_decisiones           # valida y reescribe la tabla
    python -m scripts.indice_decisiones --check   # solo valida; sale con 1 si hay errores
                                                  # o si la tabla está desactualizada
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "docs" / "golden_model" / "decisiones"
README = DIR / "README.md"
INICIO = "<!-- INDICE:INICIO (generado por scripts/indice_decisiones.py; no editar a mano) -->"
FIN = "<!-- INDICE:FIN -->"

RE_ARCHIVO = re.compile(r"^(DEC-(\d{4})(\d{2})(\d{2})-\d{4}-([a-z0-9]+))-[a-z0-9-]+\.md$")
RE_TITULO = re.compile(r"^# (DEC-\d{8}-\d{4}-[a-z0-9]+): (.+)$")
RE_CAMPO = re.compile(r"^- ([^:]+): (.+)$")
RE_ESTADO = re.compile(r"^(propuesta|aceptada|descartada|reemplazada por (DEC-\d{8}-\d{4}-[a-z0-9]+))$")
CAMPOS = ("Estado", "Fecha", "Máquina", "Bloque", "Afecta al equipo")


@dataclass
class Decision:
    id: str
    titulo: str
    archivo: str
    campos: dict


def leer(path: Path, errores: list[str]) -> Decision | None:
    """Lee una decisión y agrega a ``errores`` cualquier problema de formato."""
    m = RE_ARCHIVO.match(path.name)
    if not m:
        errores.append(f"{path.name}: el nombre no sigue DEC-AAAAMMDD-HHMM-<maquina>-<titulo-corto>.md")
        return None
    id_archivo, anio, mes, dia, maquina = m.groups()

    lineas = path.read_text(encoding="utf-8").splitlines()
    t = RE_TITULO.match(lineas[0]) if lineas else None
    if not t:
        errores.append(f"{path.name}: la primera línea debe ser '# DEC-AAAAMMDD-HHMM-<maquina>: <título>'")
        return None
    if t.group(1) != id_archivo:
        errores.append(f"{path.name}: el ID del título ({t.group(1)}) no coincide con el nombre del archivo")

    campos = {}
    for linea in lineas[1:]:
        if linea.startswith("## "):
            break
        c = RE_CAMPO.match(linea)
        if c:
            campos[c.group(1).strip()] = c.group(2).strip()
    for campo in CAMPOS:
        if campo not in campos:
            errores.append(f"{path.name}: falta el campo '- {campo}:'")
    if "Estado" in campos and not RE_ESTADO.match(campos["Estado"]):
        errores.append(f"{path.name}: estado inválido '{campos['Estado']}'")
    if campos.get("Fecha") not in (None, f"{anio}-{mes}-{dia}"):
        errores.append(f"{path.name}: la fecha no coincide con el nombre del archivo")
    if campos.get("Máquina") not in (None, maquina):
        errores.append(f"{path.name}: la máquina no coincide con el nombre del archivo")
    return Decision(t.group(1), t.group(2), path.name, campos)


def validar_conjunto(decisiones: list[Decision], errores: list[str]) -> None:
    ids = [d.id for d in decisiones]
    for i in sorted({i for i in ids if ids.count(i) > 1}):
        errores.append(f"ID repetido: {i} (sumar un minuto al más nuevo)")
    for d in decisiones:
        m = RE_ESTADO.match(d.campos.get("Estado", ""))
        if m and m.group(2) and m.group(2) not in ids:
            errores.append(f"{d.archivo}: reemplazada por {m.group(2)}, que no existe")


def tabla(decisiones: list[Decision]) -> str:
    filas = [
        "| ID | Título | Estado | Fecha | Máquina | Bloque | Afecta al equipo |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in decisiones:
        c = d.campos
        filas.append(
            f"| [{d.id}]({d.archivo}) | {d.titulo} | {c.get('Estado', '?')} | {c.get('Fecha', '?')} "
            f"| {c.get('Máquina', '?')} | {c.get('Bloque', '?')} | {c.get('Afecta al equipo', '?')} |"
        )
    return "\n".join(filas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="no escribe; falla si hay errores o la tabla está vieja")
    args = ap.parse_args()

    errores: list[str] = []
    decisiones = [d for p in sorted(DIR.glob("DEC-*.md")) if (d := leer(p, errores))]
    decisiones.sort(key=lambda d: d.id)
    validar_conjunto(decisiones, errores)

    texto = README.read_text(encoding="utf-8")
    if INICIO not in texto or FIN not in texto:
        errores.append(f"{README.name}: faltan los marcadores del índice")
    for e in errores:
        print(f"[error] {e}")
    if errores:
        return 1

    antes, resto = texto.split(INICIO, 1)
    _, despues = resto.split(FIN, 1)
    nuevo = f"{antes}{INICIO}\n{tabla(decisiones)}\n{FIN}{despues}"

    if args.check:
        if nuevo != texto:
            print("[error] la tabla del README está desactualizada: correr python -m scripts.indice_decisiones")
            return 1
        print(f"OK: {len(decisiones)} decisiones, índice al día")
        return 0
    README.write_text(nuevo, encoding="utf-8")
    print(f"Índice regenerado: {len(decisiones)} decisiones en {README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
