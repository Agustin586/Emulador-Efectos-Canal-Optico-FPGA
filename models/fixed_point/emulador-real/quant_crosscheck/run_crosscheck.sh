#!/usr/bin/env bash
################################################################################
# Corrida del test cruzado de cuantización en el contenedor de HALCON (C3.0)
#
# Uso: run_crosscheck.sh <directorio_de_datos>
#
# - <directorio_de_datos>: tiene los *_in.txt; el simulador escribe ahí los
#   *_out.txt y double_*.txt (ver src/root/root.cpp).
# - HALCON_DIR: copia de HALCON con los submódulos de lib/ inicializados
#   (por defecto /home/wanda/halcon/tutorials, decisión 28).
# - HAWK_IMAGE: imagen del contenedor (por defecto hawk_workspace).
#
# No escribe nada en el repo ni en HALCON_DIR:
# - lib/ de HALCON se monta de solo lectura en /app/lib;
# - este proyecto se monta de solo lectura y se copia a /tmp/work dentro del
#   contenedor, porque main.py se reescribe a sí mismo y la compilación crea
#   build/ y run/;
# - en /tmp/work, ../lib apunta a /app/lib, así que core_dir = "../lib/core/"
#   de main.py sirve sin cambios, y PYTHONPATH (/app/lib/tools/python) es el
#   de la imagen.
# Corre con el usuario del host, para que los archivos de salida no queden
# como root.
################################################################################

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "uso: $0 <directorio_de_datos>" >&2
    exit 2
fi

data_dir="$(cd "$1" && pwd)"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
halcon_dir="${HALCON_DIR:-/home/wanda/halcon/tutorials}"
image="${HAWK_IMAGE:-hawk_workspace}"

if [ ! -f "$halcon_dir/lib/core/inc/ac_fixed.h" ]; then
    echo "no se encontró HALCON en $halcon_dir (lib/core/inc/ac_fixed.h)" >&2
    exit 3
fi

rm -f "$data_dir"/*_out.txt "$data_dir"/double_*_*.txt

docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e QC_DATA_DIR=/data \
    -v "$halcon_dir/lib":/app/lib:ro \
    -v "$project_dir":/src/quant_crosscheck:ro \
    -v "$data_dir":/data \
    "$image" \
    bash -c 'set -e
        mkdir -p /tmp/work
        cp -r /src/quant_crosscheck /tmp/work/quant_crosscheck
        rm -rf /tmp/work/quant_crosscheck/build /tmp/work/quant_crosscheck/run
        ln -s /app/lib /tmp/work/lib
        cd /tmp/work/quant_crosscheck
        python3 main.py -f -r'

# simulator_handler usa os.system y no propaga errores: se verifica la salida
for f in product_out.txt butterfly_out.txt negation_out.txt double_14_1_conv_sat.txt; do
    if [ ! -s "$data_dir/$f" ]; then
        echo "falta la salida $f: falló la compilación o la corrida" >&2
        exit 1
    fi
done
