# DEC-20260928-2035-asus: Hardware RFSoC Zynq UltraScale+ y FFT radix-2² con P parametrizable
- Estado: aceptada
- Fecha: 2026-09-28
- Máquina: asus
- Bloque: común
- Afecta al equipo: sí

## Contexto
El PDF de fundamentos (§4.8.3) evaluaba una Artix-7 (Arty A7) con FFT SDF de P = 1. La plataforma pasa a ser un RFSoC, con muchos más bloques DSP. La arquitectura totalmente paralela de Jiang et al. (unos 4370 DSP solo para la CD) sigue sin entrar.

## Decisión
- **Plataforma:** RFSoC Zynq UltraScale+. El modelo exacto no está confirmado.
- **FFT de la CD:** radix-2², segmentada, con paralelismo P parametrizable (SDF con P = 1, o MDC con P > 1).
- **Parámetros provisorios:** P, f_clk y N_max viven en `HardwareParams` (separado de `Scenario`), marcados como provisorios. Valores provisorios: P = 1, f_clk sin definir, N_max = 2048.
- **Reemplaza** la suposición de Artix-7 del PDF de fundamentos.

## Alternativas consideradas
- Artix-7 con P = 1 (propuesta del PDF): reemplazada por esta decisión.
- FFT radix-4 pura: exige L potencia de 4 (512 y 2048 no lo son).

## Consecuencias (código, tests, notebooks, hardware)
- Golden en flotante: no cambia.
- Modelo bit a bit: tiene que replicar las etapas radix-2² y el escalado por etapa, así que **no puede cerrarse sin confirmar el dispositivo y P** (paso C1 de la guía).
- Tasa efectiva del bloque de CD: R_CD = P·f_clk/2.
- Pendiente: una decisión nueva reemplazará los valores provisorios.

## Referencias
- referencia_guia_ISI_CD.md B.8; PDF Fundamentos §4.8; AMD DS889; He y Torkelson (1996).
