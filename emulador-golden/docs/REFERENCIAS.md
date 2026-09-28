# Referencias

Lista inicial encontrada con una búsqueda web. Leerlas antes de apoyarse en ellas: no están revisadas a fondo.

## Punto de partida general

- **OptiCommPy** (Python, código abierto): simulación de comunicaciones ópticas con modulaciones PAM/QAM, propagación en fibra y bloques DSP de receptor. https://opticommpy.readthedocs.io/ — artículo: https://joss.theoj.org/papers/10.21105/joss.06600
- **Fiber-on-Chip** (Börjeson y Larsson-Edefors, J. Lightwave Technol., 2023): el enfoque que sigue esta tesis. https://research.chalmers.se/publication/531909/file/531909_Fulltext.pdf
- **CHOICE** (código VHDL de Chalmers): https://www.cse.chalmers.se/~erikbor/choice/ — sus modelos MATLAB no están publicados.

## Modelado de cada efecto

| Efecto | Dónde mirar |
|---|---|
| ISI, offset de fase | No requieren referencia especial: convolución con un FIR y multiplicación por un fasor |
| AWGN | Ruido gaussiano con potencia fijada por la SNR. Definir antes qué SNR se usa (ver README) |
| Dispersión cromática | Función de transferencia y aplicación en frecuencia: https://arxiv.org/pdf/1705.05284 |
| PMD | Introducción al emulador ideal de PMD: https://www.lightwaveonline.com/network-design/dwdm-roadm/article/16668501/understanding-polarizationmode-dispersion. Khosravani et al., emuladores con conectores rotables: https://www2.umbc.edu/photonics/publications/PdfPapers/PAJ137.pdf. Noé y Koch, emuladores de PMD para transmisión coherente: arXiv 1903.05248 |
| Ruido de fase | Modelo de Wiener del ruido de fase del láser: https://arxiv.org/pdf/2503.13199 |
| Error de timing | Filtro de retardo fraccionario e interpolación: buscar "Fractional Delay Filters Using Farrow Structures" (MathWorks) |

## Para más adelante (hardware)

Estas referencias son del paso de punto fijo y RTL; no hacen falta todavía.

- **Generadores de ruido gaussiano en hardware:** Lee et al., Box-Muller y análisis de error: https://cel.eng.sydney.edu.au/wp-content/uploads/2019/01/bm_tc06.pdf. Método de inversión (ICDF) con aproximación polinómica por tramos: https://aplicat.upv.es/exploraupv/ficha-publicacion/publicacion/231273. Alimohammad et al.: https://amir.sdsu.edu/Alimohammad08a.pdf
- **Dispersión cromática en FPGA:** diseño de filtros FIR de compensación: https://arxiv.org/pdf/2308.05747
- **Retardo fraccionario en FPGA:** estructura Farrow: https://sjee.ftn.kg.ac.rs/index.php/sjee/article/download/415/383
- **PMD digital variable en el tiempo:** Kan et al., "Digital emulation of time-varying PMD for real-time DSP evaluations" (Chalmers, 2021): buscar por título
