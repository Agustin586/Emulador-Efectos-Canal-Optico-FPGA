/*******************************************************************************
* Test cruzado de cuantización (paso C3.0 de GUIA_ISI_CD.md, decisión 33)
********************************************************************************
* Author: Wanda Molina
* Date: 11/10/2026
********************************************************************************
* Estructura tomada de tp2 de HALCON (MIT License, Copyright (c) 2024 Fundacion
* Fulgor).
*
* Lee enteros y doubles de archivos de texto, los lleva a ac_fixed con los
* caminos de cuantización del bloque de CD (decisión 32) y escribe los enteros
* resultantes (código de W bits con signo). El test de Python
* tests/test_quant_crosscheck.py del golden compara esos enteros con los de
* emulador.quantization (quantize y to_int).
********************************************************************************/

#pragma once

/*******************************************************************************
* HEADERS
********************************************************************************/

#include <string>
#include <vector>
#include "halcon.hpp"
#include "ac_fixed.h"

/*******************************************************************************
* MODULE
********************************************************************************/

class Root : public Simulator
{
public:

    Root();

private:

    /* Formatos exactos de la decisión 32 (sin redondeo ni saturación propios):
        DataFxp: datos internos de la CD, ac_fixed<20, 2>
        CoefFxp: twiddles y H, ac_fixed<18, 1>
    */
    using DataFxp = ac_fixed<20, 2, true>;
    using CoefFxp = ac_fixed<18, 1, true>;

    /* Salidas de cada camino:
        ProductOutFxp: productos y sumas de productos, AC_RND y AC_SAT
        ButterflyOutFxp: mariposas y negación, AC_RND_CONV y AC_SAT
    */
    using ProductOutFxp = ac_fixed<20, 2, true, AC_RND, AC_SAT>;
    using ButterflyOutFxp = ac_fixed<20, 2, true, AC_RND_CONV, AC_SAT>;

    /* User methods */
    void Init() override;
    void Connect() override;
    void Iteration() override;
    bool ContinueRunning() override;
    void Terminate() override;

    /* Casos del test cruzado */
    void RunDoubleCase();
    void RunProductCase();
    void RunButterflyCase();
    void RunNegationCase();

    /* Lectura y escritura de archivos */
    std::vector<double> ReadDoubles(const std::string &file_name);
    std::vector<long long> ReadCodes(const std::string &file_name);
    std::string DataPath(const std::string &file_name);

    /* Variables */
    bool enable_log { false };
    unsigned int n_iterations { 1 };
    long double fs_hz { 1.0 };
    std::string data_dir { "../data/" };

    /* Clocks */
    Clock clk;
};
