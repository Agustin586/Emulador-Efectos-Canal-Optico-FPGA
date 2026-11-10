/*******************************************************************************
* Test cruzado de cuantización (paso C3.0 de GUIA_ISI_CD.md, decisión 33)
********************************************************************************
* Author: Wanda Molina
* Date: 11/10/2026
********************************************************************************
* Estructura tomada de tp2 de HALCON (MIT License, Copyright (c) 2024 Fundacion
* Fulgor).
*
* Archivos (en data_dir, variable de entorno QC_DATA_DIR):
*   double_in.txt: un double por línea
*     -> double_<W>_<I>_<q>_<o>.txt: código de ac_fixed<W, I> desde el double,
*        q en {conv, rnd, trn} (AC_RND_CONV, AC_RND, AC_TRN) y o en {sat, wrap}
*   product_in.txt: a1 b1 a2 b2 por línea (códigos, a en <20, 2> y b en <18, 1>)
*     -> product_out.txt: a1 b1, a1 b1 + a2 b2 y a1 b1 - a2 b2, a <20, 2> con
*        AC_RND y AC_SAT
*   butterfly_in.txt: a b por línea (códigos en <20, 2>)
*     -> butterfly_out.txt: a + b y a - b con s = 0, con s = 1 y con >> 1, a
*        <20, 2> con AC_RND_CONV y AC_SAT
*   negation_in.txt: un código en <20, 2> por línea
*     -> negation_out.txt: -x a <20, 2> con AC_RND_CONV y AC_SAT
*
* Un código es el entero con signo k del valor k 2^-(W - I).
********************************************************************************/

/*******************************************************************************
* HEADERS
********************************************************************************/

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "root.hpp"

/*******************************************************************************
* AUXILIARY FUNCTIONS
********************************************************************************/

namespace
{

/* Código de W bits con signo de un ac_fixed (bits crudos, sin pasar por double) */
template<int W, int I, bool S, ac_q_mode Q, ac_o_mode O>
long long ToCode(const ac_fixed<W, I, S, Q, O> &value)
{
    return value.template slc<W>(0).to_int64();
}

/* ac_fixed<W, I> exacto a partir de su código de W bits */
template<int W, int I>
ac_fixed<W, I, true> FromCode(long long code)
{
    ac_fixed<W, I, true> value;
    value.set_slc(0, ac_int<W, true>(code));
    return value;
}

/* Desplazamiento de s bits a la derecha como cambio exacto de escala: los mismos
   W bits reinterpretados con I - s bits enteros (valor dividido por 2^s, sin
   perder bits). No se usa el operador >> de ac_fixed: conserva el tipo del
   operando y descarta los s bits menos significativos, o sea que trunca
   (ac_fixed.h, "Shift (result constrained by left operand)"). */
template<int S_BITS, int W, int I>
ac_fixed<W, I - S_BITS, true> ScaleDown(const ac_fixed<W, I, true> &value)
{
    ac_fixed<W, I - S_BITS, true> scaled;
    scaled.set_slc(0, value.template slc<W>(0));
    return scaled;
}

/* Conversión double -> ac_fixed<W, I, true, Q, O>, la misma asignación que hace
   el handler de SET (handler_ac_fixed.hpp y handler_array_ac_fixed.hpp) */
template<int W, int I, ac_q_mode Q, ac_o_mode O>
void WriteFromDoubles(const std::vector<double> &input, const std::string &path)
{
    std::ofstream file { path };
    for (size_t i { 0 }; i < input.size(); i++)
    {
        ac_fixed<W, I, true, Q, O> value { input[i] };
        file << ToCode(value) << "\n";
    }
}

template<int W, int I>
void WriteAllModes(const std::vector<double> &input, const std::string &prefix)
{
    WriteFromDoubles<W, I, AC_RND_CONV, AC_SAT>(input, prefix + "_conv_sat.txt");
    WriteFromDoubles<W, I, AC_RND, AC_SAT>(input, prefix + "_rnd_sat.txt");
    WriteFromDoubles<W, I, AC_TRN, AC_SAT>(input, prefix + "_trn_sat.txt");
    WriteFromDoubles<W, I, AC_RND_CONV, AC_WRAP>(input, prefix + "_conv_wrap.txt");
    WriteFromDoubles<W, I, AC_RND, AC_WRAP>(input, prefix + "_rnd_wrap.txt");
    WriteFromDoubles<W, I, AC_TRN, AC_WRAP>(input, prefix + "_trn_wrap.txt");
}

}

/*******************************************************************************
* CONSTRUCTOR (REFLECT MODULES, CLOCKS AND SETTINGS)
********************************************************************************/

Root::Root()
{
    /* Clocks */
    REFLECT(clk);

    /* Settings YAML */
    REFLECT_YAML(enable_log);
    REFLECT_YAML(n_iterations);
    REFLECT_YAML(fs_hz);
}

/*******************************************************************************
* CONNECTIONS
********************************************************************************/

void Root::Connect()
{
    /* Clock */
    clk.i_frequency_hz << fs_hz;
    clk.i_phase_deg.SetData(0);
    clk.i_division_factor_num.SetData(1);
    clk.i_division_factor_den.SetData(1);

    /* Command Handler Clock */
    clk_cmd_handler.i_frequency_hz << fs_hz;
    clk_cmd_handler.i_phase_deg.SetData(0);
    clk_cmd_handler.i_division_factor_num.SetData(1);
    clk_cmd_handler.i_division_factor_den.SetData(1);
};

/*******************************************************************************
* INITIALIZATION
********************************************************************************/

void Root::Init()
{
    const char *env_data_dir { std::getenv("QC_DATA_DIR") };
    if (env_data_dir != nullptr)
    {
        data_dir = env_data_dir;
    }
    if (!data_dir.empty() && data_dir.back() != '/')
    {
        data_dir += "/";
    }

    std::cout << "-- Test cruzado: datos en " << data_dir << "\n";

    RunDoubleCase();
    RunProductCase();
    RunButterflyCase();
    RunNegationCase();
}

/*******************************************************************************
* ITERATION
********************************************************************************/

void Root::Iteration()
{
}

bool Root::ContinueRunning()
{
    return GetIterationCounter() < n_iterations;
}

/*******************************************************************************
* TERMINATION
********************************************************************************/

void Root::Terminate()
{
    std::cout << "-- Test cruzado terminado\n";
}

/*******************************************************************************
* CASES
********************************************************************************/

/* Entradas de la cadena (SET con doubles): formatos de la decisión 32 */
void Root::RunDoubleCase()
{
    const std::vector<double> input { ReadDoubles("double_in.txt") };

    WriteAllModes<14, 1>(input, DataPath("double_14_1"));
    WriteAllModes<20, 2>(input, DataPath("double_20_2"));
    WriteAllModes<18, 1>(input, DataPath("double_18_1"));
    WriteAllModes<14, 2>(input, DataPath("double_14_2"));

    std::cout << "-- double: " << input.size() << " valores\n";
}

/* Camino a) productos (rotaciones no triviales y por H): producto exacto
   <38, 3>, suma o resta exacta de dos productos <39, 4> y un solo redondeo a
   <20, 2> con AC_RND y AC_SAT */
void Root::RunProductCase()
{
    using ProductFxp = decltype(DataFxp() * CoefFxp());
    using ProductSumFxp = decltype(ProductFxp() + ProductFxp());
    static_assert(ProductFxp::width == 38 && ProductFxp::i_width == 3);
    static_assert(ProductSumFxp::width == 39 && ProductSumFxp::i_width == 4);

    const std::vector<long long> codes { ReadCodes("product_in.txt") };
    if (codes.size() % 4 != 0)
    {
        throw std::runtime_error("product_in.txt: se esperan 4 códigos por línea");
    }

    std::ofstream file { DataPath("product_out.txt") };
    for (size_t i { 0 }; i < codes.size(); i += 4)
    {
        const DataFxp data_1 { FromCode<20, 2>(codes[i]) };
        const CoefFxp coef_1 { FromCode<18, 1>(codes[i + 1]) };
        const DataFxp data_2 { FromCode<20, 2>(codes[i + 2]) };
        const CoefFxp coef_2 { FromCode<18, 1>(codes[i + 3]) };

        const ProductFxp product_1 { data_1 * coef_1 };
        const ProductFxp product_2 { data_2 * coef_2 };
        const ProductSumFxp product_sum { product_1 + product_2 };
        const ProductSumFxp product_diff { product_1 - product_2 };

        const ProductOutFxp out_product { product_1 };
        const ProductOutFxp out_sum { product_sum };
        const ProductOutFxp out_diff { product_diff };

        file << ToCode(out_product) << " " << ToCode(out_sum) << " "
             << ToCode(out_diff) << "\n";
    }

    std::cout << "-- producto: " << codes.size() / 4 << " casos\n";
}

/* Camino b) mariposa: a + b y a - b exactos en <21, 3>, desplazamiento de s bits
   (cambio exacto de escala) y un redondeo a <20, 2> con AC_RND_CONV y AC_SAT.
   Se escribe además el resultado con >> 1, para documentar que trunca. */
void Root::RunButterflyCase()
{
    using ButterflySumFxp = decltype(DataFxp() + DataFxp());
    static_assert(ButterflySumFxp::width == 21 && ButterflySumFxp::i_width == 3);

    const std::vector<long long> codes { ReadCodes("butterfly_in.txt") };
    if (codes.size() % 2 != 0)
    {
        throw std::runtime_error("butterfly_in.txt: se esperan 2 códigos por línea");
    }

    const ac_int<1, false> shift_one { 1 };
    std::ofstream file { DataPath("butterfly_out.txt") };
    for (size_t i { 0 }; i < codes.size(); i += 2)
    {
        const DataFxp data_a { FromCode<20, 2>(codes[i]) };
        const DataFxp data_b { FromCode<20, 2>(codes[i + 1]) };

        const ButterflySumFxp sum { data_a + data_b };
        const ButterflySumFxp diff { data_a - data_b };

        const ButterflyOutFxp sum_s0 { ScaleDown<0>(sum) };
        const ButterflyOutFxp diff_s0 { ScaleDown<0>(diff) };
        const ButterflyOutFxp sum_s1 { ScaleDown<1>(sum) };
        const ButterflyOutFxp diff_s1 { ScaleDown<1>(diff) };
        const ButterflyOutFxp sum_shr { sum >> shift_one };
        const ButterflyOutFxp diff_shr { diff >> shift_one };

        file << ToCode(sum_s0) << " " << ToCode(diff_s0) << " "
             << ToCode(sum_s1) << " " << ToCode(diff_s1) << " "
             << ToCode(sum_shr) << " " << ToCode(diff_shr) << "\n";
    }

    std::cout << "-- mariposa: " << codes.size() / 2 << " casos\n";
}

/* Camino c) negación (por -j): -x exacto en <21, 3> y a <20, 2> con
   AC_RND_CONV y AC_SAT; -(-2) satura a 2 - q */
void Root::RunNegationCase()
{
    using NegationFxp = decltype(-DataFxp());
    static_assert(NegationFxp::width == 21 && NegationFxp::i_width == 3);

    const std::vector<long long> codes { ReadCodes("negation_in.txt") };

    std::ofstream file { DataPath("negation_out.txt") };
    for (size_t i { 0 }; i < codes.size(); i++)
    {
        const DataFxp data { FromCode<20, 2>(codes[i]) };
        const NegationFxp negated { -data };
        const ButterflyOutFxp out { negated };
        file << ToCode(out) << "\n";
    }

    std::cout << "-- negación: " << codes.size() << " casos\n";
}

/*******************************************************************************
* FILES
********************************************************************************/

std::string Root::DataPath(const std::string &file_name)
{
    return data_dir + file_name;
}

std::vector<double> Root::ReadDoubles(const std::string &file_name)
{
    std::ifstream file { DataPath(file_name) };
    if (!file)
    {
        throw std::runtime_error("No se pudo abrir " + DataPath(file_name));
    }
    std::vector<double> values;
    double value { 0.0 };
    while (file >> value)
    {
        values.push_back(value);
    }
    return values;
}

std::vector<long long> Root::ReadCodes(const std::string &file_name)
{
    std::ifstream file { DataPath(file_name) };
    if (!file)
    {
        throw std::runtime_error("No se pudo abrir " + DataPath(file_name));
    }
    std::vector<long long> codes;
    long long code { 0 };
    while (file >> code)
    {
        codes.push_back(code);
    }
    return codes;
}
