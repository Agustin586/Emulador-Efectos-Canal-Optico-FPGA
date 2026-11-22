# Caracterización de costos por pieza (D2.2). Uso:
#   /tools/Xilinx/Vivado/2024.2/bin/vivado -mode batch -nojournal -nolog -source run.tcl
# Síntesis + place & route fuera de contexto, xczu48dr-fsvg1517-2-e, reloj de 2,667 ns (375 MHz).
set part xczu48dr-fsvg1517-2-e
set here [file dirname [file normalize [info script]]]
set cases {
    {add_W14   add_reg   {W 14}}
    {add_W17   add_reg   {W 17}}
    {add_W20   add_reg   {W 20}}
    {bfly_W18  bfly_r2   {W 18}}
    {rot_m16   rot_const {WD 18 WC 16 C 32758 S -804}}
    {rot_m432  rot_const {WD 18 WC 16 C 25833 S -20160}}
    {rot_m2352 rot_const {WD 18 WC 16 C -29269 S 14733}}
    {rot_w8    rot_w8    {WD 18 WC 16 K 23170}}
    {cmul3     cmul3     {WD 18 WC 18}}
    {rotdsp_m2352      rot_const_dsp {WD 18 WC 16 C -29269 S 14733}}
    {rotdsp_m432_W17_12 rot_const_dsp {WD 17 WC 12 C 1615 S -1260}}
}
proc util {rpt label} {
    if {[regexp -line [format {^\|\s*%s\s*\|\s*(\d+)} $label] $rpt -> v]} { return $v }
    return -1
}
set out [open [file join $here resultados.csv] w]
puts $out "caso,modulo,parametros,LUT,FF,DSP,CARRY8,WNS_ns"
foreach c $cases {
    lassign $c name top params
    close_project -quiet
    create_project -in_memory -part $part
    read_verilog [file join $here rtl $top.v]
    set gen {}
    foreach {k v} $params { lappend gen -generic "$k=$v" }
    synth_design -top $top -part $part -mode out_of_context {*}$gen
    create_clock -period 2.667 -name clk [get_ports clk]
    opt_design
    place_design
    route_design
    # conteo desde el informe de utilización después de rutear
    set rpt [report_utilization -return_string]
    set lut [util $rpt {CLB LUTs\*?}]
    set ff  [util $rpt {CLB Registers}]
    set dsp [util $rpt {DSPs}]
    set c8  [util $rpt {CARRY8}]
    set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
    report_utilization -file [file join $here reportes $name.util.rpt]
    report_timing_summary -file [file join $here reportes $name.timing.rpt]
    puts $out "$name,$top,\"$params\",$lut,$ff,$dsp,$c8,$wns"
    flush $out
}
close $out
