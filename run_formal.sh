#!/bin/bash
set -u
cd "$(dirname "$0")"; ROOT=$(cd .. && pwd)
SBY=${SBY:-$ROOT/build/tools/sby/sbysrc/sby.py}
SV2V=${SV2V:-$ROOT/build/sv2v-Linux/sv2v}
cp "$ROOT/rtl/softmax_hp_lut.svh" .

flat() {
  $SV2V -DFORMAL --exclude=Assert -I"$1" "$1/pe_db.sv" "$1/attn_core_db.sv" "$1/attn_tiled.sv" stubs.sv > tiled_flat.v
}
prove() {
  local t=${1:-prove}
  rm -rf tiled_$t
  python3 "$SBY" -f tiled.sby $t > $t.log 2>&1
  grep -q "DONE (PASS" $t.log && echo PASS || echo FAIL
}

flat "$ROOT/rtl"
echo "baseline RTL N=4 MAXT=2: $(prove prove)"  | tee formal_results.run.txt
[ "${N8:-0}" = 1 ] && echo "baseline RTL N=8 MAXT=4: $(prove prove8)" | tee -a formal_results.run.txt

mutate() {
  M=$(mktemp -d); cp "$ROOT"/rtl/*.sv "$ROOT"/rtl/*.svh "$M/"
  python3 - "$M/attn_tiled.sv" "$2" "$3" <<'EOF'
import sys; p, o, n = sys.argv[1:]; s = open(p).read(); assert o in s, o; open(p, 'w').write(s.replace(o, n, 1))
EOF
  flat "$M"; r=$(prove)
  [ "$r" = FAIL ] && v="caught" || v="MISSED"
  echo "mutant $1: proof $r -> $v" | tee -a formal_results.run.txt
  rm -rf "$M"
}
mutate "M1 swap token on 2nd row  " "iss_swap <= (rr == 0);" "iss_swap <= (rr == 1);"
mutate "M2 no prob-ready check     " "(!cur_p || prdy[cur_t[0]][rr[CW-1:0]]);" "1'b1;"
mutate "M3 issue before weights    " "((rr == 0) ? (sh_full && sh_slot == s) : 1'b1) &&" "1'b1 &&"
mutate "M4 prob row never released" "prdy[cur_t[0]][rr[CW-1:0]] <= 1'b0;" ""
flat "$ROOT/rtl"
