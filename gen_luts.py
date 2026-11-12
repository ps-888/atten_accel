#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from attn_model import HP, exp2_lut, seed_lut

def main():
    e, s = exp2_lut(), seed_lut()
    out = os.path.join(os.path.dirname(__file__), '..', 'rtl', 'softmax_hp_lut.svh')
    with open(out, 'w') as f:
        f.write(f"localparam int LUT_XF = {HP['XF']}, LUT_EXP_FRAC = {HP['EXP_FRAC']}, "
                f"LUT_SEED_BITS = {HP['SEED_BITS']}, LUT_R = {HP['R']};\n")
        f.write("function automatic logic [EXP_FRAC:0] exp2_frac_lut(input logic [XF-1:0] f);\n  case (f)\n")
        for i, v in enumerate(e): f.write(f"    {i}: exp2_frac_lut = {v};\n")
        f.write("    default: exp2_frac_lut = '0;\n  endcase\nendfunction\n")
        f.write("function automatic logic [R:0] recip_seed_lut(input logic [SEED_BITS-1:0] m);\n  case (m)\n")
        for i, v in enumerate(s): f.write(f"    {i}: recip_seed_lut = {v};\n")
        f.write("    default: recip_seed_lut = '0;\n  endcase\nendfunction\n")
    print("wrote", os.path.normpath(out))

if __name__ == '__main__':
    main()
