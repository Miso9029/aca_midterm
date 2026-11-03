#!/usr/bin/env python3
"""
mix_report.py — instruction mix from a Spike PC histogram (spike -g)

Usage (in ~/project/src/nn):
    spike -g --isa=rv64imc build_x2/model.elf > x2_run_g.txt 2>&1
    python ../../scripts/mix_report.py build_x2/model.elf x2_run_g.txt ../../results/x2

Output:
    <prefix>_mix.txt        "count mnemonic" per instruction, sorted
    <prefix>_mix_class.txt  alu / load / branch/jump / mul/div / store / csr / custom

How it works:
    1. Spike -g prints "<pc-hex> <count>" for every executed PC.
    2. objdump -d gives the mnemonic at every PC of the ELF.
    3. Counts are summed per mnemonic, then per class.
    Custom-0 instructions (opcode 0x0B) are decoded from the raw word:
    funct3 = 0 -> qmac8, funct3 = 1 -> qrequant (objdump does not know them).
    PCs outside the ELF (Spike boot ROM at 0x1000) appear with an empty name.
"""
import re
import subprocess
import sys
from collections import Counter

OBJDUMP = "riscv64-unknown-elf-objdump"

LOAD = {"lb", "lbu", "lh", "lhu", "lw", "lwu", "ld"}
STORE = {"sb", "sh", "sw", "sd"}
MULDIV = {"mul", "mulh", "mulhu", "mulhsu", "mulw", "div", "divu", "divw", "divuw",
          "rem", "remu", "remw", "remuw"}
BRANCH = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez", "bltz", "bgez",
          "blez", "bgtz", "bgt", "ble", "bgtu", "bleu", "j", "jal", "jr", "jalr", "ret"}
CSR = {"csrr", "csrw", "csrs", "csrc", "csrrw", "csrrs", "csrrc", "csrrwi", "csrrsi", "csrrci"}
CUSTOM = {"qmac8", "qrequant"}
CUSTOM0 = {0: "qmac8", 1: "qrequant"}


def disassemble(elf):
    """pc -> mnemonic"""
    out = subprocess.run([OBJDUMP, "-d", elf], capture_output=True, text=True, check=True).stdout
    line_re = re.compile(r"^\s*([0-9a-f]+):\s+([0-9a-f]+)\s+(\S+)")
    table = {}
    for line in out.splitlines():
        m = line_re.match(line)
        if not m:
            continue
        pc, word, name = int(m.group(1), 16), int(m.group(2), 16), m.group(3)
        if len(m.group(2)) == 8 and (word & 0x7F) == 0x0B:      # custom-0
            name = CUSTOM0.get((word >> 12) & 7, "custom0")
        table[pc] = name
    return table


def read_hist(path):
    """pc -> count; ignores every line that is not '<hex> <int>'"""
    hist_re = re.compile(r"^([0-9a-f]+)\s+(\d+)\s*$")
    hist = Counter()
    with open(path) as f:
        for line in f:
            m = hist_re.match(line)
            if m:
                hist[int(m.group(1), 16)] += int(m.group(2))
    return hist


def classify(name):
    if name in LOAD:
        return "load"
    if name in STORE:
        return "store"
    if name in MULDIV:
        return "mul/div"
    if name in BRANCH:
        return "branch/jump"
    if name in CSR:
        return "csr"
    if name in CUSTOM or name == "custom0":
        return "custom"
    return "alu"


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    elf, hist_path, prefix = sys.argv[1:]

    table = disassemble(elf)
    hist = read_hist(hist_path)
    if not hist:
        sys.exit(f"no '<pc> <count>' lines found in {hist_path} (was spike run with -g?)")

    mix = Counter()
    for pc, n in hist.items():
        mix[table.get(pc, "")] += n

    with open(f"{prefix}_mix.txt", "w") as f:
        for name, n in mix.most_common():
            f.write(f"{n} {name}\n")

    cls = Counter()
    for name, n in mix.items():
        cls[classify(name)] += n
    total = sum(cls.values())
    with open(f"{prefix}_mix_class.txt", "w") as f:
        f.write(f"{'total':<15}{total:>10}\n")
        for name, n in cls.most_common():
            f.write(f"{name:<15}{n:>10}{100 * n / total:>7.1f}%\n")

    print(f"total {total:,} instructions")
    print(f"-> {prefix}_mix.txt, {prefix}_mix_class.txt")


if __name__ == "__main__":
    main()