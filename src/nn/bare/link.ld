/* link.ld — Spike 預設 RAM 從 0x80000000 開始；模型 weights ~5 MB + arena ~1.5 MB，給 64 MB 足夠 */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 64M
}

SECTIONS
{
    .text : {
        *(.text.init)
        *(.text .text.*)
    } > RAM

    .rodata : ALIGN(8) {
        *(.rodata .rodata.* .srodata .srodata.*)
    } > RAM

    .data : ALIGN(8) {
        *(.data .data.* .sdata .sdata.*)
    } > RAM

    .tohost : ALIGN(64) {
        *(.tohost)
    } > RAM

    .bss : ALIGN(64) {
        __bss_start = .;
        *(.sbss .sbss.* .bss .bss.* COMMON)
        . = ALIGN(8);
        __bss_end = .;
    } > RAM

    . = ALIGN(16);
    . = . + 0x10000;          /* 64 KB stack */
    _stack_top = .;
}
