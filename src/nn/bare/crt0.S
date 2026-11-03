# crt0.S — 開機：設 stack、清 .bss、呼叫 main、用回傳值結束
    .section .text.init
    .globl _start
_start:
    la      sp, _stack_top

    la      t0, __bss_start
    la      t1, __bss_end
1:  bgeu    t0, t1, 2f
    sd      zero, 0(t0)
    addi    t0, t0, 8
    j       1b
2:
    call    main
    call    plat_exit          # a0 = main 的回傳值
3:  j       3b
