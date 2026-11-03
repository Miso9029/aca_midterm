/*
 * platform_spike.c — Spike bare-metal 支援 (HTIF，跟 riscv-tests 同一套做法)
 *   印字：透過 tohost/fromhost 呼叫 Spike 的 SYS_write
 *   結束：tohost = (code << 1) | 1
 */
#include <stdint.h>
#include "../platform.h"

volatile uint64_t tohost __attribute__((section(".tohost"), aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"), aligned(64)));

#define SYS_write 64

static uint64_t htif_syscall(uint64_t which, uint64_t a0, uint64_t a1, uint64_t a2)
{
    volatile uint64_t magic[8] __attribute__((aligned(64)));
    magic[0] = which;
    magic[1] = a0;
    magic[2] = a1;
    magic[3] = a2;
    __sync_synchronize();
    tohost = (uint64_t)(uintptr_t)magic;
    while (fromhost == 0)
        ;
    fromhost = 0;
    __sync_synchronize();
    return magic[0];
}

void plat_puts(const char *s)
{
    uint64_t n = 0;
    while (s[n])
        n++;
    htif_syscall(SYS_write, 1, (uint64_t)(uintptr_t)s, n);
}

uint64_t plat_instret(void)
{
    uint64_t v;
    __asm__ volatile("csrr %0, minstret" : "=r"(v));
    return v;
}

uint64_t plat_cycle(void)
{
    uint64_t v;
    __asm__ volatile("csrr %0, mcycle" : "=r"(v));
    return v;
}

void plat_exit(int code)
{
    tohost = ((uint64_t)code << 1) | 1;
    for (;;)
        ;
}

/* gcc 可能自己產生 memcpy/memset 呼叫，bare-metal 沒有 libc，要自己提供 */
void *memcpy(void *d, const void *s, unsigned long n)
{
    char *dp = d;
    const char *sp = s;
    while (n--)
        *dp++ = *sp++;
    return d;
}

void *memset(void *d, int c, unsigned long n)
{
    char *dp = d;
    while (n--)
        *dp++ = (char)c;
    return d;
}
