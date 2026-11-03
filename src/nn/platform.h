/* platform.h — 輸出、計數器、結束程式。x86 用 platform_host.c，Spike 用 bare/platform_spike.c */
#ifndef PLATFORM_H
#define PLATFORM_H
#include <stdint.h>

void     plat_puts(const char *s);   /* 印字串 */
uint64_t plat_instret(void);         /* 已執行指令數 (Spike: minstret；x86: 0) */
uint64_t plat_cycle(void);           /* cycle 數 (Spike: mcycle；x86: 0) */
void     plat_exit(int code);

/* 小工具 (bare-metal 沒有 printf) */
void print_str(const char *s);
void print_u64(uint64_t v);
void print_i64(int64_t v);
void print_hex32(uint32_t v);
void print_pad(const char *s, int width);   /* 印字串並補空白到 width */
void print_u64_pad(uint64_t v, int width);  /* 數字靠右對齊 */
void print_flush(void);

#endif
