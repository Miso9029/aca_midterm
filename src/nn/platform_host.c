/* platform_host.c — 在 x86 (WSL) 上跑時用：stdio 輸出，沒有指令計數 */
#include <stdio.h>
#include <stdlib.h>
#include "platform.h"

void plat_puts(const char *s) { fputs(s, stdout); }
uint64_t plat_instret(void) { return 0; }
uint64_t plat_cycle(void) { return 0; }
void plat_exit(int code)
{
    fflush(stdout);
    exit(code);
}
