/* print.c — 不依賴 libc 的簡易輸出 (先存在 buffer，滿了或 flush 才送出，減少 HTIF 呼叫次數) */
#include "platform.h"

static char buf[256];
static int pos;

void print_flush(void)
{
    if (pos == 0)
        return;
    buf[pos] = 0;
    plat_puts(buf);
    pos = 0;
}

static void putc_buf(char c)
{
    if (pos >= (int)sizeof(buf) - 1)
        print_flush();
    buf[pos++] = c;
    if (c == '\n')
        print_flush();
}

void print_str(const char *s)
{
    while (*s)
        putc_buf(*s++);
}

static int u64_to_str(uint64_t v, char *tmp)
{
    int n = 0;
    do {
        tmp[n++] = (char)('0' + v % 10);
        v /= 10;
    } while (v);
    return n;   /* 反向存放 */
}

void print_u64(uint64_t v)
{
    char tmp[24];
    int n = u64_to_str(v, tmp);
    while (n)
        putc_buf(tmp[--n]);
}

void print_i64(int64_t v)
{
    if (v < 0) {
        putc_buf('-');
        print_u64((uint64_t)(-(v + 1)) + 1);
    } else {
        print_u64((uint64_t)v);
    }
}

void print_hex32(uint32_t v)
{
    static const char hx[] = "0123456789abcdef";
    print_str("0x");
    for (int i = 28; i >= 0; i -= 4)
        putc_buf(hx[(v >> i) & 0xf]);
}

void print_pad(const char *s, int width)
{
    int n = 0;
    while (s[n]) {
        putc_buf(s[n]);
        n++;
    }
    while (n++ < width)
        putc_buf(' ');
}

void print_u64_pad(uint64_t v, int width)
{
    char tmp[24];
    int n = u64_to_str(v, tmp);
    for (int i = n; i < width; i++)
        putc_buf(' ');
    while (n)
        putc_buf(tmp[--n]);
}
