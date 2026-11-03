// tests/test_xqmac.c
// Verification tests for the Xqmac8 custom instructions on Spike.
//   qmac8    : rd = sext32( int32(rd) + sum_{i=0..7} int8(rs1.byte[i]) * int8(rs2.byte[i]) )
//   qrequant : rd = sext32( clamp( MultiplyByQuantizedMultiplier(rs1, mult, shift) + out_zp ) )
//              rs2 = act_max[63:56] | act_min[55:48] | out_zp[47:40] | shift[39:32] | mult[31:0]
// Each instruction result (full 64-bit rd) is compared with an independent C reference.
// main() returns the total number of failures (0 = all pass).

#include <stdint.h>

/* =================== HTIF: talk to Spike through tohost =================== */
volatile uint64_t tohost   __attribute__((section(".tohost"), aligned(8)));
volatile uint64_t fromhost __attribute__((section(".tohost"), aligned(8)));

static void putch(char c) {
    while (tohost) ;                                      /* wait until Spike took the previous command */
    if (fromhost) fromhost = 0;                           /* drop any reply */
    tohost = (1ULL << 56) | (1ULL << 48) | (uint8_t)c;   /* device 1 (console), cmd 1 (write) */
}

void plat_exit(int code) {                                /* called by crt0.S after main */
    while (tohost) ;                                      /* let the last character finish */
    tohost = ((uint64_t)code << 1) | 1;
    while (1) ;
}
/* ========================================================================== */

static void put_str(const char *s) { while (*s) putch(*s++); }

static void put_dec(int64_t v) {
    char buf[24];
    int n = 0;
    uint64_t u;
    if (v < 0) { putch('-'); u = (uint64_t)(-(v + 1)) + 1; }
    else       { u = (uint64_t)v; }
    do { buf[n++] = (char)('0' + u % 10); u /= 10; } while (u);
    while (n) putch(buf[--n]);
}

static void put_hex(uint64_t v) {
    put_str("0x");
    for (int i = 15; i >= 0; i--) putch("0123456789abcdef"[(v >> (4 * i)) & 0xf]);
}

/* ---------------- custom instructions (custom-0, opcode 0x0b) ---------------- */
static inline int64_t qmac8(int64_t acc, uint64_t a, uint64_t b) {
    asm volatile (".insn r 0x0b, 0, 0, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(b));
    return acc;
}

static inline int64_t qrequant(int64_t acc, uint64_t p) {
    int64_t r;
    asm volatile (".insn r 0x0b, 1, 0, %0, %1, %2" : "=r"(r) : "r"(acc), "r"(p));
    return r;
}

/* ---------------- software references ---------------- */
static int64_t ref_qmac8(int64_t acc, uint64_t a, uint64_t b) {
    int32_t s = (int32_t)acc;                 /* only low 32 bits of rd are used */
    for (int i = 0; i < 8; i++) {
        int32_t x = (int8_t)(a >> (8 * i));
        int32_t w = (int8_t)(b >> (8 * i));
        s += x * w;
    }
    return (int64_t)s;                        /* sign-extended to 64 bits */
}

/* TFLite SaturatingRoundingDoublingHighMul */
static int32_t ref_srdhm(int32_t a, int32_t b) {
    if (a == INT32_MIN && b == INT32_MIN) return INT32_MAX;
    int64_t ab = (int64_t)a * (int64_t)b;
    int64_t nudge = (ab >= 0) ? (1LL << 30) : (1 - (1LL << 30));
    return (int32_t)((ab + nudge) / (1LL << 31));
}

/* TFLite RoundingDivideByPOT */
static int32_t ref_rdbpot(int32_t x, int e) {
    int32_t mask = (int32_t)((1LL << e) - 1);
    int32_t rem  = x & mask;
    int32_t thr  = (mask >> 1) + ((x < 0) ? 1 : 0);
    return (x >> e) + ((rem > thr) ? 1 : 0);
}

static int64_t ref_qrequant(int32_t acc, int32_t mult, int shift,
                            int zp, int act_min, int act_max) {
    int left  = (shift > 0) ? shift : 0;
    int right = (shift > 0) ? 0 : -shift;
    int32_t x = (int32_t)((uint32_t)acc << left);
    int32_t v = ref_rdbpot(ref_srdhm(x, mult), right) + zp;
    if (v < act_min) v = act_min;
    if (v > act_max) v = act_max;
    return (int64_t)v;
}

static uint64_t pack(int32_t mult, int shift, int zp, int act_min, int act_max) {
    return  (uint64_t)(uint32_t)mult
          | ((uint64_t)(uint8_t)shift   << 32)
          | ((uint64_t)(uint8_t)zp      << 40)
          | ((uint64_t)(uint8_t)act_min << 48)
          | ((uint64_t)(uint8_t)act_max << 56);
}

/* ---------------- test bookkeeping ---------------- */
static uint64_t rng = 0x2545F4914F6CDD1DULL;
static uint64_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

static int n_mac = 0, f_mac = 0, n_rq = 0, f_rq = 0;

static void check_mac(int64_t acc, uint64_t a, uint64_t b) {
    int64_t got = qmac8(acc, a, b), exp = ref_qmac8(acc, a, b);
    n_mac++;
    if (got != exp) {
        if (++f_mac <= 5) {
            put_str("  qmac8 FAIL acc="); put_hex((uint64_t)acc);
            put_str(" rs1="); put_hex(a); put_str(" rs2="); put_hex(b);
            put_str(" got="); put_dec(got); put_str(" exp="); put_dec(exp); putch('\n');
        }
    }
}

static void check_rq(int32_t acc, int32_t mult, int shift, int zp, int mn, int mx) {
    uint64_t p = pack(mult, shift, zp, mn, mx);
    int64_t got = qrequant((int64_t)acc, p);
    int64_t exp = ref_qrequant(acc, mult, shift, zp, mn, mx);
    n_rq++;
    if (got != exp) {
        if (++f_rq <= 5) {
            put_str("  qrequant FAIL acc="); put_dec(acc);
            put_str(" rs2="); put_hex(p);
            put_str(" got="); put_dec(got); put_str(" exp="); put_dec(exp); putch('\n');
        }
    }
}

static uint64_t bytes8(int8_t v) { return 0x0101010101010101ULL * (uint8_t)v; }

/* ---------------- qmac8 tests ---------------- */
static void test_qmac8(void) {
    /* directed: extreme values */
    check_mac(0, bytes8(-128), bytes8(-128));          /* 8 * 16384 = 131072 */
    check_mac(0, bytes8(-128), bytes8(127));
    check_mac(0, bytes8(127),  bytes8(127));
    check_mac(0, 0, bytes8(-1));
    check_mac(-1000000, bytes8(-1), bytes8(1));         /* negative accumulator */
    check_mac(0x7FFF0000LL, bytes8(1), bytes8(1));       /* large positive accumulator */
    check_mac((int64_t)0xFFFFFFFF00000005ULL, bytes8(2), bytes8(3)); /* upper 32 bits of rd ignored */
    check_mac(0, 0x0102030405060708ULL, 0x0807060504030201ULL);    /* lane order */

    /* random: accumulator kept within +-2^30 so the int32 sum never overflows */
    for (int i = 0; i < 20000; i++) {
        int64_t acc = (int64_t)(int32_t)(rnd() & 0x7FFFFFFF) - (1LL << 30);
        check_mac(acc, rnd(), rnd());
    }

    /* chained: a length-1280 dot product (largest Cin in the model) */
    for (int t = 0; t < 20; t++) {
        int64_t hw = 0, sw = 0;
        for (int k = 0; k < 1280 / 8; k++) {
            uint64_t a = rnd(), b = rnd();
            hw = qmac8(hw, a, b);
            sw = ref_qmac8(sw, a, b);
        }
        n_mac++;
        if (hw != sw && ++f_mac <= 5) {
            put_str("  qmac8 chained FAIL got="); put_dec(hw);
            put_str(" exp="); put_dec(sw); putch('\n');
        }
    }
}

/* ---------------- qrequant tests ---------------- */
static void test_qrequant(void) {
    /* directed: rounding boundaries (mult = 2^30 means x0.5), both signs */
    for (int acc = -16; acc <= 16; acc++)
        for (int sh = 0; sh >= -3; sh--)
            check_rq(acc, 1 << 30, sh, 0, -128, 127);

    /* directed: extreme accumulators */
    int32_t big[] = { INT32_MAX, INT32_MIN, INT32_MAX - 1, INT32_MIN + 1, 0, 1, -1 };
    for (int i = 0; i < 7; i++) {
        check_rq(big[i], 1518500250, -8, 0, -128, 127);
        check_rq(big[i], 1518500250, 0, 0, -128, 127);
        check_rq(big[i], 1518500250, 1, 0, -128, 127);  /* positive shift, may wrap */
        check_rq(big[i], 1518500250, -31, 0, -128, 127);
    }

    /* directed: the only saturating case of SRDHM */
    check_rq(INT32_MIN, INT32_MIN, 0, 0, -128, 127);

    /* directed: zero point and clamp */
    check_rq(100000, 1 << 30, -10, -128, -128, 127);
    check_rq(100000, 1 << 30, -10, 127, -128, 127);     /* clamps to act_max */
    check_rq(-100000, 1 << 30, -10, -128, -128, 127);   /* clamps to act_min */
    check_rq(5000, 1 << 30, -4, 0, -10, 10);            /* narrow range */
    check_rq(-5000, 1 << 30, -4, 0, -10, 10);

    /* random: mult in [2^30, 2^31), shift in [-31, 2] */
    for (int i = 0; i < 50000; i++) {
        int32_t acc = (i & 1) ? (int32_t)rnd()
                              : (int32_t)(rnd() % 2000001) - 1000000;
        int32_t mult = (int32_t)((1u << 30) + (uint32_t)(rnd() % (1u << 30)));
        int shift = (int)(rnd() % 34) - 31;
        int zp = (int8_t)rnd();
        int a = (int8_t)rnd(), b = (int8_t)rnd();
        int mn = (a < b) ? a : b, mx = (a < b) ? b : a;
        if (i % 4 == 0) { mn = -128; mx = 127; }        /* typical full int8 range */
        check_rq(acc, mult, shift, zp, mn, mx);
    }
}

int main(void) {
    put_str("== Xqmac8 verification ==\n");

    test_qmac8();
    put_str("qmac8    : "); put_dec(n_mac); put_str(" tests, ");
    put_dec(f_mac); put_str(" failed\n");

    test_qrequant();
    put_str("qrequant : "); put_dec(n_rq); put_str(" tests, ");
    put_dec(f_rq); put_str(" failed\n");

    put_str((f_mac + f_rq) ? "RESULT: FAIL\n" : "RESULT: ALL PASS\n");
    return f_mac + f_rq;
}