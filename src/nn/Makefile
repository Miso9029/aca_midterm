# Makefile — 在 ~/project/src/nn 底下執行
#   make host    在 x86 (WSL) 上編譯並執行，先確認結果正確 (快)
#   make spike   用 riscv64 gcc 編成 rv64imc 執行檔，丟到 Spike 上跑
#   make clean

MODEL_DIR = ../model
BUILD    ?= build
SRCS      = main.c kernels.c kernels_xqmac.c print.c $(MODEL_DIR)/model_data.c

# make spike XQMAC=1 → X1 (1x1 conv 用 qmac8)
# make spike XQMAC=2 → X2 (1x1 conv 用 qmac8 + qrequant)；不加 → B0
ifeq ($(XQMAC),1)
  DEFS  = -DUSE_XQMAC
  BUILD = build_x1
else ifeq ($(XQMAC),2)
  DEFS  = -DUSE_XQMAC -DUSE_QREQUANT
  BUILD = build_x2
else
  DEFS  =
endif
INC       = -I. -I$(MODEL_DIR)

# --- x86 ---
HOST_CC     = gcc
HOST_CFLAGS = -O2 -Wall -Wextra -Wno-unused-parameter $(INC) $(DEFS)

# --- RISC-V ---
RV_CC     = riscv64-unknown-elf-gcc
# zicsr 只是讓組譯器接受 csrr (讀指令計數器)，不是新的運算指令
RV_MARCH  = rv64imc_zicsr
RV_CFLAGS = -O2 -march=$(RV_MARCH) -mabi=lp64 -mcmodel=medany \
            -ffreestanding -fno-builtin -nostdlib -nostartfiles \
            -Wall -Wno-unused-parameter $(INC) $(DEFS)
SPIKE     = spike
SPIKE_ISA = rv64imc

.PHONY: host spike run-host run-spike clean dump size test-xqmac bench

host: $(BUILD)/model_host
	./$(BUILD)/model_host

$(BUILD)/model_host: $(SRCS) platform_host.c *.h Makefile
	@mkdir -p $(BUILD)
	$(HOST_CC) $(HOST_CFLAGS) $(SRCS) platform_host.c -o $@

spike: $(BUILD)/model.elf
	$(SPIKE) --isa=$(SPIKE_ISA) $(BUILD)/model.elf

$(BUILD)/model.elf: $(SRCS) bare/crt0.S bare/platform_spike.c bare/link.ld *.h Makefile
	@mkdir -p $(BUILD)
	$(RV_CC) $(RV_CFLAGS) -T bare/link.ld bare/crt0.S bare/platform_spike.c $(SRCS) -lgcc -o $@

# 反組譯，看 kernel 的內迴圈長什麼樣子
dump: $(BUILD)/model.elf
	riscv64-unknown-elf-objdump -d $(BUILD)/model.elf > $(BUILD)/model.dis
	@echo "-> $(BUILD)/model.dis"

# 程式碼大小 (作業要求要報告 code size)
size: $(BUILD)/model.elf
	riscv64-unknown-elf-size $(BUILD)/model.elf

# --- 自訂指令驗證 (qmac8 + qrequant，約 7 萬組，對照 TFLite 參考算法) ---
# test_xqmac.c 自帶 tohost / plat_exit，所以用 verification/ 自己的 crt0.S 和 link.ld
test-xqmac:
	@mkdir -p $(BUILD)
	$(RV_CC) $(RV_CFLAGS) -T ../verification/link.ld ../verification/crt0.S \
		../verification/test_xqmac.c -o $(BUILD)/test_xqmac.elf
	$(SPIKE) --isa=$(SPIKE_ISA) $(BUILD)/test_xqmac.elf

# --- 一次跑 B0 / X1 / X2，結果存到 ../../results/ ---
bench:
	@mkdir -p ../../results
	$(MAKE) spike           | tee ../../results/b0_spike.txt
	$(MAKE) spike XQMAC=1   | tee ../../results/x1_spike.txt
	$(MAKE) spike XQMAC=2   | tee ../../results/x2_spike.txt

clean:
	rm -rf build build_x1 build_x2
