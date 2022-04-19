#include <err.h>
#include <errno.h>
#include <immintrin.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/signal.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <xbyak/xbyak_util.h>

#include <cstddef>
#include <memory>

#include "xbyak/xbyak.h"

#define fatal_error(msg, ...) err(1, "[FAIL]\t" msg, ##__VA_ARGS__)
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)

#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023
#define TILE_M 16    // Number of rows in an A or C tile
#define TILE_K 32    // Number of columns in an A tile or rows in a B tile
#define TILE_N 16    // Number of columns in a B or C tile
#define KPACK 2      // Vertical K packing into dword
#define MZ 64        // (M / MT)
#define NUM_M 4      // (MZ / TILE_N)

#define ARG(x, m, n) ptr[rsp + x * 32 * 16 + m * 32 + n]
static void request_perm_xtile_data() {
  unsigned long bitmask;
  long rc;

  rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
  if (rc) fatal_error("XTILE_DATA request failed: %ld", rc);

  rc = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
  if (rc) fatal_error("prctl(ARCH_GET_XCOMP_PERM) error: %ld", rc);

  if (bitmask & XFEATURE_MASK_XTILE)
    printf("ARCH_REQ_XCOMP_PERM XTILE_DATA successful.\n");
}


typedef unsigned short __bfloat16_t;
typedef __bfloat16_t src_t;
typedef float dst_t;
typedef int64_t dim_t;

// Tile configure structure
struct tileconfig_t {
    uint8_t  palette_id;
    uint8_t  reserved[15];
    uint16_t colb[16];
    uint8_t  rows[16];
} tc = {0};

void configure_tiles() {
    // Filling tile configure structure. Could be done offline.
    tc.palette_id = 1;
    // Configure C tiles
    for (int t = 0; t < 4; ++t) {
        tc.rows[t] = TILE_M;
        tc.colb[t] = TILE_N * sizeof(dst_t);
    }
    // Configure A tiles
    for (int t = 4; t < 6; ++t) {
        tc.rows[t] = TILE_M;
        tc.colb[t] = TILE_K * sizeof(src_t);
    }
    // Configure B tile. B effectively has 64 rows and 16 columns.
    for (int t = 6; t < 8; ++t) {
        tc.rows[t] = TILE_K / KPACK;
        tc.colb[t] = TILE_N * KPACK * sizeof(src_t);
    }
    _tile_loadconfig(&tc);
}


constexpr Xbyak::Operand::Code abi_save_gpr_regs[] = {
    Xbyak::Operand::RBX, Xbyak::Operand::RBP, Xbyak::Operand::R12,
    Xbyak::Operand::R13, Xbyak::Operand::R14, Xbyak::Operand::R15,
#ifdef _WIN32
    Xbyak::Operand::RDI, Xbyak::Operand::RSI,
#endif
};

typedef struct amx_params {
  dim_t shape[1];
  dim_t blocksize[1];
  dim_t blocks_per_group;
  dim_t nnz_group;
  dim_t nrowptr;
  dim_t* colidxs;
  dim_t* group_rowptr;
} amx_params_t;

template <typename T, typename Q>
struct amx_inputs {
  T* weight;
  T* src;
  Q* dst;
  dim_t bs;
};
typedef amx_inputs<src_t, dst_t> amx_inputs_t;

#define GET_OFF(field) offsetof(amx_inputs_t, field)

struct gemm_kernel : Xbyak::CodeGenerator {
  gemm_kernel(amx_params_t params_) {
    // Init parameters and check can we generate kernel or not:
    params = params_;
    N = params.shape[0];
    K = params.shape[1];
    nnz_group = params.nnz_group;
    nrowptr = params.nrowptr;
    colidxs = params.colidxs;
    group_rowptr = params.group_rowptr;
  }

  template <typename... kernel_args_t>
  void operator()(kernel_args_t... args) const {
    using jit_kernel_func_t = void (*)(const kernel_args_t... args);
    auto* fptr = (jit_kernel_func_t)jit_ker_;
    (*fptr)(std::forward<kernel_args_t>(args)...);
  }

  virtual bool create_kernel() {
    generate();
    jit_ker_ = getCode();
    return (jit_ker_) ? true : false;
  }

  void read_inputs() {
    mov(reg_weight, ptr[reg_param + GET_OFF(weight)]);
    mov(reg_src, ptr[reg_param + GET_OFF(src)]);
    mov(reg_dst, ptr[reg_param + GET_OFF(dst)]);
    mov(reg_bs, ptr[reg_param + GET_OFF(bs)]);
  }

  void main_compute(dim_t mstart) {
    for (int b_row = 0; b_row < nrowptr - 1; ++b_row) {
      // int n_start = nt * NZ;
      tilezero(tmm0);
      tilezero(tmm1);
      tilezero(tmm2);
      tilezero(tmm3);

      for (int group = group_rowptr[b_row]; group < group_rowptr[b_row + 1];
           ++group) {
        dim_t* my_rows = colidxs + group * 32;

        tileloadd(tmm6, ptr[reg_weight + group * TILE_M * TILE_K]);

        
        for (int m = mstart; m < mstart + tileM; m += TILE_M) {
          for (int k = 0; k < 32; k += 2) {
            vmovdqu(ymm0, ptr[reg_src + m + my_rows[k]]);
            vmovdqu(ymm1, ptr[reg_src + m + my_rows[k + 1]]);
            vinserti32x8(zmm0, zmm0, zmm1, 1);
            vpermw(zmm0, reg_musk, zmm0);
            vmovdqu(ARG(m - mstart / TILE_N, k / 2, 0), zmm0);
          }
        }
        tileloadd(tmm4, ptr[rsp]);
        tdpbf16ps(tmm0, tmm6, tmm6);
        tileloadd(tmm4, ptr[rsp + 0x400]);
        tdpbf16ps(tmm1, tmm6, tmm4);
        tileloadd(tmm4, ptr[rsp + 0x800]);
        tdpbf16ps(tmm2, tmm6, tmm4);
        tileloadd(tmm4, ptr[rsp + 0xc00]);
        tdpbf16ps(tmm3, tmm6, tmm4);
      }
      mov(rax, b_row);
      shl(rax, 4);
      imul(rax, reg_bs);
      add(rax, reg_mstart);
      mov(r12, ptr[reg_dst + rax]);
      lea(r11, ptr[r12]);
      tilestored(ptr[r11], tmm0);
      lea(r11, ptr[r11 + TILE_N]);
      tilestored(ptr[r11], tmm1);
      lea(r11, ptr[r11 + TILE_N]);
      tilestored(ptr[r11], tmm2);
      lea(r11, ptr[r11 + TILE_N]);
      tilestored(ptr[r11], tmm3);
    }
  }

  // void loop_K() {
  //   L(l2);
  //   main_compute();
  //   add(reg_nstart, 1);
  //   cmp(reg_nstart, nrowptr);
  //   jb(l2);
  // }

  void loop_N() {
    dim_t mstart = 0;
    L(l1);
    add(reg_mstart, tileM);
    main_compute(mstart);
    mstart += tileM;
    cmp(reg_mstart, reg_bs);
    jl(l1, T_NEAR);
  }

  void init_param() {
      mov(reg_K, K);
      mov(reg_N, N);
      mov(reg_mstart, 0);
      mov(reg_nstart, 0);
      vpmovzxbd(reg_musk, ptr[rip + loopMusk]);
  }

  void generate() {
    Xbyak::util::StackFrame spmm_sf(this, 4, 0, 4028);
    read_inputs();
    init_param();
    loop_N();
  }

  void load_musk() {
    L(loopMusk);
    int num = 16;
    int wordlen = 4;
    const int musk[32] = {31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26,
                          10, 25, 9,  24, 8,  23, 7,  22, 6,  21, 5,
                          20, 4,  19, 3,  18, 2,  17, 1,  16, 0};
    for (int i = 0; i < num; ++i) {
      db(musk[i], wordlen);
    }
  }

 private:
  amx_params_t params;
  const Xbyak::uint8* jit_ker_ = nullptr;

  const Xbyak::Reg64& reg_param = rdi;
  const Xbyak::Reg64& reg_weight = rsi;
  const Xbyak::Reg64& reg_src = rdx;
  const Xbyak::Reg64& reg_dst = rcx;
  const Xbyak::Reg64& reg_bs = r8;
  const Xbyak::Reg64& reg_K = rbx;
  const Xbyak::Reg64& reg_N = rbp;
  const Xbyak::Reg64& reg_mstart = r9;
  const Xbyak::Reg64& reg_nstart = r10;
  const Xbyak::Zmm& reg_musk = zmm31;

  dim_t N;
  dim_t K;
  const dim_t blocks_per_group = 32;
  dim_t nnz_group;
  dim_t nrowptr;
  dim_t* colidxs;
  dim_t* group_rowptr;

  dim_t tileM = 64; // 4x16

  Xbyak::Label loopMusk;
  Xbyak::Label l1, l2, l3, l4;
  const Xbyak::uint8* getCode() {
    this->ready();
    if (!is_initialized()) return nullptr;
    const Xbyak::uint8* code = CodeGenerator::getCode();
    return code;
  }

  static inline bool is_initialized() {
    return Xbyak::GetError() == Xbyak::ERR_NONE;
  }
};

class GemmDriver {
 public:
  GemmDriver(amx_params_t params) {
    kernel_.reset(new gemm_kernel(params));
    kernel_->create_kernel();
  }

 public:
  void operator()(src_t* weight, src_t* activation, dst_t* result) const {
    (*kernel_)(weight, activation, result);
  }

 private:
  std::unique_ptr<gemm_kernel> kernel_;
};

__bfloat16_t make_bf16(float x) {
  int* res = reinterpret_cast<int*>(&x);
  *res = *res >> 16;
  return (__bfloat16_t)*res;
}

int main() {
  request_perm_xtile_data();
  configure_tiles();
  dim_t M = 1024;
  dim_t N = 1024;
  dim_t K = 1024;
  dim_t bk = 1;
  dim_t bn = 16;
  dim_t blocks_per_group = 32;
  dim_t shape[2] = {N, K};
  dim_t blocksize[2] = {bk, bn};
  dim_t* group_rowptr = (dim_t*)malloc((1 + N / bn) * sizeof(dim_t));
  for (dim_t i = 1; i < (1 + N / bn); ++i) {
    group_rowptr[i] = (dim_t)rand() % 2 + 1 + group_rowptr[i - 1];
  }
  dim_t nrowptr = N / bn;
  dim_t nnz_group = group_rowptr[N / bn];
  dim_t* colidxs =
      (dim_t*)malloc(group_rowptr[N / bn] * blocks_per_group * sizeof(dim_t));
  for (dim_t i = 0; i < (N / bn); ++i) {
    for (dim_t j = group_rowptr[i]; j < group_rowptr[i + 1]; ++j) {
      dim_t nonzeros =
          (group_rowptr[i + 1] - group_rowptr[i]) * blocks_per_group;
      dim_t temp = K / nonzeros;
      colidxs[j * blocks_per_group] = rand() % temp * N;
      for (dim_t k = 1; k < nonzeros; ++k) {
        colidxs[j * blocks_per_group + k] =
            rand() % temp * N + colidxs[j * blocks_per_group + k - 1];
        assert(colidxs[j * blocks_per_group + k] < N * K);
      }
    }
  }

  src_t* data = (src_t*)malloc(group_rowptr[N / bn] * blocks_per_group * bk *
                               bn * sizeof(src_t));
  for (dim_t i = 0; i < group_rowptr[N / bn] * blocks_per_group * bk * bn;
       ++i) {
    data[i] = make_bf16(rand() % 10 + 1);
  }

  src_t* src = (src_t*)malloc(K * M * sizeof(src_t));
  for (dim_t i = 0; i < K * M; ++i) {
    src[i] = make_bf16(rand() % 10 + 1);
  }

  dst_t* dst = (dst_t*)malloc(N * M * sizeof(dst_t));

  amx_params_t param;
  param.shape[0] = shape[0];
  param.shape[1] = shape[1];
  param.blocksize[0] = blocksize[0];
  param.blocksize[1] = blocksize[1];
  param.blocks_per_group = blocks_per_group;
  param.nnz_group = nnz_group;
  param.nrowptr = nrowptr;
  param.colidxs = colidxs;
  param.group_rowptr = group_rowptr;

  amx_inputs_t inputs;
  inputs.weight = data;
  inputs.src = src;
  inputs.dst = dst;

  GemmDriver op(param);
  op(inputs);

  free(data);
  free(src);
  free(dst);
  free(group_rowptr);
  free(colidxs);
}
