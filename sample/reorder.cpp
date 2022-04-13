#include <stdio.h>

#include <memory>

#include "xbyak/xbyak.h"

typedef unsigned short __bfloat16_t;
typedef __bfloat16_t src_t;
typedef float res_t;

constexpr Xbyak::Operand::Code abi_save_gpr_regs[] = {
    Xbyak::Operand::RBX, Xbyak::Operand::RBP, Xbyak::Operand::R12,
    Xbyak::Operand::R13, Xbyak::Operand::R14, Xbyak::Operand::R15,
#ifdef _WIN32
    Xbyak::Operand::RDI, Xbyak::Operand::RSI,
#endif
};

struct gemm_kernel : Xbyak::CodeGenerator {
  gemm_kernel() {
    // Init parameters and check can we generate kernel or not:
  }

  template <typename... kernel_args_t>
  void operator()(kernel_args_t... args) const {
    using jit_kernel_func_t = void (*)(const kernel_args_t... args);
    auto *fptr = (jit_kernel_func_t)jit_ker_;
    (*fptr)(std::forward<kernel_args_t>(args)...);
  }

  virtual bool create_kernel() {
    generate();
    jit_ker_ = getCode();
    return (jit_ker_) ? true : false;
  }

  void generate() {
    push(r12);
    push(r13);
    push(r14);
    push(r15);
    push(rbx);
    sub(rsp, 0x1090);
    Xbyak::label loopMusk;
    Xbyak::label l2, l4, l10, l12, l13, l25, l31;
    mov(r9, rsi);
    mov(eax, dword[0x10e0 + rsp]);
    mov(r11, rdi);
    mov(r10d, eax);
    xor(esi, esi);
    mov(ebx, dword[0x10c8 + rsp]);
    mov(r8, rdx);
    shl(r10d, 0x4);
    xor(cl, cl);
    movsxd(r10, r10d);
    lea(edx, dword[rax * 0x4]);
    mov(r14d, dword[0x10c8 + rsp]);
    lea(ebx, dword[rbx - 0x1]);
    vmovups(zmm0, ptr[rip + loopMusk]);
    xor(edi, edi);
    mov(qword[0x1010 + rsp], r10);
    xor(eax, eax);
    mov(qword[0x1048 + rsp], r11);
    mov(qword[0x1050 + rsp], r9);
    l(l2);
    xor(r12d, r12d);
    xor(r11d, r11d);
    xor(r9d, r9d);
    mov(r13d, 1);
    cmp(r14d, 1);
    jle(l31);
    movsxd(rsi, esi);
    mov(byte[0x1000 + rsp], cl);
    lea(r15d, dword[0x40 + rsi]);
    mov(qword[0x101e + rsp], r8);
    mov(dword[0x1096 + rsp], r15d);
    mov(qword[0x1068 + rsp], rax);
    lea(r10, qword[r8 + rsi * 4]);
    mov(dword[0x1070 + rsp], edi);
    mov(dword[0x1078 + rsp], esi);
    mov(r8, qword[0x1010 + rsp]);
    mov(rcx, qword[0x10d8 + rsp]);
    mov(r14, qword[0x10d0 + rsp]);
    l(l4);
    tilezero(tmm0);
    tilezero(tmm1);
    tilezero(tmm2);
    tilezero(tmm3);
    mov(esi, dword[rcx + r12 * 4]);
    mov(edi, esi);
    shl(edi, 5);
    mov(eax, esi);
    movsxd(rdi, edi);
    shl(eax, 9);
    movsxd(rax, eax);
    add(rax, rax);
    lea(rdi, qword[r14 + rdi * 4]);
    cmp(esi, dword[rcx + r13 * 4]);
    jge(l25);
    mov(qword[0x1018 + rsp], r10);
    mov(qword[0x1020 + rsp], r9);
    lea(r15, qword[0xc00 + rsp]);
    mov(qword[0x428 + r15], r12);
    lea(r12, qword[0x800 + rsp]);
    mov(dword[0x430 + r15], r11d);
    lea(r11, qword[0x400 + rsp]);
    mov(dword[0xc38 + r11], ebx);
    mov(dword[0xc40 + r11], edx);
    mov(r14d, dword[0xc70 + r11]);
    mov(r10d, dword[0xc78 + r11]);
    mov(r9, qword[0xc48 + r11]);

    l(l10);
    mov(ebx, 64);
    tileloadd(tmm6, qword[r9 + rax]);

    mov(dword[4184 + rsp], esi);
    mov(r8d, r10d);
    mov(rdx, qword[4200 + rsp]);
    lea(ebx, dword[r10 + r14]);
    mov(r9d, dword[4224 + rsp]);
    mov(rsi, qword[4176 + rsp]);
    mov(qword[4192 + rsp], r13);
    l(l12);
    mov(ecx, ebx);
    xor(r13d, r13d);
    sar(ecx, 3);
    shr(ecx, 28);
    add(ecx, ebx);
    sar(ecx, 4);
    movsxd(rcx, ecx);
    shl(rcx, 10);
    lea(r10, qword[rsp + rcx]);
    lea(rcx, qword[rsi + rdx * 2]);
    l(l13);
    movsxd(r14, dword[rdi + r13 * 4]);
    vmovdqu(ymm1, ymmword[rcx + r14 * 2]);
    movsxd(r14, dword[4 + rdi + r13 * 4]);
    vinserti32x8(zmm2, zmm1, ymmword[rcx + r14 * 2], 1);
    mov(r14d, r13d);
    shr(r14d, 1);
    add(r13, 2);
    vpermw(zmm3, zmm0, zmm2);
    shl(r14, 6);
    vmovdqu32(zmmword[r14 + r10], zmm3);
    cmp(r13, 32);
    jl(l13);
    add(r8d, 16);
    add(ebx, 16);
    add(rdx, 16);
    cmp(r8d, r9d);
    jl(l12);
    mov(r13, qword[4192 + rsp]);
    mov(r8d, 64);
    mov(rcx, qword[4312 + rsp]);
    lea(rbx, qword[rsp]);
    mov(esi, dword[4184 + rbx]);
    mov(r14d, dword[4208 + rbx]);
    mov(r10d, dword[4216 + rbx]);
    mov(r9, qword[4168 + rbx]);
    mov(edx, dword[rcx + r13 * 4]);
    tileloadd((rbx, r8, 1), tmm4);

    tdpbf16ps(tmm4, tmm6, tmm0);

    mov(ebx, 64);
    tileloadd((r11, rbx, 1), tmm4);
    tdpbf16ps(tmm4, tmm6, tmm1);

    tileloadd((r12, rbx, 1), tmm4);
    tdpbf16ps(tmm4, tmm6, tmm2);
    tileloadd((r15, rbx, 1), tmm4);
    tdpbf16ps(tmm4, tmm6, tmm3);
    inc(esi);
    add(rdi, 128);
    add(rax, 1024);
    cmp(esi, edx);
    jl(..B1 .10 #Prob 82 %);
    mov(r10, qword[4120 + rsp]);
    mov(r9, qword[4128 + rsp]);
    mov(r12, qword[4136 + rsp]);
    mov(r11d, dword[4144 + rsp]);
    mov(ebx, dword[4152 + rsp]);
    mov(edx, dword[4160 + rsp]);
    mov(r8, qword[4112 + rsp]);
    mov(r14, qword[4304 + rsp]);
    l(l25);
    lea(rsi, qword[r10 + r9 * 4]);
    tilestored(tmm0, (rsi, rdx, 1));

    lea(rax, qword[64 + rsi]);
    tilestored(tmm1, (rax, rdx, 1));

    lea(rax, qword[128 + rsi]);
    tilestored(tmm2, (rax, rdx, 1));

    add(rsi, 192);
    tilestored(tmm3, (rsi, rdx, 1));

    inc(r11d);
    add(r9, r8);
    inc(r13);
    inc(r12);
    cmp(r11d, ebx);
    jl(l4);
    mov(rax, qword[4200 + rsp]);
    mov(edi, dword[4208 + rsp]);
    mov(esi, dword[4216 + rsp]);
    mov(cl, byte[4096 + rsp]);
    mov(r8, qword[4104 + rsp]);
    mov(r14d, dword[4296 + rsp]);

    l(l31);
    inc(cl);
    add(edi, -64);
    add(rax, 64);
    add(esi, 64);
    cmp(cl, 16);
    jl(..B1 .2 #Prob 93 %);
    vzeroupper();
    add(rsp, 4240);
    pop(rbx);
    pop(r15);
    pop(r14);
    pop(r13);
    pop(r12);
    ret();

    l(loopMusk) : int num = 16;
    int wordlen = 4;
    const int musk[32] = [
      31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8,
      23, 7,  22, 6,  21, 5,  20, 4,  19, 3,  18, 2,  17, 1, 16, 0
    ];
    for (int i = 0; i < num; ++i) {
      db(musk[i], worklen);
    }
  }

 private:
  int nb_;

  const Xbyak::Zmm &reg_musk = zmm0;

  const Xbyak::uint8 *getCode() {
    this->ready();
    if (!is_initialized()) return nullptr;
    const Xbyak::uint8 *code = CodeGenerator::getCode();
    return code;
  }

  static inline bool is_initialized() {
    return Xbyak::GetError() == Xbyak::ERR_NONE;
  }
};

class GemmDriver {
 public:
  GemmDriver() {
    kernel_.reset(new gemm_kernel());
    kernel_->create_kernel();
  }

 public:
  void operator()(params_t args) const { (*kernel_)(&args); }

 private:
  std::unique_ptr<gemm_kernel> kernel_;
};

__bfloat16 make_bf16(float x)
{
    int* res = reinterpret_cast<int*>(&x);
    *res = *res >> 16;
    return (__bfloat16)*res;
}

int main() {

  dim_t M = 1024;
  dim_t N = 1024;
  dim_t K = 1024;
  dim_t bk = 1;
  dim_t bn = 16;
  dim_t blocks_per_group = 32;
  dim_t shape[2] = {N, k};
  dim_t blocksize[2] = {bk, bn};
  dim_t* group_rowptr = (dim_t*) malloc((1 + N / bn) * sizeof(dim_t));
  for(dim_t i = 1; i < (1 + N / bn); ++i){
    group_rowptr[i] = (dim_t) rand() % 2 + 1 + group_rowptr[i - 1];
  }
  dim_t nrowptr = N/bn;
  dim_t nnz_group = group_rowptr[N / bn];
  dim_t* colidxs = (dim_t*) malloc( group_rowptr[N / bn] * blocks_per_group * sizeof(dim_t));
  for(dim_t i = 0; i< (N / bn); ++i){
    for(dim_t j = group_rowptr[i]; j < group_rowptr[i+1]; ++j){
      dim_t nonzeros = (group_rowptr[i+1] - group_rowptr[i]) * blocks_per_group;
      dim_t temp = K / nonzeros;
      colidxs[j * blocks_per_group] = rand() % temp * N;
      for(dim_t k = 1; k < nonzeros; ++k){
        colidxs[j * blocks_per_group+k] = rand() % temp * N + colidxs[j * blocks_per_group+k-1]; 
        assert(colidxs[j * blocks_per_group+k] < N * K);
      }
    }
  }

  src_t* data = (src_t*) malloc(group_rowptr[N / bn] * blocks_per_group * bk * bn * sizeof(src_t));
  for(dim_t i = 0; i< group_rowptr[N / bn] * blocks_per_group * bk * bn;++i){
      data[i] = make_bf16(rand() % 10 + 1);
  }

  src_t* src = (src_t*) malloc(K * M * sizeof(src_t));
  for(dim_t i = 0; i< K * M;++i){
      src[i] = make_bf16(rand() % 10 + 1);
  }

  dst_t* dst = (dst_t*) malloc(N * M * sizeof(dst_t));

  GemmDriver op();
  op(data, src, dst, shape, blocksize, blocks_per_group, nnz_group, nrowptr, colidxs, group_rowptr, N);

  free(data);
  free(src);
  free(dst);
  free(group_rowptr);
  free(colidxs);
}
