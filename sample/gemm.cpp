#include <stdio.h>
#include "xbyak/xbyak.h"
#include <memory>
 
typedef struct params {
    void *A;
    void *B;
    void *C;
} params_t;
#define GET_OFF(field) offsetof(params_t, field)


constexpr Xbyak::Operand::Code abi_save_gpr_regs[] = {
        Xbyak::Operand::RBX,
        Xbyak::Operand::RBP,
        Xbyak::Operand::R12,
        Xbyak::Operand::R13,
        Xbyak::Operand::R14,
        Xbyak::Operand::R15,
#ifdef _WIN32
        Xbyak::Operand::RDI,
        Xbyak::Operand::RSI,
#endif
};


struct gemm_kernel : Xbyak::CodeGenerator {
    gemm_kernel(int m_block, int n_block, int k_block) {
        // Init parameters and check can we generate kernel or not:
        simd_w_ = 16;
        unroll_factor_ = m_block;
        m_block_ = m_block;
        n_block_ = n_block;
        k_block_ = k_block;

        if (n_block_ % simd_w_ || m_block_ > unroll_factor_)
            return;
        nb_ = n_block_ / simd_w_;
        if (nb_ > 4)
            return;
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

    void generate(){
        // Code generation
        // Read parameters
        for (size_t i = 0; i < num_abi_save_gpr_regs; ++i)
            push(Xbyak::Reg64(abi_save_gpr_regs[i]));
        mov(reg_EVEX_max_8b_offt, 2 * EVEX_max_8b_offt);
        mov(reg_A, ptr[param + GET_OFF(A)]);
        mov(reg_B, ptr[param + GET_OFF(B)]);
        mov(reg_C, ptr[param + GET_OFF(C)]);

        // Init accumulator registers
        for (int n = 0; n < nb_; n++) {
            for (int m = 0; m < m_block_; m++) {
                vpxord(accum(m,n), accum(m,n), accum(m,n));
            }
        }
        // Computations
        for (int k = 0; k < k_block_; k++) {
            for (int n = 0; n < nb_; n++) {
                vmovups(load(n), ptr[reg_B + sizeof(float) * B_offset(k, n)]);
            }
            for (int m = 0; m < m_block_; m++) {
                vbroadcastss(bcst(), ptr[reg_A +  sizeof(float) * A_offset(m, k)]);
                for (int n = 0; n < nb_; n++) {
                    vfmadd231ps(accum(m,n), load(n), bcst());
                }
            }
        }
        // Store result
        for (int n = 0; n < nb_; n++) {
            for (int m = 0; m < m_block_; m++) {
                vmovups(ptr[reg_C +  sizeof(float) * C_offset(m, n)], accum(m,n));
            }
        }
        for (size_t i = 0; i < num_abi_save_gpr_regs; ++i)
            pop(Xbyak::Reg64(abi_save_gpr_regs[num_abi_save_gpr_regs - 1 - i]));
        vzeroupper();
        ret();
    }



private:
    int nb_;
    int m_block_;
    int n_block_;
    int k_block_;

    int simd_w_;
    int unroll_factor_;

    const Xbyak::uint8 *jit_ker_ = nullptr;

    Xbyak::Reg64 param = rdi;
    Xbyak::Reg64 reg_A = r15;
    Xbyak::Reg64 reg_B = r14;
    Xbyak::Reg64 reg_C = r13;

    const size_t num_abi_save_gpr_regs = sizeof(abi_save_gpr_regs) / sizeof(abi_save_gpr_regs[0]);
    const int EVEX_max_8b_offt = 0x200;
    const Xbyak::Reg64 reg_EVEX_max_8b_offt = rbp;

    Xbyak::Zmm accum(int m, int n) { return Xbyak::Zmm(m * 4 + n); }
    Xbyak::Zmm load(int n) { return Xbyak::Zmm(unroll_factor_ * 4 + n); }
    // Xbyak::Zmm load(int n) { return Xbyak::Zmm(n);  }
    Xbyak::Zmm bcst() { return Xbyak::Zmm(31); }

    int B_offset (int k, int n) { return k * n_block_ + n * simd_w_; }
    int A_offset (int m, int k) { return m * k_block_ + k; }
    int C_offset (int m, int n) { return m * n_block_ + n * simd_w_; }

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

class GemmDriver{
public:
    GemmDriver(int m_, int n_, int k_){
        m = m_;
        n = n_;
        k = k_;
        kernel_.reset(new gemm_kernel(m, n, k));
        kernel_->create_kernel();
    }
public:
    void operator()(params_t args) const{
        (*kernel_)(&args);
    }

private:
    std::unique_ptr<gemm_kernel> kernel_;
    int m;
    int n;
    int k;
};


void dump(const void *code, size_t code_size)
{
    FILE *file = fopen("dump.bin", "wb+");
    if (file) {
        size_t unused = fwrite(code, code_size, 1, file);
        fclose(file);
    }
}

int main () {

    const int M = 4, N = 48, K = 4;

    float *A = (float*) malloc( sizeof(float) * M * K);
    float *B = (float*) malloc( sizeof(float) * K * N);
    float *C = (float*) malloc( sizeof(float) * M * N);

    for(int m = 0; m < M; ++m){
        for(int k = 0; k < K; ++k){
            A[m * K + k] = rand() % 10;
        }
    }
    for(int k = 0; k < K; ++k){
        for(int n = 0; n < N; ++n){
            B[k * N + n] = rand() % 10;
        }
    }

    params_t p;
    p.A = (void *)A;
    p.B = (void *)B;
    p.C = (void *)C;

    GemmDriver op(M, N, K);
    op(p);

   for(int m = 0; m < M; ++m){
        for(int n = 0; n < N; ++n){
            float ref = A[m * K + 0] * B[0 * N + n];
            ref += A[m * K + 1] * B[1 * N + n];
            ref += A[m * K + 2] * B[2 * N + n];
            ref += A[m * K + 3] * B[3 * N + n];
            if (C[m * N + n] != ref){printf("failed\n"); break;};
            // printf("%f ", C[m * N + n]);
            // printf("%f \n", ref);
        }
        // printf("\n");
    }
   

    free(A);
    free(B);
    free(C);
}

