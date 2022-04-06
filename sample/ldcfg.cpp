#include <stdio.h>
#include "xbyak/xbyak.h"
#include <memory>
#include <err.h>
#include <errno.h>

#define Bm rdi
#define Bn rsi
#define K rdx
#define AO rax
#define BO rbp
#define LDC r13
#define I rcx
#define FLAG r8
#define bm r9
#define bm0 r10
#define bm0w r10w
#define bm1 r11
#define bm1w r11w
#define bn r12
#define cc1 r14
#define cc2 r15
#define T0 rax
#define T0b al
#define T1 rbx
#define T1b bl
#define N qword[rsp + 0x40]
#define A qword[rsp + 0x48]
#define B qword[rsp + 0x50]
#define C qword[rsp + 0x58]
#define FinalM qword[rsp + 0x60]
#define FinalN qword[rsp + 0x68]
#define BackUp0 qword[rsp + 0x70]
#define BackUp1 qword[rsp + 0x78]
#define BackUp2 qword[rsp + 0x80]

#define ARG_X(x) qword[rsp + (STACKSIZE + (x))]

#define STACKSIZE 256
#define TILEB(X) byte[rsp + ((X) + 0xc0)]
#define TILEW(X) word[rsp + ((X) + 0xc0)]
#define TILED(X) dword[rsp + ((X) + 0xc0)]
#define TILEQ(X) qword[rsp + ((X) + 0xc0)]
 

// using namespace Xbyak::util;
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/signal.h>
#include <unistd.h>

#define fatal_error(msg, ...)   err(1, "[FAIL]\t" msg, ##__VA_ARGS__)
#define XFEATURE_XTILECFG   17
#define XFEATURE_XTILEDATA  18
#define XFEATURE_MASK_XTILECFG  (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)

#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

static void request_perm_xtile_data()
{
    unsigned long bitmask;
    long rc;

    rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (rc)
        fatal_error("XTILE_DATA request failed: %ld", rc);

    rc = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (rc)
        fatal_error("prctl(ARCH_GET_XCOMP_PERM) error: %ld", rc);

    if (bitmask & XFEATURE_MASK_XTILE)
        printf("ARCH_REQ_XCOMP_PERM XTILE_DATA successful.\n");
}

typedef int64_t dim_t;
typedef unsigned short bfloat16_t;
typedef bfloat16_t src_t;


typedef struct params {
    dim_t* m;
    dim_t* n;
    dim_t* k;
    float* alpha; 
    src_t* src;
    src_t* wgt;
    float* dst;
    dim_t* ldc;
    dim_t* col_offset;
    dim_t* row_offset;
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


struct ldcfg_kernel : Xbyak::CodeGenerator {
    ldcfg_kernel() {
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
       dim_t SHIFT_UNROLL_M = 4; // tile M in 16x32
       dim_t SHIFT_UNROLL_N = 4; // tile N in 16x32
       dim_t SHIFT_UNROLL_K = 1; // tile K in 4x64
       dim_t SHIFT_UNROLL_MM = 5, SHIFT_UNROLL_NN = 5, SHIFT_UNROLL_KK = 5;

       Xbyak::Label loopM, loopN;
       Xbyak::Label l0, l1, lb;

        sub(rsp, STACKSIZE);
           
        mov(ptr[rsp + 0x00], rbx);
        mov(ptr[rsp + 0x08], rbp);
        mov(ptr[rsp + 0x10], r12);
        mov(ptr[rsp + 0x18], r13);
        mov(ptr[rsp + 0x20], r14);
        mov(ptr[rsp + 0x28], r15);
 
        mov(Bm, qword[rdi]);
        mov(Bn, qword[rsi]);
        mov(K, qword[rdx]);
        mov(rcx, r8);
        mov(r8, r9);
        mov(r9, ARG_X(0x08));
        mov(LDC, ARG_X(0x10));
            
        /* Initializing tile. First value is pallate ID */
        mov(TILEQ(0x00), 1);
        mov(TILEQ(0x08), 0);
        mov(TILEQ(0x20), 0);
        mov(TILEQ(0x28), 0);
        mov(TILEQ(0x38), 0);

        /* Clear back up data */
        mov(BackUp0, 0);
        mov(BackUp1, 0);
        mov(BackUp2, 0);

        sal(LDC, SHIFT_C);

        /* K needs to be multiple of 4 */
        add(K, UNROLL_K - 1);
        and_(K, -UNROLL_K);

        mov(N, Bn);
        mov(A, rcx);
        mov(B, r8);
        mov(C, r9);

        /* Calculating last value for M loop */
        lea(T0, ptr[Bm - 1]);
        and_(T0, -UNROLL_MM);
        lea(T1, ptr[Bm - UNROLL_MM]);
        sub(T1, T0);
        mov(FinalM, T1);

        /* Calculating last value for N loop */
        lea(T0, ptr[Bn - 1]);
        and_(T0, -UNROLL_NN);
        lea(T1, ptr[Bn - UNROLL_NN]);
        sub(T1, T0);
        mov(FinalN, T1);
        align(4);

        L(loopM);
        /* Updating C address */
        mov(cc1, C);
        mov(cc2, LDC);
        sal(cc2, SHIFT_UNROLL_N);
        add(cc2, cc1);
        add(C, UNROLL_MM * SIZE_C);

        mov(bm, UNROLL_MM);
        cmp(Bm, UNROLL_MM);
        cmovle(bm, Bm);

        mov(bm0, UNROLL_M);
        cmp(bm, UNROLL_M);
        cmovle(bm0, bm);

        mov(bm1, bm);
        sub(bm1, bm0);
    
        /* Filling in tile information for M */
        mov(TILEB(0x30), UNROLL_KK / UNROLL_K);
        mov(TILEB(0x31), UNROLL_KK / UNROLL_K);
    
        sal(bm0, SHIFT_UNROLL_K + SHIFT_A);
        sal(bm1, SHIFT_UNROLL_K + SHIFT_A);
    
        mov(TILEW(0x10), bm0);
        mov(TILEW(0x12), bm1);
    
        sar(bm0, SHIFT_UNROLL_K + SHIFT_A - SHIFT_C);
        sar(bm1, SHIFT_UNROLL_K + SHIFT_A - SHIFT_C);
    
        mov(TILEW(0x18), bm0);
        mov(TILEW(0x1a), bm0);
        mov(TILEW(0x1c), bm1);
        mov(TILEW(0x1e), bm1);
    
        sal(bm, SHIFT_UNROLL_KK + SHIFT_A);
    
        xor_(FLAG, FLAG);
        mov(T0, 2);
        cmp(bm1, 0);
        cmovg(FLAG, T0);
    
        mov(Bn, N);
        mov(BO, B);
        align(4);
    
        L(loopN);
        mov(bn, UNROLL_NN);
        cmp(Bn, UNROLL_NN);
        cmovle(bn, Bn);
    
        mov(T0, UNROLL_N);
        cmp(bn, UNROLL_N);
        cmovle(T0, bn);
    
        mov(T1, bn);
        sub(T1, T0);
    
        /* Filling in tile information for N */
        mov(TILEW(0x14), UNROLL_KK * SIZE_B);
        mov(TILEW(0x16), UNROLL_KK * SIZE_B);
    
        mov(TILEB(0x32), T0);
        mov(TILEB(0x33), T1);
    
        mov(TILEB(0x34), T0);
        mov(TILEB(0x35), T1);
        mov(TILEB(0x36), T0);
        mov(TILEB(0x37), T1);
    
        sal(bn, SHIFT_UNROLL_KK + SHIFT_B);
    
        /* Disabling unnecessary tile */
        test(FLAG, 2);
        jg(l0);
    
        mov(TILEW(0x12), 0x00);
        mov(TILEW(0x1c), 0x00);
        mov(TILEW(0x1e), 0x00);
        mov(TILEB(0x31), 0x00);
        mov(TILEB(0x36), 0x00);
        mov(TILEB(0x37), 0x00);
    
        L(l0);
        or_(FLAG, 1);
        cmp(T1, 0);
        jg(l1);
    
        and_(FLAG, -2);
    
        mov(TILEW(0x16), 0x00);
        mov(TILEW(0x1a), 0x00);
        mov(TILEW(0x1e), 0x00);
        mov(TILEB(0x33), 0x00);
        mov(TILEB(0x35), 0x00);
        mov(TILEB(0x37), 0x00);
    
        L(l1);
        /* Configuring tile if tile has been changed */
        mov(T1, BackUp0);
        mov(T0, TILEQ(0x10));
        mov(BackUp0, T0);
        xor_(T1, T0);
    
        xor_(T1, BackUp1);
        mov(T0, TILEQ(0x18));
        mov(BackUp1, T0);
        xor_(T1, T0);
    
        xor_(T1, BackUp2);
        mov(T0, TILEQ(0x30));
        mov(BackUp2, T0);
        xor_(T1, T0);
        je(lb);
    
        ldtilecfg(TILEQ(0));
        L(lb);
    tilerelease();

    mov(rbx, ptr[rsp + 0x00]);
    mov(rbp, ptr[rsp + 0x08]);
    mov(r12, ptr[rsp + 0x10]);
    mov(r13, ptr[rsp + 0x18]);
    mov(r14, ptr[rsp + 0x20]);
    mov(r15, ptr[rsp + 0x28]);

#ifdef _WIN32
    mov(rsi, ptr[rsp + 0x30]);
    mov(rdi, ptr[rsp + 0x38]);
#endif

    add(rsp, STACKSIZE);
        ret();
    }



private:

    const Xbyak::uint8 *jit_ker_ = nullptr;

    const dim_t SHIFT_A = 1;
    const dim_t SHIFT_B = 1;
    const dim_t SHIFT_C = 2;
    const dim_t UNROLL_M = 16;
    const dim_t UNROLL_N = 16;
    const dim_t UNROLL_K = 2;
    const dim_t UNROLL_MM = 32;
    const dim_t UNROLL_NN = 32;
    const dim_t UNROLL_KK = 32;
    const dim_t SIZE_A = 2;
    const dim_t SIZE_B = 2;
    const dim_t SIZE_C = 4;

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

class LDCFGDriver{
public:
    LDCFGDriver(){
        kernel_.reset(new ldcfg_kernel());
        kernel_->create_kernel();
    }
public:
    void operator()(dim_t m, dim_t n, dim_t k, float alpha, src_t* a, src_t* b, float* c, dim_t ldc, dim_t col_offset, dim_t row_offset) const{
    printf("before kernel\n");    
    (*kernel_)(&m, &n, &k, &alpha, a, b, c, ldc, col_offset, row_offset);
    printf("after kernel\n");
    }

private:
    std::unique_ptr<ldcfg_kernel> kernel_;
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

    request_perm_xtile_data();
    dim_t m = 32, n = 32, k = 32;

    src_t *src = (src_t*) malloc( sizeof(src_t) * m * k);
    src_t *wgt = (src_t*) malloc( sizeof(src_t) * k * n);
    float *dst = (float*) malloc( sizeof(float) * m * n);

    for(dim_t i = 0; i < m; ++i){
        for(dim_t j = 0; j < n; ++j){
            src[i * k + j] = (src_t)rand() % 10;
        }
    }
    for(dim_t j = 0; j < k; ++j){
        for(dim_t l = 0; l < n; ++l){
            wgt[j * n + l] = (src_t)rand() % 10;
        }
    }

    // params_t p;
    // p.m = &m;
    // p.n = &n;
    // p.k = &k;
    // p.src = src;
    // p.wgt = wgt;
    // p.dst = dst;

    float alpha = 1.0;
    dim_t ldc = 1;
    dim_t col_offset = 1;
    dim_t row_offset = 1;
    // p.alpha = &alpha;
    // p.ldc = &ldc;
    // p.col_offset = &col_offset;
    // p.row_offset = &row_offset;

    LDCFGDriver op;
    op(m, n, k, alpha, src, wgt, dst, ldc, col_offset, row_offset);


    free(src);
    free(wgt);
    free(dst);
}

