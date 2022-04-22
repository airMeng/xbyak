#define XBYAK_NO_OP_NAMES
#include <xbyak/xbyak_util.h>

#include <climits>
#define STACKSIZE 8192

#ifdef XBYAK32
#error "this sample is for only 64-bit mode"
#endif

Xbyak::Address EVEX_compress_addr(Xbyak::Reg64 base, int raw_offt,
                                  bool bcast = false) {
  using Xbyak::Address;
  using Xbyak::Reg64;
  using Xbyak::RegExp;
  using Xbyak::Zmm;
  const int EVEX_max_8b_offt = 0x200;
  const Xbyak::Reg64 reg_EVEX_max_8b_offt = Xbyak::util::rbp;

  assert(raw_offt <= INT_MAX);
  auto offt = static_cast<int>(raw_offt);

  int scale = 0;

  if (EVEX_max_8b_offt <= offt && offt < 3 * EVEX_max_8b_offt) {
    offt = offt - 2 * EVEX_max_8b_offt;
    scale = 1;
  } else if (3 * EVEX_max_8b_offt <= offt && offt < 5 * EVEX_max_8b_offt) {
    offt = offt - 4 * EVEX_max_8b_offt;
    scale = 2;
  }

  auto re = RegExp() + base + offt;
  if (scale) re = re + reg_EVEX_max_8b_offt * scale;

  if (bcast)
    return Xbyak::util::zword_b[re];
  else
    return Xbyak::util::zword[re];
}

struct Code : public Xbyak::CodeGenerator {
  Code() {
    // see xbyak/sample/sf_test.cpp for how to use other parameter
    // Xbyak::util::StackFrame sf(this, 4);
    sub(rsp, STACKSIZE);

    // push(rdi);
    // push(rsi);
    // push(rdx);
    // push(rcx);
    mov(eax, ptr[rdi + 4]);
    mov(rax, eax);
    vmovups(zmm0, zword[rsi]);
    vmovups(zmm1, zword[rsi]);
    vaddps(zmm0, zmm1);

    // vmovdqu(zword[rsp + 4096], zmm0);
    vmovdqu32(EVEX_compress_addr(rsp, 4096), zmm0);
    mov(rdi, ptr[rsp + 4104]);
    add(rax, rdi);
    add(rax, rdx);
    mov(ptr[rcx], rax);
    add(rsp, STACKSIZE);
    // pop(rcx);
    // pop(rdx);
    // pop(rsi);
    // pop(rdi);
    ret();
    // add(rsp, STACKSIZE);
  }
};

int main() {
  Code c;
  int* a = (int*)malloc(2 * sizeof(int));
  int* b = (int*)malloc(16 * sizeof(int));
  a[0] = 3;
  a[1] = 4;
  for (int i = 0; i < 16; ++i) {
    b[i] = (int)i % 3;
  }
  int res;
  void (*f)(int*, int*, int, int*) =
      c.getCode<void (*)(int*, int*, int, int*)>();
  f(a, b, 2, &res);
  if (res == 4 + 4 + 2) {
    puts("ok");
  } else {
    printf("res = %d\n", res);
    puts("ng");
  }

  free(a);
  free(b);
}
