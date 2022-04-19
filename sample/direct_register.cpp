#define XBYAK_NO_OP_NAMES
#include <xbyak/xbyak_util.h>

#define STACKSIZE 256 

#ifdef XBYAK32
	#error "this sample is for only 64-bit mode"
#endif

struct Code : public Xbyak::CodeGenerator {
	Code()
	{
		// see xbyak/sample/sf_test.cpp for how to use other parameter
		// Xbyak::util::StackFrame sf(this, 4);
        // sub(rsp, STACKSIZE);

        // push(rdi);
        // push(rsi);
        // push(rdx);
        // push(rcx);
		mov(eax, ptr[rdi + 4]);
        mov(rax, eax);
		add(rax, rsi);
		add(rax, rdx);
        mov(ptr[rcx], rax);
        // pop(rcx);
        // pop(rdx);
        // pop(rsi);
        // pop(rdi);
        ret();
       // add(rsp, STACKSIZE);
	}
};

int main()
{
	Code c;
    int* a = (int*) malloc(2 * sizeof(int));
    a[0] = 3;
    a[1] = 4;
    int res;
	void (*f)(int*, int, int, int*) = c.getCode<void(*) (int*, int, int, int*)>();
	f(a, 5, 2, &res);
	if (res == 4 + 5 + 2) {
		puts("ok");
	} else {
        printf("res = %d\n", res);
		puts("ng");
	}
}
