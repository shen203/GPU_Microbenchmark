
Fatbin elf code:
================
arch = sm_30
code version = [1,7]
producer = <unknown>
host = linux
compile_size = 64bit

Fatbin elf code:
================
arch = sm_30
code version = [1,7]
producer = cuda
host = linux
compile_size = 64bit

Fatbin ptx code:
================
arch = sm_30
code version = [6,1]
producer = cuda
host = linux
compile_size = 64bit
compressed








.version 6.1
.target sm_30
.address_size 64



.visible .entry _Z5l1_bwPjS_PfS0_(
.param .u64 _Z5l1_bwPjS_PfS0__param_0,
.param .u64 _Z5l1_bwPjS_PfS0__param_1,
.param .u64 _Z5l1_bwPjS_PfS0__param_2,
.param .u64 _Z5l1_bwPjS_PfS0__param_3
)
{
.reg .pred %p<6>;
.reg .f32 %f<76>;
.reg .b32 %r<23>;
.reg .b64 %rd<26>;


ld.param.u64 %rd8, [_Z5l1_bwPjS_PfS0__param_0];
ld.param.u64 %rd9, [_Z5l1_bwPjS_PfS0__param_1];
ld.param.u64 %rd10, [_Z5l1_bwPjS_PfS0__param_2];
ld.param.u64 %rd11, [_Z5l1_bwPjS_PfS0__param_3];
mov.u32 %r11, %tid.x;
shl.b32 %r12, %r11, 2;
mov.f32 %f72, 0f00000000;
setp.gt.u32	%p1, %r12, 32767;
mov.f32 %f73, %f72;
mov.f32 %f74, %f72;
mov.f32 %f75, %f72;
@%p1 bra BB0_3;

shl.b32 %r20, %r11, 2;
mul.wide.u32 %rd12, %r20, 4;
add.s64 %rd24, %rd11, %rd12;
mov.f32 %f72, 0f00000000;
mov.f32 %f73, %f72;
mov.f32 %f74, %f72;
mov.f32 %f75, %f72;

BB0_2:

	{	
.reg .f32 data<4>;
ld.global.ca.v4.f32 {data0,data1,data2,data3}, [%rd24];
add.f32 %f73, data0, %f73;
add.f32 %f73, data1, %f74;
add.f32 %f73, data2, %f75;
add.f32 %f73, data3, %f72;
}

	add.s64 %rd24, %rd24, 16384;
add.s32 %r20, %r20, 4096;
setp.lt.u32	%p2, %r20, 32768;
@%p2 bra BB0_2;

BB0_3:

	bar.sync 0;

	
	mov.u32 %r14, %clock;

	mul.wide.u32 %rd14, %r12, 4;
add.s64 %rd4, %rd11, %rd14;
mov.u32 %r21, 0;

BB0_4:
setp.gt.u32	%p3, %r12, 16383;
@%p3 bra BB0_7;

mul.wide.u32 %rd15, %r21, 4;
add.s64 %rd25, %rd4, %rd15;
mov.u32 %r22, %r12;

BB0_6:

	{	
.reg .f32 data<4>;
ld.global.ca.v4.f32 {data0,data1,data2,data3}, [%rd25];
add.f32 %f73, data0, %f73;
add.f32 %f73, data1, %f74;
add.f32 %f73, data2, %f75;
add.f32 %f73, data3, %f72;
}

	add.s64 %rd25, %rd25, 16384;
add.s32 %r22, %r22, 4096;
setp.lt.u32	%p4, %r22, 16384;
@%p4 bra BB0_6;

BB0_7:
add.s32 %r21, %r21, 4;
setp.lt.u32	%p5, %r21, 16384;
@%p5 bra BB0_4;


	bar.sync 0;

	
	mov.u32 %r18, %clock;

	cvta.to.global.u64 %rd17, %rd8;
mul.wide.u32 %rd18, %r11, 4;
add.s64 %rd19, %rd17, %rd18;
st.global.u32 [%rd19], %r14;
cvta.to.global.u64 %rd20, %rd9;
add.s64 %rd21, %rd20, %rd18;
st.global.u32 [%rd21], %r18;
add.f32 %f53, %f74, %f73;
add.f32 %f54, %f75, %f53;
add.f32 %f55, %f72, %f54;
cvta.to.global.u64 %rd22, %rd10;
add.s64 %rd23, %rd22, %rd18;
st.global.f32 [%rd23], %f55;
ret;
}


