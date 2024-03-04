#include<stdio.h>
#include<cmath>
#include<time.h>
#include<stdint.h>

constexpr uint32_t M=0x04c11db7;

uint32_t calc_CRC1(const uint8_t*data,uint64_t len,uint32_t init_value=0)
{
    uint32_t r=init_value;
    for(uint64_t i=0;i<len;i++)
    {
        for(int j=0;j<8;j++)
        {
            uint32_t flag=r>>31?M:0;
            r=(r<<1)^(data[i]>>(7-j)&1);
            r^=flag;
        }
    }
    for(int i=0;i<32;i++)
    {
        r=(r<<1)^(r>>31?M:0);
    }
    return r;
}

struct Table
{
	uint32_t table[256];
	constexpr Table()
	{
		table[0]=0;
		for(int i=1;i<256;i++)
		{
			uint32_t tmp=table[i>>1];
			table[i]=(tmp<<1)^(tmp>>31?M:0)^(i&1?M:0);
		}
	}

	uint32_t operator[](int idx) const
	{
		return table[idx];
	}
};

uint32_t calc_CRC2(const uint8_t*data,uint64_t len,uint32_t init_value=0)
{
	
	
	static constexpr Table table;
	uint32_t r=init_value;

	
	for(uint64_t i=0;i<len;i++)
	{
		r=table[r>>24]^data[i]^(r<<8);
	}
	for(int i=0;i<4;i++)
	{
		r=table[r>>24]^(r<<8);
	}
	return r;
}

uint32_t calc_CRC2_nzx(const uint8_t*data,uint64_t len,uint32_t init_value=0)
{	
	static constexpr Table table;
	uint32_t r=init_value;

	for(uint64_t i=0;i<len;i++)
	{
		r=table[r>>24]^data[i]^(r<<8);
	}
	return r;
}

uint32_t calc_CRC3(const uint8_t*data,uint64_t len,uint32_t init_value=0)
{
	uint32_t table[256];

	table[0]=0;
	for(int i=1;i<256;i++)
	{
		uint32_t tmp=table[i>>1];
		table[i]=(tmp<<1)^(tmp>>31?M:0)^(i&1?M:0);
	}

	uint64_t seg_len=len/4;
	uint64_t seg0=seg_len*0;
	uint64_t seg1=seg_len*1;
	uint64_t seg2=seg_len*2;
	uint64_t seg3=seg_len*3;
	uint64_t seg4=seg_len*4;
	uint64_t seg_r=len%4;

	auto mul_mod=[&table](uint32_t x,uint32_t y)->uint32_t
	{
		uint64_t r=0;
		for(int i=0;i<32;i++)
			r^=y>>i&1?(uint64_t)x<<i:0;

		r=((uint64_t)table[r>>56&255]<<24)^r;
		r=((uint64_t)table[r>>48&255]<<16)^r;
		r=((uint64_t)table[r>>40&255]<< 8)^r;
		r=((uint64_t)table[r>>32&255]<< 0)^r;
		return r;
	};

	auto pow_mod=[&mul_mod](uint32_t x,uint64_t y)->uint32_t
	{
		uint32_t ans=1;
		while(y)
		{
			if(y&1)ans=mul_mod(ans,x);
			x=mul_mod(x,x);
			y>>=1;
		}
		return ans;
	};


	uint32_t r0=init_value;
	uint32_t r1=0;
	uint32_t r2=0;
	uint32_t r3=0;

	for(uint64_t i=0;i<seg_len;i++)
	{
		r0=table[r0>>24]^data[seg0+i]^(r0<<8);
		r1=table[r1>>24]^data[seg1+i]^(r1<<8);
		r2=table[r2>>24]^data[seg2+i]^(r2<<8);
		r3=table[r3>>24]^data[seg3+i]^(r3<<8);
	}

	uint32_t powx1=pow_mod(2,seg_len*8);
	uint32_t powx2=mul_mod(powx1,powx1);
	uint32_t powx3=mul_mod(powx1,powx2);

	uint32_t r=mul_mod(r0,powx3)^mul_mod(r1,powx2)^mul_mod(r2,powx1)^r3;

	for(uint64_t i=0;i<seg_r;i++)
		r=r3=table[r>>24]^data[seg4+i]^(r<<8);

	for(int i=0;i<4;i++)
		r=table[r>>24]^(r<<8);
	return r;
}

#define AVX512_TEST 1

#if AVX512_TEST

#include <wmmintrin.h>
#include <immintrin.h>
#pragma GCC target("aes")
#pragma GCC target("sse")
#pragma GCC target("sse2")
#pragma GCC target("pclmul")
#pragma GCC target("avx512f")
#pragma GCC target("vpclmulqdq")
#pragma GCC target("avx512bw")

struct xmm
{
	__m128i x;
	xmm(int32_t a):x(_mm_set_epi32(a,a,a,a)){}
	xmm(int64_t a):x(_mm_set_epi64x(a,a)){}
	xmm(uint32_t a):x(_mm_set_epi32(a,a,a,a)){}
	xmm(uint64_t a):x(_mm_set_epi64x(a,a)){}
	xmm(const xmm&a):x(a.x){}
	xmm(__m128i a):x(a){}
	xmm():xmm(0){};
	
	operator __m128i(){return x;}

	void store(void*p)
	{
		_mm_storeu_si128((__m128i*)p,x);
	}

};

struct zmm
{
	__m512i x;
	zmm(int32_t a):x(_mm512_set1_epi32(a)){}
	zmm(int64_t a):x(_mm512_set1_epi64(a)){}
	zmm(uint32_t a):x(_mm512_set1_epi32(a)){}
	zmm(uint64_t a):x(_mm512_set1_epi64(a)){}
	zmm(const zmm&a):x(a.x){}
	zmm(__m512i a):x(a){}
	zmm(const void*p):x(_mm512_load_epi64(p)){}
	zmm():zmm(0){}
	
	operator __m512i(){return x;}

	void store(void*p)
	{
		_mm512_storeu_si512(p,x);
	}
};



int64_t operator"" _i64(unsigned long long x)
{
	return (int64_t)x;
}

uint64_t operator"" _u64(unsigned long long x)
{
	return (uint64_t)x;
}

struct CRC32_engine
{
	uint32_t MP;
	uint64_t inv_rMP;
	zmm x2048;
	zmm x64;
	zmm x2048_2;
	zmm x64_2;
	zmm vrMP;
	zmm vMP;
	zmm vx;
	zmm x2048_64;
	CRC32_engine(uint32_t M):MP(M)
	{
		uint64_t tmp=0x8000000000000000ull;
		for(int i=1;i<64;i++)
		{
			uint64_t t=0;
			for(int j=0;j<i&&j<32;j++)
				t^=(M>>(31-j))&(tmp>>(64-i+j));
			tmp|=(t&1)<<(63-i);
		}
		inv_rMP=tmp;

		uint64_t xv=pow(2,2048);

		x2048=zmm(xv);
		x2048_2=zmm(xv<<1);
		uint64_t xv2=pow(2,64);
		x64=zmm(xv2);
		x64_2=zmm(xv2<<1);
		vrMP=zmm(inv_rMP);
		vMP=zmm(0x100000000_u64+MP);
		vx=zmm(2_u64);

		uint64_t xv3=pow(2,2048+64);

		x2048_64=_mm512_set_epi64(xv3,xv,xv3,xv,xv3,xv,xv3,xv);
	}

	uint32_t pow(uint32_t x,uint64_t y)
	{
		uint64_t ans=1;
		while(y)
		{
			if(y&1)ans=mul_mod(ans,x);
			x=mod(mul_mod(x,x));
			y>>=1;
		}
		return mod(ans);
	}

	void print(xmm x)
	{
		alignas(64) uint64_t tmp[2];
		x.store(tmp);
		printf("%llx %llx\n",tmp[0],tmp[1]);
	}

	uint32_t mod(uint64_t x)
	{
		for(int i=63;i>=32;i--)
		{
			if(x>>i&1)
			{
				x^=(uint64_t)MP<<i-32;
			}
		}
		return x;
	}

	//最大支持乘积为96位
	uint32_t mul_mod(uint64_t ta,uint64_t tb)
	{
		xmm a(ta),b(tb*2);
		alignas(64) uint64_t tmp[2];
		a=_mm_clmulepi64_si128(a,b,0);
		//printf("%llx\n",inv_rMP);
		b=_mm_clmulepi64_si128(_mm_bsrli_si128(a,4),xmm(inv_rMP),0);
		b=_mm_clmulepi64_si128(b,xmm(0x100000000_u64+MP),0x01);
		a=_mm_xor_si128(_mm_clmulepi64_si128(xmm(ta),xmm(tb),0),b);
		a.store(tmp);
		return mod(tmp[0]);
	}

	

	uint32_t mul_mod_b(uint64_t a,uint64_t b)
	{
		a=mod(a);
		b=mod(b);
		uint64_t tmp=0;
		for(int i=0;i<32;i++)
			for(int j=0;j<32;j++)
				tmp^=((a>>i&1)&(b>>j&1))<<(i+j);
		return mod(tmp);
	}

	uint32_t calc(const uint8_t*data,uint64_t len,uint32_t init_value=0)
	{
		
		//确保data对齐到64字节
		uint64_t pv=reinterpret_cast<uint64_t>(data);
		pv=(pv&63)?64-(pv&63):0;
		uint32_t r=calc_CRC2_nzx(data,pv,init_value);
		

		zmm r0(0_i64);
		zmm r1(0_i64);
		zmm r2(0_i64);
		zmm r3(0_i64);

		uint64_t i;

		//uint64_t rtest=0;

		//uint64_t rtest_2=0;

		for(i=pv;i+255<len;i+=256)
		{
			zmm d0(data+i+0);
			zmm d1(data+i+64);
			zmm d2(data+i+128);
			zmm d3(data+i+192);

			zmm reverse=_mm512_set_epi64(0x08090a0b0c0d0e0f_u64,0x0001020304050607_u64,0x08090a0b0c0d0e0f_u64,0x0001020304050607_u64,0x08090a0b0c0d0e0f_u64,0x0001020304050607_u64,0x08090a0b0c0d0e0f_u64,0x0001020304050607_u64);

			d0=_mm512_shuffle_epi8(d0,reverse);
			d1=_mm512_shuffle_epi8(d1,reverse);
			d2=_mm512_shuffle_epi8(d2,reverse);
			d3=_mm512_shuffle_epi8(d3,reverse);

			// static uint8_t test_data[256]{0};
			// rtest=calc_CRC2_nzx(test_data,248,rtest);
			// rtest=calc_CRC2_nzx(data+i,8,rtest);
			

			// rtest_2=mul_mod(rtest_2,pow(2,2048));
			// rtest_2^=*(uint64_t*)(data+i);

			zmm t00=_mm512_bsrli_epi128(_mm512_clmulepi64_epi128(r0,x2048_2,0x00),4);
			zmm t01=_mm512_bsrli_epi128(_mm512_clmulepi64_epi128(r0,x2048_2,0x11),4);
			zmm t10=_mm512_bsrli_epi128(_mm512_clmulepi64_epi128(r1,x2048_2,0x00),4);
			zmm t11=_mm512_bsrli_epi128(_mm512_clmulepi64_epi128(r1,x2048_2,0x11),4);
			zmm t20=_mm512_bsrli_epi128(_mm512_clmulepi64_epi128(r2,x2048_2,0x00),4);
			zmm t21=_mm512_bsrli_epi128(_mm512_clmulepi64_epi128(r2,x2048_2,0x11),4);
			zmm t30=_mm512_bsrli_epi128(_mm512_clmulepi64_epi128(r3,x2048_2,0x00),4);
			zmm t31=_mm512_bsrli_epi128(_mm512_clmulepi64_epi128(r3,x2048_2,0x11),4);

			zmm t0=_mm512_maskz_unpacklo_epi64(0xff,_mm512_clmulepi64_epi128(r0,x2048,0x00),_mm512_clmulepi64_epi128(r0,x2048,0x11));
			zmm t1=_mm512_maskz_unpacklo_epi64(0xff,_mm512_clmulepi64_epi128(r1,x2048,0x00),_mm512_clmulepi64_epi128(r1,x2048,0x11));
			zmm t2=_mm512_maskz_unpacklo_epi64(0xff,_mm512_clmulepi64_epi128(r2,x2048,0x00),_mm512_clmulepi64_epi128(r2,x2048,0x11));
			zmm t3=_mm512_maskz_unpacklo_epi64(0xff,_mm512_clmulepi64_epi128(r3,x2048,0x00),_mm512_clmulepi64_epi128(r3,x2048,0x11));

			t00=_mm512_clmulepi64_epi128(t00,vrMP,0x00);
			t01=_mm512_clmulepi64_epi128(t01,vrMP,0x00);
			t10=_mm512_clmulepi64_epi128(t10,vrMP,0x00);
			t11=_mm512_clmulepi64_epi128(t11,vrMP,0x00);
			t20=_mm512_clmulepi64_epi128(t20,vrMP,0x00);
			t21=_mm512_clmulepi64_epi128(t21,vrMP,0x00);
			t30=_mm512_clmulepi64_epi128(t30,vrMP,0x00);
			t31=_mm512_clmulepi64_epi128(t31,vrMP,0x00);

			t00=_mm512_clmulepi64_epi128(t00,vMP,0x01);
			t01=_mm512_clmulepi64_epi128(t01,vMP,0x01);
			t10=_mm512_clmulepi64_epi128(t10,vMP,0x01);
			t11=_mm512_clmulepi64_epi128(t11,vMP,0x01);
			t20=_mm512_clmulepi64_epi128(t20,vMP,0x01);
			t21=_mm512_clmulepi64_epi128(t21,vMP,0x01);
			t30=_mm512_clmulepi64_epi128(t30,vMP,0x01);
			t31=_mm512_clmulepi64_epi128(t31,vMP,0x01);

			r0=_mm512_xor_epi64(t0,_mm512_maskz_unpacklo_epi64(0xff,t00,t01));
			r1=_mm512_xor_epi64(t1,_mm512_maskz_unpacklo_epi64(0xff,t10,t11));
			r2=_mm512_xor_epi64(t2,_mm512_maskz_unpacklo_epi64(0xff,t20,t21));
			r3=_mm512_xor_epi64(t3,_mm512_maskz_unpacklo_epi64(0xff,t30,t31));

			r0=_mm512_xor_epi64(r0,d0);
			r1=_mm512_xor_epi64(r1,d1);
			r2=_mm512_xor_epi64(r2,d2);
			r3=_mm512_xor_epi64(r3,d3);

		}

		alignas(64) uint64_t res[32];

		r0.store(res+0);
		r1.store(res+8);
		r2.store(res+16);
		r3.store(res+24);

		//printf("%llu %u %u\n",rtest,mod(rtest_2),mod(res[0]));

		uint32_t xn=pow(2,(i-pv)*8);

		r=mul_mod(r,xn);

		xn=pow(2,64);
		uint32_t xt=1;

		for(int i=0;i<32;i++)
		{
			r^=mul_mod(res[31-i],xt);
			xt=mul_mod(xt,xn);
		}

		return calc_CRC2(data+i,len-i,r);

	}


	uint32_t calc2(const uint8_t*data,uint64_t len,uint32_t init_value=0)
	{
		
		//确保data对齐到64字节
		uint64_t pv=reinterpret_cast<uint64_t>(data);
		pv=(pv&63)?64-(pv&63):0;
		uint32_t r=calc_CRC2_nzx(data,pv,init_value);
		
		zmm r0(0_i64);
		zmm r1(0_i64);
		zmm r2(0_i64);
		zmm r3(0_i64);

		uint64_t i;

		zmm reverse=_mm512_set_epi64(
			0x0001020304050607_u64,0x08090a0b0c0d0e0f_u64,
			0x0001020304050607_u64,0x08090a0b0c0d0e0f_u64,
			0x0001020304050607_u64,0x08090a0b0c0d0e0f_u64,
			0x0001020304050607_u64,0x08090a0b0c0d0e0f_u64
		);

		for(i=pv;i+255<len;i+=256)
		{
			zmm d0(data+i+0);
			zmm d1(data+i+64);
			zmm d2(data+i+128);
			zmm d3(data+i+192);

			zmm t00=_mm512_clmulepi64_epi128(r0,x2048_64,0);
			zmm t01=_mm512_clmulepi64_epi128(r0,x2048_64,17);
			zmm t10=_mm512_clmulepi64_epi128(r1,x2048_64,0);
			zmm t11=_mm512_clmulepi64_epi128(r1,x2048_64,17);
			zmm t20=_mm512_clmulepi64_epi128(r2,x2048_64,0);
			zmm t21=_mm512_clmulepi64_epi128(r2,x2048_64,17);
			zmm t30=_mm512_clmulepi64_epi128(r3,x2048_64,0);
			zmm t31=_mm512_clmulepi64_epi128(r3,x2048_64,17);

			d0=_mm512_shuffle_epi8(d0,reverse);
			d1=_mm512_shuffle_epi8(d1,reverse);
			d2=_mm512_shuffle_epi8(d2,reverse);
			d3=_mm512_shuffle_epi8(d3,reverse);

			r0=_mm512_xor_epi64(d0,t00);
			r1=_mm512_xor_epi64(d1,t10);
			r2=_mm512_xor_epi64(d2,t20);
			r3=_mm512_xor_epi64(d3,t30);

			r0=_mm512_xor_epi64(r0,t01);
			r1=_mm512_xor_epi64(r1,t11);
			r2=_mm512_xor_epi64(r2,t21);
			r3=_mm512_xor_epi64(r3,t31);

		}

		alignas(64) uint8_t res[256];

		r0=_mm512_shuffle_epi8(r0,reverse);
		r1=_mm512_shuffle_epi8(r1,reverse);
		r2=_mm512_shuffle_epi8(r2,reverse);
		r3=_mm512_shuffle_epi8(r3,reverse);

		r0.store(res+0);
		r1.store(res+64);
		r2.store(res+128);
		r3.store(res+192);

		r=calc_CRC2_nzx(res,256,r);
		return calc_CRC2(data+i,len-i,r);

	}
};



uint32_t calc_CRC4(const uint8_t*data,uint64_t len,uint32_t init_value=0)
{
	CRC32_engine eng(M);
	return eng.calc(data,len,init_value);
}


uint32_t calc_CRC_nb(  /* AVX512+PCLMUL */
    const unsigned char *buf,
    uint64_t len,
    uint32_t crc)
{
    /*
     * Definitions of the bit-reflected domain constants k1,k2,k3,k4
     * are similar to those given at the end of the paper, and remaining
     * constants and CRC32+Barrett polynomials remain unchanged.
     *
     * Replace the index of x from 128 to 512. As follows:
     * k1 = ( x ^ ( 512 * 4 + 32 ) mod P(x) << 32 )' << 1 = 0x011542778a
     * k2 = ( x ^ ( 512 * 4 - 32 ) mod P(x) << 32 )' << 1 = 0x01322d1430
     * k3 = ( x ^ ( 512 + 32 ) mod P(x) << 32 )' << 1 = 0x0154442bd4
     * k4 = ( x ^ ( 512 - 32 ) mod P(x) << 32 )' << 1 = 0x01c6e41596
     */
    alignas(64) static const uint64_t k1k2[] = { 0x011542778a, 0x01322d1430,
                                                0x011542778a, 0x01322d1430,
                                                0x011542778a, 0x01322d1430,
                                                0x011542778a, 0x01322d1430 };
    alignas(64) static const uint64_t  k3k4[] = { 0x0154442bd4, 0x01c6e41596,
                                                0x0154442bd4, 0x01c6e41596,
                                                0x0154442bd4, 0x01c6e41596,
                                                0x0154442bd4, 0x01c6e41596 };
    alignas(16) static const uint64_t k5k6[] = { 0x01751997d0, 0x00ccaa009e };
    alignas(16) static const uint64_t k7k8[] = { 0x0163cd6124, 0x0000000000 };
    alignas(16) static const uint64_t poly[] = { 0x01db710641, 0x01f7011641 };
    __m512i x0, x1, x2, x3, x4, x5, x6, x7, x8, y5, y6, y7, y8;
    __m128i a0, a1, a2, a3;
    /*
     * There's at least one block of 256.
     */
    x1 = _mm512_loadu_si512((__m512i *)(buf + 0x00));
    x2 = _mm512_loadu_si512((__m512i *)(buf + 0x40));
    x3 = _mm512_loadu_si512((__m512i *)(buf + 0x80));
    x4 = _mm512_loadu_si512((__m512i *)(buf + 0xC0));
    x1 = _mm512_xor_si512(x1, _mm512_castsi128_si512(_mm_cvtsi32_si128(crc)));
    x0 = _mm512_load_si512((__m512i *)k1k2);
    buf += 256;
    len -= 256;
    /*
     * Parallel fold blocks of 256, if any.
     */
    while (len >= 256)
    {
        x5 = _mm512_clmulepi64_epi128(x1, x0, 0x00);
        x6 = _mm512_clmulepi64_epi128(x2, x0, 0x00);
        x7 = _mm512_clmulepi64_epi128(x3, x0, 0x00);
        x8 = _mm512_clmulepi64_epi128(x4, x0, 0x00);
        x1 = _mm512_clmulepi64_epi128(x1, x0, 0x11);
        x2 = _mm512_clmulepi64_epi128(x2, x0, 0x11);
        x3 = _mm512_clmulepi64_epi128(x3, x0, 0x11);
        x4 = _mm512_clmulepi64_epi128(x4, x0, 0x11);
        y5 = _mm512_loadu_si512((__m512i *)(buf + 0x00));
        y6 = _mm512_loadu_si512((__m512i *)(buf + 0x40));
        y7 = _mm512_loadu_si512((__m512i *)(buf + 0x80));
        y8 = _mm512_loadu_si512((__m512i *)(buf + 0xC0));
        x1 = _mm512_xor_si512(x1, x5);
        x2 = _mm512_xor_si512(x2, x6);
        x3 = _mm512_xor_si512(x3, x7);
        x4 = _mm512_xor_si512(x4, x8);
        x1 = _mm512_xor_si512(x1, y5);
        x2 = _mm512_xor_si512(x2, y6);
        x3 = _mm512_xor_si512(x3, y7);
        x4 = _mm512_xor_si512(x4, y8);
        buf += 256;
        len -= 256;
    }
    /*
     * Fold into 512-bits.
     */
    x0 = _mm512_load_si512((__m512i *)k3k4);
    x5 = _mm512_clmulepi64_epi128(x1, x0, 0x00);
    x1 = _mm512_clmulepi64_epi128(x1, x0, 0x11);
    x1 = _mm512_xor_si512(x1, x2);
    x1 = _mm512_xor_si512(x1, x5);
    x5 = _mm512_clmulepi64_epi128(x1, x0, 0x00);
    x1 = _mm512_clmulepi64_epi128(x1, x0, 0x11);
    x1 = _mm512_xor_si512(x1, x3);
    x1 = _mm512_xor_si512(x1, x5);
    x5 = _mm512_clmulepi64_epi128(x1, x0, 0x00);
    x1 = _mm512_clmulepi64_epi128(x1, x0, 0x11);
    x1 = _mm512_xor_si512(x1, x4);
    x1 = _mm512_xor_si512(x1, x5);
    /*
     * Single fold blocks of 64, if any.
     */
    while (len >= 64)
    {
        x2 = _mm512_loadu_si512((__m512i *)buf);
        x5 = _mm512_clmulepi64_epi128(x1, x0, 0x00);
        x1 = _mm512_clmulepi64_epi128(x1, x0, 0x11);
        x1 = _mm512_xor_si512(x1, x2);
        x1 = _mm512_xor_si512(x1, x5);
        buf += 64;
        len -= 64;
    }
    /*
     * Fold 512-bits to 384-bits.
     */
    a0 = _mm_load_si128((__m128i *)k5k6);
    a1 = _mm512_extracti32x4_epi32(x1, 0);
    a2 = _mm512_extracti32x4_epi32(x1, 1);
    a3 = _mm_clmulepi64_si128(a1, a0, 0x00);
    a1 = _mm_clmulepi64_si128(a1, a0, 0x11);
    a1 = _mm_xor_si128(a1, a3);
    a1 = _mm_xor_si128(a1, a2);
    /*
     * Fold 384-bits to 256-bits.
     */
    a2 = _mm512_extracti32x4_epi32(x1, 2);
    a3 = _mm_clmulepi64_si128(a1, a0, 0x00);
    a1 = _mm_clmulepi64_si128(a1, a0, 0x11);
    a1 = _mm_xor_si128(a1, a3);
    a1 = _mm_xor_si128(a1, a2);
    /*
     * Fold 256-bits to 128-bits.
     */
    a2 = _mm512_extracti32x4_epi32(x1, 3);
    a3 = _mm_clmulepi64_si128(a1, a0, 0x00);
    a1 = _mm_clmulepi64_si128(a1, a0, 0x11);
    a1 = _mm_xor_si128(a1, a3);
    a1 = _mm_xor_si128(a1, a2);
    /*
     * Fold 128-bits to 64-bits.
     */
    a2 = _mm_clmulepi64_si128(a1, a0, 0x10);
    a3 = _mm_setr_epi32(~0, 0, ~0, 0);
    a1 = _mm_srli_si128(a1, 8);
    a1 = _mm_xor_si128(a1, a2);
    a0 = _mm_loadl_epi64((__m128i*)k7k8);
    a2 = _mm_srli_si128(a1, 4);
    a1 = _mm_and_si128(a1, a3);
    a1 = _mm_clmulepi64_si128(a1, a0, 0x00);
    a1 = _mm_xor_si128(a1, a2);
    /*
     * Barret reduce to 32-bits.
     */
    a0 = _mm_load_si128((__m128i*)poly);
    a2 = _mm_and_si128(a1, a3);
    a2 = _mm_clmulepi64_si128(a2, a0, 0x10);
    a2 = _mm_and_si128(a2, a3);
    a2 = _mm_clmulepi64_si128(a2, a0, 0x00);
    a1 = _mm_xor_si128(a1, a2);
    /*
     * Return the crc32.
     */
    return _mm_extract_epi32(a1, 1);
}

uint32_t calc_CRC5(const uint8_t*data,uint64_t len,uint32_t init_value=0)
{
	CRC32_engine eng(M);
	return eng.calc2(data,len,init_value);
}

#endif



constexpr uint64_t len=(1ull<<33)+1;
uint8_t data[len];

void test()
{
	for(uint64_t i=0;i<len;i++)
		data[i]=i&255;
	constexpr int k=10;
	volatile uint32_t crc=calc_CRC3(data,len);

	int t1=clock();

	for(int i=0;i<1;i++)
	{
		volatile uint32_t crc_t=calc_CRC1(data,len);
		if(crc_t!=crc)puts("ERR1");
	}

	int t2=clock();

	for(int i=0;i<1;i++)
	{
		volatile uint32_t crc_t=calc_CRC2(data,len);
		if(crc_t!=crc)puts("ERR2");
	}

	int t3=clock();

	for(int i=0;i<k;i++)
	{
		volatile uint32_t crc_t=calc_CRC3(data,len);
		if(crc_t!=crc)puts("ERR3");
	}

	int t4=clock();

	for(int i=0;i<k;i++)
	{
		volatile uint32_t crc_t=calc_CRC4(data,len);
		if(crc_t!=crc)puts("ERR4");
	}

	int t5=clock();

	for(int i=0;i<k;i++)
	{
		volatile uint32_t crc_t=calc_CRC_nb(data,len,0);
		if(crc_t!=crc)puts("ERR5");
	}

	int t6=clock();

	for(int i=0;i<k;i++)
	{
		volatile uint32_t crc_t=calc_CRC5(data,len,0);
		if(crc_t!=crc)puts("ERR6");
	}

	int t7=clock();

	printf("CRC1: %d\nCRC2: %d\nCRC3: %d\nCRC4: %d\nCRC_nb: %d\nCRC5: %d\n",t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t7-t6);
}


int main()
{
	test();

	// uint32_t m=0x13579246;

	// CRC32_engine crc(m);
	// uint64_t a=0x123456789abcdef0ull;
	// uint64_t b=0x98765432ull;

	// uint32_t ans=crc.mul_mod(a,b);

	// printf("%x %x\n",ans,crc.mul_mod_b(a,b));


}
