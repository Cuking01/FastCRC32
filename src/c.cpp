#include<stdio.h>
#include<cmath>
#include<time.h>
#include<stdint.h>

constexpr uint32_t M=0x04c11db7;

uint32_t calc_CRC1(const uint8_t*data,int len,uint32_t init_value=0)
{
    uint32_t r=init_value;
    for(int i=0;i<len;i++)
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

uint32_t calc_CRC2(const uint8_t*data,int len,uint32_t init_value=0)
{
	
	
	static constexpr Table table;
	uint32_t r=init_value;

	
	for(int i=0;i<len;i++)
	{
		r=table[r>>24]^data[i]^(r<<8);
	}
	for(int i=0;i<4;i++)
	{
		r=table[r>>24]^(r<<8);
	}
	return r;
}

uint32_t calc_CRC3(const uint8_t*data,int len,uint32_t init_value=0)
{
	uint32_t table[256];

	table[0]=0;
	for(int i=1;i<256;i++)
	{
		uint32_t tmp=table[i>>1];
		table[i]=(tmp<<1)^(tmp>>31?M:0)^(i&1?M:0);
	}

	int seg_len=len/4;
	int seg0=seg_len*0;
	int seg1=seg_len*1;
	int seg2=seg_len*2;
	int seg3=seg_len*3;
	int seg4=seg_len*4;
	int seg_r=len%4;

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

	auto pow_mod=[&mul_mod](uint32_t x,uint32_t y)->uint32_t
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

	for(int i=0;i<seg_len;i++)
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

	for(int i=0;i<seg_r;i++)
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
	zmm(const zmm&a):x(a.x){}
	zmm(__m512i a):x(a){}
	zmm():zmm(0){};
	
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

struct CRC32_engine
{
	uint32_t MP;
	uint64_t inv_rMP;

	CRC32_engine(uint32_t M):MP(M)
	{
		uint64_t tmp=0x8000000000000000ull;
		for(int i=1;i<64;i++)
		{
			uint64_t t=0;
			for(int j=0;j<i&&j<32;j++)
				t^=(MP>>(31-j))&(tmp>>(i-j-1));
			tmp|=(t&1)<<(63-i);
		}
		inv_rMP=tmp;
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

	uint32_t mul_mod(uint64_t ta,uint64_t tb)
	{
		xmm a(ta),b(tb);
		alignas(64) uint64_t tmp[2];
		a=_mm_clmulepi64_si128(a,b,0);
		print(a);
		printf("%llx\n",inv_rMP);
		b=_mm_clmulepi64_si128(a,xmm(inv_rMP),0);
		b=_mm_clmulepi64_si128(b,xmm(0x000000000ull+MP),0x01);
		a=_mm_xor_si128(a,b);
		a.store(tmp);
		return mod((uint64_t)mod((uint64_t)mod(tmp[1])<<32)<<32)^mod(tmp[0]);
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
};

uint32_t calc_CRC4(const uint8_t*data,int len,uint32_t init_value=0)
{
	zmm a(0b111_i64),b(0b101_i64);
	zmm c=_mm512_clmulepi64_epi128(a,b,0);
	alignas(64) int64_t tmp[8];

	zmm d=_mm512_permutexvar_epi32(a,b);
	c.store(tmp);
	return tmp[0];
}

#endif

constexpr int len=(1<<24)+1;
uint8_t data[len];

void test()
{
	for(int i=0;i<len;i++)
		data[i]=i&255;
	constexpr int k=5;
	volatile uint32_t crc=calc_CRC1(data,len);

	int t1=clock();

	for(int i=0;i<k;i++)
	{
		volatile uint32_t crc_t=calc_CRC1(data,len);
		if(crc_t!=crc)puts("ERR1");
	}

	int t2=clock();

	for(int i=0;i<k;i++)
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
		volatile uint32_t crc_t=calc_CRC3(data,len);
		if(crc_t!=crc)puts("ERR4");
	}

	int t5=clock();

	printf("CRC1: %d\nCRC2: %d\nCRC3: %d\nCRC4: %d\n",t2-t1,t3-t2,t4-t3,t5-t4);
}


int main()
{
	uint32_t m=0x99824435;

	CRC32_engine crc(m);
	uint64_t a=0x123456789abcull;
	uint64_t b=0x23456789abcdull;

	uint32_t ans=crc.mul_mod(a,b);

	printf("%u %u\n",ans,crc.mul_mod_b(a,b));


}
