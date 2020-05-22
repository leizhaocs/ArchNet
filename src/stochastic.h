#ifndef _SC_H_
#define _SC_H_

#include "includes.h"

/* stochastic number, using n unsigned int internally to represent B-bit sc number */
template<int N, int B>
class SC
{
public:
    /* constructor */
    __host__ __device__ SC()
    {
        for (int i = 0; i < N; i++)
        {
            bits[i] = 0;
        }
    }
    __host__ __device__ SC(float f)
    {
        float_to_sc(f, bits);
    }

    /* computation operators */
    __host__ __device__ friend SC<N, B> operator +(SC<N, B> x, SC<N, B> y)
    {
        SC<N, B> temp;
        //for (int i = 0; i < N; i++)
        //{
        //    temp.bits[i] = (x.bits[i] & MUX) | (y.bits[i] & ~MUX);
        //}
        float xf = sc_to_float(x.bits);
        float yf = sc_to_float(y.bits);
        float_to_sc(xf+yf, temp.bits);
        return temp;
    }
    __host__ __device__ friend SC<N, B> operator -(SC<N, B> x, SC<N, B> y)
    {
        SC<N, B> temp;
        //for (int i = 0; i < N; i++)
        //{
        //    temp.bits[i] = (x.bits[i] & MUX) | (~y.bits[i] & ~MUX);
        //}
        float xf = sc_to_float(x.bits);
        float yf = sc_to_float(y.bits);
        float_to_sc(xf-yf, temp.bits);
        return temp;
    }
    __host__ __device__ friend SC<N, B> operator *(SC<N, B> x, SC<N, B> y)
    {
        SC<N, B> temp;
        for (int i = 0; i < N; i++)
        {
            temp.bits[i] = ~(x.bits[i] ^ y.bits[i]);
        }
        return temp;
    }
    __host__ __device__ friend SC<N, B> operator /(SC<N, B> x, SC<N, B> y)
    {
        SC<N, B> temp;
        printf("/ not implemented\n");
        float xf = sc_to_float(x.bits);
        float yf = sc_to_float(y.bits);
        float_to_sc(xf/yf, temp.bits);
        return temp;
    }

    /* assignment operators */
    __host__ __device__ SC<N, B> &operator =(SC<N, B> x)
    {
        for (int i = 0; i < N; i++)
        {
            bits[i] = x.bits[i];
        }
        return *this;
    }
    __host__ __device__ SC<N, B> &operator +=(SC<N, B> x)
    {
        //for (int i = 0; i < N; i++)
        //{
        //    bits[i] = (bits[i] & MUX) | (x.bits[i] & ~MUX);
        //}
        float xf = sc_to_float(x.bits);
        float f = sc_to_float(bits);
        float_to_sc(xf+f, bits);
        return *this;
    }
    __host__ __device__ SC<N, B> &operator -=(SC<N, B> x)
    {
        //for (int i = 0; i < N; i++)
        //{
        //    bits[i] = (bits[i] & MUX) | (~x.bits[i] & ~MUX);
        //}
        float xf = sc_to_float(x.bits);
        float f = sc_to_float(bits);
        float_to_sc(f-xf, bits);
        return *this;
    }
    __host__ __device__ SC<N, B> &operator *=(SC<N, B> x)
    {
        for (int i = 0; i < N; i++)
        {
            bits[i] = ~(bits[i] ^ x.bits[i]);
        }
        return *this;
    }
    __host__ __device__ SC<N, B> &operator /=(SC<N, B> x)
    {
        printf("/= not implemented\n");
        float xf = sc_to_float(x.bits);
        float f = sc_to_float(bits);
        float_to_sc(f/xf, bits);
        return *this;
    }

    /* comparison operators */
    __host__ __device__ friend bool operator ==(SC<N, B> x, SC<N, B> y)
    {
        float xf = sc_to_float(x.bits);
        float yf = sc_to_float(y.bits);
        return xf == yf;
    }
    __host__ __device__ friend bool operator !=(SC<N, B> x, SC<N, B> y)
    {
        float xf = sc_to_float(x.bits);
        float yf = sc_to_float(y.bits);
        return xf != yf;
    }
    __host__ __device__ friend bool operator >(SC<N, B> x, SC<N, B> y)
    {
        float xf = sc_to_float(x.bits);
        float yf = sc_to_float(y.bits);
        return xf > yf;
    }
    __host__ __device__ friend bool operator <(SC<N, B> x, SC<N, B> y)
    {
        float xf = sc_to_float(x.bits);
        float yf = sc_to_float(y.bits);
        return xf < yf;
    }
    __host__ __device__ friend bool operator >=(SC<N, B> x, SC<N, B> y)
    {
        float xf = sc_to_float(x.bits);
        float yf = sc_to_float(y.bits);
        return xf >= yf;
    }
    __host__ __device__ friend bool operator <=(SC<N, B> x, SC<N, B> y)
    {
        float xf = sc_to_float(x.bits);
        float yf = sc_to_float(y.bits);
        return xf <= yf;
    }

    /* sign operators */
    __host__ __device__ SC<N, B> operator +()
    {
        SC<N, B> temp;
        for (int i = 0; i < N; i++)
        {
            temp.bits[i] = bits[i];
        }
        return temp;
    }
    __host__ __device__ SC<N, B> operator -()
    {
        SC<N, B> temp;
        for (int i = 0; i < N; i++)
        {
            temp.bits[i] = ~bits[i];
        }
        return temp;
    }

    /* converter */
    __host__ __device__ operator float()
	{
        return sc_to_float(bits);
	}

private:
    /* convert from float to sc */
    __host__ __device__ static void float_to_sc(float f, unsigned int *b)
    {
        float f_ = (f + 1) / 2;
        for (int i = 0; i < N; i++)
        {
            b[i] = 0;
            if ((i+1)*32 <= B)
            {
                for (int j = 0; j < 32; j++)
                {
                    float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                    if (r < f_)
                    {
                        b[i] |= 1 << j;
                    }
                }
            }
            else
            {
                for (int j = 0; j < B%32; j++)
                {
                    float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                    if (r < f_)
                    {
                        b[i] |= 1 << j;
                    }
                }
                break;
            }
        }
    }

    /* convert from float to sc */
    __host__ __device__ static float sc_to_float(unsigned int *b)
    {
        int ones = 0;
        for (int i = 0; i < N; i++)
        {
            if ((i+1)*32 <= B)
            {
                ones += __builtin_popcount(b[i]);
            }
            else
            {
                ones += __builtin_popcount(b[i]&MASK);
                break;
            }
        }
        return (float(ones) / B ) * 2 - 1;
    }

    static const unsigned int MUX = 0xAAAAAAAA; // used for addition, need to generate dynamically for higher precision
    static const unsigned int MASK = (1 << (B%32)) - 1;// used for getting the bits if B is not a multiple of 32
    unsigned int bits[N]; // actual value
};

#endif
