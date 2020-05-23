/**********************************************************************
 *
 * Copyright Lei Zhao.
 * contact: leizhao0403@gmail.com
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 **********************************************************************/

#ifndef _FIXED_POINT_H_
#define _FIXED_POINT_H_

#include "includes.h"

template<typename B, unsigned char I, unsigned char F>
class FixedPoint
{
public:
    /* constructor */
    __host__ __device__ FixedPoint()
    {
        m = 0;
    }
    __host__ __device__ FixedPoint(float f)
    {
        m = float_to_fixed(f);
    }

    /* computation operators */
    __host__ __device__ friend FixedPoint<B, I, F> operator +(FixedPoint<B, I, F> x, FixedPoint<B, I, F> y)
    {
        FixedPoint<B, I, F> temp;
        temp.m = static_cast<B>(x.m + y.m);
        return temp;
    }
    __host__ __device__ friend FixedPoint<B, I, F> operator -(FixedPoint<B, I, F> x, FixedPoint<B, I, F> y)
    {
        FixedPoint<B, I, F> temp;
        temp.m = static_cast<B>(x.m - y.m);
        return temp;
    }
    __host__ __device__ friend FixedPoint<B, I, F> operator *(FixedPoint<B, I, F> x, FixedPoint<B, I, F> y)
    {
        FixedPoint<B, I, F> temp;
        temp.m = static_cast<B>(x.m * y.m / factor);
        return temp;
    }
    __host__ __device__ friend FixedPoint<B, I, F> operator /(FixedPoint<B, I, F> x, FixedPoint<B, I, F> y)
    {
        FixedPoint<B, I, F> temp;
        temp.m = static_cast<B>(x.m * factor / y.m);
        return temp;
    }

    /* assignment operators */
    __host__ __device__ FixedPoint<B, I, F> &operator =(FixedPoint<B, I, F> x)
    {
        m = x.m;
        return *this;
    }
    __host__ __device__ FixedPoint<B, I, F> &operator +=(FixedPoint<B, I, F> x)
    {
        m += x.m;
        return *this;
    }
    __host__ __device__ FixedPoint<B, I, F> &operator -=(FixedPoint<B, I, F> x)
    {
        m -= x.m;
        return *this;
    }
    __host__ __device__ FixedPoint<B, I, F> &operator *=(FixedPoint<B, I, F> x)
    {
        m *= x.m;
        m /= factor;
        return *this;
    }
    __host__ __device__ FixedPoint<B, I, F> &operator /=(FixedPoint<B, I, F> x)
    {
        m *= factor;
        m /= x.m;
        return *this;
    }

    /* comparison operators */
    __host__ __device__ friend bool operator ==(FixedPoint<B, I, F> x, FixedPoint<B, I, F> y)
    {
        return x.m == y.m;
    }
    __host__ __device__ friend bool operator !=(FixedPoint<B, I, F> x, FixedPoint<B, I, F> y)
    {
        return x.m != y.m;
    }
    __host__ __device__ friend bool operator >(FixedPoint<B, I, F> x, FixedPoint<B, I, F> y)
    {
        return x.m > y.m;
    }
    __host__ __device__ friend bool operator <(FixedPoint<B, I, F> x, FixedPoint<B, I, F> y)
    {
        return x.m < y.m;
    }
    __host__ __device__ friend bool operator >=(FixedPoint<B, I, F> x, FixedPoint<B, I, F> y)
    {
        return x.m >= y.m;
    }
    __host__ __device__ friend bool operator <=(FixedPoint<B, I, F> x, FixedPoint<B, I, F> y)
    {
        return x.m <= y.m;
    }

    /* sign operators */
    __host__ __device__ const FixedPoint<B, I, F> operator +()
    {
        FixedPoint<B, I, F> temp;
        temp.m = m;
        return temp;
    }
    __host__ __device__ const FixedPoint<B, I, F> operator -()
    {
        FixedPoint<B, I, F> temp;
        temp.m = -m;
        return temp;
    }

    /* converter */
    __host__ __device__ operator float()
	{
        return float(m) / factor;
	}

private:
    /* convert from float to fixed point */
    __host__ __device__ static B float_to_fixed(float d)
    {
        return static_cast<B>(d * factor);
    }

    /* convert from fixed point to float */
    __host__ __device__ static float fixed_to_float(B m)
    {
        return float(m) / factor;
    }

    static const long factor = 1L << F; // used to convert other type to B
    B m;                                // actual value
};

#endif
