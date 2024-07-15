/*
  mallocMC: Memory Allocator for Many Core Architectures.

  Copyright 2024 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Julian Johannes Lenz

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

#pragma once

#include <alpaka/atomic/Op.hpp>
#include <alpaka/atomic/Traits.hpp>
#include <iterator>
#include <type_traits>

namespace mallocMC
{
    template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<U>>>
    ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr auto ceilingDivision(T const numerator, U const denominator) -> T
    {
        return (numerator + (denominator - 1)) / denominator;
    }

    template<typename T_size>
    ALPAKA_FN_INLINE ALPAKA_FN_ACC auto indexOf(
        void const* const pointer,
        void const* const start,
        T_size const stepSize) -> std::make_signed_t<T_size>
    {
        return std::distance(reinterpret_cast<char const*>(start), reinterpret_cast<char const*>(pointer)) / stepSize;
    }

    template<typename TAcc, typename T>
    ALPAKA_FN_INLINE ALPAKA_FN_ACC auto atomicLoad(TAcc const& acc, T& target)
    {
        return alpaka::atomicCas(acc, &target, 0U, 0U);
    }
} // namespace mallocMC
