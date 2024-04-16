/*
  mallocMC: Memory Allocator for Many Core Architectures.

  Copyright 2024 Helmholtz-Zentrum Dresden - Rossendorf,
                 CERN

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

#include <iterator>
#include <type_traits>

namespace mallocMC
{
    template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<U>>>
    [[nodiscard]] constexpr auto ceilingDivision(T const numerator, U const denominator) -> T
    {
        return (numerator + (denominator - 1)) / denominator;
    }

    // power function for integers, returns base^exp
    template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<U>>>
    inline constexpr auto powInt(T const base, U const exp) -> T
    {
        auto result = 1U;
        for(auto i = 0U; i < exp; ++i)
        {
            result *= base;
        }
        return result;
    }

    // integer logarithm, i.e. "how many times can I divide value by base", its the inverse of powInt for appropriately
    // defined target spaces
    template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<U>>>
    inline constexpr auto logInt(T value, U const base) -> T
    {
        T counter = 0U;
        while(value > 0U)
        {
            counter++;
            value /= base;
        }
        return counter;
    }

    inline auto indexOf(void* const pointer, void* const start, ssize_t const stepSize) -> ssize_t
    {
        return std::distance(reinterpret_cast<char*>(start), reinterpret_cast<char*>(pointer)) / stepSize;
    }

    template<typename T, typename U>
    inline auto atomicAdd(T& lhs, U const& rhs)
    {
        lhs += rhs;
    }
} // namespace mallocMC
