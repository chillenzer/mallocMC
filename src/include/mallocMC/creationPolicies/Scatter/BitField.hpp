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

#include "mallocMC/auxiliary.hpp"

#include <bitset>
#include <cstdint>
#include <span>

namespace mallocMC::CreationPolicies::ScatterAlloc
{
    constexpr const uint32_t BitMaskSize = 32U;
    using BitMask = std::bitset<BitMaskSize>;

    struct BitFieldFlat
    {
        std::span<BitMask> data;

        [[nodiscard]] auto get(uint32_t index) const -> bool
        {
            return data[index / BitMaskSize][index % BitMaskSize];
        }

        void set(uint32_t const index, bool value = true)
        {
            data[index / BitMaskSize].set(index % BitMaskSize, value);
        }

        [[nodiscard]] auto begin() const
        {
            return std::begin(data);
        }

        [[nodiscard]] auto end() const
        {
            return std::end(data);
        }

        [[nodiscard]] auto size() const
        {
            return data.size() * BitMaskSize;
        }
    };

    inline auto noFreeBitFound(BitMask const& /*unused*/) -> uint32_t
    {
        return BitMaskSize;
    }

    inline auto noFreeBitFound(BitFieldFlat const& field) -> uint32_t
    {
        return field.size();
    }

    [[nodiscard]] constexpr inline auto firstFreeBit(BitMask const mask, uint32_t const startIndex = 0) -> uint32_t
    {
        // TODO(lenz): we are not yet caring for performance here...
        for(size_t i = startIndex; i < BitMaskSize; ++i) // NOLINT(altera-unroll-loops)
        {
            if(not mask[i])
            {
                return i;
            }
        }
        return noFreeBitFound(mask);
    }

    inline auto firstFreeBit(BitFieldFlat field) -> uint32_t
    {
        for(uint32_t i = 0; i < field.size(); ++i)
        {
            if(!field.get(i))
            {
                return i;
            }
        }
        return noFreeBitFound(field);
    }

} // namespace mallocMC::CreationPolicies::ScatterAlloc
