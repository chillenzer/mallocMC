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

#include <cstdint>
#include <limits>
#include <span>

namespace mallocMC::CreationPolicies::ScatterAlloc
{
    constexpr const uint32_t BitMaskSize = 32U;
    template<uint32_t size = BitMaskSize, typename = std::enable_if_t<BitMaskSize == 32U>> // NOLINT(*magic-number*)
    using BitMaskStorageType = uint32_t;
    constexpr const BitMaskStorageType<> allOnes = std::numeric_limits<BitMaskStorageType<BitMaskSize>>::max();

    inline auto singleBit(BitMaskStorageType<> const index) -> BitMaskStorageType<>
    {
        return 1U << index;
    }

    struct BitMask
    {
        // Convention: We start counting from the right, i.e., if mask[0] == 1 and all others are 0, then mask = 0...01
        BitMaskStorageType<BitMaskSize> mask{};
        auto operator[](auto const index) const
        {
            return (atomicLoad(mask) & singleBit(index)) != 0U;
        }

        auto set()
        {
            atomicStore(mask, allOnes);
        }

        auto set(auto const index)
        {
            return atomicOr(mask, singleBit(index));
        }

        auto unset(auto const index)
        {
            return atomicAnd(mask, allOnes ^ singleBit(index));
        }

        auto flip()
        {
            return atomicXor(mask, allOnes);
        }

        auto flip(auto const index)
        {
            return atomicXor(mask, singleBit(index));
        }

        auto operator==(auto const other) const
        {
            return (mask == other);
        }

        // clang-format off
        auto operator<=> (BitMask const other) const
        // clang-format on
        {
            return (mask - other.mask);
        }

        [[nodiscard]] auto none() const
        {
            return mask == 0U;
        }

        [[nodiscard]] auto all() const
        {
            return mask == allOnes;
        }
    };

    struct BitFieldFlat
    {
        std::span<BitMask> data;

        [[nodiscard]] auto get(uint32_t index) const -> bool
        {
            return data[index / BitMaskSize][index % BitMaskSize];
        }

        [[nodiscard]] auto operator[](uint32_t index) const -> BitMask&
        {
            return data[index];
        }

        void set(uint32_t const index) const
        {
            data[index / BitMaskSize].set(index % BitMaskSize);
        }

        void unset(uint32_t const index) const
        {
            data[index / BitMaskSize].unset(index % BitMaskSize);
        }

        [[nodiscard]] auto begin() const
        {
            return std::begin(data);
        }

        [[nodiscard]] auto end() const
        {
            return std::end(data);
        }

        [[nodiscard]] auto numMasks() const
        {
            return data.size();
        }

        [[nodiscard]] auto numBits() const
        {
            return numMasks() * BitMaskSize;
        }
    };

    inline auto noFreeBitFound(BitMask const& /*unused*/) -> uint32_t
    {
        return BitMaskSize;
    }

    inline auto noFreeBitFound(BitFieldFlat const& field) -> uint32_t
    {
        return field.numBits();
    }

    [[nodiscard]] inline auto firstFreeBit(BitMask& mask, uint32_t const startIndex = 0) -> uint32_t
    {
        // TODO(lenz): Don't iterate through all but guess first and then jump to the next free.
        for(uint32_t i = startIndex; i < BitMaskSize; ++i)
        {
            if((atomicOr(mask.mask, singleBit(i)) & singleBit(i)) == 0U)
            {
                return i;
            }
        }
        return noFreeBitFound(mask);
    }

    inline auto firstFreeBit(BitFieldFlat& field, uint32_t numValidBits = 0) -> uint32_t
    {
        if(numValidBits == 0)
        {
            numValidBits = field.numBits();
        }
        for(uint32_t i = 0; i < field.numMasks(); ++i)
        {
            auto indexInMask = firstFreeBit(field[i]);
            if(indexInMask < noFreeBitFound(BitMask{}))
            {
                uint32_t freeBitIndex = indexInMask + BitMaskSize * i;
                if(freeBitIndex < numValidBits)
                {
                    return freeBitIndex;
                }
                return noFreeBitFound(field);
            }
        }
        return noFreeBitFound(field);
    }

} // namespace mallocMC::CreationPolicies::ScatterAlloc
