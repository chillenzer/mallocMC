/*
  mallocMC: Memory Allocator for Many Core Architectures.

  Copyright 2024 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Julian Johannes Lenz, Rene Widera

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

#include "mallocMC/creationPolicies/Scatter/wrappingLoop.hpp"
#include "mallocMC/mallocMC_utils.hpp"

#include <alpaka/core/Common.hpp>
#include <alpaka/intrinsic/Traits.hpp>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace mallocMC::CreationPolicies::ScatterAlloc
{
    constexpr const uint32_t BitMaskSize = 32U;
    template<uint32_t size = BitMaskSize, typename = std::enable_if_t<BitMaskSize == 32U>> // NOLINT(*magic-number*)
    using BitMaskStorageType = uint32_t;
    static constexpr const BitMaskStorageType<BitMaskSize> allOnes
        = std::numeric_limits<BitMaskStorageType<BitMaskSize>>::max();

    ALPAKA_FN_INLINE ALPAKA_FN_ACC auto singleBit(BitMaskStorageType<> const index) -> BitMaskStorageType<>
    {
        return 1U << index;
    }

    template<typename TAcc>
    ALPAKA_FN_INLINE ALPAKA_FN_ACC auto allOnesUpTo(BitMaskStorageType<> const index) -> BitMaskStorageType<>
    {
        return index == 0 ? 0 : (allOnes >> (BitMaskSize - index));
    }

    struct BitMask
    {
        // Convention: We start counting from the right, i.e., if mask[0] == 1 and all others are 0, then mask = 0...01
        BitMaskStorageType<BitMaskSize> mask{};

        ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto noFreeBitFound() -> uint32_t
        {
            return BitMaskSize;
        }

        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto operator()(TAcc const& acc, auto const index)
        {
            return (atomicLoad(acc, mask) & singleBit(index)) != 0U;
        }

        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto set(TAcc const& acc)
        {
            alpaka::atomicOr(acc, &mask, allOnes);
        }

        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto set(TAcc const& acc, auto const index)
        {
            return alpaka::atomicOr(acc, &mask, singleBit(index));
        }

        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto unset(TAcc const& acc, auto const index)
        {
            return alpaka::atomicAnd(acc, &mask, allOnes ^ singleBit(index));
        }

        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto flip(TAcc const& acc)
        {
            return alpaka::atomicXor(acc, &mask, allOnes);
        }

        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto flip(TAcc const& acc, auto const index)
        {
            return alpaka::atomicXor(acc, &mask, singleBit(index));
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto operator==(auto const other) const
        {
            return (mask == other);
        }

        // clang-format off
         ALPAKA_FN_INLINE ALPAKA_FN_ACC auto operator<=> (BitMask const other) const
        // clang-format on
        {
            return (mask - other.mask);
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto none() const
        {
            return mask == 0U;
        }

        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto all(TAcc const& acc) const
        {
            return alpaka::atomicAnd(acc, &mask, allOnes);
        }


        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto firstFreeBit(
            TAcc const& acc,
            uint32_t const startIndex = 0,
            uint32_t const numValidBits = BitMaskSize) -> uint32_t
        {
            return firstFreeBitInBetween(acc, startIndex % BitMaskSize, numValidBits);
        }

    private:
        /**
         *
         * @tparam TAcc
         * @param acc
         * @param startIndex range [0;BitMaskSize)
         * @param endIndex range (0;BitMaskSize]
         * @return
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto firstFreeBitInBetween(
            TAcc const& acc,
            uint32_t const startIndex,
            uint32_t const endIndex) -> uint32_t
        {
            auto result = noFreeBitFound();
            auto oldMask = 0U;

            // This avoids a modulo that's not a power of two and is faster thereby.
            auto const selectedStartBit = startIndex >= endIndex ? 0U : startIndex;
            for(uint32_t i = selectedStartBit; i < endIndex and result == noFreeBitFound();)
            {
                oldMask = alpaka::atomicOr(acc, &mask, singleBit(i));
                if((oldMask & singleBit(i)) == 0U)
                {
                    result = i;
                }

                i = alpaka::ffs(acc, static_cast<std::make_signed_t<BitMaskStorageType<>>>(~oldMask)) - 1;
            }

            return result;
        }
    };

    struct BitFieldFlat
    {
        std::span<BitMask> data;

        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto get(TAcc const& acc, uint32_t index) const -> bool
        {
            return data[index / BitMaskSize](acc, index % BitMaskSize);
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto get(uint32_t index) const -> BitMask&
        {
            return data[index];
        }

        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC void set(TAcc const& acc, uint32_t const index) const
        {
            data[index / BitMaskSize].set(acc, index % BitMaskSize);
        }

        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC void unset(TAcc const& acc, uint32_t const index) const
        {
            data[index / BitMaskSize].unset(acc, index % BitMaskSize);
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto begin() const
        {
            return std::begin(data);
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto end() const
        {
            return std::end(data);
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto numMasks() const
        {
            return data.size();
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto numBits() const
        {
            return numMasks() * BitMaskSize;
        }

        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto firstFreeBit(
            TAcc const& acc,
            uint32_t numValidBits = 0U,
            uint32_t const startIndex = 0U) -> uint32_t
        {
            numValidBits = numValidBits == 0 ? numBits() : numValidBits;
            return wrappingLoop(
                acc,
                startIndex % numMasks(),
                numMasks(),
                noFreeBitFound(),
                [this, numValidBits](TAcc const& localAcc, auto const index)
                { return firstFreeBitAt(localAcc, numValidBits, index); });
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto noFreeBitFound() const -> uint32_t
        {
            return numBits();
        }

    private:
        ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto startBitIndex()
        {
            return laneid();
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto isThisLastMask(uint32_t const numValidBits, uint32_t const index)
        {
            // >= in case index == numValidBits - BitMaskSize
            return (index + 1) * BitMaskSize >= numValidBits;
        }

        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto firstFreeBitAt(
            TAcc const& acc,
            uint32_t const numValidBits,
            uint32_t const maskIdx) -> uint32_t
        {
            auto numValidBitsInLastMask = (numValidBits ? ((numValidBits - 1U) % BitMaskSize + 1U) : 0U);
            auto indexInMask = get(maskIdx).firstFreeBit(
                acc,
                startBitIndex(),
                isThisLastMask(numValidBits, maskIdx) ? numValidBitsInLastMask : BitMaskSize);
            if(indexInMask < BitMask::noFreeBitFound())
            {
                uint32_t freeBitIndex = indexInMask + BitMaskSize * maskIdx;
                if(freeBitIndex < numValidBits)
                {
                    return freeBitIndex;
                }
            }
            return noFreeBitFound();
        }
    };
} // namespace mallocMC::CreationPolicies::ScatterAlloc
