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
    /**
     * @brief Number of bits in a bit mask. Most likely you want a power of two here.
     */
    constexpr const uint32_t BitMaskSize = 32U;

    /**
     * @brief Type to store the bit masks in. It's implemented as a template in order to facilitate changing the type
     * depending on BitMaskSize. Use it with its default template argument in order to make your code agnostic of the
     * number configured in BitMaskSize. (Up to providing a template implementation, of course.)
     */
    template<uint32_t size = BitMaskSize, typename = std::enable_if_t<BitMaskSize == 32U>> // NOLINT(*magic-number*)
    using BitMaskStorageType = uint32_t;

    /**
     * @brief Represents a completely filled bit mask, i.e., all bits are one.
     */
    static constexpr const BitMaskStorageType<> allOnes = std::numeric_limits<BitMaskStorageType<>>::max();

    /**
     * @brief Return the bit mask's underlying type with a single bit set (=1) at position index and all others unset
     * (=0).
     *
     * @param index Position of the single bit set.
     * @return Bit mask's underlying type with one bit set.
     */
    ALPAKA_FN_INLINE ALPAKA_FN_ACC auto singleBit(BitMaskStorageType<> const index) -> BitMaskStorageType<>
    {
        return 1U << index;
    }

    /**
     * @brief Return the bit mask's underlying type with all bits up to index from the right are set (=1) and all
     * higher bits are unset (=0).
     *
     * @param index Number of set bits.
     * @return Bit mask's underlying type with index bits set.
     */
    template<typename TAcc>
    ALPAKA_FN_INLINE ALPAKA_FN_ACC auto allOnesUpTo(BitMaskStorageType<> const index) -> BitMaskStorageType<>
    {
        return index == 0 ? 0 : (allOnes >> (BitMaskSize - index));
    }

    /**
     * @class BitMask
     * @brief Represents a bit mask basically wrapping the BitMaskStorageType<>.
     *
     * This class basically provides a convenience interface to the (typically integer) type BitMaskStorageType<> for
     * bit manipulations. It was originally modelled closely after std::bitset which is not necessarily available on
     * device for all compilers, etc.
     *
     * Convention: We start counting from the right, i.e., if mask[0] == 1 and all others are 0, then mask = 0...01
     *
     * CAUTION: This convention is nowhere checked and we might have an implicit assumption on the endianess here. We
     * never investigated because all architectures we're interested in have the same endianess and it works on them.
     *
     */
    struct BitMask
    {
        BitMaskStorageType<> mask{};

        /**
         * @return An invalid bit index indicating the failure of a search in the bit mask.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto noFreeBitFound() -> uint32_t
        {
            return BitMaskSize;
        }

        /**
         * @brief Look up if the index-th bit is set.
         *
         * @param index Bit position to check.
         * @return true if bit is set else false.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto operator()(TAcc const& acc, auto const index) -> bool
        {
            return (atomicLoad(acc, mask) & singleBit(index)) != 0U;
        }

        /**
         * @brief Set all bits (to 1).
         *
         * @return Previous mask.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto set(TAcc const& acc) -> BitMaskStorageType<>
        {
            return alpaka::atomicOr(acc, &mask, +allOnes);
        }

        /**
         * @brief Set the index-th bit (to 1).
         *
         * @param index Bit position to set.
         * @return Previous mask.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto set(TAcc const& acc, auto const index)
        {
            return alpaka::atomicOr(acc, &mask, singleBit(index));
        }

        /**
         * @brief Unset the index-th bit (set it to 0).
         *
         * @param index Bit position to unset.
         * @return Previous mask.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto unset(TAcc const& acc, auto const index)
        {
            return alpaka::atomicAnd(acc, &mask, allOnes ^ singleBit(index));
        }

        /**
         * @brief Flip all bits in the mask.
         *
         * @return Previous mask.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto flip(TAcc const& acc)
        {
            return alpaka::atomicXor(acc, &mask, +allOnes);
        }

        /**
         * @brief Flip the index-th bits in the mask.
         *
         * @param index Bit position to flip.
         * @return Previous mask.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto flip(TAcc const& acc, auto const index)
        {
            return alpaka::atomicXor(acc, &mask, singleBit(index));
        }

        /**
         * @brief Compare with another mask represented by a BitMaskStorageType<>. CAUTION: This does not use atomics
         * and is not thread-safe!
         *
         * This is not implemented thread-safe because to do so we'd need to add the accelerator as a function argument
         * and that would not abide by the interface for operator==. It's intended use is to make (single-threaded)
         * tests more readable, so that's not an issue.
         *
         * @param other Mask to compare with.
         * @return true if all bits are identical else false.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto operator==(BitMaskStorageType<> const other) const -> bool
        {
            return (mask == other);
        }

        /**
         * @brief Spaceship operator comparing with other bit masks. CAUTION: This does not use atomics and is not
         * thread-safe! See operator== for an explanation.
         *
         * @param other Bit mask to compare with.
         * @return Positive if this mask > other mask, 0 for equality, negative otherwise.
         */
        // My version of clang cannot yet handle the spaceship operator apparently:
        // clang-format off
         ALPAKA_FN_INLINE ALPAKA_FN_ACC auto operator<=> (BitMask const other) const
        // clang-format on
        {
            return (mask - other.mask);
        }

        /**
         * @brief Check if no bit is set (=1).
         *
         * @return true if no bit is set else false.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto none() const -> bool
        {
            return mask == 0U;
        }

        /**
         * @brief Interface to the main algorithm of finding a free bit.
         *
         * This algorithm searches for an unset bit and returns its position as an index (which is supposed to be
         * translated into a corresponding chunk by the PageInterpretation). Upon doing so, it also sets this bit
         * because in a multi-threaded context we cannot separate the concerns of retrieving information and acting on
         * the information. It takes a start index that acts as an initial guess but (in the current implementation) it
         * does not implement a strict wrapping loop as the other stages do because this would waste valuable
         * information obtained from the collective operation on all bits in the mask.
         *
         * Additionally, it copes with partial masks by ignoring all bit positions beyond numValidBits.
         *
         * @param numValidBits Bit positions beyond this number will be ignored.
         * @param initialGuess Initial guess for the first look up.
         * @return Bit position of a free bit or noFreeBitFound() in the case of none found.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto firstFreeBit(
            TAcc const& acc,
            uint32_t const numValidBits = BitMaskSize,
            uint32_t const initialGuess = 0) -> uint32_t
        {
            return firstFreeBitInBetween(acc, initialGuess % BitMaskSize, numValidBits);
        }

    private:
        /**
         * @brief Implementation of the main search algorithm. See the public firstFreeBit method for general details.
         * This version assumes a valid range of the input values.
         *
         * @param initialGuess Initial guess for the first look up must be in the range [0;BitMaskSize).
         * @param endIndex Maximal position to consider. Bits further out will be ignored.
         * @return Bit position of a free bit or noFreeBitFound() in the case of none found.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto firstFreeBitInBetween(
            TAcc const& acc,
            uint32_t const initialGuess,
            uint32_t const endIndex) -> uint32_t
        {
            auto result = noFreeBitFound();
            auto oldMask = 0U;

            // This avoids a modulo that's not a power of two and is faster thereby.
            auto const selectedStartBit = initialGuess >= endIndex ? 0U : initialGuess;
            for(uint32_t i = selectedStartBit; i < endIndex and result == noFreeBitFound();)
            {
                oldMask = alpaka::atomicOr(acc, &mask, singleBit(i));
                if((oldMask & singleBit(i)) == 0U)
                {
                    result = i;
                }

                // In case of no free bit found, this will return -1. Storing it in a uint32_t will underflow and
                // result in 0xffffffff but that's okay because it also ends the loop as intended.
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

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto get(uint32_t const index) const -> BitMask&
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
            uint32_t numValidBits,
            uint32_t const startIndex = 0U) -> uint32_t
        {
            numValidBits = numValidBits == 0 ? numBits() : numValidBits;
            return wrappingLoop(
                acc,
                startIndex % numMasks(),
                numMasks(),
                noFreeBitFound(),
                [this, numValidBits](TAcc const& localAcc, auto const index)
                { return this->firstFreeBitAt(localAcc, numValidBits, index); });
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
                isThisLastMask(numValidBits, maskIdx) ? numValidBitsInLastMask : BitMaskSize,
                startBitIndex());
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
