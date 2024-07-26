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

#include "mallocMC/creationPolicies/Scatter/BitField.hpp"
#include "mallocMC/creationPolicies/Scatter/DataPage.hpp"
#include "mallocMC/mallocMC_utils.hpp"

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <unistd.h>

namespace mallocMC::CreationPolicies::ScatterAlloc
{
    template<uint32_t T_pageSize>
    struct PageInterpretation
    {
    private:
        DataPage<T_pageSize>& _data;
        uint32_t const _chunkSize;

    public:
        // this is needed to instantiate this in-place in an std::optional
        ALPAKA_FN_INLINE ALPAKA_FN_ACC PageInterpretation(DataPage<T_pageSize>& data, uint32_t chunkSize)
            : _data(data)
            , _chunkSize(chunkSize)
        {
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto bitFieldStart(DataPage<T_pageSize>& data, uint32_t const chunkSize)
            -> BitMask*
        {
            return PageInterpretation<T_pageSize>(data, chunkSize).bitFieldStart();
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr static auto numChunks(uint32_t const chunkSize) -> uint32_t
        {
            constexpr auto b = static_cast<uint32_t>(sizeof(BitMask));
            auto const numFull = T_pageSize / (BitMaskSize * chunkSize + b);
            auto const leftOverSpace = T_pageSize - numFull * (BitMaskSize * chunkSize + b);
            auto const numInRemainder = leftOverSpace > b ? (leftOverSpace - b) / chunkSize : 0U;
            return numFull * BitMaskSize + numInRemainder;
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto numChunks() const -> uint32_t
        {
            return numChunks(_chunkSize);
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto chunkPointer(uint32_t index) const -> void*
        {
            return reinterpret_cast<void*>(&_data.data[index * _chunkSize]);
        }


        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto startBitMaskIndex(uint32_t const hashValue) const
        {
            return (hashValue >> 16);
        }

        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto create(TAcc const& acc, uint32_t const hashValue = 0U) -> void*
        {
            auto field = bitField();
            auto const index = field.firstFreeBit(acc, numChunks(), startBitMaskIndex(hashValue));
            return (index < field.noFreeBitFound()) ? chunkPointer(index) : nullptr;
        }

        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto destroy(TAcc const& acc, void* pointer) -> void
        {
            if(_chunkSize == 0)
            {
#if(!defined(NDEBUG) && !BOOST_LANG_CUDA && !BOOST_LANG_HIP)
                throw std::runtime_error{
                    "Attempted to destroy a pointer with chunkSize==0. Likely this page was recently "
                    "(and potentially pre-maturely) freed."};
#endif // NDEBUG
                return;
            }
            auto chunkIndex = chunkNumberOf(pointer);
#if(!defined(NDEBUG) && !BOOST_LANG_CUDA && !BOOST_LANG_HIP)
            if(not isValid(acc, chunkIndex))
            {
                throw std::runtime_error{"Attempted to destroy an invalid pointer! Either the pointer does not point "
                                         "to a valid chunk or it is not marked as allocated."};
            }
#endif // NDEBUG
            bitField().unset(acc, chunkIndex);
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto minimalChunkSize() -> uint32_t
        {
            return 1U;
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto cleanup() -> void
        {
            PageInterpretation<T_pageSize>(_data, minimalChunkSize()).resetBitField();
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto resetBitField() -> void
        {
            // This method is not thread-safe by itself. But it is supposed to be called after acquiring a "lock" in
            // the form of setting the filling level, so that's fine.

            memset(static_cast<void*>(bitFieldStart()), 0U, bitFieldSize());
        }

        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto isValid(TAcc const& acc, void* const pointer) const -> bool
        {
            // This function is neither thread-safe nor particularly performant. It is supposed to be used in tests and
            // debug mode.
            return isValid(acc, chunkNumberOf(pointer));
        }

    private:
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto isValid(TAcc const& acc, int32_t const chunkIndex) const -> bool
        {
            return chunkIndex >= 0 and chunkIndex < static_cast<int32_t>(numChunks()) and isAllocated(acc, chunkIndex);
        }

    public:
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto isAllocated(TAcc const& acc, uint32_t const chunkIndex) const -> bool
        {
            return bitField().get(acc, chunkIndex);
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto bitField() const -> BitFieldFlat
        {
            return BitFieldFlat{{bitFieldStart(), ceilingDivision(numChunks(), BitMaskSize)}};
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto bitFieldStart() const -> BitMask*
        {
            return reinterpret_cast<BitMask*>(&_data.data[T_pageSize - bitFieldSize()]);
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto bitFieldSize() const -> uint32_t
        {
            return bitFieldSize(_chunkSize);
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto bitFieldSize(uint32_t const chunkSize) -> uint32_t
        {
            return sizeof(BitMask) * ceilingDivision(numChunks(chunkSize), BitMaskSize);
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto maxBitFieldSize() -> uint32_t
        {
            // TODO: 1U is likely too generous we need to take the mallocMC allignment policy into account to know the
            // smallest allocation size
            return PageInterpretation<T_pageSize>::bitFieldSize(minimalChunkSize());
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto chunkNumberOf(void* pointer) const -> int32_t
        {
            return indexOf(pointer, &_data, _chunkSize);
        }

        // these are supposed to be temporary objects, don't start messing around with them:
        PageInterpretation(PageInterpretation const&) = delete;
        PageInterpretation(PageInterpretation&&) = delete;
        auto operator=(PageInterpretation const&) -> PageInterpretation& = delete;
        auto operator=(PageInterpretation&&) -> PageInterpretation& = delete;
        ~PageInterpretation() = default;
    };
} // namespace mallocMC::CreationPolicies::ScatterAlloc
