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
#include "mallocMC/creationPolicies/Scatter/BitField.hpp"
#include "mallocMC/creationPolicies/Scatter/DataPage.hpp"

#include <cstdint>
#include <cstring>
#include <optional>
#include <unistd.h>

namespace mallocMC::CreationPolicies::ScatterAlloc
{
    template<size_t T_pageSize>
    struct PageInterpretation
    {
    private:
        DataPage<T_pageSize>& _data;
        uint32_t const _chunkSize;

    public:
        // this is needed to instantiate this in-place in an std::optional
        PageInterpretation(DataPage<T_pageSize>& data, uint32_t chunkSize) : _data(data), _chunkSize(chunkSize)
        {
        }

        static auto bitFieldStart(DataPage<T_pageSize>& data, uint32_t const chunkSize) -> BitMask*
        {
            return PageInterpretation<T_pageSize>(data, chunkSize).bitFieldStart();
        }

        [[nodiscard]] constexpr static auto numChunks(uint32_t const chunkSize) -> uint32_t
        {
            return BitMaskSize * T_pageSize / (static_cast<size_t>(BitMaskSize * chunkSize) + sizeof(BitMask));
        }

        [[nodiscard]] auto numChunks() const -> uint32_t
        {
            return numChunks(_chunkSize);
        }

        [[nodiscard]] auto operator[](size_t index) const -> void*
        {
            return reinterpret_cast<void*>(&_data.data[index * _chunkSize]);
        }

        template<typename TAcc>
        auto create(TAcc const& acc) -> void*
        {
            auto field = bitField();
            auto const index = firstFreeBit(acc, field, numChunks());
            return (index < noFreeBitFound(field)) ? this->operator[](index) : nullptr;
        }

        template<typename TAcc>
        auto destroy(TAcc const& acc, void* pointer) -> void
        {
            if(_chunkSize == 0)
            {
#ifdef DEBUG
                throw std::runtime_error{
                    "Attempted to destroy a pointer with chunkSize==0. Likely this page was recently "
                    "(and potentially pre-maturely) freed."};
#endif // DEBUG
                return;
            }
            auto chunkIndex = chunkNumberOf(pointer);
#ifdef DEBUG
            if(isValid(chunkIndex))
#endif // DEBUG
            {
                bitField().unset(acc, chunkIndex);
            }
        }

        auto cleanup() -> void
        {
            // This method is not thread-safe by itself. But it is supposed to be called after acquiring a "lock" in
            // the form of setting the filling level, so that's fine.
            memset(&_data.data[T_pageSize - maxBitFieldSize()], 0U, maxBitFieldSize());
        }

        template<typename TAcc>
        auto isValid(TAcc const& acc, void* pointer) -> bool
        {
            // This function is neither thread-safe nor particularly performant. It is supposed to be used in tests and
            // debug mode.
            return isValid(acc, chunkNumberOf(pointer));
        }

    private:
        template<typename TAcc>
        auto isValid(TAcc const& acc, uint32_t const chunkIndex) -> bool
        {
            if(chunkIndex >= numChunks())
            {
#ifdef DEBUG
                throw std::runtime_error{"Attempted to destroy out-of-bounds pointer. Chunk index out of range!"};
#endif // DEBUG
                return false;
            }
            if(!isAllocated(acc, chunkIndex))
            {
#ifdef DEBUG
                throw std::runtime_error{"Attempted to destroy un-allocated memory."};
#endif // DEBUG
                return false;
            }
            return true;
        }

    public:
        template<typename TAcc>
        auto isAllocated(TAcc const& acc, uint32_t const chunkIndex) -> bool
        {
            return bitField().get(acc, chunkIndex);
        }

        [[nodiscard]] auto firstFreeChunk() const -> std::optional<Chunk>
        {
            auto field = bitField();
            auto const index = firstFreeBit(field, numChunks());
            if(index < noFreeBitFound(field))
            {
                return std::optional<Chunk>({index, this->operator[](index)});
            }
            return std::nullopt;
        }

        [[nodiscard]] auto bitField() const -> BitFieldFlat
        {
            return BitFieldFlat{{bitFieldStart(), ceilingDivision(numChunks(), BitMaskSize)}};
        }

        [[nodiscard]] auto bitFieldStart() const -> BitMask*
        {
            return reinterpret_cast<BitMask*>(&_data.data[T_pageSize - bitFieldSize()]);
        }

        [[nodiscard]] auto bitFieldSize() const -> uint32_t
        {
            return bitFieldSize(_chunkSize);
        }

        [[nodiscard]] static auto bitFieldSize(uint32_t const chunkSize) -> uint32_t
        {
            return sizeof(BitMask) * ceilingDivision(numChunks(chunkSize), BitMaskSize);
        }

        [[nodiscard]] static auto maxBitFieldSize() -> uint32_t
        {
            return PageInterpretation<T_pageSize>::bitFieldSize(1U);
        }

        [[nodiscard]] auto chunkNumberOf(void* pointer) -> uint32_t
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
