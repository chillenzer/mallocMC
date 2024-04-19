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
#include <stdexcept>

namespace mallocMC::CreationPolicies::ScatterAlloc
{
    template<size_t T_pageSize>
    struct PageInterpretation
    {
        DataPage<T_pageSize>& _data;
        uint32_t& _chunkSize;
        uint32_t& _fillingLevel;

        // this is needed to instantiate this in-place in an std::optional
        PageInterpretation(DataPage<T_pageSize>& data, uint32_t& chunkSize, uint32_t& fillingLevel)
            : _data(data)
            , _chunkSize(chunkSize)
            , _fillingLevel(fillingLevel)
        {
        }

        [[nodiscard]] auto topLevelMask() const -> BitMask&
        {
            return *PageInterpretation<T_pageSize>::bitFieldStart(_data, _chunkSize);
        }

        static auto bitFieldStart(DataPage<T_pageSize>& data, uint32_t chunkSize) -> BitMask*
        {
            uint32_t fillingLevel{};
            return PageInterpretation<T_pageSize>(data, chunkSize, fillingLevel).bitFieldStart();
        }

        [[nodiscard]] auto numChunks() const -> uint32_t
        {
            return BitMaskSize * T_pageSize / (BitMaskSize * _chunkSize + sizeof(BitMask));
        }

        [[nodiscard]] auto dataSize() const -> size_t
        {
            return numChunks() * _chunkSize;
        }

        [[nodiscard]] auto hasBitField() const -> bool
        {
            return bitFieldDepth() > 0;
        }

        [[nodiscard]] auto operator[](size_t index) const -> void*
        {
            return reinterpret_cast<void*>(&_data.data[index * _chunkSize]);
        }

        auto create() -> void*
        {
            auto chunk = firstFreeChunk();
            if(chunk)
            {
                // TODO(lenz): Move this into firstFreeChunk().
                bitField().set(chunk.value().index);
                return chunk.value().pointer;
            }
            return nullptr;
        }

        auto destroy(void* pointer) -> void
        {
            if(_chunkSize == 0)
            {
                throw std::runtime_error{
                    "Attempted to destroy a pointer with chunkSize==0. Likely this page was recently "
                    "(and potentially pre-maturely) freed."};
            }
            auto chunkIndex = chunkNumberOf(pointer);
            if(isValidDestruction(chunkIndex))
            {
                bitField().set(chunkIndex, false);
                atomicSub(_fillingLevel, 1U);
                // TODO(lenz): this should use the return from atomicAdd
                if(_fillingLevel == 0U)
                {
                    memset(&_data.data[T_pageSize - maxBitFieldSize()], 0U, maxBitFieldSize());
                    // TODO(lenz): First block this page by setting a special value in chunkSize or fillingLevel.
                    // TODO(lenz): this should be atomic CAS
                    _chunkSize = 0U;
                    // TODO(lenz): Clean up full range of possible bitfield.
                }
            }
        }

    private:
        auto isValidDestruction(uint32_t const chunkIndex) -> bool
        {
            // TODO(lenz): Only enable these checks in debug mode.
            if(chunkIndex < 0 || chunkIndex >= numChunks())
            {
                throw std::runtime_error{"Attempted to destroy out-of-bounds pointer. Chunk index out of range!"};
            }
            if(!isAllocated(chunkIndex))
            {
                throw std::runtime_error{"Attempted to destroy un-allocated memory."};
            }
            return true;
        }

    public:
        auto isAllocated(uint32_t const chunkIndex) -> bool
        {
            return bitField().get(chunkIndex);
        }

        [[nodiscard]] auto firstFreeChunk() const -> std::optional<Chunk>
        {
            auto field = bitField();
            auto const index = firstFreeBit(field);
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
            return sizeof(BitMask) * numChunks();
        }

        [[nodiscard]] auto bitFieldDepth() const -> uint32_t
        {
            // We subtract one such that numChunks() == BitMaskSize yields 0.
            return logInt((numChunks() - 1) / BitMaskSize, BitMaskSize);
        }

        [[nodiscard]] auto maxBitFieldSize() -> uint32_t
        {
            return T_pageSize / BitMaskSize * sizeof(BitMask);
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
