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
#include <optional>

namespace mallocMC::CreationPolicies::ScatterAlloc
{
    constexpr const uint32_t pageTableEntrySize = 4U + 4U + sizeof(BitMask);

    // Computing the number of chunks is not quite trivial: We have to take into account the space for the hierarchical
    // bit field at the end of the page, the size of which again depends on the number of chunks. So, we kind of solve
    // this self-consistently by making a naive estimate and checking if
    inline constexpr auto selfConsistentNumChunks(
        size_t const pageSize,
        uint32_t const chunkSize,
        uint32_t const depth = 0U) -> uint32_t
    {
        auto naiveEstimate = pageSize / chunkSize;
        auto bitsAvailable = powInt(BitMaskSize, depth + 1);
        if(naiveEstimate <= bitsAvailable)
        {
            return naiveEstimate;
        }

        // otherwise let's check how the situation looks with the next level of bit field hierarchy added:
        auto numBitsInNextLevel = bitsAvailable * BitMaskSize;
        // we count memory in bytes not bits:
        auto bitFieldSpaceRequirement = numBitsInNextLevel / 8U; // NOLINT(*magic*)
        auto remainingPageSize = pageSize - bitFieldSpaceRequirement;
        return selfConsistentNumChunks(remainingPageSize, chunkSize, depth + 1);
    }

    template<size_t T_pageSize>
    struct PageInterpretation
    {
        DataPage<T_pageSize>& _data;
        uint32_t& _chunkSize;
        BitMask& _topLevelMask;
        uint32_t& _fillingLevel;

        // this is needed to instantiate this in-place in an std::optional
        PageInterpretation<T_pageSize>(
            DataPage<T_pageSize>& data,
            uint32_t& chunkSize,
            BitMask& topLevelMask,
            uint32_t& fillingLevel)
            : _data(data)
            , _chunkSize(chunkSize)
            , _topLevelMask(topLevelMask)
            , _fillingLevel(fillingLevel)
        {
        }

        [[nodiscard]] auto numChunks() const -> uint32_t
        {
            return selfConsistentNumChunks(T_pageSize, _chunkSize);
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
                atomicAdd(_fillingLevel, 1U);
                bitField().set(chunk.value().index);
                return chunk.value().pointer;
            }
            return nullptr;
        }

        [[nodiscard]] auto firstFreeChunk() const -> std::optional<Chunk>
        {
            auto tree = bitField();
            auto const index = firstFreeBit(tree);
            if(index < noFreeBitFound(tree.depth))
            {
                return std::optional<Chunk>({index, this->operator[](index)});
            }
            return std::nullopt;
        }

        [[nodiscard]] auto bitField() const -> BitFieldTree
        {
            return BitFieldTree{_topLevelMask, bitFieldStart(), bitFieldDepth()};
        }

        [[nodiscard]] auto bitFieldStart() const -> BitMask*
        {
            if(!hasBitField())
            {
                return nullptr;
            }
            return reinterpret_cast<BitMask*>(
                &_data.data[T_pageSize - sizeof(BitMask) * (treeVolume<BitMaskSize>(bitFieldDepth()) - 1)]);
        }

        [[nodiscard]] auto bitFieldDepth() const -> uint32_t
        {
            // We subtract one such that numChunks() == BitMaskSize yields 0.
            auto tmp = (numChunks() - 1) / BitMaskSize;
            uint32_t counter = 0U;
            while(tmp > 0)
            {
                counter++;
                tmp /= BitMaskSize;
            }
            return counter;
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
