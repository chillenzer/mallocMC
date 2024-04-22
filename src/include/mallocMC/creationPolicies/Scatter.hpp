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
#include "mallocMC/creationPolicies/Scatter/PageInterpretation.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace mallocMC::CreationPolicies::ScatterAlloc
{
    constexpr const uint32_t pageTableEntrySize = 4U + 4U;

    template<size_t T_numPages>
    struct PageTable
    {
        uint32_t _chunkSizes[T_numPages]{};
        uint32_t _fillingLevels[T_numPages]{};
    };

    inline auto computeHash([[maybe_unused]] uint32_t const numBytes) -> size_t
    {
        return 42U; // NOLINT(*magic*)
    }

    template<size_t T_blockSize, size_t T_pageSize>
    struct BitMasksWrapper;

    template<size_t T_blockSize, size_t T_pageSize>
    struct AccessBlock
    {
        [[nodiscard]] constexpr static auto numPages() -> size_t
        {
            return T_blockSize / (T_pageSize + pageTableEntrySize);
        }

        [[nodiscard]] constexpr static auto dataSize() -> size_t
        {
            return numPages() * T_pageSize;
        }

        [[nodiscard]] constexpr static auto metadataSize() -> size_t
        {
            return numPages() * pageTableEntrySize;
        }

        DataPage<T_pageSize> pages[numPages()];
        PageTable<numPages()> pageTable;

        auto interpret(size_t const pageIndex)
        {
            return PageInterpretation<T_pageSize>(pages[pageIndex], pageTable._chunkSizes[pageIndex]);
        }

        auto bitMasks()
        {
            return BitMasksWrapper<T_blockSize, T_pageSize>{*this};
        }

        auto create(uint32_t numBytes) -> void*
        {
            if(numBytes > T_pageSize)
            {
                return createOverMultiplePages(numBytes);
            }
            return createChunk(numBytes);
        }

    private:
        auto createOverMultiplePages(uint32_t const numBytes) -> void*
        {
            auto numPagesNeeded = ceilingDivision(numBytes, T_pageSize);
            auto chunk = createContiguousPages(numPagesNeeded);
            if(chunk)
            {
                for(uint32_t i = 0; i < numPagesNeeded; ++i)
                {
                    pageTable._chunkSizes[chunk.value().index + i] = numBytes;
                    pageTable._fillingLevels[chunk.value().index + i] = 1U;
                }
                return chunk.value().pointer;
            }
            return nullptr;
        }

        auto createChunk(uint32_t const numBytes) -> void*
        {
            auto startIndex = computeHash(numBytes);

            // TODO(lenz): This loop is dangerous. If we'd happen to be in an inconsistent state, this would get us
            // into an infinite loop. Check if we can solve this more elegantly.
            while(auto page = choosePage(
                      numBytes,
                      startIndex /*TODO: should get also "originalStart", so we know when we're done*/))
            {
                auto pointer = page.value().create();
                if(pointer != nullptr)
                {
                    return pointer;
                }
                startIndex = indexOf(&page.value()._data, pages, T_pageSize) + 1;
            }
            return nullptr;
        }

        auto choosePage(uint32_t const numBytes, size_t const startIndex = 0)
            -> std::optional<PageInterpretation<T_pageSize>>
        {
            for(size_t i = 0; i < numPages(); ++i)
            {
                auto index = (startIndex + i) % numPages();
                if(thisPageIsAppropriate(index, numBytes))
                {
                    return std::optional<PageInterpretation<T_pageSize>>{
                        std::in_place_t{},
                        pages[index],
                        pageTable._chunkSizes[index]};
                }
            }
            return std::nullopt;
        }

        auto thisPageIsAppropriate(size_t const index, uint32_t const numBytes) -> bool

        {
            auto tmp = numBytes;
            auto oldFilling = atomicAdd(pageTable._fillingLevels[index], 1U);
            if(oldFilling < PageInterpretation<T_pageSize>{pages[index], tmp}.numChunks())
            {
                auto oldChunkSize = atomicCAS(pageTable._chunkSizes[index], 0U, numBytes);
                if((oldChunkSize == 0U || oldChunkSize == numBytes))
                {
                    return true;
                }
            }
            atomicSub(pageTable._fillingLevels[index], 1U);
            return false;
        }

        auto createContiguousPages(uint32_t const numPagesNeeded) -> std::optional<Chunk>
        {
            for(size_t index = 0; index < numPages() - numPagesNeeded; ++index)
            {
                if(std::all_of(
                       &pageTable._chunkSizes[index],
                       &pageTable._chunkSizes[index + numPagesNeeded],
                       [](auto const val) { return val == 0U; }))
                {
                    return std::optional<Chunk>({static_cast<uint32_t>(index), &pages[index]});
                };
            }
            return std::nullopt;
        }

    public:
        auto destroy(void* const pointer) -> void
        {
            auto const pageIndex = indexOf(pointer, pages, T_pageSize);
            if(pageIndex > static_cast<ssize_t>(numPages()) || pageIndex < 0)
            {
                throw std::runtime_error{"Attempted to destroy invalid pointer."};
            }
            if(pageTable._chunkSizes[pageIndex] > T_pageSize)
            {
                destroyOverMultiplePages(pageIndex);
            }
            else
            {
                destroyChunk(pointer, pageIndex);
            }
        }

    private:
        void destroyChunk(void* pointer, uint32_t const pageIndex)
        {
            interpret(pageIndex).destroy(pointer);
            auto oldFilling = atomicSub(pageTable._fillingLevels[pageIndex], 1U);
            if(oldFilling == 1U)
            {
                interpret(pageIndex).cleanup();
                // TODO(lenz): First block this page by setting a special value in chunkSize or fillingLevel.
                // TODO(lenz): this should be atomic CAS
                pageTable._chunkSizes[pageIndex] = 0U;
                // TODO(lenz): Clean up full range of possible bitfield.
            }
        }

        void destroyOverMultiplePages(size_t const pageIndex)
        {
            auto numPagesNeeded = ceilingDivision(pageTable._chunkSizes[pageIndex], T_pageSize);
            for(uint32_t i = 0; i < numPagesNeeded; ++i)
            {
                pageTable._chunkSizes[pageIndex + i] = 0U;
                pageTable._fillingLevels[pageIndex + i] = 0U;
            }
        }
    };

    // TODO(lenz): The structs below are drafts part of an on-going refactoring. They would need significant hardening
    // to appear in a production version of this code.
    template<size_t T_blockSize, size_t T_pageSize>
    struct BitMasksIterator;

    template<size_t T_blockSize, size_t T_pageSize>
    struct BitMasksWrapper
    {
        AccessBlock<T_blockSize, T_pageSize>& accessBlock;

        auto operator[](size_t const index) -> BitMask&
        {
            return *accessBlock.interpret(index).bitField().begin();
        }

        auto begin() -> BitMasksIterator<T_blockSize, T_pageSize>
        {
            return {*this, 0};
        }

        auto end() -> BitMasksIterator<T_blockSize, T_pageSize>
        {
            return {*this, accessBlock.numPages()};
        }
    };

    template<size_t T_blockSize, size_t T_pageSize>
    struct BitMasksIterator
    {
        BitMasksWrapper<T_blockSize, T_pageSize>& wrapper;
        size_t index{0U};

        auto operator++()
        {
            ++index;
            return *this;
        }

        auto operator==(BitMasksIterator<T_blockSize, T_pageSize> const other)
        {
            return index == other.index;
        }

        auto operator!=(BitMasksIterator<T_blockSize, T_pageSize> const other)
        {
            return index != other.index;
        }
        auto operator*() -> BitMask&
        {
            return wrapper[index];
        }
    };


} // namespace mallocMC::CreationPolicies::ScatterAlloc
