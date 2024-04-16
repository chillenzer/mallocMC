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
    template<size_t T_numPages>
    struct PageTable
    {
        BitMask _bitMasks[T_numPages]{};
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
            return PageInterpretation<T_pageSize>(
                pages[pageIndex],
                pageTable._chunkSizes[pageIndex],
                pageTable._bitMasks[pageIndex],
                pageTable._fillingLevels[pageIndex]);
        }

        auto bitMasks()
        {
            return BitMasksWrapper<T_blockSize, T_pageSize>{*this};
        }

        auto create(uint32_t numBytes) -> void*
        {
            if(numBytes > T_pageSize)
            {
                return handleMultiplePages(numBytes);
            }
            return handleSinglePage(numBytes);
        }

    private:
        auto handleMultiplePages(uint32_t const numBytes) -> void*
        {
            auto numPagesNeeded = ceilingDivision(numBytes, T_pageSize);
            auto chunk = createContiguousPages(numPagesNeeded);
            if(chunk)
            {
                return chunk.value().pointer;
            }
            return nullptr;
        }

        auto handleSinglePage(uint32_t const numBytes) -> void*
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
                    // TODO(lenz): Increment fillingLevel, so everybody is informed that we're interest.
                    pageTable._chunkSizes[index] = numBytes;
                    return std::optional<PageInterpretation<T_pageSize>>{
                        std::in_place_t{},
                        pages[index],
                        pageTable._chunkSizes[index],
                        pageTable._bitMasks[index],
                        pageTable._fillingLevels[index]};
                }
            }
            return std::nullopt;
        }

        auto thisPageIsAppropriate(size_t const index, uint32_t const numBytes) -> bool
        {
            return (pageTable._chunkSizes[index] == numBytes
                    && pageTable._fillingLevels[index] < interpret(index).numChunks())
                || pageTable._chunkSizes[index] == 0U;
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
                    return std::optional<Chunk>({index, &pages[index]});
                };
            }
            return std::nullopt;
        }

    public:
        auto destroy(void* const pointer) -> void
        {
            auto const pageIndex = indexOf(pointer, pages, T_pageSize);
            if(pageIndex > numPages() || pageIndex < 0)
            {
                throw std::runtime_error{"Attempted to destroy invalid pointer."};
            }
            interpret(pageIndex).destroy(pointer);
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
            return accessBlock.interpret(index).bitField().headNode();
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
