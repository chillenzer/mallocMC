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

#include "mallocMC/creationPolicies/Scatter/BitField.hpp"
#include "mallocMC/creationPolicies/Scatter/PageInterpretation.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace mallocMC::CreationPolicies::ScatterAlloc
{
    constexpr const uint32_t maxChunksPerPage = BitMaskSize;

    template<size_t T_numPages>
    struct PageTable
    {
        BitMask _bitMasks[T_numPages]{};
        uint32_t _chunkSizes[T_numPages]{};
        uint32_t _fillingLevels[T_numPages]{};
    };

    inline auto indexOf(void* const pointer, void* start, size_t const stepSize) -> size_t
    {
        return std::distance(reinterpret_cast<char*>(start), reinterpret_cast<char*>(pointer)) / stepSize;
    }


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

        auto create(uint32_t numBytes) -> void*
        {
            if(numBytes > T_pageSize)
            {
                // Not yet implemented.
                return nullptr;
            }
            const auto page = choosePage(numBytes);
            if(page)
            {
                const auto chunk = page.value().firstFreeChunk();
                if(chunk)
                {
                    page.value()._topLevelMask[chunk.value().index].flip();
                    ++page.value()._fillingLevel;
                    return chunk.value().pointer;
                }
            }
            return nullptr;
        }

    private:
        auto choosePage(uint32_t numBytes) -> std::optional<PageInterpretation<T_pageSize>>
        {
            for(size_t i = 0; i < numPages(); ++i)
            {
                if(thisPageIsAppropriate(i, numBytes))
                {
                    pageTable._chunkSizes[i] = numBytes;
                    return std::optional<PageInterpretation<T_pageSize>>{
                        std::in_place_t{},
                        pages[i],
                        pageTable._chunkSizes[i],
                        pageTable._bitMasks[i],
                        pageTable._fillingLevels[i]};
                }
            }
            return std::nullopt;
        }

        auto thisPageIsAppropriate(size_t const index, uint32_t const numBytes) -> bool
        {
            return (pageTable._chunkSizes[index] == numBytes
                    && pageTable._fillingLevels[index]
                        < PageInterpretation<
                              T_pageSize>{pages[index], pageTable._chunkSizes[index], pageTable._bitMasks[index], pageTable._fillingLevels[index]}
                              .numChunks())
                || pageTable._chunkSizes[index] == 0U;
        }

    public:
        auto destroy(void* const pointer) -> void
        {
            auto const pageIndex = indexOf(pointer, pages, T_pageSize);
            auto page = PageInterpretation<T_pageSize>{
                pages[pageIndex],
                pageTable._chunkSizes[pageIndex],
                pageTable._bitMasks[pageIndex],
                pageTable._fillingLevels[pageIndex]};
            --page._fillingLevel;
            auto chunkIndex = page.chunkNumberOf(pointer);
            page._topLevelMask.set(chunkIndex, false);
        }
    };
} // namespace mallocMC::CreationPolicies::ScatterAlloc
