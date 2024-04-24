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
#include "mallocMC/creationPolicies/Scatter/DataPage.hpp"
#include "mallocMC/creationPolicies/Scatter/PageInterpretation.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

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
    class AccessBlock
    {
    public:
        [[nodiscard]] constexpr static auto numPages() -> size_t
        {
            return T_blockSize / (T_pageSize + pageTableEntrySize);
        }

        [[nodiscard]] auto getAvailableSlots(uint32_t const chunkSize) -> size_t
        {
            if(chunkSize < T_pageSize)
            {
                return getAvailableChunks(chunkSize);
            }
            return getAvailableMultiPages(chunkSize);
        }

        auto pageIndex(void* pointer) const -> size_t
        {
            return indexOf(pointer, pages, T_pageSize);
        }

        auto isValid(void* pointer) -> bool
        {
            auto const index = pageIndex(pointer);
            if(pageTable._chunkSizes[index] >= T_pageSize)
            {
                return true;
            }
            return pageTable._chunkSizes[index] == 0U ? false : interpret(index).isValid(pointer);
        }

        auto create(uint32_t numBytes) -> void*
        {
            if(numBytes >= T_pageSize)
            {
                return createOverMultiplePages(numBytes);
            }
            return createChunk(numBytes);
        }

        auto destroy(void* const pointer) -> void
        {
            auto const pageIndex = indexOf(pointer, pages, T_pageSize);
            if(pageIndex > static_cast<ssize_t>(numPages()) || pageIndex < 0)
            {
                throw std::runtime_error{"Attempted to destroy invalid pointer."};
            }
            auto const chunkSize = atomicLoad(pageTable._chunkSizes[pageIndex]);
            if(chunkSize > T_pageSize)
            {
                destroyOverMultiplePages(pageIndex);
            }
            else
            {
                destroyChunk(pointer, pageIndex);
            }
        }

    private:
        DataPage<T_pageSize> pages[numPages()];
        PageTable<numPages()> pageTable;

        auto interpret(size_t const pageIndex)
        {
            return interpret(pageIndex, atomicLoad(pageTable._chunkSizes[pageIndex]));
        }

        auto interpret(size_t const pageIndex, uint32_t const chunkSize)
        {
            return PageInterpretation<T_pageSize>(pages[pageIndex], chunkSize);
        }

        [[nodiscard]] auto getAvailableChunks(uint32_t const chunkSize) const -> size_t
        {
            // This is not thread-safe!
            return std::transform_reduce(
                std::cbegin(pageTable._chunkSizes),
                std::cend(pageTable._chunkSizes),
                std::cbegin(pageTable._fillingLevels),
                0U,
                std::plus<size_t>{},
                [chunkSize](auto const localChunkSize, auto const fillingLevel)
                {
                    return chunkSize == localChunkSize or localChunkSize == 0U
                        ? PageInterpretation<T_pageSize>::numChunks(chunkSize) - fillingLevel
                        : 0U;
                });
        }

        [[nodiscard]] auto getAvailableMultiPages(uint32_t const chunkSize) -> size_t
        {
            // This is the most inefficient but simplest and only thread-safe way I could come up with. If we ever
            // need this in production, we might want to revisit this.
            void* pointer = nullptr;
            std::vector<void*> pointers;
            while((pointer = create(chunkSize)))
            {
                pointers.push_back(pointer);
            }
            std::for_each(std::begin(pointers), std::end(pointers), [this](auto ptr) { destroy(ptr); });
            return pointers.size();
        }

        auto createOverMultiplePages(uint32_t const numBytes) -> void*
        {
            auto numPagesNeeded = ceilingDivision(numBytes, T_pageSize);
            if(numPagesNeeded > numPages())
            {
                return nullptr;
            }
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
                    return std::optional<PageInterpretation<T_pageSize>>{std::in_place_t{}, pages[index], numBytes};
                }
            }
            return std::nullopt;
        }

        auto thisPageIsAppropriate(size_t const index, uint32_t const numBytes) -> bool
        {
            auto oldFilling = atomicAdd(pageTable._fillingLevels[index], 1U);
            if(oldFilling < interpret(index, numBytes).numChunks())
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
            for(size_t index = 0; index < numPages() - (numPagesNeeded - 1); ++index)
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

        void destroyChunk(void* pointer, uint32_t const pageIndex)
        {
            auto page = interpret(pageIndex);
            page.destroy(pointer);
            auto oldFilling = atomicSub(pageTable._fillingLevels[pageIndex], 1U);
            if(oldFilling == 1U)
            {
                auto latestFilling = atomicCAS(pageTable._fillingLevels[pageIndex], 0U, page.numChunks());
                if(latestFilling == 0U)
                {
                    page.cleanup();
                    atomicStore(pageTable._chunkSizes[pageIndex], 0U);
                    atomicStore(pageTable._fillingLevels[pageIndex], 0U);
                }
                else
                {
                    atomicStore(pageTable._fillingLevels[pageIndex], latestFilling);
                }
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
} // namespace mallocMC::CreationPolicies::ScatterAlloc
