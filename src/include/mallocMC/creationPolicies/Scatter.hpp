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
#include <thread>
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

    inline auto computeHash([[maybe_unused]] uint32_t const numBytes, size_t const numPages) -> size_t
    {
        return std::hash<std::thread::id>{}(std::this_thread::get_id()) % numPages; // NOLINT(*magic*)
    }

    template<size_t T_blockSize, uint32_t T_pageSize>
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
            if(pointer == nullptr)
            {
                return false;
            }
            auto const index = pageIndex(pointer);
            auto chunkSize = atomicLoad(pageTable._chunkSizes[index]);
            if(chunkSize >= T_pageSize)
            {
                return true;
            }
            return chunkSize == 0U or atomicLoad(pageTable._fillingLevels[index]) == 0U
                ? false
                : interpret(index, chunkSize).isValid(pointer);
        }

        auto create(uint32_t const numBytes) -> void*
        {
            if(numBytes >= T_pageSize)
            {
                return createOverMultiplePages(numBytes);
            }
            return createChunk(numBytes);
        }

        auto destroy(void* const pointer) -> void
        {
            auto const index = pageIndex(pointer);
            if(index > static_cast<ssize_t>(numPages()) || index < 0)
            {
#ifdef DEBUG
                throw std::runtime_error{"Attempted to destroy invalid pointer."};
#endif // DEBUG
                return;
            }
            auto const chunkSize = atomicLoad(pageTable._chunkSizes[index]);
            if(chunkSize >= T_pageSize)
            {
                destroyOverMultiplePages(index, chunkSize);
            }
            else
            {
                destroyChunk(pointer, index, chunkSize);
            }
        }

    private:
        DataPage<T_pageSize> pages[numPages()]{};
        PageTable<numPages()> pageTable{};

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
            auto startIndex = computeHash(numBytes, numPages());

            // Under high pressure, this loop could potentially run for a long time because the information where and
            // when we started our search is not maintained and/or used. This is a feature, not a bug: Given a
            // consistent state, the loop will terminate once a free chunk is found or when all chunks are filled for
            // long enough that `choosePage` could verify that each page is filled in a single run.
            //
            // The seemingly non-terminating behaviour that we wrap around multiple times can only occur (assuming a
            // consistent, valid state of the data) when there is high demand for memory such that pages that appear
            // free to `choosePage` are repeatedly found but then the free chunks scooped away by other threads.
            //
            // In the latter case, it is considered desirable to wrap around multiple times until the thread was fast
            // enough to acquire some memory.
            while(auto page = choosePage(numBytes, startIndex))
            {
                auto pointer = page.value().create();
                if(pointer != nullptr)
                {
                    return pointer;
                }
                startIndex = pageIndex(page.value()[0]) + 1;
            }
            return nullptr;
        }

        auto choosePage(uint32_t const numBytes, size_t const startIndex = 0)
            -> std::optional<PageInterpretation<T_pageSize>>
        {
            for(size_t i = 0; i < numPages(); ++i)
            {
                // TODO(lenz): Check if an "if" statement would yield better performance here.
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
            bool result = false;
            if(enterPage(index, numBytes))
            {
                auto oldChunkSize = atomicCAS(pageTable._chunkSizes[index], 0U, numBytes);
                result = (oldChunkSize == 0U || oldChunkSize == numBytes);
            }
            else
            {
                leavePage(index, numBytes);
                result = false;
            }
            return result;
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

        void destroyChunk(void* pointer, uint32_t const pageIndex, uint32_t const chunkSize)
        {
            auto page = interpret(pageIndex, chunkSize);
            page.destroy(pointer);
            leavePage(pageIndex, chunkSize);
        }

        auto enterPage(uint32_t const pageIndex, uint32_t const chunkSize) -> bool
        {
            auto const oldFilling = atomicAdd(pageTable._fillingLevels[pageIndex], 1U);
            // We assume that this page has the correct chunk size. If not, the chunk size is either 0 (and oldFilling
            // must be 0, too) or the next check will fail.
            return oldFilling < PageInterpretation<T_pageSize>::numChunks(chunkSize);
        }

        void leavePage(uint32_t const pageIndex, uint32_t const chunkSize)
        {
            // This outermost atomicSub is an optimisation: We can fast-track this if we are not responsible for the
            // clean-up. Using 0U -> 1U in the atomicCAS and comparison further down would have the same effect (if the
            // else branch contained the simple subtraction). It's a matter of which case shall have one operation
            // less.
            if(atomicSub(pageTable._fillingLevels[pageIndex], 1U) == 1U)
            {
                // This number depends on the chunk size which will at some point get reset to 0 and might even get set
                // to another value by another thread before our task is complete here. Naively, one could expect this
                // "weakens" the lock and makes it insecure. But not the case and having it like this is a feature, not
                // a bug, as is proven in the comments below.
                auto lock = PageInterpretation<T_pageSize>::numChunks(chunkSize);
                auto latestFilling = atomicCAS(pageTable._fillingLevels[pageIndex], 0U, lock);
                if(latestFilling == 0U)
                {
                    // At this point it's guaranteed that the fiilling level is numChunks and thereby locked.
                    // Furthermore, chunkSize cannot have changed because we maintain the invariant that the filling
                    // level is always considered first, so no other thread can have passed that barrier to reset it.
                    PageInterpretation<T_pageSize>{pages[pageIndex], chunkSize}.cleanup();
                    atomicCAS(pageTable._chunkSizes[pageIndex], chunkSize, 0U);

                    // TODO(lenz): Original version had a thread fence at this point in order to invalidate potentially
                    // cached bit masks. Check if that's necessary!

                    // At this point, there might already be another thread (with another chunkSize) on this page but
                    // that's fine. It won't see the full capacity but we can just subtract what we've added before:
                    atomicSub(pageTable._fillingLevels[pageIndex], lock);
                }
            }
        }

        void destroyOverMultiplePages(size_t const pageIndex, uint32_t const chunkSize)
        {
            auto numPagesNeeded = ceilingDivision(chunkSize, T_pageSize);
            for(uint32_t i = 0; i < numPagesNeeded; ++i)
            {
                pageTable._chunkSizes[pageIndex + i] = 0U;
                pageTable._fillingLevels[pageIndex + i] = 0U;
            }
        }
    };
} // namespace mallocMC::CreationPolicies::ScatterAlloc
