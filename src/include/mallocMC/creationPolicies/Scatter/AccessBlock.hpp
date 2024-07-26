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
#include "mallocMC/creationPolicies/Scatter/PageInterpretation.hpp"
#include "mallocMC/mallocMC_utils.hpp"

#include <algorithm>
#include <alpaka/core/Common.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/idx/Accessors.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/mem/fence/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/mem/view/ViewPlainPtr.hpp>
#include <alpaka/vec/Vec.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <sys/types.h>
#include <vector>

namespace mallocMC::CreationPolicies::ScatterAlloc
{

    template<uint32_t T_numPages>
    struct PageTable
    {
        uint32_t _chunkSizes[T_numPages]{};
        uint32_t _fillingLevels[T_numPages]{};
    };

    template<uint32_t T_size>
    struct Padding
    {
        char padding[T_size]{};
    };

    template<>
    struct Padding<0U>
    {
    };

    template<typename T_HeapConfig, typename T_AlignmentPolicy>
    class AccessBlock
    {
        constexpr static uint32_t const blockSize = T_HeapConfig::accessblocksize;
        constexpr static uint32_t const pageSize = T_HeapConfig::pagesize;
        constexpr static uint32_t const wasteFactor = T_HeapConfig::wastefactor;
        constexpr static bool const resetfreedpages = T_HeapConfig::resetfreedpages;

        using MyPageInterpretation = PageInterpretation<pageSize>;

    protected:
        // This class is supposed to be reinterpeted on a piece of raw memory and not instantiated directly. We set it
        // protected, so we can still test stuff in the future easily.
        AccessBlock() = default;

    public:
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr static auto numPages() -> uint32_t
        {
            constexpr auto numberOfPages = blockSize / (pageSize + sizeof(PageTable<1>));
            // check that the page table entries does not have a padding
            static_assert(sizeof(PageTable<numberOfPages>) == numberOfPages * sizeof(PageTable<1>));
            return numberOfPages;
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto getAvailableSlots(auto const& acc, uint32_t const chunkSize) const
            -> uint32_t
        {
            if(chunkSize < multiPageThreshold())
            {
                return getAvailableChunks(chunkSize);
            }
            return getAvailableMultiPages(acc, chunkSize);
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto pageIndex(void* pointer) const -> int32_t
        {
            return indexOf(pointer, pages, pageSize);
        }

        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto isValid(TAcc const& acc, void* const pointer) -> bool
        {
            if(pointer == nullptr)
            {
                return false;
            }
            auto const index = pageIndex(pointer);
            auto chunkSize = atomicLoad(acc, pageTable._chunkSizes[index]);
            if(chunkSize >= pageSize)
            {
                return true;
            }
            return chunkSize == 0U or atomicLoad(acc, pageTable._fillingLevels[index]) == 0U
                ? false
                : interpret(index, chunkSize).isValid(acc, pointer);
        }

        //! @param  hashValue the default makes testing easier because we can avoid adding the hash to each call^^
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto create(
            TAcc const& acc,
            uint32_t const numBytes,
            uint32_t const hashValue = 0U) -> void*
        {
            void* pointer{nullptr};
            if(numBytes >= multiPageThreshold())
            {
                pointer = createOverMultiplePages(acc, numBytes);
            }
            else
            {
                pointer = createChunk(acc, numBytes, hashValue);
            }
            return pointer;
        }

        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto destroy(TAcc const& acc, void* const pointer) -> void
        {
            auto const index = pageIndex(pointer);
            if(index >= static_cast<int32_t>(numPages()) || index < 0)
            {
#if(!defined(NDEBUG) && !BOOST_LANG_CUDA && !BOOST_LANG_HIP)
                throw std::runtime_error{
                    "Attempted to destroy an invalid pointer! Pointer does not point to any page."};
#endif // NDEBUG
                return;
            }
            auto const chunkSize = atomicLoad(acc, pageTable._chunkSizes[index]);
            if(chunkSize >= multiPageThreshold())
            {
                destroyOverMultiplePages(acc, index, chunkSize);
            }
            else
            {
                destroyChunk(acc, pointer, index, chunkSize);
            }
        }

    private:
        DataPage<pageSize> pages[numPages()]{};
        PageTable<numPages()> pageTable{};
        Padding<blockSize - sizeof(pages) - sizeof(pageTable)> padding{};

        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr static auto multiPageThreshold() -> uint32_t
        {
            return pageSize - sizeof(BitMaskStorageType<>);
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto interpret(uint32_t const pageIndex, uint32_t const chunkSize)
        {
            return MyPageInterpretation(pages[pageIndex], chunkSize);
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto getAvailableChunks(uint32_t const chunkSize) const -> uint32_t
        {
            // TODO(lenz): This is not thread-safe!
            return std::transform_reduce(
                std::cbegin(pageTable._chunkSizes),
                std::cend(pageTable._chunkSizes),
                std::cbegin(pageTable._fillingLevels),
                0U,
                std::plus<uint32_t>{},
                [this, chunkSize](auto const localChunkSize, auto const fillingLevel)
                {
                    auto const numChunks
                        = MyPageInterpretation::numChunks(localChunkSize == 0 ? chunkSize : localChunkSize);
                    return ((this->isInAllowedRange(localChunkSize, chunkSize) or localChunkSize == 0U)
                            and fillingLevel < numChunks)
                        ? numChunks - fillingLevel
                        : 0U;
                });
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto getAvailableMultiPages(auto const& /*acc*/, uint32_t const chunkSize) const
            -> uint32_t
        {
            // TODO(lenz): This is not thread-safe!
            auto numPagesNeeded = ceilingDivision(chunkSize, pageSize);
            if(numPagesNeeded > numPages())
            {
                return 0U;
            }
            uint32_t sum = 0U;
            for(uint32_t i = 0; i < numPages() - numPagesNeeded + 1;)
            {
                if(std::all_of(
                       pageTable._chunkSizes + i,
                       pageTable._chunkSizes + i + numPagesNeeded,
                       [](auto const& val) { return val == 0U; }))
                {
                    sum += 1;
                    i += numPagesNeeded;
                }
                else
                {
                    ++i;
                }
            }
            return sum;
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto createOverMultiplePages(auto const& acc, uint32_t const numBytes) -> void*
        {
            auto numPagesNeeded = ceilingDivision(numBytes, pageSize);
            if(numPagesNeeded > numPages())
            {
                return nullptr;
            }

            void* result{nullptr};
            for(uint32_t firstIndex = 0; firstIndex < numPages() - (numPagesNeeded - 1) and result == nullptr;
                ++firstIndex)
            {
                auto numPagesAcquired = acquirePages(acc, firstIndex, numPagesNeeded);
                if(numPagesAcquired == numPagesNeeded)
                {
                    // At this point, we have acquired all the pages we need and nobody can mess with them anymore. We
                    // still have to set the chunk size correctly.
                    setChunkSizes(acc, firstIndex, numPagesNeeded, numBytes);
                    result = &pages[firstIndex];
                }
                else
                {
                    releasePages(acc, firstIndex, numPagesAcquired);
                };
            }
            return result;
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto acquirePages(
            auto const& acc,
            uint32_t const firstIndex,
            uint32_t const numPagesNeeded) -> uint32_t
        {
            return managePageOwnerships(acc, firstIndex, numPagesNeeded, 0U, pageSize);
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto releasePages(
            auto const& acc,
            uint32_t const firstIndex,
            uint32_t const numPagesNeeded) -> uint32_t
        {
            return managePageOwnerships(acc, firstIndex, numPagesNeeded, pageSize, 0U);
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto managePageOwnerships(
            auto const& acc,
            uint32_t const firstIndex,
            uint32_t const numPages,
            uint32_t const expectedFilling,
            uint32_t const newFilling) -> uint32_t
        {
            uint32_t index = 0U;
            uint32_t oldFilling = expectedFilling;
            for(index = 0U; index < numPages; ++index)
            {
                oldFilling = alpaka::atomicCas(
                    acc,
                    &pageTable._fillingLevels[firstIndex + index],
                    expectedFilling,
                    newFilling);
                if(oldFilling != expectedFilling)
                {
                    break;
                }
            }
            return index;
        }


        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto setChunkSizes(
            auto const& acc,
            uint32_t const firstIndex,
            uint32_t const numPagesNeeded,
            uint32_t const numBytes) -> void
        {
            for(uint32_t numPagesAcquired = 0U; numPagesAcquired < numPagesNeeded; ++numPagesAcquired)
            {
#if(!defined(NDEBUG) && !BOOST_LANG_CUDA && !BOOST_LANG_HIP)
                auto oldChunkSize =
#endif
                    alpaka::atomicCas(acc, &pageTable._chunkSizes[firstIndex + numPagesAcquired], 0U, numBytes);
#if(!defined(NDEBUG) && !BOOST_LANG_CUDA && !BOOST_LANG_HIP)
                if(oldChunkSize != 0U)
                {
                    throw std::runtime_error{"Unexpected intermediate chunkSize in multi-page allocation."};
                }
#endif // !NDEBUG
            }
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto noFreePageFound()
        {
            return numPages();
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto startPageIndex(auto const& /*acc*/, uint32_t const hashValue)
        {
            return (hashValue >> 8U) % numPages();
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto isValidPageIdx(uint32_t const index) const -> bool
        {
            return index != noFreePageFound() && index < numPages();
        }

        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto createChunk(
            TAcc const& acc,
            uint32_t const numBytes,
            uint32_t const hashValue) -> void*
        {
            auto index = startPageIndex(acc, hashValue);

            // Under high pressure, this loop could potentially run for a long time because the information where and
            // when we started our search is not maintained and/or used. This is a feature, not a bug: Given a
            // consistent state, the loop will terminate once a free chunk is found or when all chunks are filled for
            // long enough that `choosePage` could verify that each page is filled in a single run.
            //
            // The seemingly non-terminating behaviour that we wrap around multiple times can only occur (assuming a
            // consistent, valid state of the data) when there is high demand for memory such that pages that appear
            // free to `choosePage` are repeatedly found but then the free chunks are scooped away by other threads.
            //
            // In the latter case, it is considered desirable to wrap around multiple times until the thread was fast
            // enough to acquire some memory.
            void* pointer = nullptr;
            do
            {
                // TODO(lenz): This can probably be index++.
                index = (index + 1) % numPages();
                uint32_t chunkSize = numBytes;
                index = choosePage(acc, numBytes, index, chunkSize);
                if(isValidPageIdx(index))
                {
                    pointer = MyPageInterpretation{pages[index], chunkSize}.create(acc, hashValue);
                    if(pointer == nullptr)
                    {
                        leavePage(acc, index);
                    }
                }
            } while(isValidPageIdx(index) and pointer == nullptr);
            return pointer;
        }


        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto choosePage(
            TAcc const& acc,
            uint32_t const numBytes,
            uint32_t const startIndex,
            uint32_t& chunkSizeCache) -> uint32_t
        {
            return wrappingLoop(
                acc,
                startIndex,
                numPages(),
                noFreePageFound(),
                [this, numBytes, &chunkSizeCache](auto const& localAcc, auto const index) {
                    return this->thisPageIsAppropriate(localAcc, index, numBytes, chunkSizeCache) ? index
                                                                                                  : noFreePageFound();
                });
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto isInAllowedRange(uint32_t const chunkSize, uint32_t const numBytes) const
        {
            return (chunkSize >= numBytes && chunkSize <= wasteFactor * numBytes);
        }

        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto thisPageIsAppropriate(
            TAcc const& acc,
            uint32_t const index,
            uint32_t const numBytes,
            uint32_t& chunkSizeCache) -> bool
        {
            bool appropriate = false;
            if(enterPage(acc, index, numBytes))
            {
                uint32_t oldChunkSize = alpaka::atomicCas(acc, &pageTable._chunkSizes[index], 0U, numBytes);
                appropriate = (oldChunkSize == 0U || isInAllowedRange(oldChunkSize, numBytes));
                chunkSizeCache = std::max(oldChunkSize, numBytes);
            }
            if(not appropriate)
            {
                leavePage(acc, index);
            }
            return appropriate;
        }

        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC void destroyChunk(
            TAcc const& acc,
            void* pointer,
            uint32_t const pageIndex,
            uint32_t const chunkSize)
        {
            auto page = interpret(pageIndex, chunkSize);
            page.destroy(acc, pointer);
            leavePage(acc, pageIndex);
        }

        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto enterPage(
            TAcc const& acc,
            uint32_t const pageIndex,
            uint32_t const expectedChunkSize) -> bool
        {
            auto const oldFilling = alpaka::atomicAdd(acc, &pageTable._fillingLevels[pageIndex], 1U);
            // We assume that this page has the correct chunk size. If not, the chunk size is either 0 (and oldFilling
            // must be 0, too) or the next check will fail.
            return oldFilling < MyPageInterpretation::numChunks(expectedChunkSize);
        }

        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC void leavePage(TAcc const& acc, uint32_t const pageIndex)
        {
            // This outermost atomicSub is an optimisation: We can fast-track this if we are not responsible for the
            // clean-up. Using 0U -> 1U in the atomicCAS and comparison further down would have the same effect (if the
            // else branch contained the simple subtraction). It's a matter of which case shall have one operation
            // less.
            auto originalFilling = alpaka::atomicSub(acc, &pageTable._fillingLevels[pageIndex], 1U);
            if constexpr(resetfreedpages)
            {
                if(originalFilling == 1U)
                {
                    // This number depends on the chunk size which will at some point get reset to 0 and might even get
                    // set to another value by another thread before our task is complete here. Naively, one could
                    // expect this "weakens" the lock and makes it insecure. But not the case and having it like this
                    // is a feature, not a bug, as is proven in the comments below.
                    auto lock = pageSize;
                    auto latestFilling = alpaka::atomicCas(acc, &pageTable._fillingLevels[pageIndex], 0U, lock);
                    if(latestFilling == 0U)
                    {
                        auto chunkSize = atomicLoad(acc, pageTable._chunkSizes[pageIndex]);
                        if(chunkSize != 0)
                        {
                            // At this point it's guaranteed that the fiilling level is numChunks and thereby locked.
                            // Furthermore, chunkSize cannot have changed because we maintain the invariant that the
                            // filling level is always considered first, so no other thread can have passed that
                            // barrier to reset it.
                            MyPageInterpretation{pages[pageIndex], chunkSize}.cleanup();
                            alpaka::mem_fence(acc, alpaka::memory_scope::Device{});

                            // It is important to keep this after the clean-up line above: Otherwise another thread
                            // with a smaller chunk size might circumvent our lock and already start allocating before
                            // we're done cleaning up.
                            alpaka::atomicCas(acc, &pageTable._chunkSizes[pageIndex], chunkSize, 0U);
                        }

                        // TODO(lenz): Original version had a thread fence at this point in order to invalidate
                        // potentially cached bit masks. Check if that's necessary!

                        // At this point, there might already be another thread (with another chunkSize) on this page
                        // but that's fine. It won't see the full capacity but we can just subtract what we've added
                        // before:
                        alpaka::atomicSub(acc, &pageTable._fillingLevels[pageIndex], lock);
                    }
                }
            }
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC void destroyOverMultiplePages(
            auto const& acc,
            uint32_t const pageIndex,
            uint32_t const chunkSize)
        {
            auto numPagesNeeded = ceilingDivision(chunkSize, pageSize);
            for(uint32_t i = 0; i < numPagesNeeded; ++i)
            {
                auto myIndex = pageIndex + i;
                if constexpr(resetfreedpages)
                {
                    MyPageInterpretation{pages[myIndex], 1U}.cleanup();
                    alpaka::mem_fence(acc, alpaka::memory_scope::Device{});
                    alpaka::atomicCas(acc, &pageTable._chunkSizes[myIndex], chunkSize, 0U);
                }
                alpaka::atomicSub(acc, &pageTable._fillingLevels[myIndex], +pageSize);
            }
        }
    };

} // namespace mallocMC::CreationPolicies::ScatterAlloc
