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

#include "mallocMC/creationPolicies/Scatter/AccessBlock.hpp"

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
#include <limits>
#include <numeric>
#include <sys/types.h>
#include <type_traits>

namespace mallocMC::CreationPolicies::ScatterAlloc
{
    template<typename T_HeapConfig, typename T_HashConfig, typename T_AlignmentPolicy>
    struct Heap
    {
        using MyAccessBlock = AccessBlock<T_HeapConfig, T_AlignmentPolicy>;

        static_assert(
            T_HeapConfig::accessblocksize
                < std::numeric_limits<std::make_signed_t<decltype(T_HeapConfig::accessblocksize)>>::max(),
            "Your access block size must be smaller than the maximal value of its signed type because we are using "
            "differences in the code occasionally.");

        static_assert(
            T_HeapConfig::pagesize < std::numeric_limits<std::make_signed_t<decltype(T_HeapConfig::pagesize)>>::max(),
            "Your page size must be smaller than the maximal value of its signed type because we are using "
            "differences in the code occasionally.");

        static_assert(
            T_HeapConfig::accessblocksize == sizeof(MyAccessBlock),
            "The real access block must have the same size as configured in order to make alignment more easily "
            "predictable.");

        size_t heapSize{};
        MyAccessBlock* accessBlocks{};
        volatile uint32_t block = 0U;

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto numBlocks() const -> uint32_t
        {
            return heapSize / T_HeapConfig::accessblocksize;
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto noFreeBlockFound() const -> uint32_t
        {
            return numBlocks();
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto startBlockIndex(
            auto const& /*acc*/,
            uint32_t const blockValue,
            uint32_t const hashValue)
        {
            return ((hashValue % T_HashConfig::blockStride) + (blockValue * T_HashConfig::blockStride)) % numBlocks();
        }

        template<typename AlpakaAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto create(const AlpakaAcc& acc, uint32_t const bytes) -> void*
        {
            auto blockValue = block;
            auto hashValue = T_HashConfig::template hash<T_HeapConfig::pagesize>(acc, bytes);
            auto startIdx = startBlockIndex(acc, blockValue, hashValue);
            return wrappingLoop(
                acc,
                startIdx,
                numBlocks(),
                static_cast<void*>(nullptr),
                [this, bytes, startIdx, &hashValue, blockValue](auto const& localAcc, auto const index) mutable
                {
                    auto ptr = accessBlocks[index].create(localAcc, bytes, hashValue);
                    if(!ptr && index == startIdx)
                    {
                        // This is not thread-safe but we're fine with that. It's just a fuzzy thing to occasionally
                        // increment and it's totally okay if its value is not quite deterministic.
                        if(blockValue == block)
                        {
                            block = blockValue + 1;
                        }
                    }
                    return ptr;
                });
        }

        template<typename AlpakaAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto destroy(const AlpakaAcc& acc, void* pointer) -> void
        {
            // indexOf requires the access block size instead of blockSize in case the reinterpreted AccessBlock
            // object is smaller than blockSize.
            auto blockIndex = indexOf(pointer, accessBlocks, sizeof(MyAccessBlock));
            accessBlocks[blockIndex].destroy(acc, pointer);
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto getAvailableSlotsDeviceFunction(auto const& acc, uint32_t const chunkSize)
            -> size_t
        {
            // TODO(lenz): Not thread-safe.
            return std::transform_reduce(
                accessBlocks,
                accessBlocks + numBlocks(),
                0U,
                std::plus<size_t>{},
                [&acc, chunkSize](auto& accessBlock) { return accessBlock.getAvailableSlots(acc, chunkSize); });
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto getAvailableSlotsAccelerator(auto const& acc, uint32_t const chunkSize)
            -> size_t
        {
            return getAvailableSlotsDeviceFunction(acc, chunkSize);
        }

    protected:
        // This class is supposed to be reinterpeted on a piece of raw memory and not instantiated directly. We set it
        // protected, so we can still test stuff in the future easily.
        Heap() = default;
    };

    struct DefaultScatterHashConfig
    {
        template<uint32_t T_pageSize>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto hash(auto const& /*acc*/, uint32_t const numBytes) -> uint32_t
        {
            const uint32_t relative_offset = warpSize * numBytes / T_pageSize;
            return (
                numBytes * hashingK + hashingDistMP * smid()
                + (hashingDistWP + hashingDistWPRel * relative_offset) * warpid());
        }

        static constexpr uint32_t hashingK = 38183;
        static constexpr uint32_t hashingDistMP = 17497;
        static constexpr uint32_t hashingDistWP = 1;
        static constexpr uint32_t hashingDistWPRel = 1;
        static constexpr uint32_t blockStride = 4;
    };

    struct InitKernel
    {
        template<typename T_HeapConfig, typename T_HashConfig, typename T_AlignmentPolicy>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto operator()(
            auto const& /*unused*/,
            Heap<T_HeapConfig, T_HashConfig, T_AlignmentPolicy>* m_heap,
            void* m_heapmem,
            size_t const m_memsize) const
        {
            m_heap->accessBlocks
                = static_cast<Heap<T_HeapConfig, T_HashConfig, T_AlignmentPolicy>::MyAccessBlock*>(m_heapmem);
            m_heap->heapSize = m_memsize;
        }
    };

} // namespace mallocMC::CreationPolicies::ScatterAlloc

namespace mallocMC::CreationPolicies
{
    template<typename T_HeapConfig, typename T_HashConfig = ScatterAlloc::DefaultScatterHashConfig>
    struct Scatter
    {
        template<typename T_AlignmentPolicy>
        using AlignmentAwarePolicy = ScatterAlloc::Heap<T_HeapConfig, T_HashConfig, T_AlignmentPolicy>;

        static auto classname() -> std::string
        {
            return "Scatter";
        }

        constexpr static auto const providesAvailableSlots = true;

        ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto isOOM(void* pointer, uint32_t const /*unused size*/) -> bool
        {
            return pointer == nullptr;
        }

        template<typename TAcc>
        static void initHeap(auto& dev, auto& queue, auto* heap, void* pool, size_t memsize)
        {
            using Dim = typename alpaka::trait::DimType<TAcc>::type;
            using Idx = typename alpaka::trait::IdxType<TAcc>::type;
            using VecType = alpaka::Vec<Dim, Idx>;

            auto poolView = alpaka::createView(dev, reinterpret_cast<char*>(pool), alpaka::Vec<Dim, Idx>(memsize));
            alpaka::memset(queue, poolView, 0U);
            alpaka::wait(queue);

            auto workDivSingleThread
                = alpaka::WorkDivMembers<Dim, Idx>{VecType::ones(), VecType::ones(), VecType::ones()};
            alpaka::exec<TAcc>(queue, workDivSingleThread, ScatterAlloc::InitKernel{}, heap, pool, memsize);
            alpaka::wait(queue);
        }

        template<typename AlpakaAcc, typename AlpakaDevice, typename AlpakaQueue, typename T_DeviceAllocator>
        static auto getAvailableSlotsHost(
            AlpakaDevice& dev,
            AlpakaQueue& queue,
            uint32_t const slotSize,
            T_DeviceAllocator* heap) -> unsigned
        {
            using Dim = typename alpaka::trait::DimType<AlpakaAcc>::type;
            using Idx = typename alpaka::trait::IdxType<AlpakaAcc>::type;
            using VecType = alpaka::Vec<Dim, Idx>;

            auto d_slots = alpaka::allocBuf<size_t, Idx>(dev, Idx{1});
            alpaka::memset(queue, d_slots, 0, Idx{1});
            auto d_slotsPtr = alpaka::getPtrNative(d_slots);

            auto getAvailableSlotsKernel = [heap, slotSize, d_slotsPtr] ALPAKA_FN_ACC(const AlpakaAcc& acc) -> void
            { *d_slotsPtr = heap->getAvailableSlotsDeviceFunction(acc, slotSize); };

            alpaka::wait(queue);
            alpaka::exec<AlpakaAcc>(
                queue,
                alpaka::WorkDivMembers<Dim, Idx>{VecType::ones(), VecType::ones(), VecType::ones()},
                getAvailableSlotsKernel);
            alpaka::wait(queue);

            auto const platform = alpaka::Platform<alpaka::DevCpu>{};
            const auto hostDev = alpaka::getDevByIdx(platform, 0);

            auto h_slots = alpaka::allocBuf<size_t, Idx>(hostDev, Idx{1});
            alpaka::memcpy(queue, h_slots, d_slots);
            alpaka::wait(queue);

            return *alpaka::getPtrNative(h_slots);
        }
    };
} // namespace mallocMC::CreationPolicies
