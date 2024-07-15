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
#include <sys/types.h>

namespace mallocMC::CreationPolicies::ScatterAlloc
{
    template<typename T_HeapConfig, typename T_HashConfig>
    struct Heap
    {
        using MyAccessBlock
            = AccessBlock<T_HeapConfig::accessblocksize, T_HeapConfig::pagesize, T_HeapConfig::wastefactor>;

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

        template<typename AlignmentPolicy, typename AlpakaAcc>
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
                [this, bytes, &acc, startIdx, &hashValue, blockValue](auto const& localAcc, auto const index) mutable
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
        template<typename T_HeapConfig, typename T_HashConfig>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto operator()(
            auto const& /*unused*/,
            Heap<T_HeapConfig, T_HashConfig>* m_heap,
            void* m_heapmem,
            size_t const m_memsize) const
        {
            m_heap->accessBlocks = static_cast<Heap<T_HeapConfig, T_HashConfig>::MyAccessBlock*>(m_heapmem);
            m_heap->heapSize = m_memsize;
        }
    };

} // namespace mallocMC::CreationPolicies::ScatterAlloc

namespace mallocMC::CreationPolicies
{

    template<typename T_HeapConfig, typename T_HashConfig = ScatterAlloc::DefaultScatterHashConfig>
    struct Scatter : public ScatterAlloc::Heap<T_HeapConfig, T_HashConfig>
    {
        static_assert(T_HeapConfig::resetfreedpages, "resetfreedpages = false is no longer implemented.");
        static_assert(T_HeapConfig::wastefactor == 1, "A wastefactor is no yet implemented.");

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

        constexpr const static bool providesAvailableSlots = false;
        static auto classname() -> std::string
        {
            return "Scatter";
        }
    };
} // namespace mallocMC::CreationPolicies
