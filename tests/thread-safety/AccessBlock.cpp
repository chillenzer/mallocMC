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


#include <algorithm>
#include <alpaka/acc/AccCpuSerial.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/mem/buf/BufCpu.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/mem/view/ViewPlainPtr.hpp>
#include <alpaka/platform/Traits.hpp>
#include <alpaka/queue/Properties.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <mallocMC/creationPolicies/Scatter.hpp>
#include <numeric>
#include <type_traits>

using mallocMC::CreationPolicies::ScatterAlloc::AccessBlock;
using mallocMC::CreationPolicies::ScatterAlloc::BitMaskSize;

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;
using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;


constexpr uint32_t pageSize = 1024;
constexpr size_t numPages = 4;
// Page table entry size = sizeof(chunkSize) + sizeof(fillingLevel):
constexpr uint32_t pteSize = 4 + 4;
constexpr size_t blockSize = numPages * (pageSize + pteSize);

// Fill all pages of the given access block with occupied chunks of the given size. This is useful to test the
// behaviour near full filling but also to have a deterministic page and chunk where an allocation must happen
// regardless of the underlying access optimisations etc.
template<size_t T_blockSize, uint32_t T_pageSize>
auto fillWith(AccessBlock<T_blockSize, T_pageSize>& accessBlock, uint32_t const chunkSize) -> std::vector<void*>
{
    std::vector<void*> pointers(accessBlock.getAvailableSlots(chunkSize));
    std::generate(
        std::begin(pointers),
        std::end(pointers),
        [&accessBlock, chunkSize]()
        {
            void* pointer = accessBlock.create(acc, chunkSize);
            REQUIRE(pointer != nullptr);
            return pointer;
        });
    return pointers;
}

template<bool parallel = true>
struct Runner
{
    Runner() : dev(alpaka::getDevByIdx(platform, 0)){};
    ~Runner() = default;
    Runner(const Runner&) = delete;
    Runner(Runner&&) = delete;
    auto operator=(const Runner&) -> Runner& = delete;
    auto operator=(Runner&&) -> Runner& = delete;

    alpaka::Platform<Acc> const platform = {};
    alpaka::Dev<alpaka::Platform<Acc>> const dev;
    alpaka::Queue<Acc, std::conditional_t<parallel, alpaka::NonBlocking, alpaka::Blocking>> queue{dev};
    alpaka::WorkDivMembers<Dim, Idx> const workDiv{Idx{1}, Idx{1}, Idx{1}};

    template<typename T_Functor, typename... T>
    auto run(T_Functor task, T... pars) -> Runner&
    {
        alpaka::enqueue(queue, alpaka::createTaskKernel<Acc>(workDiv, task, pars...));
        return *this;
    }

    auto join()
    {
        if constexpr(parallel)
        {
            alpaka::wait(queue);
        }
    }
};

struct ContentGenerator
{
    uint32_t counter{0U};

    auto operator()() -> uint32_t
    {
        return counter++;
    }
};

struct Create
{
    auto operator()(Acc const& acc, auto* accessBlock, void** pointers, auto chunkSize) const
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        pointers[idx[0]] = accessBlock->create(acc, chunkSize);
    };

    auto operator()(Acc const& acc, auto* accessBlock, void** pointers, auto* chunkSizes) const
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        pointers[idx[0]] = accessBlock->create(acc, chunkSizes[idx[0]]);
    };
};

struct Destroy
{
    auto operator()(Acc const& acc, auto* accessBlock, void** pointers) const
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        accessBlock->destroy(acc, pointers[idx[0]]);
    };
};

TEST_CASE("Threaded AccessBlock")
{
    AccessBlock<blockSize, pageSize> accessBlock{};

    auto const destroy = [&accessBlock](Acc const& acc, void* pointer) { accessBlock.destroy(acc, pointer); };
    std::vector<void*> pointers(2, reinterpret_cast<void*>(1U));
    auto& pointer1 = pointers[0];
    auto& pointer2 = pointers[1];
    std::vector<uint32_t> const chunkSizes({32U, 512U});
    uint32_t const& chunkSize1 = chunkSizes[0];
    uint32_t const& chunkSize2 = chunkSizes[1];

    alpaka::Platform<Acc> const platformAcc = {};
    alpaka::Platform<alpaka::AccCpuSerial<Dim, Idx>> const platformHost = {};
    alpaka::Dev<alpaka::Platform<Acc>> const devAcc(alpaka::getDevByIdx(platformAcc, 0));
    alpaka::Dev<alpaka::Platform<Acc>> const devHost(alpaka::getDevByIdx(platformHost, 0));
    alpaka::Queue<Acc, alpaka::Blocking> queue{devAcc};

    SECTION("creates second memory somewhere else.")
    {
        auto extents = alpaka::Vec<Dim, Idx>(2U);
        alpaka::Buf<Acc, void*, Dim, Idx> devPointers(alpaka::allocBuf<void*, Idx>(devAcc, extents));

        alpaka::WorkDivMembers<Dim, Idx> const workDiv{Idx{1}, Idx{2}, Idx{1}};
        alpaka::exec<Acc>(queue, workDiv, Create{}, &accessBlock, alpaka::getPtrNative(devPointers), chunkSize1);
        alpaka::wait(queue);

        auto hostPointers = alpaka::createView(devHost, pointers.data(), extents);
        alpaka::memcpy(queue, hostPointers, devPointers);
        alpaka::wait(queue);

        CHECK(pointer1 != pointer2);
    }

    SECTION("creates memory of different chunk size in different pages.")
    {
        auto extents = alpaka::Vec<Dim, Idx>(2U);
        alpaka::Buf<Acc, void*, Dim, Idx> devPointers(alpaka::allocBuf<void*, Idx>(devAcc, extents));
        alpaka::Buf<Acc, uint32_t, Dim, Idx> devChunkSizes(alpaka::allocBuf<uint32_t, Idx>(devAcc, extents));
        alpaka::memcpy(queue, devChunkSizes, alpaka::createView(devHost, chunkSizes.data(), extents));
        alpaka::wait(queue);

        alpaka::WorkDivMembers<Dim, Idx> const workDiv{Idx{1}, Idx{2}, Idx{1}};
        alpaka::exec<Acc>(
            queue,
            workDiv,
            Create{},
            &accessBlock,
            alpaka::getPtrNative(devPointers),
            alpaka::getPtrNative(devChunkSizes));

        alpaka::wait(queue);

        auto hostPointers = alpaka::createView(devHost, pointers.data(), extents);
        alpaka::memcpy(queue, hostPointers, devPointers);
        alpaka::wait(queue);

        CHECK(accessBlock.pageIndex(pointer1) != accessBlock.pageIndex(pointer2));
    }

    SECTION("can handle many different chunk sizes.")
    {
        auto const chunkSizes = []()
        {
            // We want to stay within chunked allocation, so we will always have at least one bit mask present in the
            // page.
            std::vector<uint32_t> tmp(pageSize - BitMaskSize);
            std::iota(std::begin(tmp), std::end(tmp), 1U);
            return tmp;
        }();

        std::vector<void*> pointers(numPages);
        auto runner = Runner<>{};

        for(auto& pointer : pointers)
        {
            runner.run(
                [&accessBlock, &pointer, &chunkSizes](Acc const& acc)
                {
                    // This is assumed to always succeed. We only need some valid pointer to formulate the loop nicely.
                    pointer = accessBlock.create(acc, 1U);
                    for(auto chunkSize : chunkSizes)
                    {
                        accessBlock.destroy(pointer);
                        pointer = nullptr;
                        while(pointer == nullptr)
                        {
                            pointer = accessBlock.create(acc, chunkSize);
                        }
                    }
                });
        }

        runner.join();

        std::sort(std::begin(pointers), std::end(pointers));
        CHECK(std::unique(std::begin(pointers), std::end(pointers)) == std::end(pointers));
    }
}
