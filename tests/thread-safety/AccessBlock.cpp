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


#include "mallocMC/auxiliary.hpp"

#include <algorithm>
#include <alpaka/acc/AccCpuSerial.hpp>
#include <alpaka/acc/AccCpuThreads.hpp>
#include <alpaka/acc/Tag.hpp>
#include <alpaka/acc/TagAccIsEnabled.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/mem/alloc/Traits.hpp>
#include <alpaka/mem/buf/BufCpu.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/mem/view/ViewPlainPtr.hpp>
#include <alpaka/platform/Traits.hpp>
#include <alpaka/queue/Properties.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <mallocMC/creationPolicies/Scatter.hpp>
#include <span>
#include <tuple>

using mallocMC::CreationPolicies::ScatterAlloc::AccessBlock;
using mallocMC::CreationPolicies::ScatterAlloc::BitMaskSize;

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;


constexpr uint32_t pageSize = 1024;
constexpr size_t numPages = 4;
// Page table entry size = sizeof(chunkSize) + sizeof(fillingLevel):
constexpr uint32_t pteSize = 4 + 4;
constexpr size_t blockSize = numPages * (pageSize + pteSize);

using MyAccessBlock = AccessBlock<blockSize, pageSize>;

// Fill all pages of the given access block with occupied chunks of the given size. This is useful to test the
// behaviour near full filling but also to have a deterministic page and chunk where an allocation must happen
// regardless of the underlying access optimisations etc.

struct FillWith
{
    template<typename TAcc, size_t T_blockSize, uint32_t T_pageSize>
    auto operator()(
        TAcc const& acc,
        AccessBlock<T_blockSize, T_pageSize>* accessBlock,
        uint32_t const chunkSize,
        void** result,
        uint32_t const size) const -> void
    {
        std::generate(
            result,
            result + size,
            [&acc, accessBlock, chunkSize]()
            {
                void* pointer{nullptr};
                while(pointer == nullptr)
                {
                    pointer = accessBlock->create(acc, chunkSize);
                }
                return pointer;
            });
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
    template<typename TAcc>
    auto operator()(TAcc const& acc, auto* accessBlock, void** pointers, auto chunkSize) const
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        pointers[idx[0]] = accessBlock->create(acc, chunkSize);
    };

    template<typename TAcc>
    auto operator()(TAcc const& acc, auto* accessBlock, void** pointers, auto* chunkSizes) const
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        pointers[idx[0]] = accessBlock->create(acc, chunkSizes[idx[0]]);
    };
};

struct CreateUntilSuccess
{
    template<typename TAcc>
    auto operator()(TAcc const& acc, auto* accessBlock, void** pointers, auto chunkSize) const
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto& myPointer = pointers[idx[0]];
        while(myPointer == nullptr)
        {
            myPointer = accessBlock->create(acc, chunkSize);
        }
    };
};


struct Destroy
{
    template<typename TAcc>
    auto operator()(TAcc const& acc, auto* accessBlock, void** pointers) const
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        accessBlock->destroy(acc, pointers[idx[0]]);
    };
};

struct IsValid
{
    template<typename TAcc>
    auto operator()(TAcc const& acc, auto* accessBlock, void** pointers, bool* results, size_t const size) const
    {
        std::span<void*> tmpPointers(pointers, size);
        std::span<bool> tmpResults(results, size);
        std::transform(
            std::begin(tmpPointers),
            std::end(tmpPointers),
            std::begin(tmpResults),
            [&acc, accessBlock](auto pointer) { return accessBlock->isValid(acc, pointer); });
    }
};


using Host = alpaka::AccCpuSerial<Dim, Idx>;

template<typename TElem, typename TDevHost, typename TDevAcc>
struct Buffer
{
    TDevAcc m_devAcc;
    TDevHost m_devHost;

    alpaka::Vec<Dim, Idx> m_extents;

    alpaka::Buf<TDevAcc, TElem, Dim, Idx> m_onDevice;
    alpaka::Buf<TDevHost, TElem, Dim, Idx> m_onHost;

    Buffer(TDevHost const& devHost, TDevAcc const& devAcc, auto extents)
        : m_devAcc{devAcc}
        , m_devHost{devHost}
        , m_extents{extents}
        , m_onDevice(alpaka::allocBuf<TElem, Idx>(devAcc, m_extents))
        , m_onHost(alpaka::allocBuf<TElem, Idx>(devHost, m_extents))
    {
    }
};

template<typename TElem, typename TDevHost, typename TDevAcc>
auto makeBuffer(TDevHost const& devHost, TDevAcc const& devAcc, auto extents)
{
    return Buffer<TElem, TDevHost, TDevAcc>{devHost, devAcc, extents};
}

auto createChunkSizes(auto const& devHost, auto const& devAcc, auto& queue)
{
    auto chunkSizes = makeBuffer<uint32_t>(devHost, devAcc, 2U);
    chunkSizes.m_onHost[0] = 32U;
    chunkSizes.m_onHost[1] = 512U;
    alpaka::memcpy(queue, chunkSizes.m_onDevice, chunkSizes.m_onHost);
    return chunkSizes;
}

auto createPointers(auto const& devHost, auto const& devAcc, auto& queue, size_t const size)
{
    auto pointers = makeBuffer<void*>(devHost, devAcc, size);
    std::span<void*> tmp(alpaka::getPtrNative(pointers.m_onHost), pointers.m_extents[0]);
    std::fill(std::begin(tmp), std::end(tmp), reinterpret_cast<void*>(1U));
    alpaka::memcpy(queue, pointers.m_onDevice, pointers.m_onHost);
    return pointers;
}

template<typename TAcc>
auto setup()
{
    alpaka::Platform<TAcc> const platformAcc = {};
    alpaka::Platform<alpaka::AccCpuSerial<Dim, Idx>> const platformHost = {};
    alpaka::Dev<alpaka::Platform<TAcc>> const devAcc(alpaka::getDevByIdx(platformAcc, 0));
    alpaka::Dev<alpaka::Platform<Host>> const devHost(alpaka::getDevByIdx(platformHost, 0));
    alpaka::Queue<TAcc, alpaka::NonBlocking> queue{devAcc};
    return std::make_tuple(platformAcc, platformHost, devAcc, devHost, queue);
}

using Acc = alpaka::AccCpuThreads<Dim, Idx>;
auto fillWith(auto& queue, auto* accessBlock, auto const& chunkSize, auto& pointers)
{
    alpaka::WorkDivMembers<Dim, Idx> const workDivSingleThread{Idx{1}, Idx{1}, Idx{1}};
    alpaka::exec<Acc>(
        queue,
        workDivSingleThread,
        FillWith{},
        accessBlock,
        chunkSize,
        alpaka::getPtrNative(pointers.m_onDevice),
        pointers.m_extents[0]);
    alpaka::wait(queue);
    alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
    alpaka::wait(queue);
}

auto fillAllButOne(auto& queue, auto* accessBlock, auto const& chunkSize, auto& pointers)
{
    fillWith(queue, accessBlock, chunkSize, pointers);
    auto* pointer1 = pointers.m_onHost[0];

    // Destroy exactly one pointer (i.e. the first). This is non-destructive on the actual values in
    // devPointers, so we don't need to wait for the copy before to finish.
    alpaka::WorkDivMembers<Dim, Idx> const workDivSingleThread{Idx{1}, Idx{1}, Idx{1}};
    alpaka::exec<Acc>(queue, workDivSingleThread, Destroy{}, accessBlock, alpaka::getPtrNative(pointers.m_onDevice));
    alpaka::wait(queue);
    return pointer1;
}

template<size_t T_blockSize, uint32_t T_pageSize>
auto freeAllButOneOnFirstPage(auto& queue, AccessBlock<T_blockSize, T_pageSize>* accessBlock, auto& pointers)
{
    std::span<void*> tmp(alpaka::getPtrNative(pointers.m_onHost), pointers.m_extents[0]);
    std::sort(std::begin(tmp), std::end(tmp));
    // This points to the first chunk of page 0.
    auto* pointer1 = tmp[0];
    alpaka::wait(queue);
    alpaka::memcpy(queue, pointers.m_onDevice, pointers.m_onHost);
    alpaka::wait(queue);

    // Delete all other chunks on page 0.
    alpaka::WorkDivMembers<Dim, Idx> const workDiv{
        Idx{1},
        Idx{pointers.m_extents[0] / AccessBlock<T_blockSize, T_pageSize>::numPages() - 1},
        Idx{1}};
    alpaka::exec<Acc>(queue, workDiv, Destroy{}, accessBlock, alpaka::getPtrNative(pointers.m_onDevice) + 1U);
    alpaka::wait(queue);
    return pointer1;
}

auto checkContent(
    auto& devHost,
    auto& devAcc,
    auto& queue,
    auto& pointers,
    auto& content,
    auto& workDiv,
    auto const chunkSize)
{
    auto results = makeBuffer<bool>(devHost, devAcc, pointers.m_extents[0]);
    alpaka::exec<Acc>(
        queue,
        workDiv,
        [](Acc const& acc, auto* content, auto* pointers, auto* results, auto chunkSize)
        {
            auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            auto* begin = reinterpret_cast<uint32_t*>(pointers[idx[0]]);
            auto* end = begin + chunkSize / sizeof(uint32_t);
            results[idx[0]] = std::all_of(begin, end, [idx, content](auto val) { return val == content[idx[0]]; });
        },
        alpaka::getPtrNative(content.m_onDevice),
        alpaka::getPtrNative(pointers.m_onDevice),
        alpaka::getPtrNative(results.m_onDevice),
        chunkSize);
    alpaka::wait(queue);
    alpaka::memcpy(queue, results.m_onHost, results.m_onDevice);
    alpaka::wait(queue);


    std::span<bool> tmpResults(alpaka::getPtrNative(results.m_onHost), results.m_extents[0]);
    auto writtenCorrectly = std::reduce(
        std::cbegin(tmpResults),
        std::cend(tmpResults),
        true,
        [](auto const lhs, auto const rhs) { return lhs && rhs; });

    return writtenCorrectly;
}

auto getAvailableSlots(auto* accessBlock, auto& queue, auto const& devHost, auto const& devAcc, auto chunkSize)
{
    alpaka::WorkDivMembers<Dim, Idx> const workDivSingleThread{Idx{1}, Idx{1}, Idx{1}};
    alpaka::wait(queue);
    auto result = makeBuffer<size_t>(devHost, devAcc, 1U);
    alpaka::wait(queue);
    alpaka::exec<Acc>(
        queue,
        workDivSingleThread,
        [](Acc const& acc, auto* accessBlock, auto chunkSize, auto* result)
        { *result = accessBlock->getAvailableSlots(chunkSize); },
        accessBlock,
        chunkSize,
        alpaka::getPtrNative(result.m_onDevice));
    alpaka::wait(queue);
    alpaka::memcpy(queue, result.m_onHost, result.m_onDevice);
    alpaka::wait(queue);
    auto tmp = result.m_onHost[0];
    alpaka::wait(queue);
    return tmp;
}


template<size_t T_blockSize, uint32_t T_pageSize>
auto pageIndex(AccessBlock<T_blockSize, T_pageSize>* accessBlock, auto* pointer)
{
    // This is a bit dirty: What we should do here is enqueue a kernel that calls accessBlock->pageIndex().
    // But we assume that the access block starts with the first page, so the pointer to the first page equals the
    // pointer to the access block. Not sure if this is reliable if the pointers are device pointers.
    return mallocMC::indexOf(pointer, accessBlock, T_pageSize);
}

TEMPLATE_LIST_TEST_CASE("Threaded AccessBlock", "", alpaka::EnabledAccTags)
{
    using Acc = alpaka::TagToAcc<TestType, Dim, Idx>;
    auto [platformAcc, platformHost, devAcc, devHost, queue] = setup<Acc>();
    auto accessBlockBuf = alpaka::allocBuf<MyAccessBlock, Idx>(devAcc, alpaka::Vec<Dim, Idx>{1U});
    alpaka::memset(queue, accessBlockBuf, 0x00);
    auto* accessBlock = alpaka::getPtrNative(accessBlockBuf);
    auto const chunkSizes = createChunkSizes(devHost, devAcc, queue);
    auto pointers = createPointers(
        devHost,
        devAcc,
        queue,
        getAvailableSlots(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[0]));
    alpaka::wait(queue);

    SECTION("creates second memory somewhere else.")
    {
        alpaka::WorkDivMembers<Dim, Idx> const workDiv{Idx{1}, Idx{2}, Idx{1}};
        alpaka::exec<Acc>(
            queue,
            workDiv,
            Create{},
            accessBlock,
            alpaka::getPtrNative(pointers.m_onDevice),
            chunkSizes.m_onDevice[0]);
        alpaka::wait(queue);

        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        CHECK(pointers.m_onHost[0] != pointers.m_onHost[1]);
    }

    SECTION("creates memory of different chunk size in different pages.")
    {
        alpaka::WorkDivMembers<Dim, Idx> const workDiv{Idx{1}, Idx{2}, Idx{1}};
        alpaka::exec<Acc>(
            queue,
            workDiv,
            Create{},
            accessBlock,
            alpaka::getPtrNative(pointers.m_onDevice),
            alpaka::getPtrNative(chunkSizes.m_onDevice));
        alpaka::wait(queue);

        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        CHECK(pageIndex(accessBlock, pointers.m_onHost[0]) != pageIndex(accessBlock, pointers.m_onHost[1]));
    }

    SECTION("creates partly for insufficient memory with same chunk size.")
    {
        auto* lastFreeChunk = fillAllButOne(queue, accessBlock, chunkSizes.m_onHost[0], pointers);

        // Okay, so here we start the actual test. The situation is the following:
        // There is a single chunk available.
        // We try to do two allocations.
        // So, we expect one to succeed and one to fail.
        alpaka::WorkDivMembers<Dim, Idx> const workDiv{Idx{1}, Idx{2}, Idx{1}};
        alpaka::exec<Acc>(
            queue,
            workDiv,
            Create{},
            accessBlock,
            alpaka::getPtrNative(pointers.m_onDevice),
            chunkSizes.m_onHost[0]);
        alpaka::wait(queue);

        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        CHECK(
            ((pointers.m_onHost[0] == lastFreeChunk and pointers.m_onHost[1] == nullptr)
             or (pointers.m_onHost[1] == lastFreeChunk and pointers.m_onHost[0] == nullptr)));
    }

    SECTION("does not race between clean up and create.")
    {
        fillWith(queue, accessBlock, chunkSizes.m_onHost[0], pointers);
        auto freePage = pageIndex(accessBlock, freeAllButOneOnFirstPage(queue, accessBlock, pointers));

        // Now, pointer1 is the last valid pointer to page 0. Destroying it will clean up the page.
        alpaka::WorkDivMembers<Dim, Idx> const workDivSingleThread{Idx{1}, Idx{1}, Idx{1}};

        alpaka::exec<Acc>(
            queue,
            workDivSingleThread,
            Destroy{},
            accessBlock,
            alpaka::getPtrNative(pointers.m_onDevice));

        alpaka::exec<Acc>(
            queue,
            workDivSingleThread,
            CreateUntilSuccess{},
            accessBlock,
            alpaka::getPtrNative(pointers.m_onDevice),
            chunkSizes.m_onHost[0]);

        alpaka::wait(queue);

        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        CHECK(pageIndex(accessBlock, pointers.m_onHost[0]) == freePage);
    }

    SECTION("destroys two pointers of different size.")
    {
        alpaka::WorkDivMembers<Dim, Idx> const workDiv{Idx{1}, Idx{2}, Idx{1}};
        alpaka::exec<Acc>(
            queue,
            workDiv,
            Create{},
            accessBlock,
            alpaka::getPtrNative(pointers.m_onDevice),
            alpaka::getPtrNative(chunkSizes.m_onDevice));
        alpaka::wait(queue);

        alpaka::exec<Acc>(queue, workDiv, Destroy{}, accessBlock, alpaka::getPtrNative(pointers.m_onDevice));
        alpaka::wait(queue);

        auto result = makeBuffer<bool>(devHost, devAcc, 2U);
        alpaka::WorkDivMembers<Dim, Idx> const workDivSingleThread{Idx{1}, Idx{1}, Idx{1}};
        alpaka::exec<Acc>(
            queue,
            workDivSingleThread,
            IsValid{},
            accessBlock,
            alpaka::getPtrNative(pointers.m_onDevice),
            alpaka::getPtrNative(result.m_onDevice),
            result.m_extents[0]);
        alpaka::wait(queue);

        alpaka::memcpy(queue, result.m_onHost, result.m_onDevice);
        alpaka::wait(queue);

        CHECK(not result.m_onHost[0]);
        CHECK(not result.m_onHost[1]);
    }

    SECTION("destroys two pointers of same size.")
    {
        alpaka::WorkDivMembers<Dim, Idx> const workDiv{Idx{1}, Idx{2}, Idx{1}};
        alpaka::exec<Acc>(
            queue,
            workDiv,
            Create{},
            accessBlock,
            alpaka::getPtrNative(pointers.m_onDevice),
            chunkSizes.m_onHost[0]);
        alpaka::wait(queue);

        alpaka::exec<Acc>(queue, workDiv, Destroy{}, accessBlock, alpaka::getPtrNative(pointers.m_onDevice));
        alpaka::wait(queue);

        auto result = makeBuffer<bool>(devHost, devAcc, 2U);
        alpaka::WorkDivMembers<Dim, Idx> const workDivSingleThread{Idx{1}, Idx{1}, Idx{1}};
        alpaka::exec<Acc>(
            queue,
            workDivSingleThread,
            IsValid{},
            accessBlock,
            alpaka::getPtrNative(pointers.m_onDevice),
            alpaka::getPtrNative(result.m_onDevice),
            result.m_extents[0]);
        alpaka::wait(queue);

        alpaka::memcpy(queue, result.m_onHost, result.m_onDevice);
        alpaka::wait(queue);

        CHECK(not result.m_onHost[0]);
        CHECK(not result.m_onHost[1]);
    }

    SECTION("fills up all chunks in parallel and writes to them.")
    {
        auto content = makeBuffer<uint32_t>(
            devHost,
            devAcc,
            getAvailableSlots(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[0]));
        std::span<uint32_t> tmp(alpaka::getPtrNative(content.m_onHost), content.m_extents[0]);
        std::generate(std::begin(tmp), std::end(tmp), ContentGenerator{});
        alpaka::memcpy(queue, content.m_onDevice, content.m_onHost);
        alpaka::wait(queue);

        alpaka::WorkDivMembers<Dim, Idx> const workDiv{Idx{1}, Idx{pointers.m_extents[0]}, Idx{1}};

        alpaka::exec<Acc>(
            queue,
            workDiv,
            [](Acc const& acc, auto* accessBlock, auto* content, auto* pointers, auto chunkSize)
            {
                auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
                pointers[idx[0]] = accessBlock->create(acc, chunkSize);
                auto* begin = reinterpret_cast<uint32_t*>(pointers[idx[0]]);
                auto* end = begin + chunkSize / sizeof(uint32_t);
                std::fill(begin, end, content[idx[0]]);
            },
            accessBlock,
            alpaka::getPtrNative(content.m_onDevice),
            alpaka::getPtrNative(pointers.m_onDevice),
            chunkSizes.m_onHost[0]);

        alpaka::wait(queue);

        auto writtenCorrectly
            = checkContent(devHost, devAcc, queue, pointers, content, workDiv, chunkSizes.m_onHost[0]);
        CHECK(writtenCorrectly);
    }

    SECTION("destroys all pointers simultaneously.")
    {
        auto const allSlots = getAvailableSlots(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[0]);
        auto const allSlotsOfDifferentSize
            = getAvailableSlots(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[1]);
        fillWith(queue, accessBlock, chunkSizes.m_onHost[0], pointers);

        alpaka::WorkDivMembers<Dim, Idx> const workDiv{Idx{1}, Idx{pointers.m_extents[0]}, Idx{1}};
        alpaka::exec<Acc>(queue, workDiv, Destroy{}, accessBlock, alpaka::getPtrNative(pointers.m_onDevice));
        alpaka::wait(queue);

        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        auto result = makeBuffer<bool>(devHost, devAcc, pointers.m_extents[0]);
        alpaka::WorkDivMembers<Dim, Idx> const workDivSingleThread{Idx{1}, Idx{1}, Idx{1}};
        alpaka::exec<Acc>(
            queue,
            workDivSingleThread,
            IsValid{},
            accessBlock,
            alpaka::getPtrNative(pointers.m_onDevice),
            alpaka::getPtrNative(result.m_onDevice),
            result.m_extents[0]);
        alpaka::wait(queue);

        alpaka::memcpy(queue, result.m_onHost, result.m_onDevice);
        alpaka::wait(queue);

        std::span<bool> tmpResults(alpaka::getPtrNative(result.m_onHost), result.m_extents[0]);
        CHECK(std::none_of(std::cbegin(tmpResults), std::cend(tmpResults), [](auto const val) { return val; }));

        CHECK(getAvailableSlots(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[0]) == allSlots);
        CHECK(
            getAvailableSlots(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[1]) == allSlotsOfDifferentSize);
    }

    SECTION("creates and destroys multiple times.")
    {
        alpaka::WorkDivMembers<Dim, Idx> const workDiv{Idx{1}, Idx{pointers.m_extents[0]}, Idx{1}};

        alpaka::exec<Acc>(
            queue,
            workDiv,
            [](Acc const& acc, auto* accessBlock, auto* pointers, auto chunkSize)
            {
                auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
                for(uint32_t j = 0; j < i; ++j)
                {
                    // `.isValid()` is not thread-safe, so we use this direct assessment:
                    while(pointers[i] == nullptr)
                    {
                        pointers[i] = accessBlock->create(acc, chunkSize);
                    }
                    accessBlock->destroy(acc, pointers[i]);
                    pointers[i] = nullptr;
                }
                while(pointers[i] == nullptr)
                {
                    pointers[i] = accessBlock->create(acc, chunkSize);
                }
            },
            accessBlock,
            alpaka::getPtrNative(pointers.m_onDevice),
            chunkSizes.m_onHost[0]);
        alpaka::wait(queue);
        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        std::span<void*> tmpPointers(alpaka::getPtrNative(pointers.m_onHost), pointers.m_extents[0]);
        std::sort(std::begin(tmpPointers), std::end(tmpPointers));
        CHECK(std::unique(std::begin(tmpPointers), std::end(tmpPointers)) == std::end(tmpPointers));
    }

    SECTION("creates and destroys multiple times with different sizes.")
    {
        SKIP("This test appears to be brittle due to reasons described in the comments.");
        // CAUTION: This test can fail because we are currently using exactly as much space as is available but with
        // multiple different chunk sizes. That means that if one of them occupies more pages than it minimally needs,
        // the other one will lack pages to with their respective chunk size. This seems not to be a problem currently
        // but it might be more of a problem once we move to device and once we include proper scattering.

        // TODO(lenz): This could be solved by "fixing the chunk sizes beforehand", e.g., one could allocate one chunk
        // on each page such that half the pages have one chunk size and the other half has the other chunk size.

        // Make sure that num2 > num1.
        auto num1 = getAvailableSlots(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[0]);
        auto num2 = getAvailableSlots(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[1]);
        auto totalSlots = num1 / 2 + num2 / 2;

        alpaka::WorkDivMembers<Dim, Idx> const workDiv{Idx{1}, Idx{totalSlots}, Idx{1}};

        alpaka::exec<Acc>(
            queue,
            workDiv,
            [num2](Acc const& acc, auto* accessBlock, auto* pointers, auto* chunkSizes)
            {
                auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
                auto myChunkSize = i % 2 == 1 and i <= num2 ? chunkSizes[1] : chunkSizes[0];
                for(uint32_t j = 0; j < i; ++j)
                {
                    // `.isValid()` is not thread-safe, so we use this direct assessment:
                    while(pointers[i] == nullptr)
                    {
                        pointers[i] = accessBlock->create(acc, myChunkSize);
                    }
                    accessBlock->destroy(acc, pointers[i]);
                    pointers[i] = nullptr;
                }
                while(pointers[i] == nullptr)
                {
                    pointers[i] = accessBlock->create(acc, myChunkSize);
                }
            },
            accessBlock,
            alpaka::getPtrNative(pointers.m_onDevice),
            alpaka::getPtrNative(chunkSizes.m_onDevice));
        alpaka::wait(queue);

        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        std::span<void*> tmpPointers(alpaka::getPtrNative(pointers.m_onHost), MyAccessBlock::numPages());
        std::sort(std::begin(tmpPointers), std::end(tmpPointers));
        CHECK(std::unique(std::begin(tmpPointers), std::end(tmpPointers)) == std::end(tmpPointers));
    }


    SECTION("can handle oversubscription.")
    {
        uint32_t oversubscriptionFactor = 2U;
        auto availableSlots = getAvailableSlots(accessBlock, queue, devHost, devAcc, chunkSizes.m_onHost[0]);

        // This is oversubscribed but we will only hold keep less than 1/oversubscriptionFactor of the memory in the
        // end.
        auto manyPointers = makeBuffer<void*>(devHost, devAcc, oversubscriptionFactor * availableSlots);

        alpaka::WorkDivMembers<Dim, Idx> const workDiv{Idx{1}, Idx{manyPointers.m_extents[0]}, Idx{1}};

        alpaka::exec<Acc>(
            queue,
            workDiv,
            [oversubscriptionFactor, availableSlots](Acc const& acc, auto* accessBlock, auto* pointers, auto chunkSize)
            {
                auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
                for(uint32_t j = 0; j < i + 1; ++j)
                {
                    // `.isValid()` is not thread-safe, so we use this direct assessment:
                    while(pointers[i] == nullptr)
                    {
                        pointers[i] = accessBlock->create(acc, chunkSize);
                    }
                    accessBlock->destroy(acc, pointers[i]);
                    pointers[i] = nullptr;
                }

                // We only keep some of the memory. In particular, we keep one chunk less than is available, such
                // that threads looking for memory after we've finished can still find some.
                while(pointers[i] == nullptr and i > (oversubscriptionFactor - 1) * availableSlots + 1)
                {
                    pointers[i] = accessBlock->create(acc, chunkSize);
                }
            },
            accessBlock,
            alpaka::getPtrNative(manyPointers.m_onDevice),
            chunkSizes.m_onHost[0]);
        alpaka::wait(queue);

        alpaka::memcpy(queue, manyPointers.m_onHost, manyPointers.m_onDevice);
        alpaka::wait(queue);

        // We only let the last (availableSlots-1) keep their memory. So, the rest at the beginning should have a
        // nullptr.
        std::span<void*> tmpManyPointers(alpaka::getPtrNative(manyPointers.m_onHost), manyPointers.m_extents[0]);
        auto beginNonNull = std::begin(tmpManyPointers) + (oversubscriptionFactor - 1) * availableSlots + 1;

        CHECK(std::all_of(
            std::begin(tmpManyPointers),
            beginNonNull,
            [](auto const pointer) { return pointer == nullptr; }));

        std::sort(beginNonNull, std::end(tmpManyPointers));
        CHECK(std::unique(beginNonNull, std::end(tmpManyPointers)) == std::end(tmpManyPointers));
    }

    SECTION("can handle many different chunk sizes.")
    {
        auto chunkSizes = makeBuffer<uint32_t>(devHost, devAcc, pageSize - BitMaskSize);
        std::span<uint32_t> tmp(alpaka::getPtrNative(chunkSizes.m_onHost), chunkSizes.m_extents[0]);
        std::iota(std::begin(tmp), std::end(tmp), 1U);
        alpaka::memcpy(queue, chunkSizes.m_onDevice, chunkSizes.m_onHost);
        alpaka::wait(queue);

        alpaka::WorkDivMembers<Dim, Idx> const workDiv{Idx{1}, Idx{MyAccessBlock::numPages()}, Idx{1}};

        alpaka::exec<Acc>(
            queue,
            workDiv,
            [](Acc const& acc, auto* accessBlock, auto* pointers, auto* chunkSizes)
            {
                auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
                pointers[i] = accessBlock->create(acc, 1U);

                std::span<uint32_t> tmpChunkSizes(chunkSizes, pageSize - BitMaskSize);
                for(auto chunkSize : tmpChunkSizes)
                {
                    accessBlock->destroy(acc, pointers[i]);
                    pointers[i] = nullptr;

                    // `.isValid()` is not thread-safe, so we use this direct assessment:
                    while(pointers[i] == nullptr)
                    {
                        pointers[i] = accessBlock->create(acc, chunkSize);
                    }
                }
            },
            accessBlock,
            alpaka::getPtrNative(pointers.m_onDevice),
            alpaka::getPtrNative(chunkSizes.m_onDevice));

        alpaka::wait(queue);

        alpaka::memcpy(queue, pointers.m_onHost, pointers.m_onDevice);
        alpaka::wait(queue);

        std::span<void*> tmpPointers(alpaka::getPtrNative(pointers.m_onHost), MyAccessBlock::numPages());
        std::sort(std::begin(tmpPointers), std::end(tmpPointers));
        CHECK(std::unique(std::begin(tmpPointers), std::end(tmpPointers)) == std::end(tmpPointers));
    }
    alpaka::wait(queue);
}
