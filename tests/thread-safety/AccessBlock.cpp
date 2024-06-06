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


#include "catch2/generators/catch_generators.hpp"

#include <algorithm>
#include <alpaka/acc/AccCpuSerial.hpp>
#include <alpaka/acc/AccCpuThreads.hpp>
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
#include <catch2/generators/catch_generators_all.hpp>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <mallocMC/creationPolicies/Scatter.hpp>

using mallocMC::CreationPolicies::ScatterAlloc::AccessBlock;
using mallocMC::CreationPolicies::ScatterAlloc::BitMaskSize;

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;
// using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
using Acc = alpaka::AccCpuThreads<Dim, Idx>;


constexpr uint32_t pageSize = 1024;
constexpr size_t numPages = 4;
// Page table entry size = sizeof(chunkSize) + sizeof(fillingLevel):
constexpr uint32_t pteSize = 4 + 4;
constexpr size_t blockSize = numPages * (pageSize + pteSize);

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
            [&acc, &accessBlock, chunkSize]()
            {
                void* pointer = accessBlock->create(acc, chunkSize);
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
    std::vector<uint32_t> const chunkSizes({32U, 512U});
    uint32_t const& chunkSize1 = chunkSizes[0];
    uint32_t const& chunkSize2 = chunkSizes[1];
    std::vector<void*> pointers(accessBlock.getAvailableSlots(chunkSizes[0]), reinterpret_cast<void*>(1U));
    auto& pointer1 = pointers[0];
    auto& pointer2 = pointers[1];

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

    SECTION("creates partly for insufficient memory with same chunk size.")
    {
        auto extents = alpaka::Vec<Dim, Idx>(accessBlock.getAvailableSlots(chunkSize1));
        alpaka::Buf<Acc, void*, Dim, Idx> devPointers(alpaka::allocBuf<void*, Idx>(devAcc, extents));

        alpaka::WorkDivMembers<Dim, Idx> const workDivSingleThread{Idx{1}, Idx{1}, Idx{1}};
        alpaka::exec<Acc>(
            queue,
            workDivSingleThread,
            FillWith{},
            &accessBlock,
            chunkSize1,
            alpaka::getPtrNative(devPointers),
            extents[0]);
        alpaka::wait(queue);

        auto hostPointers = alpaka::createView(devHost, pointers.data(), pointers.size());
        auto devPointersView = alpaka::createView(devHost, alpaka::getPtrNative(devPointers), pointers.size());
        alpaka::memcpy(queue, hostPointers, devPointersView);
        auto* pointer1 = hostPointers[0];
        auto* pointer2 = hostPointers[1];
        alpaka::wait(queue);

        // Destroy exactly one pointer (i.e. the first). This is non-destructive on the actual values in devPointers,
        // so we don't need to wait for the copy before to finish.
        alpaka::exec<Acc>(queue, workDivSingleThread, Destroy{}, &accessBlock, alpaka::getPtrNative(devPointers));
        alpaka::wait(queue);

        // Okay, so here we start the actual test. The situation is the following:
        // There is a single chunk available.
        // We try to do two allocations.
        // So, we expect one to succeed and one to fail.
        alpaka::WorkDivMembers<Dim, Idx> const workDiv{Idx{1}, Idx{2}, Idx{1}};
        alpaka::exec<Acc>(queue, workDiv, Create{}, &accessBlock, alpaka::getPtrNative(devPointers), chunkSize1);
        alpaka::wait(queue);

        alpaka::memcpy(queue, hostPointers, devPointersView);
        alpaka::wait(queue);
        CHECK(
            ((hostPointers[0] == pointer1 and hostPointers[1] == nullptr)
             or (hostPointers[1] == pointer1 and hostPointers[0] == nullptr)));
    }

    SECTION("does not race between clean up and create.")
    {
        for(auto chunkSize : chunkSizes)
        {
            auto extents = alpaka::Vec<Dim, Idx>(accessBlock.getAvailableSlots(chunkSize1));
            alpaka::Buf<Acc, void*, Dim, Idx> devPointers(alpaka::allocBuf<void*, Idx>(devAcc, extents));

            alpaka::WorkDivMembers<Dim, Idx> const workDivSingleThread{Idx{1}, Idx{1}, Idx{1}};
            alpaka::exec<Acc>(
                queue,
                workDivSingleThread,
                FillWith{},
                &accessBlock,
                chunkSize1,
                alpaka::getPtrNative(devPointers),
                extents[0]);
            alpaka::wait(queue);

            auto hostPointers = alpaka::createView(devHost, pointers.data(), pointers.size());
            auto devPointersView = alpaka::createView(devHost, alpaka::getPtrNative(devPointers), pointers.size());
            alpaka::memcpy(queue, hostPointers, devPointersView);
            alpaka::wait(queue);

            std::sort(std::begin(pointers), std::end(pointers));
            // This points to the first chunk of page 0.
            pointer1 = pointers[0];
            // Delete all other chunks on page 0.
            alpaka::WorkDivMembers<Dim, Idx> const workDiv{
                Idx{1},
                Idx{pointers.size() / accessBlock.numPages() - 1},
                Idx{1}};
            alpaka::exec<Acc>(queue, workDiv, Destroy{}, &accessBlock, alpaka::getPtrNative(devPointers) + 1U);

            // Now, pointer1 is the last valid pointer to page 0. Destroying it will clean up the page.
            alpaka::exec<Acc>(queue, workDivSingleThread, Destroy{}, &accessBlock, alpaka::getPtrNative(devPointers));
            alpaka::exec<Acc>(
                queue,
                workDivSingleThread,
                [&chunkSize](Acc const& acc, auto* accessBlock, void** pointer2)
                {
                    while(*pointer2 == nullptr)
                    {
                        *pointer2 = accessBlock->create(acc, chunkSize);
                    }
                },
                &accessBlock,
                alpaka::getPtrNative(devPointers));

            CHECK(accessBlock.pageIndex(pointer1) == accessBlock.pageIndex(pointer2));
        }
    }

    //    SECTION("destroys two pointers of different size.")
    //    {
    //        pointer1 = accessBlock.create(acc, chunkSize1);
    //        pointer2 = accessBlock.create(acc, chunkSize2);
    //        Runner{}.run(destroy, pointer1).run(destroy, pointer2).join();
    //        CHECK(not accessBlock.isValid(acc, pointer1));
    //        CHECK(not accessBlock.isValid(acc, pointer2));
    //    }
    //
    //    SECTION("destroys two pointers of same size.")
    //    {
    //        pointer1 = accessBlock.create(acc, chunkSize1);
    //        pointer2 = accessBlock.create(acc, chunkSize1);
    //        Runner{}.run(destroy, pointer1).run(destroy, pointer2).join();
    //        CHECK(not accessBlock.isValid(acc, pointer1));
    //        CHECK(not accessBlock.isValid(acc, pointer2));
    //    }
    //
    //    SECTION("fills up all blocks in parallel and writes to them.")
    //    {
    //        auto const content = [&accessBlock]()
    //        {
    //            std::vector<uint32_t> tmp(accessBlock.getAvailableSlots(chunkSize1));
    //            std::generate(std::begin(tmp), std::end(tmp), ContentGenerator{});
    //            return tmp;
    //        }();
    //
    //        std::vector<void*> pointers(content.size());
    //        auto runner = Runner{};
    //
    //        for(size_t i = 0; i < content.size(); ++i)
    //        {
    //            runner.run(
    //                [&accessBlock, i, &content, &pointers](Acc const& acc)
    //                {
    //                    pointers[i] = accessBlock.create(acc, chunkSize1);
    //                    auto* begin = reinterpret_cast<uint32_t*>(pointers[i]);
    //                    auto* end = begin + chunkSize1 / sizeof(uint32_t);
    //                    std::fill(begin, end, content[i]);
    //                });
    //        }
    //
    //        runner.join();
    //
    //        CHECK(std::transform_reduce(
    //            std::cbegin(pointers),
    //            std::cend(pointers),
    //            std::cbegin(content),
    //            true,
    //            [](auto const lhs, auto const rhs) { return lhs && rhs; },
    //            [](void* pointer, uint32_t const value)
    //            {
    //                auto* start = reinterpret_cast<uint32_t*>(pointer);
    //                auto end = start + chunkSize1 / sizeof(uint32_t);
    //                return std::all_of(start, end, [value](auto const val) { return val == value; });
    //            }));
    //    }
    //
    //    SECTION("destroys all pointers simultaneously.")
    //    {
    //        auto const allSlots = accessBlock.getAvailableSlots(chunkSize1);
    //        auto const allSlotsOfDifferentSize = accessBlock.getAvailableSlots(chunkSize2);
    //        auto pointers = fillWith(accessBlock, chunkSize1);
    //        auto runner = Runner{};
    //
    //        for(auto* pointer : pointers)
    //        {
    //            runner.run(destroy, pointer);
    //        }
    //        runner.join();
    //
    //        CHECK(std::transform_reduce(
    //            std::cbegin(pointers),
    //            std::cend(pointers),
    //            true,
    //            [](auto const lhs, auto const rhs) { return lhs && rhs; },
    //            [&accessBlock](void* pointer) { return not accessBlock.isValid(acc, pointer); }));
    //        CHECK(accessBlock.getAvailableSlots(chunkSize1) == allSlots);
    //        CHECK(accessBlock.getAvailableSlots(chunkSize2) == allSlotsOfDifferentSize);
    //    }
    //
    //    SECTION("creates and destroys multiple times.")
    //    {
    //        std::vector<void*> pointers(accessBlock.getAvailableSlots(chunkSize1));
    //        auto runner = Runner<>{};
    //
    //        for(size_t i = 0; i < pointers.size(); ++i)
    //        {
    //            runner.run(
    //                [&accessBlock, i, &pointers](Acc const& acc)
    //                {
    //                    for(uint32_t j = 0; j < i; ++j)
    //                    {
    //                        // `.isValid()` is not thread-safe, so we use this direct assessment:
    //                        while(pointers[i] == nullptr)
    //                        {
    //                            pointers[i] = accessBlock.create(acc, chunkSize1);
    //                        }
    //                        accessBlock.destroy(acc, pointers[i]);
    //                        pointers[i] = nullptr;
    //                    }
    //                    while(pointers[i] == nullptr)
    //                    {
    //                        pointers[i] = accessBlock.create(acc, chunkSize1);
    //                    }
    //                });
    //        }
    //
    //        runner.join();
    //
    //        std::sort(std::begin(pointers), std::end(pointers));
    //        CHECK(std::unique(std::begin(pointers), std::end(pointers)) == std::end(pointers));
    //    }
    //
    //    SECTION("creates and destroys multiple times with different sizes.")
    //    {
    //        // CAUTION: This test can fail because we are currently using exactly as much space as is available but
    //        with
    //        // multiple different chunk sizes. That means that if one of them occupies more pages than it minimally
    //        needs,
    //        // the other one will lack pages to with their respective chunk size. This seems not to be a problem
    //        currently
    //        // but it might be more of a problem once we move to device and once we include proper scattering.
    //
    //        // Make sure that num2 > num1.
    //        auto num1 = accessBlock.getAvailableSlots(chunkSize1);
    //        auto num2 = accessBlock.getAvailableSlots(chunkSize2);
    //
    //        std::vector<void*> pointers(num1 / 2 + num2 / 2);
    //        auto runner = Runner<>{};
    //
    //        for(size_t i = 0; i < pointers.size(); ++i)
    //        {
    //            runner.run(
    //                [&accessBlock, i, &pointers, num2](Acc const& acc)
    //                {
    //                    auto myChunkSize = i % 2 == 1 and i <= num2 ? chunkSize2 : chunkSize1;
    //                    for(uint32_t j = 0; j < i; ++j)
    //                    {
    //                        // `.isValid()` is not thread-safe, so we use this direct assessment:
    //                        while(pointers[i] == nullptr)
    //                        {
    //                            pointers[i] = accessBlock.create(acc, myChunkSize);
    //                        }
    //                        accessBlock.destroy(acc, pointers[i]);
    //                        pointers[i] = nullptr;
    //                    }
    //                    while(pointers[i] == nullptr)
    //                    {
    //                        pointers[i] = accessBlock.create(acc, myChunkSize);
    //                    }
    //                });
    //        }
    //
    //        runner.join();
    //
    //        std::sort(std::begin(pointers), std::end(pointers));
    //        CHECK(std::unique(std::begin(pointers), std::end(pointers)) == std::end(pointers));
    //    }
    //
    //    SECTION("can handle oversubscription.")
    //    {
    //        uint32_t oversubscriptionFactor = 2U;
    //        auto availableSlots = accessBlock.getAvailableSlots(chunkSize1);
    //
    //        // This is oversubscribed but we will only hold keep less 1/oversubscriptionFactor of the memory in the
    //        end. std::vector<void*> pointers(oversubscriptionFactor * availableSlots); auto runner = Runner<>{};
    //
    //        for(size_t i = 0; i < pointers.size(); ++i)
    //        {
    //            runner.run(
    //                [&accessBlock, i, &pointers, oversubscriptionFactor, availableSlots](Acc const& acc)
    //                {
    //                    for(uint32_t j = 0; j < i; ++j)
    //                    {
    //                        // `.isValid()` is not thread-safe, so we use this direct assessment:
    //                        while(pointers[i] == nullptr)
    //                        {
    //                            pointers[i] = accessBlock.create(acc, chunkSize1);
    //                        }
    //                        accessBlock.destroy(acc, pointers[i]);
    //                        pointers[i] = nullptr;
    //                    }
    //
    //                    // We only keep some of the memory. In particular, we keep one chunk less than is available,
    //                    such
    //                    // that threads looking for memory after we've finished can still find some.
    //                    while(pointers[i] == nullptr and i > (oversubscriptionFactor - 1) * availableSlots + 1)
    //                    {
    //                        pointers[i] = accessBlock.create(acc, chunkSize1);
    //                    }
    //                });
    //        }
    //
    //        runner.join();
    //
    //        // We only let the last (availableSlots-1) keep their memory. So, the rest at the beginning should have a
    //        // nullptr.
    //        auto beginNonNull = std::begin(pointers) + (oversubscriptionFactor - 1) * availableSlots + 1;
    //
    //        CHECK(std::all_of(std::begin(pointers), beginNonNull, [](auto const pointer) { return pointer == nullptr;
    //        }));
    //
    //        std::sort(beginNonNull, std::end(pointers));
    //        CHECK(std::unique(beginNonNull, std::end(pointers)) == std::end(pointers));
    //    }
    //
    //    SECTION("can handle many different chunk sizes.")
    //    {
    //        auto const chunkSizes = []()
    //        {
    //            // We want to stay within chunked allocation, so we will always have at least one bit mask present in
    //            the
    //            // page.
    //            std::vector<uint32_t> tmp(pageSize - BitMaskSize);
    //            std::iota(std::begin(tmp), std::end(tmp), 1U);
    //            return tmp;
    //        }();
    //
    //        std::vector<void*> pointers(numPages);
    //        auto runner = Runner<>{};
    //
    //        for(auto& pointer : pointers)
    //        {
    //            runner.run(
    //                [&accessBlock, &pointer, &chunkSizes](Acc const& acc)
    //                {
    //                    // This is assumed to always succeed. We only need some valid pointer to formulate the loop
    //                    nicely. pointer = accessBlock.create(acc, 1U); for(auto chunkSize : chunkSizes)
    //                    {
    //                        accessBlock.destroy(acc, pointer);
    //                        pointer = nullptr;
    //                        while(pointer == nullptr)
    //                        {
    //                            pointer = accessBlock.create(acc, chunkSize);
    //                        }
    //                    }
    //                });
    //        }
    //
    //        runner.join();
    //
    //        std::sort(std::begin(pointers), std::end(pointers));
    //        CHECK(std::unique(std::begin(pointers), std::end(pointers)) == std::end(pointers));
    //    }
}
