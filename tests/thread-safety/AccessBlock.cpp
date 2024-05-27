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
#include <catch2/catch.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <mallocMC/creationPolicies/Scatter.hpp>
#include <thread>

using mallocMC::CreationPolicies::ScatterAlloc::AccessBlock;

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
            void* pointer = accessBlock.create(chunkSize);
            REQUIRE(pointer != nullptr);
            return pointer;
        });
    return pointers;
}

template<bool parallel = true>
struct Runner
{
    std::vector<std::thread> threads{};

    template<typename... T>
    auto run(T... pars) -> Runner&
    {
        threads.emplace_back(pars...);
        if constexpr(not parallel)
        {
            threads.back().join();
        }
        return *this;
    }

    auto join()
    {
        if constexpr(parallel)
        {
            std::for_each(std::begin(threads), std::end(threads), [](auto& thread) { thread.join(); });
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

TEST_CASE("Threaded AccessBlock")
{
    AccessBlock<blockSize, pageSize> accessBlock{};
    auto create = [&accessBlock](void** pointer, auto chunkSize) { *pointer = accessBlock.create(chunkSize); };
    auto destroy = [&accessBlock](void* pointer) { accessBlock.destroy(pointer); };
    void* pointer1 = nullptr;
    void* pointer2 = nullptr;
    constexpr uint32_t const chunkSize1 = 32U;
    constexpr uint32_t const chunkSize2 = 512U;

    SECTION("creates second memory somewhere else.")
    {
        Runner{}.run(create, &pointer1, chunkSize1).run(create, &pointer2, chunkSize1).join();
        CHECK(pointer1 != pointer2);
    }

    SECTION("creates memory of different chunk size in different pages.")
    {
        Runner{}.run(create, &pointer1, chunkSize1).run(create, &pointer2, chunkSize2).join();
        CHECK(accessBlock.pageIndex(pointer1) != accessBlock.pageIndex(pointer2));
    }

    SECTION("creates partly for insufficient memory with same chunk size.")
    {
        fillWith(accessBlock, chunkSize1);

        // This is a pointer to the first chunk of the first page. It is valid because we have manually filled up the
        // complete accessBlock. So, we're effectively opening one slot:
        accessBlock.destroy(reinterpret_cast<void*>(&accessBlock));

        Runner{}.run(create, &pointer1, chunkSize1).run(create, &pointer2, chunkSize1).join();

        // Chained comparisons are not supported by Catch2:
        if(pointer1 != nullptr)
        {
            CHECK(pointer2 == nullptr);
        }
        if(pointer2 != nullptr)
        {
            CHECK(pointer1 == nullptr);
        }
        // this excludes that both are nullptr:
        CHECK(pointer1 != pointer2);
    }

    SECTION("does not race between clean up and create.")
    {
        auto pointers = fillWith(accessBlock, chunkSize1);
        std::sort(std::begin(pointers), std::end(pointers));
        // This points to the first chunk of page 0.
        pointer1 = pointers[0];
        // Delete all other chunks on page 0.
        std::for_each(
            std::begin(pointers) + 1,
            std::begin(pointers) + pointers.size() / accessBlock.numPages(),
            destroy);
        // Now, pointer1 is the last valid pointer to page 0. Destroying it will clean up the page.

        Runner{}
            .run(destroy, &pointer1)
            .run(
                [&accessBlock, &pointer2]()
                {
                    while(pointer2 == nullptr)
                    {
                        pointer2 = accessBlock.create(chunkSize1);
                    }
                })
            .join();

        CHECK(accessBlock.pageIndex(pointer1) == accessBlock.pageIndex(pointer2));
    }

    SECTION("destroys two pointers of different size.")
    {
        pointer1 = accessBlock.create(chunkSize1);
        pointer2 = accessBlock.create(chunkSize2);
        Runner{}.run(destroy, pointer1).run(destroy, pointer2).join();
        CHECK(not accessBlock.isValid(pointer1));
        CHECK(not accessBlock.isValid(pointer2));
    }

    SECTION("destroys two pointers of same size.")
    {
        pointer1 = accessBlock.create(chunkSize1);
        pointer2 = accessBlock.create(chunkSize1);
        Runner{}.run(destroy, pointer1).run(destroy, pointer2).join();
        CHECK(not accessBlock.isValid(pointer1));
        CHECK(not accessBlock.isValid(pointer2));
    }

    SECTION("fills up all blocks in parallel and writes to them.")
    {
        auto const content = [&accessBlock]()
        {
            std::vector<uint32_t> tmp(accessBlock.getAvailableSlots(chunkSize1));
            std::generate(std::begin(tmp), std::end(tmp), ContentGenerator{});
            return tmp;
        }();

        std::vector<void*> pointers(content.size());
        auto runner = Runner{};

        for(size_t i = 0; i < content.size(); ++i)
        {
            runner.run(
                [&accessBlock, i, &content, &pointers]()
                {
                    pointers[i] = accessBlock.create(chunkSize1);
                    auto* begin = reinterpret_cast<uint32_t*>(pointers[i]);
                    auto* end = begin + chunkSize1 / sizeof(uint32_t);
                    std::fill(begin, end, content[i]);
                });
        }

        runner.join();

        CHECK(std::transform_reduce(
            std::cbegin(pointers),
            std::cend(pointers),
            std::cbegin(content),
            true,
            [](auto const lhs, auto const rhs) { return lhs && rhs; },
            [](void* pointer, uint32_t const value)
            {
                auto* start = reinterpret_cast<uint32_t*>(pointer);
                auto end = start + chunkSize1 / sizeof(uint32_t);
                return std::all_of(start, end, [value](auto const val) { return val == value; });
            }));
    }

    SECTION("destroys all pointers simultaneously.")
    {
        auto const allSlots = accessBlock.getAvailableSlots(chunkSize1);
        auto const allSlotsOfDifferentSize = accessBlock.getAvailableSlots(chunkSize2);
        auto pointers = fillWith(accessBlock, chunkSize1);
        auto runner = Runner{};

        for(auto* pointer : pointers)
        {
            runner.run(destroy, pointer);
        }
        runner.join();

        CHECK(std::transform_reduce(
            std::cbegin(pointers),
            std::cend(pointers),
            true,
            [](auto const lhs, auto const rhs) { return lhs && rhs; },
            [&accessBlock](void* pointer) { return not accessBlock.isValid(pointer); }));
        CHECK(accessBlock.getAvailableSlots(chunkSize1) == allSlots);
        CHECK(accessBlock.getAvailableSlots(chunkSize2) == allSlotsOfDifferentSize);
    }

    SECTION("creates and destroys multiple times.")
    {
        std::vector<void*> pointers(accessBlock.getAvailableSlots(chunkSize1));
        auto runner = Runner<>{};

        for(size_t i = 0; i < pointers.size(); ++i)
        {
            runner.run(
                [&accessBlock, i, &pointers]()
                {
                    for(uint32_t j = 0; j < i; ++j)
                    {
                        // `.isValid()` is not thread-safe, so we use this direct assessment:
                        while(pointers[i] == nullptr)
                        {
                            pointers[i] = accessBlock.create(chunkSize1);
                        }
                        accessBlock.destroy(pointers[i]);
                        pointers[i] = nullptr;
                    }
                    while(pointers[i] == nullptr)
                    {
                        pointers[i] = accessBlock.create(chunkSize1);
                    }
                });
        }

        runner.join();

        std::sort(std::begin(pointers), std::end(pointers));
        CHECK(std::unique(std::begin(pointers), std::end(pointers)) == std::end(pointers));
    }

    SECTION("creates and destroys multiple times with different sizes.")
    {
        // CAUTION: This test can fail because we are currently using exactly as much space as is available but with
        // multiple different chunk sizes. That means that if one of them occupies more pages than it minimally needs,
        // the other one will lack pages to with their respective chunk size. This seems not to be a problem currently
        // but it might be more of a problem once we move to device and once we include proper scattering.

        // Make sure that num2 > num1.
        auto num1 = accessBlock.getAvailableSlots(chunkSize1);
        auto num2 = accessBlock.getAvailableSlots(chunkSize2);

        std::vector<void*> pointers(num1 / 2 + num2 / 2);
        auto runner = Runner<>{};

        for(size_t i = 0; i < pointers.size(); ++i)
        {
            runner.run(
                [&accessBlock, i, &pointers, num2]()
                {
                    auto myChunkSize = i % 2 == 1 and i <= num2 ? chunkSize2 : chunkSize1;
                    for(uint32_t j = 0; j < i; ++j)
                    {
                        // `.isValid()` is not thread-safe, so we use this direct assessment:
                        while(pointers[i] == nullptr)
                        {
                            pointers[i] = accessBlock.create(myChunkSize);
                        }
                        accessBlock.destroy(pointers[i]);
                        pointers[i] = nullptr;
                    }
                    while(pointers[i] == nullptr)
                    {
                        pointers[i] = accessBlock.create(myChunkSize);
                    }
                });
        }

        runner.join();

        std::sort(std::begin(pointers), std::end(pointers));
        CHECK(std::unique(std::begin(pointers), std::end(pointers)) == std::end(pointers));
    }

    SECTION("can handle oversubscription.")
    {
        uint32_t oversubscriptionFactor = 2U;
        auto availableSlots = accessBlock.getAvailableSlots(chunkSize1);

        // This is oversubscribed but we will only hold keep less 1/oversubscriptionFactor of the memory in the end.
        std::vector<void*> pointers(oversubscriptionFactor * availableSlots);
        auto runner = Runner<>{};

        for(size_t i = 0; i < pointers.size(); ++i)
        {
            runner.run(
                [&accessBlock, i, &pointers, oversubscriptionFactor, availableSlots]()
                {
                    for(uint32_t j = 0; j < i; ++j)
                    {
                        // `.isValid()` is not thread-safe, so we use this direct assessment:
                        while(pointers[i] == nullptr)
                        {
                            pointers[i] = accessBlock.create(chunkSize1);
                        }
                        accessBlock.destroy(pointers[i]);
                        pointers[i] = nullptr;
                    }

                    // We only keep some of the memory. In particular, we keep one chunk less than is available, such
                    // that threads looking for memory after we've finished can still find some.
                    while(pointers[i] == nullptr and i > (oversubscriptionFactor - 1) * availableSlots + 1)
                    {
                        pointers[i] = accessBlock.create(chunkSize1);
                    }
                });
        }

        runner.join();

        // We only let the last (availableSlots-1) keep their memory. So, the rest at the beginning should have a
        // nullptr.
        auto beginNonNull = std::begin(pointers) + (oversubscriptionFactor - 1) * availableSlots + 1;

        CHECK(std::all_of(std::begin(pointers), beginNonNull, [](auto const pointer) { return pointer == nullptr; }));

        std::sort(beginNonNull, std::end(pointers));
        CHECK(std::unique(beginNonNull, std::end(pointers)) == std::end(pointers));
    }
}
