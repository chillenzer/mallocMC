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
#include <iterator>
#include <mallocMC/creationPolicies/Scatter.hpp>
#include <thread>

using mallocMC::CreationPolicies::ScatterAlloc::AccessBlock;

constexpr size_t pageSize = 1024;
constexpr size_t numPages = 4;
// Page table entry size = sizeof(chunkSize) + sizeof(fillingLevel):
constexpr size_t pteSize = 4 + 4;
constexpr size_t blockSize = numPages * (pageSize + pteSize);

// Fill all pages of the given access block with occupied chunks of the given size. This is useful to test the
// behaviour near full filling but also to have a deterministic page and chunk where an allocation must happen
// regardless of the underlying access optimisations etc.
template<size_t T_blockSize, size_t T_pageSize>
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
    AccessBlock<blockSize, pageSize> accessBlock;
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
}
