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
#include "mallocMC/creationPolicies/Scatter/BitField.hpp"

#include <catch2/catch.hpp>
#include <cstddef>
#include <cstdint>
#include <mallocMC/creationPolicies/Scatter.hpp>
#include <thread>

using mallocMC::indexOf;
using mallocMC::CreationPolicies::ScatterAlloc::AccessBlock;
using mallocMC::CreationPolicies::ScatterAlloc::BitMask;

constexpr size_t pageSize = 1024;
constexpr size_t numPages = 4;
// Page table entry size = sizeof(chunkSize) + sizeof(fillingLevel):
constexpr size_t pteSize = 4 + 4;
constexpr size_t blockSize = numPages * (pageSize + pteSize);

// Fill all pages of the given access block with occupied chunks of the given size. This is useful to test the
// behaviour near full filling but also to have a deterministic page and chunk where an allocation must happen
// regardless of the underlying access optimisations etc.
template<size_t T_blockSize, size_t T_pageSize>
void fillWith(AccessBlock<T_blockSize, T_pageSize>& accessBlock, uint32_t const chunkSize)
{
    for(auto& tmpChunkSize : accessBlock.pageTable._chunkSizes)
    {
        tmpChunkSize = chunkSize;
    }

    auto maxFillingLevel = accessBlock.interpret(0).numChunks();
    for(auto& fillingLevel : accessBlock.pageTable._fillingLevels)
    {
        fillingLevel = maxFillingLevel;
    }

    for(size_t i = 0; i < accessBlock.numPages(); ++i)
    {
        auto page = accessBlock.interpret(i);
        for(auto& mask : page.bitField())
        {
            mask.set();
        }
    }
}

TEST_CASE("Threaded AccessBlock.create")
{
    AccessBlock<blockSize, pageSize> accessBlock;
    auto create = [&accessBlock](void** pointer, auto chunkSize) { *pointer = accessBlock.create(chunkSize); };

    SECTION("creates second memory somewhere else.")
    {
        void* pointer1 = nullptr;
        void* pointer2 = nullptr;
        constexpr uint32_t const chunkSize = 32U;

        auto thread1 = std::thread(create, &pointer1, chunkSize);
        auto thread2 = std::thread(create, &pointer2, chunkSize);
        thread1.join();
        thread2.join();

        CHECK(accessBlock.create(32U) != accessBlock.create(32U));
    }

    SECTION("creates memory of different chunk size in different pages.")
    {
        void* pointer1 = nullptr;
        void* pointer2 = nullptr;

        auto thread1 = std::thread(create, &pointer1, 32U);
        auto thread2 = std::thread(create, &pointer2, 512U);
        thread1.join();
        thread2.join();

        CHECK(
            indexOf(pointer1, &accessBlock.pages[0], pageSize) != indexOf(pointer2, &accessBlock.pages[0], pageSize));
    }

    SECTION("creates partly for insufficient memory with same chunk size.")
    {
        constexpr uint32_t const chunkSize = 32U;
        fillWith(accessBlock, chunkSize);

        void* pointer1 = nullptr;
        void* pointer2 = nullptr;

        // This is a pointer to the first chunk of the first page. It is valid because we have manually filled up the
        // complete accessBlock. So, we're effectively opening one slot:
        accessBlock.destroy(reinterpret_cast<void*>(&accessBlock));

        auto thread1 = std::thread(create, &pointer1, chunkSize);
        auto thread2 = std::thread(create, &pointer2, chunkSize);
        thread1.join();
        thread2.join();

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

    // TEST_CASE("AccessBlock.destroy")
    //{
    //    // TODO(lenz): Remove reference to .bitmasks() and check that these tests did the correct thing after all.
    //
    //    AccessBlock<blockSize, pageSize> accessBlock;
    //
    //    SECTION("destroys a previously created pointer.")
    //    {
    //        constexpr const uint32_t numBytes = 32U;
    //        void* pointer = accessBlock.create(numBytes);
    //        accessBlock.destroy(pointer);
    //        SUCCEED("Just check that you can do this at all.");
    //    }
    //
    //    SECTION("frees up the page upon destroying the last element without hierarchy.")
    //    {
    //        constexpr const uint32_t numBytes = 32U;
    //        void* pointer = accessBlock.create(numBytes);
    //        auto pageIndex = indexOf(pointer, std::begin(accessBlock.pages), pageSize);
    //        auto chunkIndex = indexOf(pointer, &accessBlock.pages[pageIndex], numBytes);
    //        REQUIRE(accessBlock.bitMasks()[pageIndex][chunkIndex]);
    //
    //        accessBlock.destroy(pointer);
    //        CHECK(accessBlock.pageTable._chunkSizes[pageIndex] == 0U);
    //    }
    //
    //    SECTION("resets one bit without touching others upon destroy without hierarchy.")
    //    {
    //        constexpr const uint32_t numBytes = 32U;
    //
    //        void* untouchedPointer = accessBlock.create(numBytes);
    //        auto const untouchedPageIndex = indexOf(untouchedPointer, &accessBlock.pages[0], pageSize);
    //        auto const untouchedChunkIndex = indexOf(untouchedPointer, &accessBlock.pages[untouchedPageIndex],
    //        numBytes); REQUIRE(accessBlock.bitMasks()[untouchedPageIndex][untouchedChunkIndex]);
    //
    //        void* pointer = accessBlock.create(numBytes);
    //        auto const pageIndex = indexOf(pointer, &accessBlock.pages[0], pageSize);
    //        auto const chunkIndex = indexOf(pointer, &accessBlock.pages[pageIndex], numBytes);
    //        REQUIRE(accessBlock.bitMasks()[pageIndex][chunkIndex]);
    //
    //        accessBlock.destroy(pointer);
    //
    //        CHECK(accessBlock.bitMasks()[untouchedPageIndex][untouchedChunkIndex]);
    //        CHECK(!accessBlock.bitMasks()[pageIndex][chunkIndex]);
    //    }
    //
    //    SECTION("decreases one filling level upon destroy without hierarchy.")
    //    {
    //        constexpr const uint32_t numBytes = 32U;
    //        void* pointer = accessBlock.create(numBytes);
    //        auto pageIndex = indexOf(pointer, &accessBlock.pages[0], pageSize);
    //        REQUIRE(accessBlock.pageTable._fillingLevels[pageIndex] == 1U);
    //
    //        accessBlock.destroy(pointer);
    //
    //        CHECK(accessBlock.pageTable._fillingLevels[pageIndex] == 0U);
    //    }
    //
    //    SECTION("decreases one filling level without touching others upon destroy without hierarchy.")
    //    {
    //        constexpr const uint32_t numBytes = 32U;
    //
    //        void* untouchedPointer = accessBlock.create(numBytes);
    //        auto untouchedPageIndex = indexOf(untouchedPointer, &accessBlock.pages[0], pageSize);
    //        REQUIRE(accessBlock.pageTable._fillingLevels[untouchedPageIndex] == 1U);
    //
    //        void* pointer = accessBlock.create(numBytes);
    //        auto pageIndex = indexOf(pointer, &accessBlock.pages[0], pageSize);
    //
    //        uint32_t fillingLevel = pageIndex == untouchedPageIndex ? 2U : 1U;
    //        REQUIRE(accessBlock.pageTable._fillingLevels[pageIndex] == fillingLevel);
    //
    //        accessBlock.destroy(pointer);
    //
    //        CHECK(accessBlock.pageTable._fillingLevels[pageIndex] == fillingLevel - 1);
    //        CHECK(accessBlock.pageTable._fillingLevels[untouchedPageIndex] > 0);
    //    }
    //
    //    SECTION("throws if given an invalid pointer.")
    //    {
    //        void* pointer = nullptr;
    //        CHECK_THROWS_WITH(accessBlock.destroy(pointer), Catch::Contains("Attempted to destroy invalid pointer"));
    //    }
    //
    //    SECTION("can destroy multiple pages.")
    //    {
    //        auto pagesNeeded = 2U;
    //        auto chunkSize = pagesNeeded * pageSize;
    //        auto* pointer = accessBlock.create(chunkSize);
    //        auto index = indexOf(pointer, &accessBlock.pages[0], pageSize);
    //
    //        for(uint32_t i = 0; i < pagesNeeded; ++i)
    //        {
    //            REQUIRE(accessBlock.interpret(index + i)._chunkSize == chunkSize);
    //            REQUIRE(accessBlock.interpret(index + i)._fillingLevel == 1U);
    //        }
    //
    //        accessBlock.destroy(pointer);
    //
    //        for(uint32_t i = 0; i < pagesNeeded; ++i)
    //        {
    //            CHECK(accessBlock.interpret(index + i)._chunkSize == 0U);
    //            CHECK(accessBlock.interpret(index + i)._fillingLevel == 0U);
    //        }
    //    }
}
