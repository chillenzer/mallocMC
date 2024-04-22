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

#include <algorithm>
#include <catch2/catch.hpp>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <mallocMC/creationPolicies/Scatter.hpp>

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

TEST_CASE("AccessBlock (reporting)")
{
    AccessBlock<blockSize, pageSize> accessBlock;

    SECTION("has pages.")
    {
        // This check is mainly leftovers from TDD. Keep it as long as `pages` is public interface.
        CHECK(accessBlock.pages); // NOLINT(*-array-*decay)
    }

    SECTION("knows its data size.")
    {
        CHECK(accessBlock.dataSize() == numPages * pageSize);
    }

    SECTION("knows its metadata size.")
    {
        CHECK(accessBlock.metadataSize() == numPages * pteSize);
    }

    SECTION("stores page table after pages.")
    {
        CHECK(reinterpret_cast<void*>(accessBlock.pages) < reinterpret_cast<void*>(&accessBlock.pageTable));
    }

    SECTION("uses an allowed amount of memory.")
    {
        CHECK(accessBlock.dataSize() + accessBlock.metadataSize() <= blockSize);
    }

    SECTION("knows its number of pages.")
    {
        CHECK(accessBlock.numPages() == numPages);
    }

    SECTION("correctly reports a different number of pages.")
    {
        constexpr size_t localNumPages = 5U;
        constexpr size_t localBlockSize = localNumPages * (pageSize + pteSize);
        AccessBlock<localBlockSize, pageSize> localAccessBlock;
        CHECK(localAccessBlock.numPages() == localNumPages);
    }
}

TEST_CASE("AccessBlock.create")
{
    AccessBlock<blockSize, pageSize> accessBlock;

    SECTION("does not return nullptr if memory is available.")
    {
        // This is not a particularly hard thing to do because any uninitialised pointer that could be returned is most
        // likely not exactly the nullptr. We just leave this in as it currently doesn't hurt anybody to keep it.
        CHECK(accessBlock.create(32U) != nullptr);
    }

    SECTION("creates memory that can be written to and read from.")
    {
        uint32_t const arbitraryValue = 42;
        auto* ptr = static_cast<uint32_t*>(accessBlock.create(4U));
        *ptr = arbitraryValue;
        CHECK(*ptr == arbitraryValue);
    }

    SECTION("creates second memory somewhere else.")
    {
        CHECK(accessBlock.create(32U) != accessBlock.create(32U));
    }

    SECTION("creates memory with same chunk size in same page (if there is no other).")
    {
        constexpr size_t localNumPages = 1U;
        constexpr size_t localBlockSize = localNumPages * (pageSize + pteSize);
        constexpr const uint32_t chunkSize = 32U;
        AccessBlock<localBlockSize, pageSize> localAccessBlock{};
        void* result1 = localAccessBlock.create(chunkSize);
        REQUIRE(result1 != nullptr);
        void* result2 = localAccessBlock.create(chunkSize);
        REQUIRE(result2 != nullptr);
        CHECK(indexOf(result1, &accessBlock.pages[0], pageSize) == indexOf(result2, &accessBlock.pages[0], pageSize));
    }

    SECTION("creates memory of different chunk size in different pages.")
    {
        CHECK(
            indexOf(accessBlock.create(32U), &accessBlock.pages[0], pageSize)
            != indexOf(accessBlock.create(512U), &accessBlock.pages[0], pageSize));
    }

    SECTION("fails to create memory if there's no page with fitting chunk size")
    {
        constexpr size_t localNumPages = 1U;
        constexpr size_t localBlockSize = localNumPages * (pageSize + pteSize);
        AccessBlock<localBlockSize, pageSize> localAccessBlock;
        const uint32_t chunkSize = 32U;
        const uint32_t otherChunkSize = 512U;

        // set the chunk size of the only available page:
        localAccessBlock.create(chunkSize);
        CHECK(localAccessBlock.create(otherChunkSize) == nullptr);
    }

    SECTION("fails to create memory if all pages have full filling level even if bit masks are empty.")
    {
        constexpr const uint32_t chunkSize = 32U;
        fillWith(accessBlock, chunkSize);
        for(auto& bitMask : accessBlock.bitMasks())
        {
            // reverse previous filling
            bitMask.flip();
        }
        CHECK(accessBlock.create(chunkSize) == nullptr);
    }

    SECTION("finds last remaining chunk for creation without hierarchical bit fields.")
    {
        constexpr const uint32_t chunkSize = 32U;
        fillWith(accessBlock, chunkSize);

        const uint32_t index1 = GENERATE(0, 1, 2, 3);
        const uint32_t index2 = GENERATE(0, 1, 2, 3);
        accessBlock.pageTable._fillingLevels[index1] -= 1;
        accessBlock.bitMasks()[index1].flip(index2);


        void* result = accessBlock.create(chunkSize);
        REQUIRE(result != nullptr);

        CHECK(
            std::distance(reinterpret_cast<char*>(&accessBlock.pages[0]), reinterpret_cast<char*>(result))
            == static_cast<long>(index1 * pageSize + index2 * chunkSize));
    }

    SECTION("increases filling level upon creation.")
    {
        REQUIRE(
            std::count(
                std::begin(accessBlock.pageTable._fillingLevels),
                std::end(accessBlock.pageTable._fillingLevels),
                0U)
            == accessBlock.numPages());
        accessBlock.create(32U);
        CHECK(
            std::count(
                std::begin(accessBlock.pageTable._fillingLevels),
                std::end(accessBlock.pageTable._fillingLevels),
                1U)
            == 1U);
    }

    SECTION("recovers from not finding a free chunk in page.")
    {
        uint32_t const chunkSize = 32U;
        uint32_t const pageIndex = GENERATE(1, 2);
        uint32_t const chunkIndex = GENERATE(2, 3, 13);
        fillWith(accessBlock, chunkSize);

        for(auto& fillingLevel : accessBlock.pageTable._fillingLevels)
        {
            // we lie here and set all filling levels to 0, so accessBlock thinks that the pages are free but they are
            // not
            fillingLevel = 0U;
        }
        // But we show some mercy and make one chunk available.
        accessBlock.interpret(pageIndex).bitField().set(chunkIndex, false);

        auto* pointer = accessBlock.create(chunkSize);

        CHECK(indexOf(pointer, &accessBlock.pages[0], pageSize) == pageIndex);
        CHECK(indexOf(pointer, &accessBlock.pages[pageIndex], chunkSize) == chunkIndex);
    }

    SECTION("can create memory larger than page size.")
    {
        // We give a strong guarantee that this searches the first possible block, so we know the index here.
        CHECK(indexOf(accessBlock.create(2U * pageSize), &accessBlock.pages[0], pageSize) == 0);
    }

    SECTION("sets correct chunk size for larger than page size.")
    {
        auto pagesNeeded = 2U;
        auto chunkSize = pagesNeeded * pageSize;
        auto* pointer = accessBlock.create(chunkSize);
        auto index = indexOf(pointer, &accessBlock.pages[0], pageSize);
        for(uint32_t i = 0; i < pagesNeeded; ++i)
        {
            CHECK(accessBlock.interpret(index + i)._chunkSize == chunkSize);
        }
    }

    SECTION("sets correct filling level for larger than page size.")
    {
        auto pagesNeeded = 2U;
        auto chunkSize = pagesNeeded * pageSize;
        auto* pointer = accessBlock.create(chunkSize);
        auto index = indexOf(pointer, &accessBlock.pages[0], pageSize);
        for(uint32_t i = 0; i < pagesNeeded; ++i)
        {
            CHECK(accessBlock.pageTable._fillingLevels[index + i] == 1U);
        }
    }

    SECTION("finds contiguous memory for larger than page size.")
    {
        constexpr size_t localNumPages = 5U;
        constexpr size_t localBlockSize = localNumPages * (pageSize + pteSize);
        AccessBlock<localBlockSize, pageSize> localAccessBlock;
        uint32_t const anythingNonzero = 1U;
        localAccessBlock.pageTable._chunkSizes[1] = anythingNonzero;
        // We give a strong guarantee that this searches the first possible block, so we know the index here.
        CHECK(indexOf(localAccessBlock.create(2U * pageSize), &localAccessBlock.pages[0], pageSize) == 2U);
    }
}

TEST_CASE("AccessBlock.destroy")
{
    // TODO(lenz): Remove reference to .bitmasks() and check that these tests did the correct thing after all.

    AccessBlock<blockSize, pageSize> accessBlock;

    SECTION("destroys a previously created pointer.")
    {
        constexpr const uint32_t numBytes = 32U;
        void* pointer = accessBlock.create(numBytes);
        accessBlock.destroy(pointer);
        SUCCEED("Just check that you can do this at all.");
    }

    SECTION("frees up the page upon destroying the last element without hierarchy.")
    {
        constexpr const uint32_t numBytes = 32U;
        void* pointer = accessBlock.create(numBytes);
        auto pageIndex = indexOf(pointer, std::begin(accessBlock.pages), pageSize);
        auto chunkIndex = indexOf(pointer, &accessBlock.pages[pageIndex], numBytes);
        REQUIRE(accessBlock.bitMasks()[pageIndex][chunkIndex]);

        accessBlock.destroy(pointer);
        CHECK(accessBlock.pageTable._chunkSizes[pageIndex] == 0U);
    }

    SECTION("resets one bit without touching others upon destroy without hierarchy.")
    {
        constexpr const uint32_t numBytes = 32U;

        void* untouchedPointer = accessBlock.create(numBytes);
        auto const untouchedPageIndex = indexOf(untouchedPointer, &accessBlock.pages[0], pageSize);
        auto const untouchedChunkIndex = indexOf(untouchedPointer, &accessBlock.pages[untouchedPageIndex], numBytes);
        REQUIRE(accessBlock.bitMasks()[untouchedPageIndex][untouchedChunkIndex]);

        void* pointer = accessBlock.create(numBytes);
        auto const pageIndex = indexOf(pointer, &accessBlock.pages[0], pageSize);
        auto const chunkIndex = indexOf(pointer, &accessBlock.pages[pageIndex], numBytes);
        REQUIRE(accessBlock.bitMasks()[pageIndex][chunkIndex]);

        accessBlock.destroy(pointer);

        CHECK(accessBlock.bitMasks()[untouchedPageIndex][untouchedChunkIndex]);
        CHECK(!accessBlock.bitMasks()[pageIndex][chunkIndex]);
    }

    SECTION("decreases one filling level upon destroy without hierarchy.")
    {
        constexpr const uint32_t numBytes = 32U;
        void* pointer = accessBlock.create(numBytes);
        auto pageIndex = indexOf(pointer, &accessBlock.pages[0], pageSize);
        REQUIRE(accessBlock.pageTable._fillingLevels[pageIndex] == 1U);

        accessBlock.destroy(pointer);

        CHECK(accessBlock.pageTable._fillingLevels[pageIndex] == 0U);
    }

    SECTION("decreases one filling level without touching others upon destroy without hierarchy.")
    {
        constexpr const uint32_t numBytes = 32U;

        void* untouchedPointer = accessBlock.create(numBytes);
        auto untouchedPageIndex = indexOf(untouchedPointer, &accessBlock.pages[0], pageSize);
        REQUIRE(accessBlock.pageTable._fillingLevels[untouchedPageIndex] == 1U);

        void* pointer = accessBlock.create(numBytes);
        auto pageIndex = indexOf(pointer, &accessBlock.pages[0], pageSize);

        uint32_t fillingLevel = pageIndex == untouchedPageIndex ? 2U : 1U;
        REQUIRE(accessBlock.pageTable._fillingLevels[pageIndex] == fillingLevel);

        accessBlock.destroy(pointer);

        CHECK(accessBlock.pageTable._fillingLevels[pageIndex] == fillingLevel - 1);
        CHECK(accessBlock.pageTable._fillingLevels[untouchedPageIndex] > 0);
    }

    SECTION("throws if given an invalid pointer.")
    {
        void* pointer = nullptr;
        CHECK_THROWS_WITH(accessBlock.destroy(pointer), Catch::Contains("Attempted to destroy invalid pointer"));
    }

    SECTION("can destroy multiple pages.")
    {
        auto pagesNeeded = 2U;
        auto chunkSize = pagesNeeded * pageSize;
        auto* pointer = accessBlock.create(chunkSize);
        auto index = indexOf(pointer, &accessBlock.pages[0], pageSize);

        for(uint32_t i = 0; i < pagesNeeded; ++i)
        {
            REQUIRE(accessBlock.pageTable._chunkSizes[index + i] == chunkSize);
            REQUIRE(accessBlock.pageTable._fillingLevels[index + i] == 1U);
        }

        accessBlock.destroy(pointer);

        for(uint32_t i = 0; i < pagesNeeded; ++i)
        {
            CHECK(accessBlock.pageTable._chunkSizes[index + i] == 0U);
            CHECK(accessBlock.pageTable._fillingLevels[index + i] == 0U);
        }
    }

    SECTION("resets chunk size when page is abandoned.")
    {
        constexpr const uint32_t numBytes = 32U;
        void* pointer = accessBlock.create(numBytes);
        auto pageIndex = indexOf(pointer, &accessBlock.pages[0], pageSize);
        REQUIRE(accessBlock.pageTable._chunkSizes[pageIndex] == numBytes);

        // this is the only allocation on this page, so page is abandoned afterwards
        accessBlock.destroy(pointer);
        CHECK(accessBlock.pageTable._chunkSizes[pageIndex] == 0U);
    }
}
