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

#include "mallocMC/creationPolicies/Scatter/PageInterpretation.hpp"

#include <catch2/catch.hpp>
#include <cstdint>
#include <iterator>
#include <mallocMC/creationPolicies/Scatter.hpp>
#include <type_traits>

using mallocMC::CreationPolicies::ScatterAlloc::AccessBlock;
using mallocMC::CreationPolicies::ScatterAlloc::PageInterpretation;

constexpr size_t const pageTableEntrySize = 8U;
constexpr size_t const pageSize1 = 1024U;
constexpr size_t const pageSize2 = 4096U;

using BlockAndPageSizes = std::tuple<
    // single page:
    std::tuple<
        std::integral_constant<size_t, 1U * (pageSize1 + pageTableEntrySize)>,
        std::integral_constant<size_t, pageSize1>>,
    // multiple pages:
    std::tuple<
        std::integral_constant<size_t, 4U * (pageSize1 + pageTableEntrySize)>,
        std::integral_constant<size_t, pageSize1>>,
    // multiple pages with some overhead:
    std::tuple<
        std::integral_constant<size_t, 4U * (pageSize1 + pageTableEntrySize) + 100U>,
        std::integral_constant<size_t, pageSize1>>,
    // other page size:
    std::tuple<
        std::integral_constant<size_t, 3U * (pageSize2 + pageTableEntrySize) + 100U>,
        std::integral_constant<size_t, pageSize2>>>;

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

TEMPLATE_LIST_TEST_CASE("AccessBlock", "", BlockAndPageSizes)
{
    constexpr auto const blockSize = std::get<0>(TestType{}).value;
    constexpr auto const pageSize = std::get<1>(TestType{}).value;

    AccessBlock<blockSize, pageSize> accessBlock{};

    SECTION("knows its number of pages.")
    {
        // The overhead from the metadata is small enough that this just happens to round down to the correct values.
        // If you choose weird numbers, it might no longer.
        CHECK(accessBlock.numPages() == blockSize / pageSize);
    }

    SECTION("knows its available slots.")
    {
        uint32_t const chunkSize = GENERATE(1U, 2U, 32U, 57U, 1024U);
        // This is not exactly true. It is only true because the largest chunk size we chose above is exactly the size
        // of one page. In general, this number would be fractional for larger than page size chunks but I don't want
        // to bother right now:
        size_t slotsPerPage = chunkSize < pageSize ? PageInterpretation<pageSize>::numChunks(chunkSize) : 1U;

        uint32_t numOccupied = GENERATE(0U, 1U, 10U);
        for(uint32_t i = 0; i < numOccupied; ++i)
        {
            if(not accessBlock.create(chunkSize))
            {
                {
                }
                numOccupied--;
            }
        }

        auto totalSlots = accessBlock.numPages() * slotsPerPage;
        if(totalSlots > numOccupied)
        {
            CHECK(accessBlock.getAvailableSlots(chunkSize) == totalSlots - numOccupied);
        }
        else
        {
            CHECK(accessBlock.getAvailableSlots(chunkSize) == 0U);
        }
    }

    constexpr uint32_t const chunkSize = 32U;

    SECTION("creates")
    {
        SECTION("no nullptr if memory is available.")
        {
            // This is not a particularly hard thing to do because any uninitialised pointer that could be returned is
            // most likely not exactly the nullptr. We just leave this in as it currently doesn't hurt anybody to keep
            // it.
            CHECK(accessBlock.create(chunkSize) != nullptr);
        }

        SECTION("memory that can be written to and read from.")
        {
            uint32_t const arbitraryValue = 42;
            auto* ptr = static_cast<uint32_t*>(accessBlock.create(chunkSize));
            *ptr = arbitraryValue;
            CHECK(*ptr == arbitraryValue);
        }

        SECTION("second memory somewhere else.")
        {
            CHECK(accessBlock.create(chunkSize) != accessBlock.create(chunkSize));
        }

        SECTION("memory of different chunk size in different pages.")
        {
            constexpr uint32_t const chunkSize2 = 512U;
            REQUIRE(chunkSize != chunkSize2);
            // To be precise, the second call will actually return a nullptr if there is only a single page (which is
            // one of the test cases at the time of writing). But that technically passes this test, too.
            CHECK(
                accessBlock.pageIndex(accessBlock.create(chunkSize))
                != accessBlock.pageIndex(accessBlock.create(chunkSize2)));
        }

        SECTION("nullptr if there's no page with fitting chunk size")
        {
            // This requests one chunk of a different chunk size for each page. As a new page is required each time,
            // all pages have a chunk size set at the end. And none of those is `chunkSize`.
            for(size_t index = 0; index < accessBlock.numPages(); ++index)
            {
                const auto differentChunkSize = chunkSize + 1U + index;
                REQUIRE(chunkSize != differentChunkSize);
                accessBlock.create(differentChunkSize);
            }

            CHECK(accessBlock.create(chunkSize) == nullptr);
        }

        SECTION("nullptr if all pages have full filling level.")
        {
            fillWith(accessBlock, chunkSize);
            CHECK(accessBlock.create(chunkSize) == nullptr);
        }

        SECTION("last remaining chunk.")
        {
            auto pointers = fillWith(accessBlock, chunkSize);
            size_t const index = GENERATE(0U, 1U, 42U);
            void* pointer = pointers[std::min(index, pointers.size() - 1)];
            accessBlock.destroy(pointer);
            CHECK(accessBlock.create(chunkSize) == pointer);
        }

        SECTION("memory larger than page size.")
        {
            // We give a strong guarantee that this searches the first possible block, so we know the index here.
            if(accessBlock.numPages() >= 2U)
            {
                CHECK(accessBlock.pageIndex(accessBlock.create(2U * pageSize)) == 0U);
            }
        }

        SECTION("nullptr if chunkSize is larger than total available memory in pages.")
        {
            // larger than the available memory but in some cases smaller than the block size even after subtracting
            // the space for the page table:
            uint32_t const excessiveChunkSize = accessBlock.numPages() * pageSize + 1U;
            CHECK(accessBlock.create(excessiveChunkSize) == nullptr);
        }

        SECTION("in the correct place for larger than page size.")
        {
            // we want to allocate 2 pages:
            if(accessBlock.numPages() > 1U)
            {
                // Let's fill up everything first:
                std::vector<void*> pointers(accessBlock.numPages());
                std::generate(
                    std::begin(pointers),
                    std::end(pointers),
                    [&accessBlock]()
                    {
                        // so we really exactly block one page:
                        void* pointer = accessBlock.create(pageSize);
                        REQUIRE(pointer != nullptr);
                        return pointer;
                    });

                // Now, we free two contiguous chunks such that there is one deterministic spot wherefrom our request
                // can be served.
                size_t index = GENERATE(0U, 1U, 5U);
                index = std::min(index, pointers.size() - 2);
                accessBlock.destroy(pointers[index]);
                accessBlock.destroy(pointers[index + 1]);

                // Must be exactly where we free'd the pages:
                CHECK(accessBlock.pageIndex(accessBlock.create(2U * pageSize)) == index);
            }
        }

        SECTION("a pointer and knows it's valid afterwards.")
        {
            void* pointer = accessBlock.create(chunkSize);
            CHECK(accessBlock.isValid(pointer));
        }
    }

    SECTION("destroys")
    {
        void* pointer = accessBlock.create(chunkSize);
        REQUIRE(accessBlock.isValid(pointer));

        SECTION("a pointer thereby invalidating it.")
        {
            accessBlock.destroy(pointer);
            CHECK(not accessBlock.isValid(pointer));
        }
    }
}
