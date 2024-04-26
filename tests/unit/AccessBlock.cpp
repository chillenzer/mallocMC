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
        uint32_t actualNumOccupied = numOccupied;
        for(uint32_t i = 0; i < numOccupied; ++i)
        {
            if(accessBlock.create(chunkSize) == nullptr)
            {
                actualNumOccupied--;
            }
        }

        auto totalSlots = accessBlock.numPages() * slotsPerPage;
        if(totalSlots > actualNumOccupied)
        {
            CHECK(accessBlock.getAvailableSlots(chunkSize) == totalSlots - actualNumOccupied);
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
                auto pointers = fillWith(accessBlock, pageSize);
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

        SECTION("the last pointer in page and its allocation does not reach into the bit field.")
        {
            auto slots = accessBlock.getAvailableSlots(chunkSize);
            // Find the last allocation on the first page:
            auto pointers = fillWith(accessBlock, chunkSize);
            std::sort(std::begin(pointers), std::end(pointers));
            auto lastOfPage0 = pointers[slots / accessBlock.numPages() - 1];

            // Free the first bit of the bit field by destroying the first allocation in the first page:
            accessBlock.destroy(pointers[0]);
            REQUIRE(not accessBlock.isValid(pointers[0]));

            // Write all ones to the last of the first page: If there is an overlap between the region of the last
            // chunk and the bit field, our recently free'd first chunk will have its bit set by this operation.
            char* begin = reinterpret_cast<char*>(lastOfPage0);
            auto* end = begin + chunkSize;
            std::fill(begin, end, 255U);

            // Now, we try to allocate one more chunk. It must be the one we free'd before.
            CHECK(accessBlock.create(chunkSize) == pointers[0]);
            REQUIRE(accessBlock.isValid(pointers[0]));
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

        SECTION("the whole page if last pointer is destroyed.")
        {
            REQUIRE(chunkSize != pageSize);
            REQUIRE(accessBlock.getAvailableSlots(pageSize) == accessBlock.numPages() - 1);
            accessBlock.destroy(pointer);
            CHECK(accessBlock.getAvailableSlots(pageSize) == accessBlock.numPages());
        }

        SECTION("not the whole page if there still exists a valid pointer.")
        {
            REQUIRE(chunkSize != pageSize);
            auto unOccupiedPages = accessBlock.numPages();
            void* newPointer{nullptr};
            // We can't be sure which page is used for any allocation, so we allocate again and again until we have hit
            // a page that already has an allocation:
            while(accessBlock.getAvailableSlots(pageSize) != unOccupiedPages)
            {
                unOccupiedPages = accessBlock.getAvailableSlots(pageSize);
                newPointer = accessBlock.create(chunkSize);
            }
            accessBlock.destroy(newPointer);
            CHECK(accessBlock.getAvailableSlots(pageSize) == unOccupiedPages);
        }

        SECTION("one slot without touching the others.")
        {
            // this won't be touched:
            accessBlock.create(chunkSize);
            auto originalSlots = accessBlock.getAvailableSlots(chunkSize);
            accessBlock.destroy(pointer);
            CHECK(accessBlock.getAvailableSlots(chunkSize) == originalSlots + 1U);
        }

        SECTION("no invalid pointer but throws instead.")
        {
            pointer = nullptr;
            CHECK_THROWS_WITH(accessBlock.destroy(pointer), Catch::Contains("Attempted to destroy invalid pointer"));
        }

        SECTION("pointer for larger than page size")
        {
            if(accessBlock.numPages() > 1U)
            {
                accessBlock.destroy(pointer);
                REQUIRE(accessBlock.getAvailableSlots(pageSize) == accessBlock.numPages());

                pointer = accessBlock.create(2U * pageSize);
                REQUIRE(accessBlock.getAvailableSlots(pageSize) == accessBlock.numPages() - 2);
                REQUIRE(accessBlock.isValid(pointer));

                accessBlock.destroy(pointer);

                SECTION("thereby invalidating it.")
                {
                    CHECK(not accessBlock.isValid(pointer));
                }

                SECTION("thereby freeing up their pages.")
                {
                    CHECK(accessBlock.getAvailableSlots(pageSize) == accessBlock.numPages());
                }
            }
        }
    }
}
