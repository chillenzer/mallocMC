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

#include "mallocMC/creationPolicies/Scatter/AccessBlock.hpp"

#include "mallocMC/creationPolicies/Scatter/BitField.hpp"
#include "mallocMC/creationPolicies/Scatter/PageInterpretation.hpp"
#include "mocks.hpp"

#include <alpaka/acc/AccCpuSerial.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/platform/PlatformCpu.hpp>
#include <alpaka/platform/Traits.hpp>
#include <alpaka/queue/Properties.hpp>
#include <alpaka/queue/Traits.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <type_traits>

template<uint32_t T_blockSize, uint32_t T_pageSize, uint32_t T_wasteFactor = 1U, bool resetfreedpages = true>
struct AccessBlock
    : public mallocMC::CreationPolicies::ScatterAlloc::
          AccessBlock<HeapConfig<T_blockSize, T_pageSize, T_wasteFactor, resetfreedpages>, AlignmentPolicy>
{
    // This is already the default but we want to make it explicit: Just for testing purposes we expose the constructor
    // in order to easily get a test instance.
public:
    AccessBlock() = default;
};

using mallocMC::CreationPolicies::ScatterAlloc::BitMaskStorageType;
using mallocMC::CreationPolicies::ScatterAlloc::PageInterpretation;

constexpr uint32_t const pageTableEntrySize = 8U;
constexpr uint32_t const pageSize1 = 1024U;
constexpr uint32_t const pageSize2 = 4096U;

using BlockAndPageSizes = std::tuple<
    // single page:
    std::tuple<
        std::integral_constant<uint32_t, 1U * (pageSize1 + pageTableEntrySize)>,
        std::integral_constant<uint32_t, pageSize1>>,
    // multiple pages:
    std::tuple<
        std::integral_constant<uint32_t, 4U * (pageSize1 + pageTableEntrySize)>,
        std::integral_constant<uint32_t, pageSize1>>,
    // multiple pages with some overhead:
    std::tuple<
        std::integral_constant<uint32_t, 4U * (pageSize1 + pageTableEntrySize) + 100U>,
        std::integral_constant<uint32_t, pageSize1>>,
    // other page size:
    std::tuple<
        std::integral_constant<uint32_t, 3U * (pageSize2 + pageTableEntrySize) + 100U>,
        std::integral_constant<uint32_t, pageSize2>>>;

template<uint32_t T_blockSize, uint32_t T_pageSize, uint32_t T_wastefactor = 1U>
auto fillWith(AccessBlock<T_blockSize, T_pageSize, T_wastefactor>& accessBlock, uint32_t const chunkSize)
    -> std::vector<void*>
{
    std::vector<void*> pointers(accessBlock.getAvailableSlots(accSerial, chunkSize));
    std::generate(
        std::begin(pointers),
        std::end(pointers),
        [&accessBlock, chunkSize]()
        {
            void* pointer = accessBlock.create(accSerial, chunkSize);
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
        uint32_t slotsPerPage = chunkSize < pageSize ? PageInterpretation<pageSize>::numChunks(chunkSize) : 1U;

        uint32_t numOccupied = GENERATE(0U, 1U, 10U);
        uint32_t actualNumOccupied = numOccupied;
        for(uint32_t i = 0; i < numOccupied; ++i)
        {
            if(accessBlock.create(accSerial, chunkSize) == nullptr)
            {
                actualNumOccupied--;
            }
        }

        auto totalSlots = accessBlock.numPages() * slotsPerPage;
        if(totalSlots > actualNumOccupied)
        {
            CHECK(accessBlock.getAvailableSlots(accSerial, chunkSize) == totalSlots - actualNumOccupied);
        }
        else
        {
            CHECK(accessBlock.getAvailableSlots(accSerial, chunkSize) == 0U);
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
            CHECK(accessBlock.create(accSerial, chunkSize) != nullptr);
        }

        SECTION("memory that can be written to and read from.")
        {
            uint32_t const arbitraryValue = 42;
            auto* ptr = static_cast<uint32_t*>(accessBlock.create(accSerial, chunkSize));
            REQUIRE(ptr != nullptr);
            *ptr = arbitraryValue;
            CHECK(*ptr == arbitraryValue);
        }

        SECTION("second memory somewhere else.")
        {
            CHECK(accessBlock.create(accSerial, chunkSize) != accessBlock.create(accSerial, chunkSize));
        }

        SECTION("memory of different chunk size in different pages.")
        {
            constexpr uint32_t const chunkSize2 = 512U;
            REQUIRE(chunkSize != chunkSize2);
            // To be precise, the second call will actually return a nullptr if there is only a single page (which is
            // one of the test cases at the time of writing). But that technically passes this test, too.

            CHECK(
                accessBlock.pageIndex(accessBlock.create(accSerial, chunkSize))
                != accessBlock.pageIndex(accessBlock.create(accSerial, chunkSize2)));
        }

        SECTION("nullptr if there's no page with fitting chunk size")
        {
            // This requests one chunk of a different chunk size for each page. As a new page is required each time,
            // all pages have a chunk size set at the end. And none of those is `chunkSize`.
            for(uint32_t index = 0; index < accessBlock.numPages(); ++index)
            {
                const auto differentChunkSize = chunkSize + 1U + index;
                REQUIRE(chunkSize != differentChunkSize);
                accessBlock.create(accSerial, differentChunkSize);
            }

            CHECK(accessBlock.create(accSerial, chunkSize) == nullptr);
        }

        SECTION("nullptr if all pages have full filling level.")
        {
            fillWith(accessBlock, chunkSize);
            CHECK(accessBlock.create(accSerial, chunkSize) == nullptr);
        }

        SECTION("last remaining chunk.")
        {
            auto pointers = fillWith(accessBlock, chunkSize);
            uint32_t const index = GENERATE(0U, 1U, 42U);
            void* pointer = pointers[std::min(index, static_cast<uint32_t>(pointers.size()) - 1)];
            accessBlock.destroy(accSerial, pointer);
            CHECK(accessBlock.create(accSerial, chunkSize) == pointer);
        }

        SECTION("memory larger than page size.")
        {
            if(accessBlock.numPages() >= 2U)
            {
                CHECK(accessBlock.isValid(accSerial, accessBlock.create(accSerial, 2U * pageSize)));
            }
        }

        SECTION("nullptr if chunkSize is larger than total available memory in pages.")
        {
            // larger than the available memory but in some cases smaller than the block size even after subtracting
            // the space for the page table:
            uint32_t const excessiveChunkSize = accessBlock.numPages() * pageSize + 1U;
            CHECK(accessBlock.create(accSerial, excessiveChunkSize) == nullptr);
        }

        SECTION("in the correct place for larger than page size.")
        {
            // we want to allocate 2 pages:
            if(accessBlock.numPages() > 1U)
            {
                auto pointers = fillWith(accessBlock, pageSize);
                std::sort(std::begin(pointers), std::end(pointers));

                // Now, we free two contiguous chunks such that there is one deterministic spot wherefrom our request
                // can be served.
                uint32_t index = GENERATE(0U, 1U, 5U);
                index = std::min(index, static_cast<uint32_t>(pointers.size()) - 2U);
                accessBlock.destroy(accSerial, pointers[index]);
                accessBlock.destroy(accSerial, pointers[index + 1]);

                // Must be exactly where we free'd the pages:
                CHECK(
                    accessBlock.pageIndex(accessBlock.create(accSerial, 2U * pageSize))
                    == static_cast<int32_t>(index));
            }
        }

        SECTION("a pointer and knows it's valid afterwards.")
        {
            void* pointer = accessBlock.create(accSerial, chunkSize);
            CHECK(accessBlock.isValid(accSerial, pointer));
        }

        SECTION("the last pointer in page and its allocation does not reach into the bit field.")
        {
            auto slots = accessBlock.getAvailableSlots(accSerial, chunkSize);
            // Find the last allocation on the first page:
            auto pointers = fillWith(accessBlock, chunkSize);
            std::sort(std::begin(pointers), std::end(pointers));
            auto lastOfPage0 = pointers[slots / accessBlock.numPages() - 1];

            // Free the first bit of the bit field by destroying the first allocation in the first page:
            accessBlock.destroy(accSerial, pointers[0]);
            REQUIRE(not accessBlock.isValid(accSerial, pointers[0]));

            // Write all ones to the last of the first page: If there is an overlap between the region of the last
            // chunk and the bit field, our recently free'd first chunk will have its bit set by this operation.
            char* begin = reinterpret_cast<char*>(lastOfPage0);
            auto* end = begin + chunkSize;
            std::fill(begin, end, 255U);

            // Now, we try to allocate one more chunk. It must be the one we free'd before.
            CHECK(accessBlock.create(accSerial, chunkSize) == pointers[0]);
            REQUIRE(accessBlock.isValid(accSerial, pointers[0]));
        }

        SECTION("and writes something very close to page size.")
        {
            // This is a regression test. The original version of the code started to use multi-page mode when numBytes
            // >= pageSize. That is too late because if we're not in multi-page mode, we need to leave some space for
            // the bit mask. Thus, the following test would corrupt the bit mask, if we were to allocate this in
            // chunked mode.

#ifndef NDEBUG
            REQUIRE(sizeof(BitMaskStorageType<>) > 1U);
            auto localChunkSize = pageSize - 1U;
            auto slots = accessBlock.getAvailableSlots(accSerial, localChunkSize);
            auto pointer = accessBlock.create(accSerial, localChunkSize);
            REQUIRE(slots == accessBlock.getAvailableSlots(accSerial, localChunkSize) + 1);
            memset(pointer, 0, localChunkSize);
            CHECK_NOTHROW(accessBlock.destroy(accSerial, pointer));
#else
            SUCCEED("This bug actually never had any observable behaviour in NDEBUG mode because the corrupted bit "
                    "mask is never read again.");
#endif // NDEBUG
        }

        SECTION("with waste factor")
        {
            constexpr uint32_t const wastefactor = 3U;
            AccessBlock<blockSize, pageSize, wastefactor> wastedAccessBlock{};
            auto pointers = fillWith(wastedAccessBlock, chunkSize);

            auto smallerChunkSize = chunkSize / (wastefactor - 1U);
            REQUIRE(smallerChunkSize < chunkSize);

            wastedAccessBlock.destroy(accSerial, pointers[0]);
            SECTION("knows its available slots.")
            {
                REQUIRE(
                    reinterpret_cast<AccessBlock<blockSize, pageSize, 1U>*>(&wastedAccessBlock)
                        ->getAvailableSlots(accSerial, smallerChunkSize)
                    == 0U);

                CHECK(wastedAccessBlock.getAvailableSlots(accSerial, smallerChunkSize) == 1U);
            }

            SECTION("creates a smaller chunk size.")
            {
                REQUIRE(
                    reinterpret_cast<AccessBlock<blockSize, pageSize, 1U>*>(&wastedAccessBlock)
                        ->create(accSerial, smallerChunkSize)
                    == nullptr);

                CHECK(wastedAccessBlock.create(accSerial, smallerChunkSize) == pointers[0]);
            }

            SECTION("fails to create too many smaller chunks.")
            {
                REQUIRE(
                    reinterpret_cast<AccessBlock<blockSize, pageSize, 1U>*>(&wastedAccessBlock)
                        ->create(accSerial, smallerChunkSize)
                    == nullptr);

                CHECK(wastedAccessBlock.create(accSerial, smallerChunkSize) == pointers[0]);
                CHECK(wastedAccessBlock.create(accSerial, smallerChunkSize) == nullptr);
            }

            SECTION("is not misled by mixing above and below multi-page threshold.")
            {
                auto const aboveMultiPageThreshold = pageSize - 2 * sizeof(BitMaskStorageType<>);
                auto const belowMultiPageThreshold = aboveMultiPageThreshold / (wastefactor - 1U);
                for(auto const pointer : pointers)
                {
                    // free one page we want to operate on
                    if(wastedAccessBlock.isValid(accSerial, pointer) and wastedAccessBlock.pageIndex(pointer) == 0U)
                    {
                        wastedAccessBlock.destroy(accSerial, pointer);
                    }
                }
                REQUIRE(wastedAccessBlock.getAvailableSlots(accSerial, belowMultiPageThreshold) == 2U);
                REQUIRE(wastedAccessBlock.getAvailableSlots(accSerial, aboveMultiPageThreshold) == 1U);

                // This allocates in multi-page mode.
                CHECK(wastedAccessBlock.pageIndex(wastedAccessBlock.create(accSerial, aboveMultiPageThreshold)) == 0U);
                // This tries to allocate in chunked mode but the waste factor would allow to create on the just
                // allocated page. This is, of course, forbidden.
                CHECK(wastedAccessBlock.create(accSerial, aboveMultiPageThreshold) == nullptr);
            }
        }
    }

    SECTION("destroys")
    {
        void* pointer = accessBlock.create(accSerial, chunkSize);
        REQUIRE(accessBlock.isValid(accSerial, pointer));

        SECTION("a pointer thereby invalidating it.")
        {
            accessBlock.destroy(accSerial, pointer);
            CHECK(not accessBlock.isValid(accSerial, pointer));
        }

        SECTION("the whole page if last pointer is destroyed.")
        {
            REQUIRE(chunkSize != pageSize);
            REQUIRE(accessBlock.getAvailableSlots(accSerial, pageSize) == accessBlock.numPages() - 1);
            accessBlock.destroy(accSerial, pointer);
            CHECK(accessBlock.getAvailableSlots(accSerial, pageSize) == accessBlock.numPages());
        }

        SECTION("not the whole page if there still exists a valid pointer.")
        {
            REQUIRE(chunkSize != pageSize);
            auto unOccupiedPages = accessBlock.numPages();
            void* newPointer{nullptr};
            // We can't be sure which page is used for any allocation, so we allocate again and again until we have hit
            // a page that already has an allocation:
            while(accessBlock.getAvailableSlots(accSerial, pageSize) != unOccupiedPages)
            {
                unOccupiedPages = accessBlock.getAvailableSlots(accSerial, pageSize);
                newPointer = accessBlock.create(accSerial, chunkSize);
            }
            accessBlock.destroy(accSerial, newPointer);
            CHECK(accessBlock.getAvailableSlots(accSerial, pageSize) == unOccupiedPages);
        }

        SECTION("one slot without touching the others.")
        {
            // this won't be touched:
            accessBlock.create(accSerial, chunkSize);
            auto originalSlots = accessBlock.getAvailableSlots(accSerial, chunkSize);
            accessBlock.destroy(accSerial, pointer);
            CHECK(accessBlock.getAvailableSlots(accSerial, chunkSize) == originalSlots + 1U);
        }

        SECTION("no invalid pointer but throws instead.")
        {
#ifndef NDEBUG
            pointer = nullptr;
            CHECK_THROWS(
                accessBlock.destroy(accSerial, pointer),
                std::runtime_error{"Attempted to destroy an invalid pointer!"});
#endif // NDEBUG
        }

        SECTION("pointer for larger than page size")
        {
            if(accessBlock.numPages() > 1U)
            {
                accessBlock.destroy(accSerial, pointer);
                REQUIRE(accessBlock.getAvailableSlots(accSerial, pageSize) == accessBlock.numPages());

                pointer = accessBlock.create(accSerial, 2U * pageSize);
                REQUIRE(accessBlock.getAvailableSlots(accSerial, pageSize) == accessBlock.numPages() - 2);
                REQUIRE(accessBlock.isValid(accSerial, pointer));

                accessBlock.destroy(accSerial, pointer);

                SECTION("thereby invalidating it.")
                {
                    CHECK(not accessBlock.isValid(accSerial, pointer));
                }

                SECTION("thereby freeing up their pages.")
                {
                    CHECK(accessBlock.getAvailableSlots(accSerial, pageSize) == accessBlock.numPages());
                }
            }
        }

        SECTION("and doesn't reset the page.")
        {
            auto& unresettingAccessBlock
                = *reinterpret_cast<AccessBlock<blockSize, pageSize, 1U, /*resetfreedpages=*/false>*>(&accessBlock);
            auto const differentChunkSize = GENERATE(17, 2048);
            REQUIRE(differentChunkSize != chunkSize);
            auto const slots = unresettingAccessBlock.getAvailableSlots(accSerial, differentChunkSize);

            unresettingAccessBlock.destroy(accSerial, pointer);
            CHECK(unresettingAccessBlock.getAvailableSlots(accSerial, differentChunkSize) == slots);
        }

        SECTION("and doesn't reset the page for larger than page size.")
        {
            auto* largePointer = accessBlock.create(accSerial, pageSize);
            if(largePointer != nullptr)
            {
                auto& unresettingAccessBlock
                    = *reinterpret_cast<AccessBlock<blockSize, pageSize, 1U, /*resetfreedpages=*/false>*>(
                        &accessBlock);
                auto const differentChunkSize = GENERATE(17, 2048);
                REQUIRE(differentChunkSize != chunkSize);
                auto const slots = unresettingAccessBlock.getAvailableSlots(accSerial, differentChunkSize);

                unresettingAccessBlock.destroy(accSerial, largePointer);
                CHECK(unresettingAccessBlock.getAvailableSlots(accSerial, differentChunkSize) == slots);
            }
        }
    }
}
