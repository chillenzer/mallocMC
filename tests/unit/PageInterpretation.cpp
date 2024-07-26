/*
  mallocMC: Memory Allocator for Many Core Architectures.

  Copyright 2024 Helmholtz-Zentrum Dresden - Rossendorf

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

#include "mallocMC/creationPolicies/Scatter/BitField.hpp"
#include "mallocMC/creationPolicies/Scatter/DataPage.hpp"
#include "mallocMC/mallocMC_utils.hpp"
#include "mocks.hpp"

#include <alpaka/acc/AccCpuSerial.hpp>
#include <alpaka/acc/AccCpuThreads.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/platform/PlatformCpu.hpp>
#include <alpaka/platform/Traits.hpp>
#include <alpaka/queue/Properties.hpp>
#include <alpaka/queue/Traits.hpp>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>

using mallocMC::CreationPolicies::ScatterAlloc::BitMask;
using mallocMC::CreationPolicies::ScatterAlloc::BitMaskSize;
using mallocMC::CreationPolicies::ScatterAlloc::DataPage;
using mallocMC::CreationPolicies::ScatterAlloc::PageInterpretation;
using std::distance;

TEST_CASE("PageInterpretation")
{
    constexpr uint32_t const pageSize = 1024U;
    constexpr uint32_t const chunkSize = 32U;
    DataPage<pageSize> data{};
    PageInterpretation<pageSize> page{data, chunkSize};

    SECTION("refers to the same data it was created with.")
    {
        CHECK(&data == page.chunkPointer(0));
    }

    SECTION("returns start of data as first chunk.")
    {
        CHECK(page.chunkPointer(0) == &data);
    }

    SECTION("computes correct number of pages.")
    {
        std::array<uint32_t, 9> chunkSizes{1, 2, 4, 5, 10, 11, 31, 32, 512};
        std::array<uint32_t, 9> expectedNumChunks{908, 480, 248, 199, 100, 92, 32, 31, 1};

        for(uint32_t i = 0U; i < 9; ++i)
        {
            CHECK(PageInterpretation<pageSize>::numChunks(chunkSizes[i]) == expectedNumChunks[i]);
        }
    }

    SECTION("jumps by chunkSize between indices.")
    {
        for(auto i = 0U; i < (pageSize / chunkSize) - 1; ++i)
        {
            CHECK(
                distance(
                    reinterpret_cast<char*>(page.chunkPointer(i)),
                    reinterpret_cast<char*>(page.chunkPointer(i + 1)))
                == chunkSize);
        }
    }

    SECTION("knows the maximal bit field size.")
    {
        // 116 allows for 29 bit masks capable of representing 928 1-byte chunks. This would give a total addressable
        // size of 1044, so in this scenario the last 20 bits of the mask would be invalid.
        CHECK(page.maxBitFieldSize() == 116U);
    }

    SECTION("reports numChunks that fit the page.")
    {
        CHECK(
            page.numChunks() * chunkSize + mallocMC::ceilingDivision(page.numChunks(), BitMaskSize) * sizeof(BitMask)
            <= pageSize);
    }

    SECTION("knows correct bit field size.")
    {
        uint32_t const numChunks = GENERATE(2, BitMaskSize - 1, BitMaskSize, 2 * BitMaskSize);
        uint32_t localChunkSize = pageSize / numChunks;
        PageInterpretation<pageSize> localPage{data, localChunkSize};
        CHECK(localPage.bitFieldSize() == sizeof(BitMask) * mallocMC::ceilingDivision(numChunks, BitMaskSize));
    }
}

TEST_CASE("PageInterpretation.create")
{
    // Such that we can fit up to four levels of hierarchy in there:
    constexpr uint32_t const pageSize
        = BitMaskSize * BitMaskSize * BitMaskSize * BitMaskSize + BitMaskSize * sizeof(BitMask);
    // This is more than 8MB which is a typical stack's size. Let's save us some trouble and create it on the heap.
    std::unique_ptr<DataPage<pageSize>> actualData{new DataPage<pageSize>};
    DataPage<pageSize>& data{*actualData};

    SECTION("regardless of hierarchy")
    {
        uint32_t numChunks = GENERATE(BitMaskSize * BitMaskSize, BitMaskSize);
        uint32_t chunkSize{0U};
        // this is a bit weird because we have to make sure that we are always handling full bit masks (until the
        // handling of partially filled bit masks is implemented)
        if(numChunks == BitMaskSize * BitMaskSize)
        {
            chunkSize = 1024U; // NOLINT(*magic-number*)
        }
        else if(numChunks == BitMaskSize)
        {
            chunkSize = 32771U; // NOLINT(*magic-number*)
        }
        PageInterpretation<pageSize> page{data, chunkSize};

        SECTION("returns a pointer to within the data.")
        {
            auto* pointer = page.create(accSerial);
            CHECK(
                std::distance(reinterpret_cast<char*>(page.chunkPointer(0)), reinterpret_cast<char*>(pointer))
                < std::distance(
                    reinterpret_cast<char*>(page.chunkPointer(0)),
                    reinterpret_cast<char*>(page.bitFieldStart())));
        }

        SECTION("returns a pointer to the start of a chunk.")
        {
            auto* pointer = page.create(accSerial);
            CHECK(
                std::distance(reinterpret_cast<char*>(page.chunkPointer(0)), reinterpret_cast<char*>(pointer))
                    % chunkSize
                == 0U);
        }

        SECTION("returns nullptr if everything is full.")
        {
            for(auto& mask : page.bitField())
            {
                mask.set(accSerial);
            }
            auto* pointer = page.create(accSerial);
            CHECK(pointer == nullptr);
        }

        SECTION("can provide numChunks pieces of memory and returns nullptr afterwards.")
        {
            for(uint32_t i = 0; i < page.numChunks(); ++i)
            {
                auto* pointer = page.create(accSerial);
                CHECK(pointer != nullptr);
            }
            auto* pointer = page.create(accSerial);
            CHECK(pointer == nullptr);
        }
    }

    SECTION("without hierarchy")
    {
        uint32_t const numChunks = BitMaskSize;
        uint32_t chunkSize = pageSize / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize};

        SECTION("updates bit field.")
        {
            BitMask& mask{page.bitField().get(0)};
            REQUIRE(mask.none());
            auto* pointer = page.create(accSerial);
            auto const index = page.chunkNumberOf(pointer);
            CHECK(mask(accSerial, index));
        }
    }
}

TEST_CASE("PageInterpretation.destroy")
{
    // Such that we can fit up to four levels of hierarchy in there:
    constexpr uint32_t const pageSize = BitMaskSize * BitMaskSize * BitMaskSize * BitMaskSize
        + BitMaskSize * BitMaskSize * BitMaskSize * sizeof(BitMask);
    // This is more than 8MB which is a typical stack's size. Let's save us some trouble and create it on the heap.
    std::unique_ptr<DataPage<pageSize>> actualData{new DataPage<pageSize>};
    DataPage<pageSize>& data{*actualData};

    SECTION("regardless of hierarchy")
    {
        uint32_t numChunks = GENERATE(BitMaskSize * BitMaskSize, BitMaskSize);
        uint32_t chunkSize = pageSize / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize};
        auto* pointer = page.create(accSerial);

#ifndef NDEBUG
        SECTION("throws if given an invalid pointer.")
        {
            pointer = nullptr;
            CHECK_THROWS(
                page.destroy(accSerial, pointer),
                throw std::runtime_error{"Attempted to destroy an invalid pointer! Either the pointer does not point "
                                         "to a valid chunk or it is not marked as allocated."});
        }

        SECTION("allows pointers to anywhere in the chunk.")
        {
            // This test documents the state as is. We haven't defined this outcome as a requirement but if we change
            // it, we might still want to be aware of this because users might want to be informed.
            pointer = reinterpret_cast<void*>(reinterpret_cast<char*>(pointer) + chunkSize / 2);
            CHECK_NOTHROW(page.destroy(accSerial, pointer));
        }
#endif // NDEBUG

        SECTION("only ever unsets (and never sets) bits in top-level bit mask.")
        {
            // We extract the position of the mask before destroying the pointer because technically speaking the whole
            // concept of a mask doesn't apply anymore after that pointer was destroyed because that will automatically
            // free the page.
            auto mask = page.bitField().get(0);
            auto value = mask;
            page.destroy(accSerial, pointer);
            CHECK(mask <= value);
        }


        SECTION("cleans up in bit field region of page.")
        {
            memset(std::begin(data.data), std::numeric_limits<char>::max(), page.numChunks() * chunkSize);
            page.cleanup();

            for(uint32_t i = pageSize - page.maxBitFieldSize(); i < pageSize; ++i)
            {
                CHECK(data.data[i] == 0U);
            }
        }
    }
}
// NOLINTEND(*widening*)
