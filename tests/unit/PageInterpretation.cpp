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

// This is fine. We're mixing uint32_t and size_t from time to time to do manual index calculations. That will not
// happen in production code.
// NOLINTBEGIN(*widening*)
#include "mallocMC/creationPolicies/Scatter/PageInterpretation.hpp"

#include "mallocMC/creationPolicies/Scatter/BitField.hpp"
#include "mallocMC/creationPolicies/Scatter/DataPage.hpp"

#include <catch2/catch.hpp>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <optional>

using mallocMC::CreationPolicies::ScatterAlloc::BitMask;
using mallocMC::CreationPolicies::ScatterAlloc::BitMaskSize;
using mallocMC::CreationPolicies::ScatterAlloc::DataPage;
using mallocMC::CreationPolicies::ScatterAlloc::PageInterpretation;
using std::distance;


TEST_CASE("PageInterpretation")
{
    constexpr size_t pageSize = 1024U;
    uint32_t chunkSize = 32U; // NOLINT(*magic-number*)
    DataPage<pageSize> data{};
    uint32_t fillingLevel{};
    PageInterpretation<pageSize> page{data, chunkSize, fillingLevel};

    SECTION("refers to the same data it was created with.")
    {
        CHECK(&data == &page._data);
    }

    SECTION("returns start of data as first chunk.")
    {
        CHECK(page[0] == &data);
    }

    SECTION("computes correct number of pages.")
    {
        CHECK(page.numChunks() == 31U);
    }

    SECTION("detects correctly if page should contain bitfield.")
    {
        uint32_t localChunkSize = GENERATE(8U, 128U);
        PageInterpretation<pageSize> localPage{data, localChunkSize, fillingLevel};
        CHECK(localPage.hasBitField() == (pageSize / localChunkSize) > 32U);
    }

    SECTION("jumps by chunkSize between indices.")
    {
        for(auto i = 0U; i < (pageSize / chunkSize) - 1; ++i)
        {
            CHECK(distance(reinterpret_cast<char*>(page[i]), reinterpret_cast<char*>(page[i + 1])) == chunkSize);
        }
    }

    SECTION("finds first free chunk.")
    {
        BitMask& mask{page.topLevelMask()};
        mask.flip();
        size_t const index = GENERATE(0, 2);
        mask.flip(index);
        auto const chunk = page.firstFreeChunk();

        if(chunk)
        {
            CHECK(chunk.value().pointer == page[index]);
        }
        else
        {
            FAIL("Expected to get a valid chunk but didn't.");
        }
    }

    SECTION("returns nullopt if all chunks are full.")
    {
        for(auto& mask : page.bitField())
        {
            mask.set();
        }
        CHECK(page.firstFreeChunk() == std::nullopt);
    }

    SECTION("knows the maximal bit field size.")
    {
        // pageSize = 1024 with chunks of size one allows for more than 32 but less than 32^2 chunks, so maximal bit
        // field size should be
        CHECK(page.maxBitFieldSize() == 256);
    }
}

TEST_CASE("PageInterpretation.bitFieldDepth")
{
    uint32_t fillingLevel{};
    // Such that we can fit up to four levels of hierarchy in there:
    constexpr size_t const pageSize = BitMaskSize * BitMaskSize * BitMaskSize * BitMaskSize
        + BitMaskSize * BitMaskSize * BitMaskSize * sizeof(BitMask);
    // This is more than 8MB which is a typical stack's size. Let's save us some trouble and create it on the heap.
    std::unique_ptr<DataPage<pageSize>> actualData{new DataPage<pageSize>};
    DataPage<pageSize>& data{*actualData};

    SECTION("knows correct bit field depths for depth 0.")
    {
        uint32_t const numChunks = BitMaskSize;
        uint32_t chunkSize = pageSize / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, fillingLevel};

        CHECK(page.bitFieldDepth() == 0U);
    }

    SECTION("knows correct bit field depths for depth 0 with less chunks.")
    {
        uint32_t const numChunks = BitMaskSize - 1;
        uint32_t chunkSize = pageSize / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, fillingLevel};

        CHECK(page.bitFieldDepth() == 0U);
    }

    SECTION("knows correct bit field depths for depth 1.")
    {
        uint32_t const numChunks = BitMaskSize * BitMaskSize;
        // choose chunk size such that bit field fits behind it:
        uint32_t chunkSize = (pageSize - BitMaskSize * sizeof(BitMask)) / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, fillingLevel};

        CHECK(page.bitFieldDepth() == 1U);
    }

    SECTION("knows correct bit field depths for depth 1 with less chunks.")
    {
        uint32_t const numChunks = BitMaskSize * BitMaskSize - 1;
        // choose chunk size such that bit field fits behind it:
        uint32_t chunkSize = (pageSize - BitMaskSize * sizeof(BitMask)) / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, fillingLevel};

        CHECK(page.bitFieldDepth() == 1U);
    }
}

TEST_CASE("PageInterpretation.create")
{
    uint32_t fillingLevel{};
    // Such that we can fit up to four levels of hierarchy in there:
    constexpr size_t const pageSize
        = BitMaskSize * BitMaskSize * BitMaskSize * BitMaskSize + BitMaskSize * sizeof(BitMask);
    // This is more than 8MB which is a typical stack's size. Let's save us some trouble and create it on the heap.
    std::unique_ptr<DataPage<pageSize>> actualData{new DataPage<pageSize>};
    DataPage<pageSize>& data{*actualData};

    SECTION("regardless of hierarchy")
    {
        uint32_t numChunks = GENERATE(BitMaskSize * BitMaskSize, BitMaskSize);
        uint32_t chunkSize = (pageSize - numChunks / sizeof(BitMask)) / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, fillingLevel};

        SECTION("returns a pointer to within the data.")
        {
            auto* pointer = page.create();
            CHECK(
                std::distance(&page._data.data[0], reinterpret_cast<char*>(pointer))
                < static_cast<long>(page.dataSize()));
        }

        SECTION("returns a pointer to the start of a chunk.")
        {
            auto* pointer = page.create();
            CHECK(std::distance(&page._data.data[0], reinterpret_cast<char*>(pointer)) % chunkSize == 0U);
        }

        SECTION("returns nullptr if everything is full.")
        {
            for(auto& mask : page.bitField())
            {
                mask.set();
            }
            auto* pointer = page.create();
            CHECK(pointer == nullptr);
        }

        SECTION("can provide numChunks pieces of memory and returns nullptr afterwards.")
        {
            for(uint32_t i = 0; i < page.numChunks(); ++i)
            {
                auto* pointer = page.create();
                CHECK(pointer != nullptr);
            }
            auto* pointer = page.create();
            CHECK(pointer == nullptr);
        }
    }

    SECTION("without hierarchy")
    {
        uint32_t const numChunks = BitMaskSize;
        uint32_t chunkSize = pageSize / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, fillingLevel};

        SECTION("updates top-level bit field.")
        {
            BitMask& mask{page.topLevelMask()};
            REQUIRE(mask.none());
            auto* pointer = page.create();
            auto const index = page.chunkNumberOf(pointer);
            CHECK(mask[index]);
        }
    }
}

TEST_CASE("PageInterpretation.destroy")
{
    uint32_t fillingLevel{};
    // Such that we can fit up to four levels of hierarchy in there:
    constexpr size_t const pageSize = BitMaskSize * BitMaskSize * BitMaskSize * BitMaskSize
        + BitMaskSize * BitMaskSize * BitMaskSize * sizeof(BitMask);
    // This is more than 8MB which is a typical stack's size. Let's save us some trouble and create it on the heap.
    std::unique_ptr<DataPage<pageSize>> actualData{new DataPage<pageSize>};
    DataPage<pageSize>& data{*actualData};

    SECTION("regardless of hierarchy")
    {
        uint32_t numChunks = GENERATE(BitMaskSize * BitMaskSize, BitMaskSize);
        uint32_t chunkSize = pageSize / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, fillingLevel};
        auto* pointer = page.create();
        // this was originally done in the page but now that's responsibility of the AccessBlock:
        page._fillingLevel = 1U;

        SECTION("throws if given an invalid pointer.")
        {
            pointer = nullptr;
            CHECK_THROWS_WITH(
                page.destroy(pointer),
                Catch::Contains("Attempted to destroy out-of-bounds pointer. Chunk index out of range!"));
        }

        SECTION("allows pointers to anywhere in the chunk.")
        {
            // This test documents the state as is. We haven't defined this outcome as a requirement but if we change
            // it, we might still want to be aware of this because users might want to be informed.
            pointer = reinterpret_cast<void*>(reinterpret_cast<char*>(pointer) + chunkSize / 2);
            CHECK_NOTHROW(page.destroy(pointer));
        }

        SECTION("only ever unsets (and never sets) bits in top-level bit mask.")
        {
            // We extract the position of the mask before destroying the pointer because technically speaking the whole
            // concept of a mask doesn't apply anymore after that pointer was destroyed because that will automatically
            // free the page.
            auto mask = page.topLevelMask();
            auto count = mask.count();
            page.destroy(pointer);
            CHECK(mask.count() <= count);
        }



        SECTION("resets chunk size when page is abandoned.")
        {
            // this is the only allocation on this page, so page is abandoned afterwards
            page.destroy(pointer);
            CHECK(page._chunkSize == 0U);
        }

        SECTION("cleans up in bit field region of page when page is abandoned.")
        {
            memset(std::begin(data.data), 255U, page.numChunks() * chunkSize);
            // This is the only allocation on this page, so the page is abandoned afterwards.
            page.destroy(pointer);

            // TODO(lenz): Check for off-by-one error in lower bound.
            for(size_t i = pageSize - page.maxBitFieldSize(); i < pageSize; ++i)
            {
                CHECK(static_cast<uint32_t>(reinterpret_cast<char*>(page._data.data)[i]) == 0U);
            }
        }
    }
}


TEST_CASE("PageInterpretation.destroy (failing)", "[!shouldfail]")
{
    uint32_t fillingLevel{};
    // Such that we can fit up to four levels of hierarchy in there:
    constexpr size_t const pageSize = BitMaskSize * BitMaskSize * BitMaskSize * BitMaskSize
        + BitMaskSize * BitMaskSize * BitMaskSize * sizeof(BitMask);
    // This is more than 8MB which is a typical stack's size. Let's save us some trouble and create it on the heap.
    std::unique_ptr<DataPage<pageSize>> actualData{new DataPage<pageSize>};
    DataPage<pageSize>& data{*actualData};

    uint32_t numChunks = GENERATE(BitMaskSize * BitMaskSize, BitMaskSize);
    uint32_t chunkSize = pageSize / numChunks;
    PageInterpretation<pageSize> page{data, chunkSize, fillingLevel};
    auto* pointer = page.create();


    SECTION("initialises invalid bits to filled.")
    {
        FAIL("does not handle cases where numChunks is not a multiple of the BitMaskSize.");
    }
}

// NOLINTEND(*widening*)
