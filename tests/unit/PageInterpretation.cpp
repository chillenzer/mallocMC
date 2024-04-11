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

#include <catch2/catch.hpp>
#include <cstdint>
#include <optional>

using mallocMC::CreationPolicies::ScatterAlloc::BitMask;
using mallocMC::CreationPolicies::ScatterAlloc::BitMaskSize;
using mallocMC::CreationPolicies::ScatterAlloc::DataPage;
using mallocMC::CreationPolicies::ScatterAlloc::PageInterpretation;
using mallocMC::CreationPolicies::ScatterAlloc::treeVolume;
using std::distance;


TEST_CASE("PageInterpretation")
{
    constexpr size_t pageSize = 1024U;
    uint32_t chunkSize = 32U; // NOLINT(*magic-number*)
    DataPage<pageSize> data{};
    BitMask mask{};
    uint32_t fillingLevel{};
    PageInterpretation<pageSize> page{data, chunkSize, mask, fillingLevel};

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
        CHECK(page.numChunks() == pageSize / chunkSize);
    }

    SECTION("detects correctly if page should contain bitfield.")
    {
        uint32_t localChunkSize = GENERATE(8U, 128U);
        PageInterpretation<pageSize> localPage{data, localChunkSize, mask, fillingLevel};
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
        mask.flip();
        CHECK(page.firstFreeChunk() == std::nullopt);
    }

    SECTION("recognises if there is no bit field at the end.")
    {
        CHECK(page.bitField().levels == nullptr);
        CHECK(page.bitField().depth == 0U);
    }

    SECTION("recognises if there is a bit field at the end.")
    {
        uint32_t localChunkSize = 1U;
        PageInterpretation<pageSize> localPage{data, localChunkSize, mask, fillingLevel};
        CHECK(localPage.bitField().levels != nullptr);
        CHECK(localPage.bitField().depth == 1U);
    }
}

TEST_CASE("PageInterpretation.bitFieldDepth")
{
    uint32_t fillingLevel{};
    constexpr uint32_t const BitMaskBytes = BitMaskSize / 8U;
    // Such that we can fit up to four levels of hierarchy in there:
    constexpr size_t const pageSize
        = BitMaskSize * BitMaskSize * BitMaskSize * BitMaskSize + treeVolume<BitMaskSize>(4) * BitMaskBytes;
    DataPage<pageSize> data{};
    BitMask mask{};

    SECTION("knows correct bit field depths for depth 0.")
    {
        uint32_t const numChunks = BitMaskSize;
        uint32_t chunkSize = pageSize / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, mask, fillingLevel};

        CHECK(page.bitFieldDepth() == 0U);
    }

    SECTION("knows correct bit field depths for depth 0 with less chunks.")
    {
        uint32_t const numChunks = BitMaskSize - 1;
        uint32_t chunkSize = pageSize / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, mask, fillingLevel};

        CHECK(page.bitFieldDepth() == 0U);
    }

    SECTION("knows correct bit field depths for depth 1.")
    {
        uint32_t const numChunks = BitMaskSize * BitMaskSize;
        // choose chunk size such that bit field fits behind it:
        uint32_t chunkSize = (pageSize - BitMaskSize * BitMaskBytes) / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, mask, fillingLevel};

        CHECK(page.bitFieldDepth() == 1U);
    }

    SECTION("knows correct bit field depths for depth 1 with less chunks.")
    {
        uint32_t const numChunks = BitMaskSize * BitMaskSize - 1;
        // choose chunk size such that bit field fits behind it:
        uint32_t chunkSize = (pageSize - BitMaskSize * BitMaskBytes) / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, mask, fillingLevel};

        CHECK(page.bitFieldDepth() == 1U);
    }

    SECTION(
        "knows correct bit field depths for depth 1 with slightly too big chunks such that we get less than expected.")
    {
        uint32_t const numChunks = BitMaskSize * BitMaskSize - 1;
        // choose chunk size such that bit field fits behind it:
        uint32_t chunkSize = pageSize / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, mask, fillingLevel};

        CHECK(page.bitFieldDepth() == 1U);
    }

    SECTION("knows correct bit field depths for depth 2.")
    {
        uint32_t const numChunks = BitMaskSize * BitMaskSize * BitMaskSize;
        // choose chunk size such that bit field fits behind it:
        uint32_t chunkSize
            = (pageSize - BitMaskSize * BitMaskBytes - BitMaskSize * BitMaskSize * BitMaskBytes) / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, mask, fillingLevel};

        CHECK(page.bitFieldDepth() == 2U);
    }

    SECTION("knows correct bit field depths for depth 3.")
    {
        uint32_t const numChunks = BitMaskSize * BitMaskSize * BitMaskSize * BitMaskSize;
        // choose chunk size such that bit field fits behind it:
        uint32_t chunkSize = (pageSize - BitMaskSize * BitMaskBytes - BitMaskSize * BitMaskSize * BitMaskBytes
                              - BitMaskSize * BitMaskSize * BitMaskSize * BitMaskBytes)
            / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, mask, fillingLevel};

        CHECK(page.bitFieldDepth() == 3U);
    }
}
// NOLINTEND(*widening*)
