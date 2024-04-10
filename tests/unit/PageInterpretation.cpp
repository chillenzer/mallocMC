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

#include <catch2/catch.hpp>
#include <cstdint>
#include <mallocMC/creationPolicies/Scatter.hpp>
#include <optional>

using mallocMC::CreationPolicies::ScatterAlloc::BitMask;
using mallocMC::CreationPolicies::ScatterAlloc::DataPage;
using mallocMC::CreationPolicies::ScatterAlloc::PageInterpretation;
using std::distance;

constexpr size_t pageSize = 1024U;

TEST_CASE("PageInterpretation")
{
    uint32_t chunkSize = 32U; // NOLINT(*magic-number*)
    DataPage<pageSize> data{};
    BitMask mask{};
    PageInterpretation<pageSize> page{data, chunkSize, mask};

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
        PageInterpretation<pageSize> localPage{data, localChunkSize, mask};
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
        CHECK(page.bitField() == std::nullopt);
    }

    SECTION("recognises if there is a bit field at the end.")
    {
        uint32_t localChunkSize = 4U;
        PageInterpretation<pageSize> localPage{data, localChunkSize, mask};
        CHECK(localPage.bitField() != std::nullopt);
    }
}
