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
#include <mallocMC/creationPolicies/Scatter.hpp>

using mallocMC::CreationPolicies::ScatterAlloc::DataPage;
using mallocMC::CreationPolicies::ScatterAlloc::PageInterpretation;
using std::distance;

constexpr size_t pageSize = 4096U;
constexpr size_t chunkSize = 32U;

TEST_CASE("PageInterpretation")
{
    DataPage<pageSize> data{};
    PageInterpretation<pageSize> page{data, chunkSize};

    SECTION("refers to the same data it was created with.")
    {
        CHECK(&data == &page.data);
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
        size_t localChunkSize = GENERATE(32U, 512U);
        PageInterpretation<pageSize> localPage{data, localChunkSize};
        CHECK(localPage.hasBitField() == (pageSize / localChunkSize) > 32U);
    }

    SECTION("jumps by chunkSize between indices.")
    {
        for(auto i = 0U; i < (pageSize / chunkSize) - 1; ++i)
        {
            CHECK(distance(reinterpret_cast<char*>(page[i]), reinterpret_cast<char*>(page[i + 1])) == chunkSize);
        }
    }
}
