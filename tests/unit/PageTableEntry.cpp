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

using mallocMC::CreationPolicies::ScatterAlloc::PageTable;

constexpr const size_t numPages = 3;

TEST_CASE("PageTable")
{
    PageTable<numPages> pageTable{};

    SECTION("initialises bit masks to 0.")
    {
        for(auto const& bitMask : pageTable._bitMasks)
        {
            CHECK(bitMask == 0U);
        }
    }

    SECTION("initialises chunk sizes to 0.")
    {
        for(auto const& chunkSize : pageTable._chunkSizes)
        {
            CHECK(chunkSize == 0U);
        }
    }

    SECTION("initialises filling levels to 0.")
    {
        for(auto const& fillingLevel : pageTable._fillingLevels)
        {
            CHECK(fillingLevel == 0U);
        }
    }
}
