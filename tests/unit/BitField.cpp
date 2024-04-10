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

using mallocMC::CreationPolicies::ScatterAlloc::BitFieldTree;
using mallocMC::CreationPolicies::ScatterAlloc::BitMask;
using mallocMC::CreationPolicies::ScatterAlloc::BitMaskSize;
using mallocMC::CreationPolicies::ScatterAlloc::firstFreeBit;

TEST_CASE("BitMask")
{
    BitMask mask{};

    SECTION("is initialised to 0.")
    {
        CHECK(mask == 0U);
    }

    SECTION("can have individual bits read.")
    {
        for(size_t i = 0; i < BitMaskSize; ++i)
        {
            CHECK(mask[i] == false);
        }
    }

    SECTION("allows to write individual bits.")
    {
        for(size_t i = 0; i < BitMaskSize; ++i)
        {
            mask.set(i, true);
            CHECK(mask[i]);
        }
    }

    SECTION("knows the first free bit.")
    {
        mask.flip();
        size_t const index = GENERATE(0, 3);
        mask.flip(index);
        CHECK(firstFreeBit(mask) == index);
    }

    SECTION("returns BitMaskSize as first free bit if there is none.")
    {
        mask.flip();
        CHECK(firstFreeBit(mask) == BitMaskSize);
    }
}

TEST_CASE("BitFieldTree")
{
    SECTION("knows its first free bit with depth 0.")
    {
        BitMask mask{};
        mask.flip();
        uint32_t const index = GENERATE(0, 3);
        mask.flip(index);

        BitFieldTree tree{mask, nullptr, 0U};

        CHECK(firstFreeBit(tree) == index);
    }

    SECTION("knows its first free bit with depth 1.")
    {
        BitMask head{};
        BitMask main[BitMaskSize]{};
        uint32_t const firstLevelIndex = GENERATE(0, 3);
        uint32_t const secondLevelIndex = GENERATE(2, 5);

        head.flip();
        head.flip(firstLevelIndex);
        for(auto& bitMask : main)
        {
            bitMask.flip();
        }
        main[firstLevelIndex].flip(secondLevelIndex);

        BitFieldTree tree{head, &(main[0]), 1U};

        CHECK(firstFreeBit(tree) == firstLevelIndex * BitMaskSize + secondLevelIndex);
    }

    SECTION("knows its first free bit with depth 2.")
    {
        BitMask head{};
        BitMask main[BitMaskSize * (1 + BitMaskSize)]{};
        uint32_t const firstLevelIndex = GENERATE(0, 5);
        uint32_t const secondLevelIndex = GENERATE(2, 7);
        uint32_t const thirdLevelIndex = GENERATE(3, 11);

        head.flip();
        for(auto& bitMask : main)
        {
            bitMask.flip();
        }

        head.flip(firstLevelIndex);
        main[firstLevelIndex].flip(secondLevelIndex);
        main[BitMaskSize * (1 + firstLevelIndex) + secondLevelIndex].flip(thirdLevelIndex);

        BitFieldTree tree{head, &(main[0]), 2U};

        CHECK(
            firstFreeBit(tree) == BitMaskSize * (BitMaskSize * firstLevelIndex + secondLevelIndex) + thirdLevelIndex);
    }
}
