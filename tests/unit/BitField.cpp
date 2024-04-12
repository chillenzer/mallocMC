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
#include <mallocMC/creationPolicies/Scatter/BitField.hpp>

using mallocMC::CreationPolicies::ScatterAlloc::BitFieldTree;
using mallocMC::CreationPolicies::ScatterAlloc::BitMask;
using mallocMC::CreationPolicies::ScatterAlloc::BitMaskSize;
using mallocMC::CreationPolicies::ScatterAlloc::firstFreeBit;
using mallocMC::CreationPolicies::ScatterAlloc::treeVolume;

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

TEST_CASE("treeVolume")
{
    constexpr uint32_t numChildren = 3;
    const uint32_t depth = GENERATE(0, 1, 2, 3, 4, 5, 6, 7);
    const uint32_t expected[] = {1, 4, 13, 40, 121, 364, 1093, 3280};
    CHECK(treeVolume<numChildren>(depth) == expected[depth]);
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

    SECTION("sets top-level mask bits for depth 0.")
    {
        BitMask head{};
        BitFieldTree tree{head, nullptr, 0U};
        uint32_t index = GENERATE(0, 1, BitMaskSize - 1);
        tree.set(index);
        for(uint32_t i = 0; i < BitMaskSize; ++i)
        {
            CHECK(tree.head[i] == (i == index));
        }
    }

    SECTION("sets lowest-level mask bits for depth not 0.")
    {
        BitMask head{};
        BitMask main[BitMaskSize * (1 + BitMaskSize)]{};
        BitFieldTree tree{head, &(main[0]), 2U};
        uint32_t index = GENERATE(0, 1, BitMaskSize - 1, BitMaskSize * BitMaskSize - 1);

        tree.set(index);

        for(uint32_t i = 0; i < BitMaskSize; ++i)
        {
            for(uint32_t j = 0; j < BitMaskSize; ++j)
            {
                CHECK(tree[tree.depth][i][j] == (i * BitMaskSize + j == index));
            }
        }
    }

    SECTION("sets bits in top-level mask as appropriate for depth 1.")
    {
        BitMask head{};
        BitMask main[BitMaskSize];
        BitFieldTree tree{head, &(main[0]), 1U};
        uint32_t index = GENERATE(0, 1, BitMaskSize - 1);

        for(uint32_t i = 0; i < BitMaskSize; ++i)
        {
            // fills up this complete bitmask such that the corresponding bit in head needs flipping
            tree.set(BitMaskSize * index + i);
        }

        for(uint32_t i = 0; i < BitMaskSize; ++i)
        {
            CHECK(tree.head[i] == (i == index));
        }
    }

    SECTION("sets bits in higher-level mask as appropriate for depth 2.")
    {
        BitMask head{};
        BitMask main[BitMaskSize * (1 + BitMaskSize)];
        BitFieldTree tree{head, &(main[0]), 2U};
        uint32_t index = GENERATE(0, 1, BitMaskSize - 1);

        for(uint32_t i = 0; i < BitMaskSize * BitMaskSize; ++i)
        {
            // fills up `BitMaskSize` complete bitmasks such that the corresponding bits in higher levels need flipping
            tree.set(BitMaskSize * BitMaskSize * index + i);
        }

        for(uint32_t i = 0; i < BitMaskSize; ++i)
        {
            CHECK(tree.head[i] == (i == index));
            if(i == index)
            {
                CHECK(tree[1U][i].all());
            }
            else
            {
                CHECK(tree[1U][i].none());
            }
        }
    }

    SECTION("unsets top-level mask bits for depth 0.")
    {
        BitMask head{};
        BitFieldTree tree{head, nullptr, 0U};
        uint32_t index = GENERATE(0, 1, BitMaskSize - 1);
        tree.set(index);
        for(uint32_t i = 0; i < BitMaskSize; ++i)
        {
            REQUIRE(tree.head[i] == (i == index));
        }
        tree.set(index, false);
        CHECK(tree.head.none());
    }

    SECTION("unsets lowest-level mask bits for depth not 0.")
    {
        BitMask head{};
        BitMask main[BitMaskSize * (1 + BitMaskSize)]{};
        BitFieldTree tree{head, &(main[0]), 2U};
        uint32_t index = GENERATE(0, 1, BitMaskSize - 1, BitMaskSize * BitMaskSize - 1);

        tree.set(index);

        for(uint32_t i = 0; i < BitMaskSize; ++i)
        {
            for(uint32_t j = 0; j < BitMaskSize; ++j)
            {
                REQUIRE(tree[tree.depth][i][j] == (i * BitMaskSize + j == index));
            }
        }

        tree.set(index, false);
        for(uint32_t i = 0; i < BitMaskSize; ++i)
        {
            CHECK(tree[tree.depth][i].none());
        }
    }


    SECTION("unsets bits in higher-level mask as appropriate for depth 2.")
    {
        BitMask head{};
        BitMask main[BitMaskSize * (1 + BitMaskSize)];
        BitFieldTree tree{head, &(main[0]), 2U};
        uint32_t index = 3U;
        uint32_t unsetIndex = 0;

        for(uint32_t i = 0; i < BitMaskSize * BitMaskSize; ++i)
        {
            // fills up `BitMaskSize` complete bitmasks such that the corresponding bits in higher levels need flipping
            tree.set(BitMaskSize * BitMaskSize * index + i);
        }

        for(uint32_t i = 0; i < BitMaskSize; ++i)
        {
            REQUIRE(tree.head[i] == (i == index));
            if(i == index)
            {
                REQUIRE(tree[1U][i].all());
            }
            else
            {
                REQUIRE(tree[1U][i].none());
            }
        }

        for(uint32_t i = 0; i < BitMaskSize; ++i)
        {
            // We unset one bit from each lowest-level set of children, so all level-1 nodes should report that each
            // child has a free spot. (Amd so does the top-level mask then, of course.)
            tree.set(index * BitMaskSize * BitMaskSize + i * BitMaskSize + unsetIndex, false);
        }
        for(uint32_t i = 0; i < BitMaskSize; ++i)
        {
            CHECK(tree[1U][i].none());
        }
        CHECK(tree.head.none());
    }
}
