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

#include "mallocMC/auxiliary.hpp"

#include <catch2/catch.hpp>
#include <cstdint>
#include <iterator>
#include <mallocMC/creationPolicies/Scatter/BitField.hpp>

using mallocMC::CreationPolicies::ScatterAlloc::BitFieldFlat;
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
    // This is potentially larger than we actually need but that's okay:
    BitMask data[1 + BitMaskSize * (1 + BitMaskSize)]{};
    BitMask& head{data[0]};
    std::span<BitMask, BitMaskSize * (1 + BitMaskSize)> main{&data[1], BitMaskSize * (1 + BitMaskSize)};

    SECTION("knows its first free bit with depth 0.")
    {
        head.flip();
        uint32_t const index = GENERATE(0, 3);
        head.flip(index);

        BitFieldTree tree{std::begin(data), 0U};

        CHECK(firstFreeBit(tree) == index);
    }

    SECTION("knows its first free bit with depth 1.")
    {
        uint32_t const firstLevelIndex = GENERATE(0, 3);
        uint32_t const secondLevelIndex = GENERATE(2, 5);

        head.flip();
        head.flip(firstLevelIndex);
        for(auto& bitMask : main)
        {
            bitMask.flip();
        }
        main[firstLevelIndex].flip(secondLevelIndex);

        BitFieldTree tree{std::begin(data), 1U};

        CHECK(firstFreeBit(tree) == firstLevelIndex * BitMaskSize + secondLevelIndex);
    }

    SECTION("knows its first free bit with depth 2.")
    {
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

        BitFieldTree tree{std::begin(data), 2U};

        CHECK(
            firstFreeBit(tree) == BitMaskSize * (BitMaskSize * firstLevelIndex + secondLevelIndex) + thirdLevelIndex);
    }

    SECTION("sets top-level mask bits for depth 0.")
    {
        BitFieldTree tree{std::begin(data), 0U};
        uint32_t index = GENERATE(0, 1, BitMaskSize - 1);
        tree.set(index);
        for(uint32_t i = 0; i < BitMaskSize; ++i)
        {
            CHECK(tree.headNode()[i] == (i == index));
        }
    }

    SECTION("sets lowest-level mask bits for depth not 0.")
    {
        BitFieldTree tree{std::begin(data), 2U};
        uint32_t index = GENERATE(0, 1, BitMaskSize - 1, BitMaskSize * BitMaskSize - 1);

        tree.set(index);

        for(uint32_t i = 0; i < BitMaskSize; ++i)
        {
            for(uint32_t j = 0; j < BitMaskSize; ++j)
            {
                CHECK(tree.level(tree._depth)[i][j] == (i * BitMaskSize + j == index));
            }
        }
    }

    SECTION("sets bits in top-level mask as appropriate for depth 1.")
    {
        BitFieldTree tree{std::begin(data), 1U};
        uint32_t index = GENERATE(0, 1, BitMaskSize - 1);

        for(uint32_t i = 0; i < BitMaskSize; ++i)
        {
            // fills up this complete bitmask such that the corresponding bit in head needs flipping
            tree.set(BitMaskSize * index + i);
        }

        for(uint32_t i = 0; i < BitMaskSize; ++i)
        {
            CHECK(tree.headNode()[i] == (i == index));
        }
    }

    SECTION("sets bits in higher-level mask as appropriate for depth 2.")
    {
        BitFieldTree tree{std::begin(data), 2U};
        uint32_t index = GENERATE(0, 1, BitMaskSize - 1);

        for(uint32_t i = 0; i < BitMaskSize * BitMaskSize; ++i)
        {
            // fills up `BitMaskSize` complete bitmasks such that the corresponding bits in higher levels need flipping
            tree.set(BitMaskSize * BitMaskSize * index + i);
        }

        for(uint32_t i = 0; i < BitMaskSize; ++i)
        {
            CHECK(tree.headNode()[i] == (i == index));
            if(i == index)
            {
                CHECK(tree.level(1U)[i].all());
            }
            else
            {
                CHECK(tree.level(1U)[i].none());
            }
        }
    }

    SECTION("unsets top-level mask bits for depth 0.")
    {
        BitFieldTree tree{std::begin(data), 0U};
        uint32_t index = GENERATE(0, 1, BitMaskSize - 1);
        tree.set(index);
        for(uint32_t i = 0; i < BitMaskSize; ++i)
        {
            REQUIRE(tree.headNode()[i] == (i == index));
        }
        tree.set(index, false);
        CHECK(tree.headNode().none());
    }

    SECTION("unsets lowest-level mask bits for depth not 0.")
    {
        BitFieldTree tree{std::begin(data), 2U};
        uint32_t index = GENERATE(0, 1, BitMaskSize - 1, BitMaskSize * BitMaskSize - 1);

        tree.set(index);

        for(uint32_t i = 0; i < BitMaskSize; ++i)
        {
            for(uint32_t j = 0; j < BitMaskSize; ++j)
            {
                REQUIRE(tree.level(tree._depth)[i][j] == (i * BitMaskSize + j == index));
            }
        }

        tree.set(index, false);
        for(uint32_t i = 0; i < BitMaskSize; ++i)
        {
            CHECK(tree.level(tree._depth)[i].none());
        }
    }


    SECTION("unsets bits in higher-level mask as appropriate for depth 2.")
    {
        BitFieldTree tree{std::begin(data), 2U};
        uint32_t index = 3U;
        uint32_t unsetIndex = 0;

        for(uint32_t i = 0; i < BitMaskSize * BitMaskSize; ++i)
        {
            // fills up `BitMaskSize` complete bitmasks such that the corresponding bits in higher levels need flipping
            tree.set(BitMaskSize * BitMaskSize * index + i);
        }

        for(uint32_t i = 0; i < BitMaskSize; ++i)
        {
            REQUIRE(tree.headNode()[i] == (i == index));
            if(i == index)
            {
                REQUIRE(tree.level(1U)[i].all());
            }
            else
            {
                REQUIRE(tree.level(1U)[i].none());
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
            CHECK(tree.level(1U)[i].none());
        }
        CHECK(tree.headNode().none());
    }

    SECTION("recovers from incorrect higher-level bits when finding a free bit.")
    {
        BitFieldTree tree{std::begin(data), 2U};
        uint32_t const index = 7 * BitMaskSize * BitMaskSize + 5;

        for(uint32_t i = 0; i < BitMaskSize * BitMaskSize; ++i)
        {
            // fill up lowest level but don't tell the higher levels, so that we get into a neatly inconsistent state
            tree.level(tree._depth)[i].set();
        }
        tree.set(index, false);
        CHECK(firstFreeBit(tree) == index);
    }

    SECTION("provides a get function to access chunk indices directly.")
    {
        BitFieldTree tree{std::begin(data), 2U};
        uint32_t index = GENERATE(1U, 42U);
        CHECK(tree.get(index) == false);
        tree.set(index);
        CHECK(tree.get(index) == true);
    }

    SECTION("provides a levels function to access individual levels directly.")
    {
        BitFieldTree tree{std::begin(data), 2U};
        CHECK(tree.level(0) == &tree.headNode());
        CHECK(tree.level(1) == &data[1]);
        CHECK(tree.level(2) == &data[1 + BitMaskSize]);
    }
}

TEST_CASE("BitFieldFlat")
{
    // This is potentially larger than we actually need but that's okay:
    constexpr uint32_t const numChunks = 128U;
    constexpr uint32_t const numMasks = mallocMC::ceilingDivision(numChunks, BitMaskSize);
    BitMask data[numMasks];

    SECTION("knows its only free bit.")
    {
        uint32_t const index = GENERATE(0, 1, numChunks / 2, numChunks - 1);
        for(auto& mask : data)
        {
            mask.set();
        }
        data[index / BitMaskSize].set(index % BitMaskSize, false);

        BitFieldFlat field{data};

        CHECK(firstFreeBit(field) == index);
    }

    SECTION("knows its first free bit if later ones are free, too.")
    {
        uint32_t const index = GENERATE(0, 1, numChunks / 2, numChunks - 1);
        for(auto& mask : std::span{data, index / BitMaskSize})
        {
            mask.set();
        }
        for(uint32_t i = 0; i < index % BitMaskSize; ++i)
        {
            data[index / BitMaskSize].set(i);
        }

        BitFieldFlat field{data};

        CHECK(firstFreeBit(field) == index);
    }

    SECTION("knows its first free bit for different numChunks.")
    {
        auto localNumChunks = numChunks / GENERATE(1, 2, 3);
        std::span localData{data, localNumChunks};
        uint32_t const index = GENERATE(0, 1, 10, 12);
        for(auto& mask : localData)
        {
            mask.set();
        }
        localData[index / BitMaskSize].set(index % BitMaskSize, false);

        BitFieldFlat field{localData};

        CHECK(firstFreeBit(field) == index);
    }

    SECTION("sets a bit.")
    {
        BitFieldFlat field{data};
        uint32_t const index = GENERATE(0, 1, numChunks / 2, numChunks - 1);
        field.set(index);
        for(uint32_t i = 0; i < numChunks; ++i)
        {
            CHECK(field.get(i) == (i == index));
        }
    }

    SECTION("sets two bits.")
    {
        BitFieldFlat field{data};
        uint32_t const firstIndex = GENERATE(0, 1, numChunks / 2, numChunks - 1);
        uint32_t const secondIndex = GENERATE(2, numChunks / 3, numChunks / 2, numChunks - 1);
        field.set(firstIndex);
        field.set(secondIndex);
        for(uint32_t i = 0; i < numChunks; ++i)
        {
            CHECK(field.get(i) == (i == firstIndex || i == secondIndex));
        }
    }

    SECTION("returns numChunks if no free bit is found.")
    {
        BitFieldFlat field{data};
        for(uint32_t i = 0; i < numChunks; ++i)
        {
            field.set(i);
        }
        CHECK(firstFreeBit(field) == numChunks);
    }
}
