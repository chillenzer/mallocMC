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
