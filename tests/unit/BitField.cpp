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

#include "mallocMC/auxiliary.hpp"
#include "mocks.hpp"

#include <alpaka/acc/AccCpuSerial.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdint>
#include <mallocMC/creationPolicies/Scatter/BitField.hpp>

using mallocMC::CreationPolicies::ScatterAlloc::BitFieldFlat;
using mallocMC::CreationPolicies::ScatterAlloc::BitMask;
using mallocMC::CreationPolicies::ScatterAlloc::BitMaskSize;

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
            CHECK(mask(accSerial, i) == false);
        }
    }

    SECTION("allows to write individual bits.")
    {
        for(size_t i = 0; i < BitMaskSize; ++i)
        {
            mask.set(accSerial, i);
            CHECK(mask(accSerial, i));
        }
    }

    SECTION("allows to unset individual bits afterwards.")
    {
        for(size_t i = 0; i < BitMaskSize; ++i)
        {
            mask.set(accSerial, i);
            for(size_t j = 0; j < BitMaskSize; ++j)
            {
                CHECK(mask(accSerial, j) == (i == j));
            }
            mask.unset(accSerial, i);
        }
    }


    SECTION("knows the first free bit.")
    {
        mask.flip(accSerial);
        size_t const index = GENERATE(0, 3);
        mask.flip(accSerial, index);
        CHECK(mask.firstFreeBit(accSerial) == index);
    }

    SECTION("returns BitMaskSize as first free bit if there is none.")
    {
        mask.flip(accSerial);
        CHECK(mask.firstFreeBit(accSerial) == BitMaskSize);
    }

    SECTION("knows the first free bit with startIndex.")
    {
        // The search is supposed to wrap around. So, the "first free bit" is always the smaller one unless startIndex
        // lies in between the two indices.
        mask.set(accSerial);
        size_t index1 = GENERATE(0, 5);
        size_t index2 = GENERATE(0, 11);
        if(index1 > index2)
        {
            std::swap(index1, index2);
        }
        size_t const startIndex = GENERATE(0, 4, 5, 6);
        mask.unset(accSerial, index1);
        mask.unset(accSerial, index2);
        CHECK(
            mask.firstFreeBit(accSerial, startIndex)
            == ((startIndex > index1 and startIndex <= index2) ? index2 : index1));
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
            mask.set(accSerial);
        }
        data[index / BitMaskSize].unset(accSerial, index % BitMaskSize);

        BitFieldFlat field{data};

        CHECK(field.firstFreeBit(accSerial) == index);
    }

    SECTION("knows a free bit if later ones are free, too.")
    {
        uint32_t const index = GENERATE(0, 1, numChunks / 2, numChunks - 1);
        for(auto& mask : std::span{static_cast<BitMask*>(data), index / BitMaskSize})
        {
            mask.set(accSerial);
        }
        for(uint32_t i = 0; i < index % BitMaskSize; ++i)
        {
            data[index / BitMaskSize].set(accSerial, i);
        }

        BitFieldFlat field{data};

        CHECK(field.firstFreeBit(accSerial) >= index);
    }

    SECTION("knows its first free bit for different numChunks.")
    {
        auto localNumChunks = numChunks / GENERATE(1, 2, 3);
        std::span localData{static_cast<BitMask*>(data), mallocMC::ceilingDivision(localNumChunks, BitMaskSize)};
        uint32_t const index = GENERATE(0, 1, 10, 12);
        for(auto& mask : localData)
        {
            mask.set(accSerial);
        }
        localData[index / BitMaskSize].unset(accSerial, index % BitMaskSize);

        BitFieldFlat field{localData};

        CHECK(field.firstFreeBit(accSerial) == index);
    }

    SECTION("sets a bit.")
    {
        BitFieldFlat field{data};
        uint32_t const index = GENERATE(0, 1, numChunks / 2, numChunks - 1);
        field.set(accSerial, index);
        for(uint32_t i = 0; i < numChunks; ++i)
        {
            CHECK(field.get(accSerial, i) == (i == index));
        }
    }

    SECTION("sets two bits.")
    {
        BitFieldFlat field{data};
        uint32_t const firstIndex = GENERATE(0, 1, numChunks / 2, numChunks - 1);
        uint32_t const secondIndex = GENERATE(2, numChunks / 3, numChunks / 2, numChunks - 1);
        field.set(accSerial, firstIndex);
        field.set(accSerial, secondIndex);
        for(uint32_t i = 0; i < numChunks; ++i)
        {
            CHECK(field.get(accSerial, i) == (i == firstIndex || i == secondIndex));
        }
    }

    SECTION("returns numChunks if no free bit is found.")
    {
        BitFieldFlat field{data};
        for(uint32_t i = 0; i < numChunks; ++i)
        {
            field.set(accSerial, i);
        }
        CHECK(field.firstFreeBit(accSerial) == numChunks);
    }

    SECTION("returns numChunks if free bit is not valid.")
    {
        BitFieldFlat field{data};
        uint32_t const numValidBits = GENERATE(1, numChunks / 2, numChunks - 1);
        for(uint32_t i = 0; i < numValidBits; ++i)
        {
            // We are filling up all valid bits.
            field.set(accSerial, i);
        }
        CHECK(field.firstFreeBit(accSerial, numValidBits) == numChunks);
    }

    SECTION("returns numChunks if free bit is not valid.")
    {
        BitFieldFlat field{data};
        uint32_t const numValidBits = GENERATE(1, numChunks / 2, numChunks - 1);
        for(uint32_t i = 0; i < numValidBits; ++i)
        {
            // We are filling up all valid bits.
            field.set(accSerial, i);
        }
        CHECK(field.firstFreeBit(accSerial, numValidBits) == numChunks);
    }
}
