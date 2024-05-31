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

#include <alpaka/acc/AccCpuSerial.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdint>
#include <mallocMC/creationPolicies/Scatter/BitField.hpp>
#include <type_traits>

using mallocMC::CreationPolicies::ScatterAlloc::BitFieldFlat;
using mallocMC::CreationPolicies::ScatterAlloc::BitMask;
using mallocMC::CreationPolicies::ScatterAlloc::BitMaskSize;
using mallocMC::CreationPolicies::ScatterAlloc::firstFreeBit;

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;
using Acc = alpaka::AccCpuSerial<Dim, Idx>;

struct Executor
{
    Executor() : dev(alpaka::getDevByIdx(platform, 0)){};
    ~Executor() = default;
    Executor(const Executor&) = delete;
    Executor(Executor&&) = delete;
    auto operator=(const Executor&) -> Executor& = delete;
    auto operator=(Executor&&) -> Executor& = delete;

    alpaka::Platform<Acc> const platform = {};
    alpaka::Dev<alpaka::Platform<Acc>> const dev;
    alpaka::Queue<Acc, alpaka::Blocking> queue{dev};
    alpaka::WorkDivMembers<Dim, Idx> const workDiv{Idx{1}, Idx{1}, Idx{1}};

    template<typename T_Functor>
    auto operator()(T_Functor task) -> std::invoke_result_t<T_Functor, Acc>
    {
        using ResultType = std::invoke_result_t<T_Functor, Acc>;
        if constexpr(std::is_same_v<ResultType, void>)
        {
            alpaka::enqueue(queue, alpaka::createTaskKernel<Acc>(workDiv, task));
        }
        else
        {
            ResultType result;
            auto const kernel = [&](auto const& acc) -> void { result = task(acc); };
            auto tmp = alpaka::createTaskKernel<Acc>(workDiv, kernel);
            alpaka::enqueue(queue, tmp);
            return result;
        }
    }
};

Executor exec{};

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
            auto result = exec([&](Acc const& acc) { return mask(acc, i); });
            CHECK(result == false);
        }
    }

    SECTION("allows to write individual bits.")
    {
        for(size_t i = 0; i < BitMaskSize; ++i)
        {
            exec([&](Acc const& acc) { return mask.set(acc, i); });
            auto result = exec([&](Acc const& acc) { return mask(acc, i); });
            CHECK(result);
        }
    }

    SECTION("knows the first free bit.")
    {
        exec([&](Acc const& acc) { return mask.flip(acc); });
        size_t const index = GENERATE(0, 3);
        exec([&](Acc const& acc) { return mask.flip(acc, index); });
        auto result = exec([&](Acc const& acc) { return firstFreeBit(acc, mask); });
        CHECK(result == index);
    }

    SECTION("returns BitMaskSize as first free bit if there is none.")
    {
        exec([&](Acc const& acc) { return mask.flip(acc); });
        auto result = exec([&](Acc const& acc) { return firstFreeBit(acc, mask); });
        CHECK(result == BitMaskSize);
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
            exec([&](Acc const& acc) { return mask.set(acc); });
        }
        exec([&](Acc const& acc) { return data[index / BitMaskSize].unset(acc, index % BitMaskSize); });

        BitFieldFlat field{data};

        auto result = exec([&](Acc const& acc) { return firstFreeBit(acc, field); });
        CHECK(result == index);
    }

    SECTION("knows its first free bit if later ones are free, too.")
    {
        uint32_t const index = GENERATE(0, 1, numChunks / 2, numChunks - 1);
        for(auto& mask : std::span{static_cast<BitMask*>(data), index / BitMaskSize})
        {
            exec([&](Acc const& acc) { return mask.set(acc); });
        }
        for(uint32_t i = 0; i < index % BitMaskSize; ++i)
        {
            exec([&](Acc const& acc) { return data[index / BitMaskSize].set(acc, i); });
        }

        BitFieldFlat field{data};

        auto result = exec([&](Acc const& acc) { return firstFreeBit(acc, field); });
        CHECK(result == index);
    }

    SECTION("knows its first free bit for different numChunks.")
    {
        auto localNumChunks = numChunks / GENERATE(1, 2, 3);
        std::span localData{static_cast<BitMask*>(data), mallocMC::ceilingDivision(localNumChunks, BitMaskSize)};
        uint32_t const index = GENERATE(0, 1, 10, 12);
        for(auto& mask : localData)
        {
            exec([&](Acc const& acc) { return mask.set(acc); });
        }
        exec([&](Acc const& acc) { return localData[index / BitMaskSize].unset(acc, index % BitMaskSize); });

        BitFieldFlat field{localData};

        auto result = exec([&](Acc const& acc) { return firstFreeBit(acc, field); });
        CHECK(result == index);
    }

    SECTION("sets a bit.")
    {
        BitFieldFlat field{data};
        uint32_t const index = GENERATE(0, 1, numChunks / 2, numChunks - 1);
        exec([&](Acc const& acc) { return field.set(acc, index); });
        for(uint32_t i = 0; i < numChunks; ++i)
        {
            auto result = exec([&](Acc const& acc) { return field.get(acc, i); });
            CHECK(result == (i == index));
        }
    }

    SECTION("sets two bits.")
    {
        BitFieldFlat field{data};
        uint32_t const firstIndex = GENERATE(0, 1, numChunks / 2, numChunks - 1);
        uint32_t const secondIndex = GENERATE(2, numChunks / 3, numChunks / 2, numChunks - 1);
        exec([&](Acc const& acc) { return field.set(acc, firstIndex); });
        exec([&](Acc const& acc) { return field.set(acc, secondIndex); });
        for(uint32_t i = 0; i < numChunks; ++i)
        {
            auto result = exec([&](Acc const& acc) { return field.get(acc, i); });
            CHECK(result == (i == firstIndex || i == secondIndex));
        }
    }

    SECTION("returns numChunks if no free bit is found.")
    {
        BitFieldFlat field{data};
        for(uint32_t i = 0; i < numChunks; ++i)
        {
            exec([&](Acc const& acc) { return field.set(acc, i); });
        }
        auto result = exec([&](Acc const& acc) { return firstFreeBit(acc, field); });
        CHECK(result == numChunks);
    }
}
