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

#pragma once

#include "mallocMC/auxiliary.hpp"

#include <bitset>
#include <cstdint>

namespace mallocMC::CreationPolicies::ScatterAlloc
{
    constexpr const uint32_t BitMaskSize = 32U;
    using BitMask = std::bitset<BitMaskSize>;

    // Compute the volume (number of nodes) of a complete N-ary tree
    template<uint32_t N>
    inline constexpr auto treeVolume(uint32_t const depth) -> uint32_t
    {
        // Analytical formula: Sum_n=0^depth N^n = (N^(depth+1) - 1) / (N - 1)
        return (powInt(N, depth + 1) - 1) / (N - 1);
    }

    struct BitFieldTree
    {
        BitMask& head; // NOLINT(*ref*member*)
        BitMask* levels{nullptr};
        // CAUTION: This might be slightly unintuitive but `depth` refers to the number of levels below `head` (or the
        // number of edges down to a leave). This is convenient due to our weird storage situation where the head is
        // situated somewhere else.
        uint32_t depth{0U};

        // Return a pointer to the level-th level in the tree.
        auto operator[](uint32_t level) -> BitMask*
        {
            if(level == 0)
            {
                return &head;
            }
            // We subtract one because the head node is stored separately.
            return &levels[treeVolume<BitMaskSize>(level - 1) - 1];
        }

        // Set the bit corresponding to chunk `index`. If that fills the corresponding bit mask, the function takes
        // care of propagating up the information.
        void set(uint32_t const index, bool value = true)
        {
            auto& mask = this->operator[](depth)[index / BitMaskSize];
            mask.set(index % BitMaskSize, value);
            if(depth > 0 && (value == mask.all()))
            {
                BitFieldTree{head, levels, depth - 1U}.set(index / BitMaskSize, value);
            }
        }
    };

    [[nodiscard]] constexpr inline auto firstFreeBit(BitMask const mask) -> uint32_t
    {
        // TODO(lenz): we are not yet caring for performance here...
        for(size_t i = 0; i < BitMaskSize; ++i) // NOLINT(altera-unroll-loops)
        {
            if(not mask[i])
            {
                return i;
            }
        }
        return BitMaskSize;
    }

    inline constexpr auto firstFreeBit(BitFieldTree tree) -> uint32_t
    {
        // this is one past the end of the chunks, so no valid index:
        auto noFreeBitFound = powInt(BitMaskSize, tree.depth + 1);

        auto result = firstFreeBit(tree.head);
        // This means that we didn't find any free bit:
        if(result == BitMaskSize)
        {
            return noFreeBitFound;
        }

        for(uint32_t currentDepth = 0U; currentDepth < tree.depth; currentDepth++)
        {
            const auto index = firstFreeBit(tree[currentDepth + 1][result]);

            // This means that we didn't find any free bit:
            if(index == BitMaskSize)
            {
                return noFreeBitFound;
            }

            result = (BitMaskSize * result) + index;
        }
        return result;
    }
} // namespace mallocMC::CreationPolicies::ScatterAlloc
