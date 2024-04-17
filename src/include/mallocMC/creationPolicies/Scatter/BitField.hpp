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
        // This class is about to be removed. But until it is, we'll document that this weird structure of separate
        // head and levels is a relict of the original idea of storing the top-level bit mask in the AccessBlock and
        // the potential further bit masks in the page itself.
        BitMask& _head; // NOLINT(*ref*member*)
        BitMask* _levels{nullptr};
        // CAUTION: This might be slightly unintuitive but `depth` refers to the number of levels below `head` (or the
        // number of edges down to a leave). This is convenient due to our weird storage situation where the head is
        // situated somewhere else.
        uint32_t _depth{0U};

        BitFieldTree(BitMask& head, BitMask* levels, uint32_t depth) : _head(head), _levels(levels), _depth(depth)
        {
        }

        BitFieldTree(BitMask* data, uint32_t depth) : _head(data[0]), _levels(&data[1]), _depth(depth)
        {
        }

        [[nodiscard]] auto headNode() const -> BitMask&
        {
            return _head;
        }

        // Return a pointer to the level-th level in the tree.
        auto level(uint32_t index) const -> BitMask*
        {
            if(index == 0)
            {
                return &_head;
            }
            // We subtract one because the head node is stored separately.
            return &_levels[treeVolume<BitMaskSize>(index - 1) - 1];
        }

        auto operator[](uint32_t index) const -> BitMask*
        {
            return level(index);
        }

        [[nodiscard]] auto get(uint32_t index) const -> bool
        {
            return this->operator[](_depth)[index / BitMaskSize][index % BitMaskSize];
        }

        // Set the bit corresponding to chunk `index`. If that fills the corresponding bit mask, the function takes
        // care of propagating up the information.
        void set(uint32_t const index, bool value = true)
        {
            auto& mask = this->operator[](_depth)[index / BitMaskSize];
            mask.set(index % BitMaskSize, value);
            if(_depth > 0 && (value == mask.all()))
            {
                BitFieldTree{_head, _levels, _depth - 1U}.set(index / BitMaskSize, value);
            }
        }
    };

    constexpr inline auto noFreeBitFound(uint32_t const depth) -> uint32_t
    {
        return powInt(BitMaskSize, depth + 1);
    }

    [[nodiscard]] constexpr inline auto firstFreeBit(BitMask const mask, uint32_t const startIndex = 0) -> uint32_t
    {
        // TODO(lenz): we are not yet caring for performance here...
        for(size_t i = startIndex; i < BitMaskSize; ++i) // NOLINT(altera-unroll-loops)
        {
            if(not mask[i])
            {
                return i;
            }
        }
        return noFreeBitFound(0);
    }

    inline auto firstFreeBit(BitFieldTree tree) -> uint32_t
    {
        // TODO(lenz): Allow for arbitrary starting point for hashing.
        uint32_t indexOnLevel[tree._depth + 2];
        indexOnLevel[0] = 0U;
        uint32_t startIndex = 0U;

        for(uint32_t currentDepth = 0U; currentDepth <= tree._depth; currentDepth++)
        {
            const auto index = firstFreeBit(tree[currentDepth][indexOnLevel[currentDepth]], startIndex);

            if(index == noFreeBitFound(0))
            {
                if(currentDepth == 0)
                {
                    return noFreeBitFound(tree._depth);
                }
                startIndex = indexOnLevel[currentDepth] + 1;
                // move up twice because the next iteration step will execute currentDepth++, so we're effectively
                // moving up one level
                currentDepth -= 2;
            }
            else
            {
                startIndex = 0;
                indexOnLevel[currentDepth + 1] = (BitMaskSize * indexOnLevel[currentDepth]) + index;
            }
        }
        return indexOnLevel[tree._depth + 1];
    }
} // namespace mallocMC::CreationPolicies::ScatterAlloc
