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

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace mallocMC::CreationPolicies::ScatterAlloc
{
    template<size_t T_pageSize>
    struct DataPage
    {
        char data[T_pageSize];
    };

    template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    [[nodiscard]] constexpr auto ceilingDivision(T const numerator, T const denominator) -> T
    {
        return (numerator + (denominator - 1)) / denominator;
    }


    constexpr const uint32_t BitMaskSize = 32U;
    constexpr const uint32_t maxChunksPerPage = BitMaskSize;
    constexpr const uint32_t pageTableEntrySize = 4U + 4U + ceilingDivision(BitMaskSize, 8U);

    using BitMask = std::bitset<BitMaskSize>;

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

    struct Chunk
    {
        uint32_t index;
        void* pointer;
    };
    template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    inline constexpr auto powInt(T const base, T const exp) -> T
    {
        auto result = 1U;
        for(auto i = 0U; i < exp; ++i)
        {
            result *= base;
        }
        return result;
    }


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
    };

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

    // Computing the number of chunks is not quite trivial: We have to take into account the space for the hierarchical
    // bit field at the end of the page, the size of which again depends on the number of chunks. So, we kind of solve
    // this self-consistently by making a naive estimate and checking if
    inline constexpr auto selfConsistentNumChunks(
        size_t const pageSize,
        uint32_t const chunkSize,
        uint32_t const depth = 0U) -> uint32_t
    {
        auto naiveEstimate = pageSize / chunkSize;
        auto bitsAvailable = powInt(BitMaskSize, depth + 1);
        if(naiveEstimate <= bitsAvailable)
        {
            return naiveEstimate;
        }

        // otherwise let's check how the situation looks with the next level of bit field hierarchy added:
        auto numBitsInNextLevel = bitsAvailable * BitMaskSize;
        // we count memory in bytes not bits:
        auto bitFieldSpaceRequirement = numBitsInNextLevel / 8U; // NOLINT(*magic*)
        auto remainingPageSize = pageSize - bitFieldSpaceRequirement;
        return selfConsistentNumChunks(remainingPageSize, chunkSize, depth + 1);
    }

    template<size_t T_pageSize>
    struct PageInterpretation
    {
        DataPage<T_pageSize>& _data;
        uint32_t& _chunkSize;
        BitMask& _topLevelMask;

        // this is needed to instantiate this in-place in an std::optional
        PageInterpretation<T_pageSize>(DataPage<T_pageSize>& data, uint32_t& chunkSize, BitMask& topLevelMask)
            : _data(data)
            , _chunkSize(chunkSize)
            , _topLevelMask(topLevelMask)
        {
        }

        [[nodiscard]] auto numChunks() const -> uint32_t
        {
            return selfConsistentNumChunks(T_pageSize, _chunkSize);
        }

        [[nodiscard]] auto hasBitField() const -> bool
        {
            return bitFieldDepth() > 0;
        }

        [[nodiscard]] auto operator[](size_t index) const -> void*
        {
            return reinterpret_cast<void*>(&_data.data[index * _chunkSize]);
        }

        [[nodiscard]] auto firstFreeChunk() const -> std::optional<Chunk>
        {
            auto const index = firstFreeBit(bitField());
            auto noFreeBitFound = powInt(BitMaskSize, bitField().depth + 1);
            if(index < noFreeBitFound)
            {
                return std::optional<Chunk>({index, this->operator[](index)});
            }
            return std::nullopt;
        }

        [[nodiscard]] auto bitField() const -> BitFieldTree
        {
            return BitFieldTree{_topLevelMask, bitFieldStart(), bitFieldDepth()};
        }

        [[nodiscard]] auto bitFieldStart() const -> BitMask*
        {
            if(!hasBitField())
            {
                return nullptr;
            }
            auto BitMaskBytes = BitMaskSize / 8U; // NOLINT(*magic*)
            return reinterpret_cast<BitMask*>(
                &_data.data[T_pageSize - (treeVolume<BitMaskSize>(bitFieldDepth()) - 1) * BitMaskBytes]);
        }

        [[nodiscard]] auto bitFieldDepth() const -> uint32_t
        {
            // We subtract one such that numChunks() == BitMaskSize yields 0.
            auto tmp = (numChunks() - 1) / BitMaskSize;
            uint32_t counter = 0U;
            while(tmp > 0)
            {
                counter++;
                tmp /= BitMaskSize;
            }
            return counter;
        }

        // these are supposed to be temporary objects, don't start messing around with them:
        PageInterpretation(PageInterpretation const&) = delete;
        PageInterpretation(PageInterpretation&&) = delete;
        auto operator=(PageInterpretation const&) -> PageInterpretation& = delete;
        auto operator=(PageInterpretation&&) -> PageInterpretation& = delete;
        ~PageInterpretation() = default;
    };

    template<size_t T_numPages>
    struct PageTable
    {
        BitMask _bitMasks[T_numPages]{};
        uint32_t _chunkSizes[T_numPages]{};
        uint32_t _fillingLevels[T_numPages]{};
    };

    template<size_t T_blockSize, size_t T_pageSize>
    struct AccessBlock
    {
        [[nodiscard]] constexpr static auto numPages() -> size_t
        {
            return T_blockSize / (T_pageSize + pageTableEntrySize);
        }

        [[nodiscard]] constexpr static auto dataSize() -> size_t
        {
            return numPages() * T_pageSize;
        }

        [[nodiscard]] constexpr static auto metadataSize() -> size_t
        {
            return numPages() * pageTableEntrySize;
        }

        DataPage<T_pageSize> pages[numPages()];
        PageTable<numPages()> pageTable;

        auto create(uint32_t numBytes) -> void*
        {
            if(numBytes > T_pageSize)
            {
                // Not yet implemented.
                return nullptr;
            }
            const auto page = choosePage(numBytes);
            if(page)
            {
                const auto chunk = page.value().firstFreeChunk();
                if(chunk)
                {
                    page.value()._topLevelMask[chunk.value().index].flip();
                    return chunk.value().pointer;
                }
            }
            return nullptr;
        }

    private:
        auto choosePage(uint32_t numBytes) -> std::optional<PageInterpretation<T_pageSize>>
        {
            for(size_t i = 0; i < numPages(); ++i)
            {
                if(thisPageIsAppropriate(i, numBytes))
                {
                    pageTable._chunkSizes[i] = numBytes;
                    return std::optional<PageInterpretation<T_pageSize>>{
                        std::in_place_t{},
                        pages[i],
                        pageTable._chunkSizes[i],
                        pageTable._bitMasks[i]};
                }
            }
            return std::nullopt;
        }

        auto thisPageIsAppropriate(size_t const index, uint32_t const numBytes) -> bool
        {
            return (pageTable._chunkSizes[index] == numBytes
                    && pageTable._fillingLevels[index]
                        < PageInterpretation<
                              T_pageSize>{pages[index], pageTable._chunkSizes[index], pageTable._bitMasks[index]}
                              .numChunks())
                || pageTable._chunkSizes[index] == 0U;
        }
    };
} // namespace mallocMC::CreationPolicies::ScatterAlloc
