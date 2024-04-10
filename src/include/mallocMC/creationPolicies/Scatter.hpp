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

    [[nodiscard]] inline auto firstFreeBit(BitMask const mask) -> uint32_t
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

    // Compute the volume (number of nodes) of a complete N-ary tree
    template<uint32_t N>
    inline auto treeVolume(uint32_t const depth) -> uint32_t
    {
        if(depth == 0)
        {
            return 1U;
        }
        return (N + 1) * treeVolume<BitMaskSize>(depth - 1);
    }

    struct BitFieldTree
    {
        BitMask& head;
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

    inline auto firstFreeBit(BitFieldTree tree) -> uint32_t
    {
        auto result = firstFreeBit(tree.head);
        for(uint32_t currentDepth = 0U; currentDepth < tree.depth; currentDepth++)
        {
            const auto index = firstFreeBit(tree[currentDepth + 1][result]);

            // This means that we didn't find any free bit:
            if(index == BitMaskSize)
            {
                return treeVolume<BitMaskSize>(tree.depth) - 1;
            }

            result = (BitMaskSize * result) + index;
        }
        return result;
    }

    struct BitFieldTreeDummy
    {
    };

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
            return T_pageSize / _chunkSize;
        }

        [[nodiscard]] auto hasBitField() const -> bool
        {
            return numChunks() > maxChunksPerPage;
        }

        [[nodiscard]] auto operator[](size_t index) const -> void*
        {
            return reinterpret_cast<void*>(&_data.data[index * _chunkSize]);
        }

        [[nodiscard]] auto firstFreeChunk() const -> std::optional<Chunk>
        {
            auto const index = firstFreeBit(_topLevelMask);
            if(index < BitMaskSize)
            {
                return std::optional<Chunk>({index, this->operator[](index)});
            }
            return std::nullopt;
        }

        [[nodiscard]] auto bitField() const -> std::optional<BitFieldTreeDummy>
        {
            if(numChunks() > BitMaskSize)
            {
                return BitFieldTreeDummy{};
            }
            return std::nullopt;
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
                if(pageTable._chunkSizes[i] == numBytes || pageTable._chunkSizes[i] == 0U)
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
    };
} // namespace mallocMC::CreationPolicies::ScatterAlloc
