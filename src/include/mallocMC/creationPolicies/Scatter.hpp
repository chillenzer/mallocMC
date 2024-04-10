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

    template<size_t T_pageSize>
    struct PageInterpretation
    {
        DataPage<T_pageSize>& data;
        uint32_t& chunkSize;
        BitMask& topLevelMask;

        [[nodiscard]] auto numChunks() const -> uint32_t
        {
            return T_pageSize / chunkSize;
        }

        [[nodiscard]] auto hasBitField() const -> bool
        {
            return numChunks() > maxChunksPerPage;
        }

        [[nodiscard]] auto operator[](size_t index) const -> void*
        {
            return reinterpret_cast<void*>(&data.data[index * chunkSize]);
        }

        [[nodiscard]] auto firstFreeChunk() const -> std::optional<Chunk>
        {
            auto const index = firstFreeBit(topLevelMask);
            if(index < BitMaskSize)
            {
                return std::optional<Chunk>({index, this->operator[](index)});
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
            const auto chunk = page.firstFreeChunk();
            if(chunk)
            {
                page.topLevelMask[chunk.value().index].flip();
                return chunk.value().pointer;
            }
            return nullptr;
        }

    private:
        auto choosePage(uint32_t numBytes) -> PageInterpretation<T_pageSize>
        {
            return {pages[0], numBytes, pageTable._bitMasks[0]};
        }
    };
} // namespace mallocMC::CreationPolicies::ScatterAlloc
