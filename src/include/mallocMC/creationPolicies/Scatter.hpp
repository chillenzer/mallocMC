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

#include <cstddef>
#include <cstdint>

namespace mallocMC::CreationPolicies::ScatterAlloc
{
    template<size_t T_pageSize>
    struct DataPage
    {
        char data[T_pageSize];
    };

    constexpr const size_t maxChunksPerPage = 32U;

    template<size_t T_pageSize>
    struct PageInterpretation
    {
        DataPage<T_pageSize>& data;
        size_t chunkSize{1U};

        auto numChunks() -> size_t
        {
            return T_pageSize / chunkSize;
        }
        auto hasBitField() -> bool
        {
            return numChunks() > maxChunksPerPage;
        }

        // these are supposed to be temporary objects, don't start messing around with them:
        PageInterpretation(PageInterpretation const&) = delete;
        PageInterpretation(PageInterpretation&&) = delete;
        auto operator=(PageInterpretation const&) -> PageInterpretation& = delete;
        auto operator=(PageInterpretation&&) -> PageInterpretation& = delete;
        ~PageInterpretation() = default;

        auto operator[](size_t index) -> void*
        {
            return reinterpret_cast<void*>(&data.data[index * chunkSize]);
        }
    };

    using BitMask = uint32_t;

    // TODO(lenz): Make this a struct of array (discussion with Rene, 2024-04-09)
    struct PageTableEntry // NOLINT(altera-struct-pack-align)
    {
        BitMask _bitMask{};
        uint32_t _chunkSize{0U};
        uint32_t _fillingLevel{0U};

        void init(uint32_t chunkSize)
        {
            _chunkSize = chunkSize;
        }

        static auto size() -> size_t
        {
            // contains 3x 32-bit values
            return 12; // NOLINT(*-magic-numbers)
        }
    };

    template<size_t T_blockSize, size_t T_pageSize>
    struct AccessBlock
    {
        [[nodiscard]] constexpr static auto numPages() -> size_t
        {
            return 4;
        }

        [[nodiscard]] auto dataSize() const -> size_t
        {
            return numPages() * T_pageSize;
        }

        [[nodiscard]] auto metadataSize() const -> size_t
        {
            return numPages() * PageTableEntry::size();
        }

        DataPage<T_pageSize> pages[numPages()];
        PageTableEntry pageTable[numPages()];

        auto create(uint32_t numBytes) -> void*
        {
            auto page = choosePage(numBytes);
            return findChunkIn(page);
        }

    private:
        auto choosePage(uint32_t numBytes) -> PageInterpretation<T_pageSize>
        {
            return {pages[0], numBytes};
        }

        auto findChunkIn(PageInterpretation<T_pageSize>& page) -> void*
        {
            return page[0];
        }
    };
} // namespace mallocMC::CreationPolicies::ScatterAlloc
