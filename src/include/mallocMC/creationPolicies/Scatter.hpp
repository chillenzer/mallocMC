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

namespace mallocMC::CreationPolicies::ScatterAlloc {
  template<size_t T_pageSize>
  struct DataPage {
    char data[T_pageSize];
  };

  enum class HasBitField {Yes,No};

  template<size_t T_pageSize>
  struct PageInterpretation {
    DataPage<T_pageSize>& data;
    size_t chunkSize{1u};
    HasBitField hasBitField{HasBitField::No};


    void* operator[](size_t i) {
      return (void*)&data.data[i*chunkSize];
    }
  };

  using BitMask = uint32_t;

  struct PageTableEntry{
    BitMask _bitMask{};
    uint32_t _chunkSize{0u};
    uint32_t _fillingLevel{0u};

    void init(uint32_t chunkSize) {
      _chunkSize=chunkSize;
    }

    static size_t size() {
      return 12;
    }
  };

  template <size_t T_blockSize, size_t T_pageSize>
  struct AccessBlock {
    constexpr static size_t numPages() {
      return 4;
    }

    size_t dataSize() const {
      return numPages() * T_pageSize;
    }

    size_t metadataSize() const {
      return numPages() * PageTableEntry::size();
    }

    DataPage<T_pageSize> pages[numPages()];
    PageTableEntry pageTable[numPages()];

    void* create(uint32_t numBytes) {
      auto page = choosePage(numBytes);
      return findChunkIn(page);
    }

    private:
    PageInterpretation<T_pageSize> choosePage(uint32_t numBytes) {
      return {pages[0], numBytes, HasBitField::No};
    }

    void* findChunkIn(PageInterpretation<T_pageSize>& page) {
      return page[0];
    }
  };
}
