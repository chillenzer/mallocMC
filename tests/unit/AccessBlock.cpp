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

#include <catch2/catch.hpp>
#include <mallocMC/creationPolicies/Scatter.hpp>

using mallocMC::CreationPolicies::ScatterAlloc::AccessBlock;

TEST_CASE ("AccessBlock") {
  constexpr size_t pageSize = 1024;
  constexpr size_t numPages = 4;
  // bitmask, chunksize, filling level
  constexpr size_t pteSize = 4 + 4 + 4;
  constexpr size_t blockSize = numPages * (pageSize + pteSize);

  AccessBlock<blockSize, pageSize> accessBlock;

  SECTION ("has pages.") {
    CHECK(accessBlock.pages != nullptr);
  }

  SECTION ("has page table.") {
    CHECK(accessBlock.pageTable != nullptr);
  }

  SECTION ("knows its data size.") {
    CHECK(accessBlock.dataSize() == numPages * pageSize);
  }

  SECTION ("knows its metadata size.") {
    CHECK(accessBlock.metadataSize() == numPages * pteSize);
  }

  SECTION ("stores page table after pages.") {
    CHECK(reinterpret_cast<void*>(accessBlock.pages) < reinterpret_cast<void*>(accessBlock.pageTable));
  }

  SECTION("uses an allowed amount of memory.") {
    CHECK(accessBlock.dataSize() + accessBlock.metadataSize() <= blockSize);
  }

  SECTION("knows its number of pages.") {
    CHECK(accessBlock.numPages() == numPages);
  }
} 
