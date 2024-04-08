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

using mallocMC::CreationPolicies::ScatterAlloc::DataPage;
using mallocMC::CreationPolicies::ScatterAlloc::NoBitField;
using mallocMC::CreationPolicies::ScatterAlloc::PageInterpretation;

constexpr size_t pageSize=4096u;
constexpr size_t chunkSize=32u;

TEST_CASE ("PageInterpretation") {
  DataPage<pageSize> data;
  PageInterpretation<pageSize, NoBitField> page{data, chunkSize};

  SECTION ("refers to the same data it was created with.") {
    CHECK(&data == &page.data);
  }

  SECTION ("returns start of data as first chunk.") {
    CHECK(page[0] == &data);
  }

  SECTION ("returns successively increasing pointers when indexing.") {
    for (auto i=0u; i<(pageSize/chunkSize) - 1; ++i) CHECK(page[i] < page[i+1]);
  }
}
