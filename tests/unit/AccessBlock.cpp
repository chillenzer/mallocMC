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
using mallocMC::CreationPolicies::ScatterAlloc::DataPage;

constexpr size_t pageSize = 1024;
constexpr size_t numPages = 4;
// bitmask, chunksize, filling level
constexpr size_t pteSize = 4 + 4 + 4;
constexpr size_t blockSize = numPages * (pageSize + pteSize);

auto pageNumberOf(void* const pointer, DataPage<pageSize>* pages) -> size_t
{
    return std::distance(reinterpret_cast<char*>(pages), reinterpret_cast<char*>(pointer)) / pageSize;
}

TEST_CASE("AccessBlock")
{
    AccessBlock<blockSize, pageSize> accessBlock;

    SECTION("has pages.")
    {
        // This check is mainly leftovers from TDD. Keep them as long as `pages` is public interface.
        CHECK(accessBlock.pages); // NOLINT(*-array-*decay)
    }

    SECTION("has page table.")
    {
        // This check is mainly leftovers from TDD. Keep them as long as `pageTable` is public interface.
        CHECK(accessBlock.pageTable); // NOLINT(*-array-*decay)
    }

    SECTION("knows its data size.")
    {
        CHECK(accessBlock.dataSize() == numPages * pageSize);
    }

    SECTION("knows its metadata size.")
    {
        CHECK(accessBlock.metadataSize() == numPages * pteSize);
    }

    SECTION("stores page table after pages.")
    {
        CHECK(reinterpret_cast<void*>(accessBlock.pages) < reinterpret_cast<void*>(accessBlock.pageTable));
    }

    SECTION("uses an allowed amount of memory.")
    {
        CHECK(accessBlock.dataSize() + accessBlock.metadataSize() <= blockSize);
    }

    SECTION("knows its number of pages.")
    {
        CHECK(accessBlock.numPages() == numPages);
    }

    SECTION("correctly reports a different number of pages.")
    {
        constexpr size_t localNumPages = 5U;
        constexpr size_t localBlockSize = localNumPages * (pageSize + pteSize);
        AccessBlock<localBlockSize, pageSize> localAccessBlock;
        CHECK(localAccessBlock.numPages() == localNumPages);
    }
    SECTION("does not create nullptr.")
    {
        // This is not a particularly hard thing to do because any uninitialised pointer that could be returned is most
        // likely not exactly the nullptr. We just leave this in as it currently doesn't hurt anybody to keep it.
        CHECK(accessBlock.create(32U) != nullptr);
    }

    SECTION("creates memory that can be written to and read from.")
    {
        uint32_t const arbitraryValue = 42;
        auto* ptr = static_cast<uint32_t*>(accessBlock.create(4U));
        *ptr = arbitraryValue;
        CHECK(*ptr == arbitraryValue);
    }

    SECTION("creates second memory somewhere else.")
    {
        CHECK(accessBlock.create(32U) != accessBlock.create(32U));
    }
}

TEST_CASE("AccessBlock (failing)", "[!shouldfail]")
{
    AccessBlock<blockSize, pageSize> accessBlock;

    SECTION("creates memory of different chunk size in different pages.")
    {
        CHECK(
            pageNumberOf(accessBlock.create(32U), &accessBlock.pages[0])
            != pageNumberOf(accessBlock.create(512U), &accessBlock.pages[0]));
    }

    SECTION("can create memory larger than page size.")
    {
        CHECK(accessBlock.create(2U * pageSize));
    }
}
