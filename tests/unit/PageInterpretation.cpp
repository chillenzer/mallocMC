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

// This is fine. We're mixing uint32_t and size_t from time to time to do manual index calculations. That will not
// happen in production code.
// NOLINTBEGIN(*widening*)
#include "mallocMC/creationPolicies/Scatter/PageInterpretation.hpp"

#include "mallocMC/auxiliary.hpp"
#include "mallocMC/creationPolicies/Scatter/BitField.hpp"
#include "mallocMC/creationPolicies/Scatter/DataPage.hpp"

#include <catch2/catch.hpp>
#include <cstdint>
#include <iterator>
#include <optional>

using mallocMC::indexOf;
using mallocMC::CreationPolicies::ScatterAlloc::BitMask;
using mallocMC::CreationPolicies::ScatterAlloc::BitMaskSize;
using mallocMC::CreationPolicies::ScatterAlloc::DataPage;
using mallocMC::CreationPolicies::ScatterAlloc::PageInterpretation;
using mallocMC::CreationPolicies::ScatterAlloc::treeVolume;
using std::distance;


TEST_CASE("PageInterpretation")
{
    constexpr size_t pageSize = 1024U;
    uint32_t chunkSize = 32U; // NOLINT(*magic-number*)
    DataPage<pageSize> data{};
    BitMask mask{};
    uint32_t fillingLevel{};
    PageInterpretation<pageSize> page{data, chunkSize, mask, fillingLevel};

    SECTION("refers to the same data it was created with.")
    {
        CHECK(&data == &page._data);
    }

    SECTION("returns start of data as first chunk.")
    {
        CHECK(page[0] == &data);
    }

    SECTION("computes correct number of pages.")
    {
        CHECK(page.numChunks() == pageSize / chunkSize);
    }

    SECTION("detects correctly if page should contain bitfield.")
    {
        uint32_t localChunkSize = GENERATE(8U, 128U);
        PageInterpretation<pageSize> localPage{data, localChunkSize, mask, fillingLevel};
        CHECK(localPage.hasBitField() == (pageSize / localChunkSize) > 32U);
    }

    SECTION("jumps by chunkSize between indices.")
    {
        for(auto i = 0U; i < (pageSize / chunkSize) - 1; ++i)
        {
            CHECK(distance(reinterpret_cast<char*>(page[i]), reinterpret_cast<char*>(page[i + 1])) == chunkSize);
        }
    }

    SECTION("finds first free chunk.")
    {
        mask.flip();
        size_t const index = GENERATE(0, 2);
        mask.flip(index);
        auto const chunk = page.firstFreeChunk();

        if(chunk)
        {
            CHECK(chunk.value().pointer == page[index]);
        }
        else
        {
            FAIL("Expected to get a valid chunk but didn't.");
        }
    }

    SECTION("returns nullopt if all chunks are full.")
    {
        mask.flip();
        CHECK(page.firstFreeChunk() == std::nullopt);
    }

    SECTION("recognises if there is no bit field at the end.")
    {
        // this is no longer true:
        // CHECK(page.bitField().levels == nullptr);
        CHECK(page.bitField().depth == 0U);
    }

    SECTION("recognises if there is a bit field at the end.")
    {
        uint32_t localChunkSize = 1U;
        PageInterpretation<pageSize> localPage{data, localChunkSize, mask, fillingLevel};
        CHECK(localPage.bitField().levels != nullptr);
        CHECK(localPage.bitField().depth == 1U);
    }

    SECTION("knows the maximal bit field size.")
    {
        // pageSize = 1024 with chunks of size one allows for more than 32 but less than 32^2 chunks, so maximal bit
        // field size should be
        CHECK(page.maxBitFieldSize() == 32U * sizeof(BitMask));
    }
}

TEST_CASE("PageInterpretation.bitFieldDepth")
{
    uint32_t fillingLevel{};
    // Such that we can fit up to four levels of hierarchy in there:
    constexpr size_t const pageSize
        = BitMaskSize * BitMaskSize * BitMaskSize * BitMaskSize + treeVolume<BitMaskSize>(4) * sizeof(BitMask);
    // This is more than 8MB which is a typical stack's size. Let's save us some trouble and create it on the heap.
    std::unique_ptr<DataPage<pageSize>> actualData{new DataPage<pageSize>};
    DataPage<pageSize>& data{*actualData};
    BitMask mask{};

    SECTION("knows correct bit field depths for depth 0.")
    {
        uint32_t const numChunks = BitMaskSize;
        uint32_t chunkSize = pageSize / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, mask, fillingLevel};

        CHECK(page.bitFieldDepth() == 0U);
    }

    SECTION("knows correct bit field depths for depth 0 with less chunks.")
    {
        uint32_t const numChunks = BitMaskSize - 1;
        uint32_t chunkSize = pageSize / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, mask, fillingLevel};

        CHECK(page.bitFieldDepth() == 0U);
    }

    SECTION("knows correct bit field depths for depth 1.")
    {
        uint32_t const numChunks = BitMaskSize * BitMaskSize;
        // choose chunk size such that bit field fits behind it:
        uint32_t chunkSize = (pageSize - BitMaskSize * sizeof(BitMask)) / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, mask, fillingLevel};

        CHECK(page.bitFieldDepth() == 1U);
    }

    SECTION("knows correct bit field depths for depth 1 with less chunks.")
    {
        uint32_t const numChunks = BitMaskSize * BitMaskSize - 1;
        // choose chunk size such that bit field fits behind it:
        uint32_t chunkSize = (pageSize - BitMaskSize * sizeof(BitMask)) / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, mask, fillingLevel};

        CHECK(page.bitFieldDepth() == 1U);
    }

    SECTION(
        "knows correct bit field depths for depth 1 with slightly too big chunks such that we get less than expected.")
    {
        uint32_t const numChunks = BitMaskSize * BitMaskSize - 1;
        // choose chunk size such that bit field fits behind it:
        uint32_t chunkSize = pageSize / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, mask, fillingLevel};

        CHECK(page.bitFieldDepth() == 1U);
    }

    SECTION("knows correct bit field depths for depth 2.")
    {
        uint32_t const numChunks = BitMaskSize * BitMaskSize * BitMaskSize;
        // choose chunk size such that bit field fits behind it:
        uint32_t chunkSize
            = (pageSize - BitMaskSize * sizeof(BitMask) - BitMaskSize * BitMaskSize * sizeof(BitMask)) / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, mask, fillingLevel};

        CHECK(page.bitFieldDepth() == 2U);
    }

    SECTION("knows correct bit field depths for depth 3.")
    {
        uint32_t const numChunks = BitMaskSize * BitMaskSize * BitMaskSize * BitMaskSize;
        // choose chunk size such that bit field fits behind it:
        uint32_t chunkSize = (pageSize - BitMaskSize * sizeof(BitMask) - BitMaskSize * BitMaskSize * sizeof(BitMask)
                              - BitMaskSize * BitMaskSize * BitMaskSize * sizeof(BitMask))
            / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, mask, fillingLevel};

        CHECK(page.bitFieldDepth() == 3U);
    }
}

TEST_CASE("PageInterpretation.create")
{
    uint32_t fillingLevel{};
    // Such that we can fit up to four levels of hierarchy in there:
    constexpr size_t const pageSize
        = BitMaskSize * BitMaskSize * BitMaskSize * BitMaskSize + treeVolume<BitMaskSize>(4) * sizeof(BitMask);
    // This is more than 8MB which is a typical stack's size. Let's save us some trouble and create it on the heap.
    std::unique_ptr<DataPage<pageSize>> actualData{new DataPage<pageSize>};
    DataPage<pageSize>& data{*actualData};
    BitMask mask{};

    SECTION("regardless of hierarchy")
    {
        uint32_t numChunks = GENERATE(BitMaskSize * BitMaskSize, BitMaskSize);
        uint32_t chunkSize = pageSize / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, mask, fillingLevel};

        SECTION("returns a pointer to within the data.")
        {
            auto* pointer = page.create();
            CHECK(
                std::distance(&page._data.data[0], reinterpret_cast<char*>(pointer))
                < static_cast<long>(page.dataSize()));
        }

        SECTION("returns a pointer to the start of a chunk.")
        {
            auto* pointer = page.create();
            CHECK(std::distance(&page._data.data[0], reinterpret_cast<char*>(pointer)) % chunkSize == 0U);
        }

        SECTION("returns nullptr if everything is full.")
        {
            mask.set();
            auto* pointer = page.create();
            CHECK(pointer == nullptr);
        }

        SECTION("updates filling level.")
        {
            page.create();
            CHECK(fillingLevel == 1U);
        }

        SECTION("can provide numChunks pieces of memory and returns nullptr afterwards.")
        {
            for(uint32_t i = 0; i < page.numChunks(); ++i)
            {
                auto* pointer = page.create();
                CHECK(pointer != nullptr);
            }
            auto* pointer = page.create();
            CHECK(pointer == nullptr);
        }
    }

    SECTION("without hierarchy")
    {
        uint32_t const numChunks = BitMaskSize;
        uint32_t chunkSize = pageSize / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, mask, fillingLevel};

        SECTION("updates top-level bit field.")
        {
            REQUIRE(mask.none());
            auto* pointer = page.create();
            auto const index = page.chunkNumberOf(pointer);
            CHECK(mask[index]);
        }
    }

    SECTION("with hierarchy")
    {
        uint32_t const numChunks = BitMaskSize * BitMaskSize;
        uint32_t chunkSize = pageSize / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, mask, fillingLevel};

        SECTION("recovers from not finding a free chunk in a lower level.")
        {
            auto tree = page.bitField();
            uint32_t const index = 2 * BitMaskSize + 1;
            for(uint32_t i = 0; i < numChunks / BitMaskSize; ++i)
            {
                // We fill only the lowest level leaving the upper ones empty, so the PageInterpretation will fail to
                // find a free bit on the first attempt and has to recover from it.
                tree[tree.depth][i].set();
            }
            tree.set(index, false);
            REQUIRE(mask.none());
            auto* pointer = page.create();
            CHECK(indexOf(pointer, std::begin(page._data.data), chunkSize) == index);
        }
    }
}

TEST_CASE("PageInterpretation.destroy")
{
    uint32_t fillingLevel{};
    // Such that we can fit up to four levels of hierarchy in there:
    constexpr size_t const pageSize
        = BitMaskSize * BitMaskSize * BitMaskSize * BitMaskSize + treeVolume<BitMaskSize>(4) * sizeof(BitMask);
    // This is more than 8MB which is a typical stack's size. Let's save us some trouble and create it on the heap.
    std::unique_ptr<DataPage<pageSize>> actualData{new DataPage<pageSize>};
    DataPage<pageSize>& data{*actualData};
    BitMask mask{};

    SECTION("regardless of hierarchy")
    {
        uint32_t numChunks = GENERATE(BitMaskSize * BitMaskSize, BitMaskSize);
        uint32_t chunkSize = pageSize / numChunks;
        PageInterpretation<pageSize> page{data, chunkSize, mask, fillingLevel};
        auto* pointer = page.create();

        SECTION("throws if given an invalid pointer.")
        {
            pointer = nullptr;
            CHECK_THROWS_WITH(
                page.destroy(pointer),
                Catch::Contains("Attempted to destroy out-of-bounds pointer. Chunk index out of range!"));
        }

        SECTION("allows pointers to anywhere in the chunk.")
        {
            // This test documents the state as is. We haven't defined this outcome as a requirement but if we change
            // it, we might still want to be aware of this because users might want to be informed.
            pointer = reinterpret_cast<void*>(reinterpret_cast<char*>(pointer) + chunkSize / 2);
            CHECK_NOTHROW(page.destroy(pointer));
        }

        SECTION("updates filling level.")
        {
            REQUIRE(page._fillingLevel == 1U);
            page.destroy(pointer);
            CHECK(fillingLevel == 0U);
        }

        SECTION("only ever unsets bits in top-level bit mask.")
        {
            auto count = page._topLevelMask.count();
            page.destroy(pointer);
            CHECK(page._topLevelMask.count() <= count);
        }

        SECTION("unsets correct bit in lowest-level bit mask.")
        {
            auto const tree = page.bitField();
            auto const index = mallocMC::indexOf(pointer, &page._data, chunkSize);
            for(uint32_t i = 0; i < numChunks; ++i)
            {
                REQUIRE(tree[tree.depth][i / BitMaskSize][i % BitMaskSize] == (i == index));
            }

            page.destroy(pointer);

            for(uint32_t i = 0; i < numChunks / BitMaskSize; ++i)
            {
                CHECK(tree[tree.depth][i].none());
            }
        }

        SECTION("throws if pointer does not point to allocated memory.")
        {
            // create another pointer, so the page doesn't get freed after invalidating one pointer
            page.create();
            // destroying this invalidates the pointer and deallocates the memory
            page.destroy(pointer);
            // attempting to do so again is an invalid operation
            CHECK_THROWS_WITH(page.destroy(pointer), Catch::Contains("Attempted to destroy un-allocated memory"));
        }

        SECTION("throws if chunk size is 0.")
        {
            // The following might be the case if the page was recently free'd. For testing purposes, we set it
            // directly.
            chunkSize = 0U;
            CHECK_THROWS_WITH(
                page.destroy(pointer),
                Catch::Contains("Attempted to destroy a pointer with chunkSize==0. Likely this page was recently (and "
                                "potentially pre-maturely) freed."));
        }

        SECTION("resets chunk size when page is abandoned.")
        {
            // this is the only allocation on this page, so page is abandoned afterwards
            page.destroy(pointer);
            CHECK(page._chunkSize == 0U);
        }
    }
}

TEST_CASE("PageInterpretation.destroy (failing)", "[!shouldfail]")
{
    uint32_t fillingLevel{};
    // Such that we can fit up to four levels of hierarchy in there:
    constexpr size_t const pageSize
        = BitMaskSize * BitMaskSize * BitMaskSize * BitMaskSize + treeVolume<BitMaskSize>(4) * sizeof(BitMask);
    // This is more than 8MB which is a typical stack's size. Let's save us some trouble and create it on the heap.
    std::unique_ptr<DataPage<pageSize>> actualData{new DataPage<pageSize>};
    DataPage<pageSize>& data{*actualData};
    BitMask mask{};

    uint32_t numChunks = GENERATE(BitMaskSize * BitMaskSize, BitMaskSize);
    uint32_t chunkSize = pageSize / numChunks;
    PageInterpretation<pageSize> page{data, chunkSize, mask, fillingLevel};
    auto* pointer = page.create();

    SECTION("cleans up in bit field region of page when page is abandoned.")
    {
        // This test does not really do anything and no such functionality is implemented currently because in a
        // single-threaded environment this is a trivial condition (as long as you use the create()/destroy()
        // interface). This might change once we enter multi-threaded realms.
        // TODO(lenz): Come back to this and check if this still holds true in the final implementation.

        // This is the only allocation on this page, so page is abandoned afterwards.
        page.destroy(pointer);

        // TODO(lenz): Check for off-by-one error in lower bound.
        for(size_t i = pageSize - 1; i >= pageSize - page.maxBitFieldSize(); i--)
        {
            CHECK(reinterpret_cast<char*>(page._data.data)[i] == 0U);
        }

        FAIL("Not yet implemented, needs to check what happens when a smaller chunkSize, i.e. more metadata, is "
             "used next time.");
    }
}

// NOLINTEND(*widening*)
