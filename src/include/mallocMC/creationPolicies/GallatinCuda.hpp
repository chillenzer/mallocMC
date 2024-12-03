/*
  mallocMC: Memory Allocator for Many Core Architectures.

  Copyright 2014-2024 Institute of Radiation Physics,
                 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Carlchristian Eckert - c.eckert ( at ) hzdr.de
              Julian Lenz - j.lenz ( at ) hzdr.de

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

#include <alpaka/alpaka.hpp>

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#    include <gallatin/allocators/gallatin.cuh>
#endif

namespace mallocMC
{
    namespace CreationPolicies
    {
        /**
         * @brief classic malloc/free behaviour known from CUDA
         *
         * This CreationPolicy implements the classic device-side malloc and
         * free system calls that is offered by CUDA-capable accelerator of
         * compute capability 2.0 and higher
         */
        template<
            size_t bytes_per_segment = 16ULL * 1024 * 1024,
            size_t smallest_slice = 16,
            size_t largest_slice = 4096>
        class GallatinCuda
        {
            using Gallatin = gallatin::allocators::Gallatin<bytes_per_segment, smallest_slice, largest_slice>;

        public:
            Gallatin* heap{nullptr};
            template<typename T_AlignmentPolicy>
            using AlignmentAwarePolicy = GallatinCuda;

            static constexpr auto providesAvailableSlots = false;

            template<typename AlpakaAcc>
            ALPAKA_FN_ACC auto create(AlpakaAcc const& acc, uint32_t bytes) const -> void*
            {
                return heap->malloc(static_cast<size_t>(bytes));
            }

            template<typename AlpakaAcc>
            ALPAKA_FN_ACC void destroy(AlpakaAcc const& /*acc*/, void* mem) const
            {
                heap->free(mem);
            }

            ALPAKA_FN_ACC auto isOOM(void* p, size_t s) const -> bool
            {
                return s != 0 && (p == nullptr);
            }

            template<typename AlpakaAcc, typename AlpakaDevice, typename AlpakaQueue, typename T_DeviceAllocator>
            static void initHeap(
                AlpakaDevice& dev,
                AlpakaQueue& queue,
                T_DeviceAllocator* devAllocator,
                void*,
                size_t memsize)
            {
                static_assert(
                    std::is_same_v<alpaka::AccToTag<AlpakaAcc>, alpaka::TagGpuCudaRt>,
                    "The GallatinCuda creation policy is only available on CUDA architectures. Please choose a "
                    "different one.");

                auto devHost = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);
                using Dim = typename alpaka::trait::DimType<AlpakaAcc>::type;
                using Idx = typename alpaka::trait::IdxType<AlpakaAcc>::type;
                using VecType = alpaka::Vec<Dim, Idx>;

                auto tmp = Gallatin::generate_on_device(memsize, 42, true);
                auto workDivSingleThread
                    = alpaka::WorkDivMembers<Dim, Idx>{VecType::ones(), VecType::ones(), VecType::ones()};
                alpaka::exec<AlpakaAcc>(
                    queue,
                    workDivSingleThread,
                    [tmp, devAllocator] ALPAKA_FN_ACC(AlpakaAcc const&) { devAllocator->heap = tmp; });
            }

            static auto classname() -> std::string
            {
                return "GallatinCuda";
            }
        };

    } // namespace CreationPolicies
} // namespace mallocMC
