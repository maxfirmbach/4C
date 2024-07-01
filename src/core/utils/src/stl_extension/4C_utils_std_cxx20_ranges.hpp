/*----------------------------------------------------------------------------*/
/*! \file

\brief  Provides range functionality for any given container. This is a forward implementation of
c++20 ranges::views

\level 0
    */
/*----------------------------------------------------------------------------*/

#ifndef FOUR_C_UTILS_STD_CXX20_RANGES_HPP
#define FOUR_C_UTILS_STD_CXX20_RANGES_HPP

#include "4C_config.hpp"

#include <iterator>

FOUR_C_NAMESPACE_OPEN

namespace std_20  // NOLINT
{
  namespace ranges::views  // NOLINT
  {
    namespace views = ranges::views;
    namespace INTERNAL
    {
      template <typename Iterator>
      class IteratorRange
      {
       public:
        IteratorRange(Iterator begin, Iterator end) : begin_(begin), end_(end) {}

        [[nodiscard]] Iterator begin() const { return begin_; }  // NOLINT
        [[nodiscard]] Iterator end() const { return end_; }      // NOLINT

       private:
        Iterator begin_;
        Iterator end_;
      };
    }  // namespace INTERNAL

    /**
     * \brief Returns a view that includes all elements of the given @p container.
     *
     */
    template <typename Container>
    auto all(Container& container)  // NOLINT
    {
      return INTERNAL::IteratorRange(std::begin(container), std::end(container));
    }
  }  // namespace ranges::views
}  // namespace std_20

FOUR_C_NAMESPACE_CLOSE

#endif