/*! \file
\brief Interface for scalar functions
\level 0
*/

#ifndef FOUR_C_UTILS_FUNCTION_OF_SCALAR_HPP
#define FOUR_C_UTILS_FUNCTION_OF_SCALAR_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::UTILS
{
  /**
   * This interface encodes functions \f$ y = f(s) \f$ which take a single scalar \f$ s \f$ and
   * return a single scalar \f$ y \f$.
   */
  class FunctionOfScalar
  {
   public:
    /**
     * Virtual destructor.
     */
    virtual ~FunctionOfScalar() = default;

    /**
     * Evaluate the function for the given @p scalar.
     */
    [[nodiscard]] virtual double evaluate(double scalar) const = 0;

    /**
     * Evaluate the @deriv_order derivative of the function for the given @p scalar.
     */
    [[nodiscard]] virtual double EvaluateDerivative(double scalar, int deriv_order) const = 0;
  };
}  // namespace Core::UTILS

FOUR_C_NAMESPACE_CLOSE

#endif