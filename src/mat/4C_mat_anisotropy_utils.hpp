/*----------------------------------------------------------------------*/
/*! \file

\brief Utilities for the Anisotropy classes

\level 3


*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_MAT_ANISOTROPY_UTILS_HPP
#define FOUR_C_MAT_ANISOTROPY_UTILS_HPP

#include "4C_config.hpp"

#include "4C_io_linedefinition.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

#include <Teuchos_RCPDecl.hpp>

#include <string>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::Communication
{
  class PackBuffer;
}
namespace Input
{
  class LineDefinition;
}

namespace Mat
{
  // forward declaration
  namespace Elastic
  {
    class StructuralTensorStrategyBase;
  }
  /*!
   * Reads a fiber with a specification from the input file definition
   *
   * @param linedef (in) : Input line definition
   * @param specifier (in) : Identifier of the fiber
   * @param fiber_vector (out) : Fiber vector
   */
  void read_anisotropy_fiber(Input::LineDefinition* linedef, std::string specifier,
      Core::LinAlg::Matrix<3, 1>& fiber_vector);

  /*!
   * \brief Compute structural tensors of a 2D vector of fibers with the structural tensor
   * strategy
   *
   * \tparam T Output type of the structural tensor (either matrix notation or stress-like Voigt
   * notation)
   *
   * \tparam numfib number of fibers
   *
   * \param fibers 2D vector of fibers (3x1 matrices)
   * \param structural_tensor 2D vector of structural tensors (3x3 or 6x1 matrices)
   * \param strategy Reference to the structural tensor strategy
   */
  template <typename T, unsigned int numfib>
  void compute_structural_tensors(
      std::vector<std::array<Core::LinAlg::Matrix<3, 1>, numfib>>& fibers,
      std::vector<std::array<T, numfib>>& structural_tensor,
      const Teuchos::RCP<Elastic::StructuralTensorStrategyBase>& strategy);

  /*!
   * \brief Pack 2D vector of fibers and structural tensors
   *
   * \tparam T Type of the fiber (3x1, 6x1 or 3x3 matrices)
   * \param buffer buffer where to pack the data
   * \param vct vector
   */
  template <typename T>
  void pack_fiber_vector(
      Core::Communication::PackBuffer& buffer, const std::vector<std::vector<T>>& vct);

  /*!
   * \brief Pack 2D vector of fibers and structural tensors
   *
   * \tparam T Type of the fiber (3x1, 6x1 or 3x3 matrices)
   * \tparam numfib number of fibers
   * \param buffer buffer where to pack the data
   * \param vct vector
   */
  template <typename T, unsigned int numfib>
  void pack_fiber_array(
      Core::Communication::PackBuffer& buffer, const std::vector<std::array<T, numfib>>& vct);

  /*!
   * \brief Unpack 2D vector of fibers and structural tensors
   *
   * \tparam T Type of the fiber (3x1, 6x1 or 3x3 matrices)
   * \param position Position where to start to unpack the data
   * \param data data where to unpack the data from
   * \param vct destination 2D array
   */
  template <typename T>
  void unpack_fiber_vector(std::vector<char>::size_type& position, const std::vector<char>& data,
      std::vector<std::vector<T>>& vct);

  /*!
   * \brief Unpack 2D vector of fibers and structural tensors
   *
   * \tparam T Type of the fiber (3x1, 6x1 or 3x3 matrices)
   * \tparam numfib number of fibers
   * \param position Position where to start to unpack the data
   * \param data data where to unpack the data from
   * \param vct destination 2D array
   */
  template <typename T, unsigned int numfib>
  void unpack_fiber_array(std::vector<char>::size_type& position, const std::vector<char>& data,
      std::vector<std::array<T, numfib>>& vct);
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif