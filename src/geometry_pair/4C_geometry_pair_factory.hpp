/*----------------------------------------------------------------------*/
/*! \file

\brief functions to create geometry pairs.

\level 1

*/
// End doxygen header.


#ifndef FOUR_C_GEOMETRY_PAIR_FACTORY_HPP
#define FOUR_C_GEOMETRY_PAIR_FACTORY_HPP


#include "4C_config.hpp"

#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

// Forward declarations.
namespace Core::Elements
{
  class Element;
}
namespace GEOMETRYPAIR
{
  class GeometryPair;
  class GeometryEvaluationDataBase;
}  // namespace GEOMETRYPAIR


namespace GEOMETRYPAIR
{
  /**
   * \brief Create the correct geometry pair for line to volume coupling.
   * @return RCP to created geometry pair.
   */
  template <typename scalar_type, typename line, typename volume>
  Teuchos::RCP<GeometryPair> GeometryPairLineToVolumeFactory(
      const Core::Elements::Element* element1, const Core::Elements::Element* element2,
      const Teuchos::RCP<GeometryEvaluationDataBase>& geometry_evaluation_data_ptr);

  /**
   * \brief Create the correct geometry pair for line to surface coupling.
   * @return RCP to created geometry pair.
   */
  template <typename scalar_type, typename line, typename surface>
  Teuchos::RCP<GeometryPair> GeometryPairLineToSurfaceFactory(
      const Core::Elements::Element* element1, const Core::Elements::Element* element2,
      const Teuchos::RCP<GeometryEvaluationDataBase>& geometry_evaluation_data_ptr);

  /**
   * \brief Create the correct geometry pair for line to surface coupling with FAD scalar types.
   *
   * The default GeometryPairLineToSurfaceFactory would be sufficient for this, however, for
   * performance reasons it is better use the wrapped pairs, which are created in this function.
   *
   * @return RCP to created geometry pair.
   */
  template <typename scalar_type, typename line, typename surface>
  Teuchos::RCP<GeometryPair> GeometryPairLineToSurfaceFactoryFAD(
      const Core::Elements::Element* element1, const Core::Elements::Element* element2,
      const Teuchos::RCP<GeometryEvaluationDataBase>& geometry_evaluation_data_ptr);
}  // namespace GEOMETRYPAIR

FOUR_C_NAMESPACE_CLOSE

#endif