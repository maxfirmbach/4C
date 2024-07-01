/*----------------------------------------------------------------------*/
/*! \file
\brief input parameters for VTP output of structural problem at runtime

\level 2

*/
/*----------------------------------------------------------------------*/
/* definitions */
#ifndef FOUR_C_INPAR_IO_RUNTIME_VTP_OUTPUT_STRUCTURE_HPP
#define FOUR_C_INPAR_IO_RUNTIME_VTP_OUTPUT_STRUCTURE_HPP


/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
namespace Inpar
{
  namespace IORuntimeVTPStructure
  {
    /// data format for written numeric data
    enum OutputDataFormat
    {
      binary,
      ascii,
      vague
    };

    /// set the valid parameters related to writing of VTP output at runtime
    void SetValidParameters(Teuchos::RCP<Teuchos::ParameterList> list);

  }  // namespace IORuntimeVTPStructure
}  // namespace Inpar

FOUR_C_NAMESPACE_CLOSE

#endif