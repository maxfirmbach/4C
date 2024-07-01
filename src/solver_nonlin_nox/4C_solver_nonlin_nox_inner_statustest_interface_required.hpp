/*-----------------------------------------------------------*/
/*! \file



\level 3

*/
/*-----------------------------------------------------------*/

#ifndef FOUR_C_SOLVER_NONLIN_NOX_INNER_STATUSTEST_INTERFACE_REQUIRED_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_INNER_STATUSTEST_INTERFACE_REQUIRED_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_forward_decl.hpp"

#include <NOX_Abstract_Group.H>
#include <NOX_StatusTest_Generic.H>

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    namespace Inner
    {
      namespace StatusTest
      {
        enum StatusType : int;
        namespace Interface
        {
          class Required
          {
           public:
            //! constructor
            Required(){};

            //! destructor
            virtual ~Required() = default;

            //! Get the number of inner-loop nonlinear iterations (e.g. line search iterations)
            virtual int GetNumIterations() const = 0;

            //! Get the objective or meritfunction
            virtual const ::NOX::MeritFunction::Generic& GetMeritFunction() const = 0;

            //! Execute the inner status test
            virtual NOX::Nln::Inner::StatusTest::StatusType CheckInnerStatus(
                const ::NOX::Solver::Generic& solver, const ::NOX::Abstract::Group& grp,
                ::NOX::StatusTest::CheckType checkType) const = 0;
          };  // class Required
        }     // namespace Interface
      }       // namespace StatusTest
    }         // namespace Inner
  }           // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif