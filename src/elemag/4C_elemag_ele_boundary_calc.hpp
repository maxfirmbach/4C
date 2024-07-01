/*--------------------------------------------------------------------------*/
/*! \file

\brief Routines for elemag boundary elements

The routines are used in the creation of boundary elements for the electromagnetic module. The
correct implementation is still missing.

\level 2

*/
/*--------------------------------------------------------------------------*/


#ifndef FOUR_C_ELEMAG_ELE_BOUNDARY_CALC_HPP
#define FOUR_C_ELEMAG_ELE_BOUNDARY_CALC_HPP

#include "4C_config.hpp"

#include "4C_fem_condition.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{
  namespace ELEMENTS
  {
    class ElemagBoundary;

    /// Interface base class for ElemagBoundaryImpl
    /*!
      This class exists to provide a common interface for all template
      versions of ElemagBoundaryImpl. The only function
      this class actually defines is Impl, which returns a pointer to
      the appropriate version of ElemagBoundaryImpl.
     */
    class ElemagBoundaryImplInterface
    {
     public:
      /// Empty constructor
      ElemagBoundaryImplInterface() {}
      /// Empty destructor
      virtual ~ElemagBoundaryImplInterface() = default;
      /// Evaluate a Neumann boundary condition
      /*!
        This class does not provide a definition for this function, it
        must be defined in ElemagBoundaryImpl.
       */
      virtual int evaluate_neumann(Discret::ELEMENTS::ElemagBoundary* ele,
          Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          Core::Conditions::Condition& condition, std::vector<int>& lm,
          Core::LinAlg::SerialDenseVector& elevec1_epetra,
          Core::LinAlg::SerialDenseMatrix* elemat1) = 0;

      /// Evaluate routine for boundary elements inteface
      virtual int evaluate(Discret::ELEMENTS::ElemagBoundary* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, std::vector<int>& lm,
          Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
          Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
          Core::LinAlg::SerialDenseVector& elevec1_epetra,
          Core::LinAlg::SerialDenseVector& elevec2_epetra,
          Core::LinAlg::SerialDenseVector& elevec3_epetra) = 0;

      /// Internal implementation class for ElemagBoundary elements
      static ElemagBoundaryImplInterface* Impl(const Core::Elements::Element* ele);

    };  // class ElemagBoundaryImplInterface


    template <Core::FE::CellType distype>
    class ElemagBoundaryImpl : public ElemagBoundaryImplInterface
    {
     public:
      /// Singleton access method
      static ElemagBoundaryImpl<distype>* Instance(
          Core::UTILS::SingletonAction action = Core::UTILS::SingletonAction::create);

      /// Constructor
      ElemagBoundaryImpl();

      //! number of element nodes
      static constexpr int bdrynen_ = Core::FE::num_nodes<distype>;

      //! number of space dimensions of the ElemagBoundary element
      static constexpr int bdrynsd_ = Core::FE::dim<distype>;

      //! number of space dimensions of the parent element
      static constexpr int nsd_ = bdrynsd_ + 1;

      //! Evaluate a Neumann boundary condition
      int evaluate_neumann(Discret::ELEMENTS::ElemagBoundary* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
          std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1_epetra,
          Core::LinAlg::SerialDenseMatrix* elemat1) override;

      /// Evaluate routine for boundary elements
      int evaluate(Discret::ELEMENTS::ElemagBoundary* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, std::vector<int>& lm,
          Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
          Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
          Core::LinAlg::SerialDenseVector& elevec1_epetra,
          Core::LinAlg::SerialDenseVector& elevec2_epetra,
          Core::LinAlg::SerialDenseVector& elevec3_epetra) override;

     private:
      //! node coordinates for boundary element
      Core::LinAlg::Matrix<nsd_, bdrynen_> xyze_;
      //! coordinates of current integration point in reference coordinates
      Core::LinAlg::Matrix<bdrynsd_, 1> xsi_;
      //! array for shape functions for boundary element
      Core::LinAlg::Matrix<bdrynen_, 1> funct_;
      //! array for shape function derivatives for boundary element
      Core::LinAlg::Matrix<bdrynsd_, bdrynen_> deriv_;
      //! normal vector pointing out of the domain
      Core::LinAlg::Matrix<nsd_, 1> unitnormal_;
      //! velocity vector at integration point
      Core::LinAlg::Matrix<nsd_, 1> velint_;
      //! infinitesimal area element drs
      double drs_;
      //! integration factor
      double fac_;

    };  // class ElemagBoundaryImpl

  }  // namespace ELEMENTS
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif