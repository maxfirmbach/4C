// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ART_NET_ARTERY_ELE_CALC_PRES_BASED_HPP
#define FOUR_C_ART_NET_ARTERY_ELE_CALC_PRES_BASED_HPP

#include "4C_config.hpp"

#include "4C_art_net_artery.hpp"
#include "4C_art_net_artery_ele_action.hpp"
#include "4C_art_net_artery_ele_calc.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Discret
{
  namespace Elements
  {
    /// Internal artery implementation
    /*!
      This internal class keeps all the working arrays needed to
      calculate the artery element. Additionally the method Sysmat()
      provides a clean and fast element implementation.

      <h3>Purpose</h3>

    */

    template <Core::FE::CellType distype>
    class ArteryEleCalcPresBased : public ArteryEleCalc<distype>
    {
     private:
      using my = ArteryEleCalc<distype>;

      /// private constructor, since we are a Singleton.
      ArteryEleCalcPresBased(const int numdofpernode, const std::string& disname);

     public:
      //! Singleton access method
      static ArteryEleCalcPresBased<distype>* instance(
          const int numdofpernode, const std::string& disname);

      int evaluate(Artery* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
          Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
          Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3,
          std::shared_ptr<Core::Mat::Material> mat) override;

      int scatra_evaluate(Artery* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, std::vector<int>& lm,
          Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
          Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3,
          std::shared_ptr<Core::Mat::Material> mat) override;

      int evaluate_service(Artery* ele, const Arteries::Action action,
          Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2, Core::LinAlg::SerialDenseVector& elevec3,
          std::shared_ptr<Core::Mat::Material> mat) override;


     protected:
      /*!
        \brief calculate element matrix and rhs

        \param ele[in]            the element which is evaluated
        \param discretization[in] discretization to which element belongs
        \param la[in]             element location array
        \param sysmat[in,out]     element matrix to calculate
        \param rhs[in,out]        element rhs to calculate
        \param material[in]       artery material
        */
      void sysmat(Artery* ele, Core::FE::Discretization& discretization,
          Core::Elements::LocationArray& la, Core::LinAlg::Matrix<my::iel_, my::iel_>& sysmat,
          Core::LinAlg::Matrix<my::iel_, 1>& rhs,
          std::shared_ptr<const Core::Mat::Material> material);

      /*!
        \brief Evaluate volumetric flow inside the element (for post-processing)

        \param ele[in]            the element which is evaluated
        \param discretization[in] discretization to which element belongs
        \param la[in]             element location array
        \param flowVec[in,out]    element flow to calculate
        \param ele_length[in,out] element length to calculate
        \param material[in]       artery material/dimesion

        \note  only checked for line2 elements
        */
      void evaluate_flow(Artery* ele, Core::FE::Discretization& discretization,
          Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseVector& flowVec,
          Core::LinAlg::SerialDenseVector& ele_length,
          std::shared_ptr<const Core::Mat::Material> material);

      /*!
        \brief Calculate element length (either in current or deformed configuration)
        \param ele[in]            the element which is evaluated
        \param discretization[in] discretization to which element belongs
        \param la[in]             element location array
        \return                   element length (either in current or deformed configuration)
       */
      double calculate_ele_length(
          Artery* ele, Core::FE::Discretization& discretization, Core::Elements::LocationArray& la);
    };

  }  // namespace Elements

}  // namespace Discret



FOUR_C_NAMESPACE_CLOSE

#endif
