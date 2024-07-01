/*----------------------------------------------------------------------*/
/*! \file
 \brief main file containing routines for calculation of scatra element with advanced reaction terms

 \level 2

 *----------------------------------------------------------------------*/


#include "4C_scatra_ele_calc_advanced_reaction.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_growth_law.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_list_reactions.hpp"
#include "4C_mat_scatra.hpp"
#include "4C_mat_scatra_reaction.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::ELEMENTS::ScaTraEleCalcAdvReac<distype, probdim>*
Discret::ELEMENTS::ScaTraEleCalcAdvReac<distype, probdim>::Instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = Core::UTILS::MakeSingletonMap<std::pair<std::string, int>>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleCalcAdvReac<distype, probdim>>(
            new ScaTraEleCalcAdvReac<distype, probdim>(numdofpernode, numscal, disname));
      });

  return singleton_map[std::make_pair(disname, numdofpernode)].Instance(
      Core::UTILS::SingletonAction::create, numdofpernode, numscal, disname);
}


/*----------------------------------------------------------------------*
 *  constructor---------------------------                   thon 02/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::ELEMENTS::ScaTraEleCalcAdvReac<distype, probdim>::ScaTraEleCalcAdvReac(
    const int numdofpernode, const int numscal, const std::string& disname)
    : Discret::ELEMENTS::ScaTraEleCalc<distype, probdim>::ScaTraEleCalc(
          numdofpernode, numscal, disname)
{
  my::reamanager_ = Teuchos::rcp(new ScaTraEleReaManagerAdvReac(my::numscal_));

  for (unsigned i = 0; i < numdim_gp_; ++i) gpcoord_[i] = 0.0;

  // safety check
  if (not my::scatrapara_->TauGP())
    FOUR_C_THROW(
        "For advanced reactions, tau needs to be evaluated by integration-point evaluations!");
}

/*----------------------------------------------------------------------*
 |  get the material constants  (private)                      thon 09/14|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::ELEMENTS::ScaTraEleCalcAdvReac<distype, probdim>::get_material_params(
    const Core::Elements::Element* ele,  //!< the element we are dealing with
    std::vector<double>& densn,          //!< density at t_(n)
    std::vector<double>& densnp,         //!< density at t_(n+1) or t_(n+alpha_F)
    std::vector<double>& densam,         //!< density at t_(n+alpha_M)
    double& visc,                        //!< fluid viscosity
    const int iquad                      //!< id of current gauss point
)
{
  // get the material
  Teuchos::RCP<Core::Mat::Material> material = ele->Material();

  // We may have some reactive and some non-reactive elements in one discretisation.
  // But since the calculation classes are singleton, we have to reset all reactive stuff in case
  // of non-reactive elements:
  rea_manager()->Clear(my::numscal_);

  if (material->MaterialType() == Core::Materials::m_matlist)
  {
    const Teuchos::RCP<const Mat::MatList> actmat =
        Teuchos::rcp_dynamic_cast<const Mat::MatList>(material);
    if (actmat->NumMat() != my::numscal_) FOUR_C_THROW("Not enough materials in MatList.");

    for (int k = 0; k < my::numscal_; ++k)
    {
      int matid = actmat->MatID(k);
      Teuchos::RCP<Core::Mat::Material> singlemat = actmat->MaterialById(matid);

      materials(singlemat, k, densn[k], densnp[k], densam[k], visc, iquad);
    }
  }

  else if (material->MaterialType() == Core::Materials::m_matlist_reactions)
  {
    const Teuchos::RCP<Mat::MatListReactions> actmat =
        Teuchos::rcp_dynamic_cast<Mat::MatListReactions>(material);
    if (actmat->NumMat() != my::numscal_) FOUR_C_THROW("Not enough materials in MatList.");

    for (int k = 0; k < my::numscal_; ++k)
    {
      int matid = actmat->MatID(k);
      Teuchos::RCP<Core::Mat::Material> singlemat = actmat->MaterialById(matid);

      // Note: order is important here!!
      materials(singlemat, k, densn[k], densnp[k], densam[k], visc, iquad);

      set_advanced_reaction_terms(
          k, actmat, get_gp_coord());  // every reaction calculation stuff happens in here!!
    }
  }

  else
  {
    materials(material, 0, densn[0], densnp[0], densam[0], visc, iquad);
  }

  return;
}  // ScaTraEleCalc::get_material_params

/*----------------------------------------------------------------------*
 |  evaluate single material  (protected)                    thon 02/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::ELEMENTS::ScaTraEleCalcAdvReac<distype, probdim>::materials(
    const Teuchos::RCP<const Core::Mat::Material> material,  //!< pointer to current material
    const int k,                                             //!< id of current scalar
    double& densn,                                           //!< density at t_(n)
    double& densnp,  //!< density at t_(n+1) or t_(n+alpha_F)
    double& densam,  //!< density at t_(n+alpha_M)
    double& visc,    //!< fluid viscosity
    const int iquad  //!< id of current gauss point
)
{
  switch (material->MaterialType())
  {
    case Core::Materials::m_scatra:
      my::mat_scatra(material, k, densn, densnp, densam, visc, iquad);
      break;
    default:
      FOUR_C_THROW("Material type %i is not supported", material->MaterialType());
      break;
  }
  return;
}


/*----------------------------------------------------------------------*
 |  Get right hand side including reaction bodyforce term    thon 02/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::ELEMENTS::ScaTraEleCalcAdvReac<distype, probdim>::get_rhs_int(
    double& rhsint,       //!< rhs containing bodyforce at Gauss point
    const double densnp,  //!< density at t_(n+1)
    const int k           //!< index of current scalar
)
{
  //... + all advanced reaction terms
  rhsint = my::bodyforce_[k].dot(my::funct_) + densnp * rea_manager()->GetReaBodyForce(k);

  return;
}

/*--------------------------------------------------------------------------- *
 |  calculation of reactive element matrix for coupled reactions  thon 02/14  |
 *----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::ELEMENTS::ScaTraEleCalcAdvReac<distype, probdim>::calc_mat_react(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const double timefacfac,
    const double timetaufac, const double taufac, const double densnp,
    const Core::LinAlg::Matrix<nen_, 1>& sgconv, const Core::LinAlg::Matrix<nen_, 1>& diff)
{
  // -----------------first care for 'easy' reaction terms K*(\partial_c
  // c)=Id*K-------------------------------------- NOTE: K_i must not depend on any concentrations!!
  // Otherwise we loose the corresponding linearisations.

  my::calc_mat_react(emat, k, timefacfac, timetaufac, taufac, densnp, sgconv, diff);

  const Core::LinAlg::Matrix<nen_, 1>& conv = my::scatravarmanager_->Conv(k);

  // -----------------second care for advanced reaction terms ( - (\partial_c f(c) )------------
  // NOTE: The shape of f(c) can be arbitrary. So better consider using this term for new
  // implementations

  const Teuchos::RCP<ScaTraEleReaManagerAdvReac> remanager = rea_manager();

  Core::LinAlg::Matrix<nen_, 1> functint = my::funct_;
  if (not my::scatrapara_->MatGP()) functint = funct_elementcenter_;

  for (int j = 0; j < my::numscal_; j++)
  {
    const double fac_reac =
        timefacfac * densnp * (-remanager->get_rea_body_force_deriv_matrix(k, j));
    const double timetaufac_reac =
        timetaufac * densnp * (-remanager->get_rea_body_force_deriv_matrix(k, j));

    //----------------------------------------------------------------
    // standard Galerkin reactive term
    //----------------------------------------------------------------
    for (unsigned vi = 0; vi < nen_; ++vi)
    {
      const double v = fac_reac * functint(vi);
      const int fvi = vi * my::numdofpernode_ + k;

      for (unsigned ui = 0; ui < nen_; ++ui)
      {
        const int fui = ui * my::numdofpernode_ + j;

        emat(fvi, fui) += v * my::funct_(ui);
      }
    }

    //----------------------------------------------------------------
    // stabilization of reactive term
    //----------------------------------------------------------------
    if (my::scatrapara_->StabType() != Inpar::ScaTra::stabtype_no_stabilization)
    {
      double densreataufac = timetaufac_reac * densnp;
      // convective stabilization of reactive term (in convective form)
      for (unsigned vi = 0; vi < nen_; ++vi)
      {
        const double v = densreataufac * (conv(vi) + sgconv(vi) +
                                             my::scatrapara_->USFEMGLSFac() * 1.0 /
                                                 my::scatraparatimint_->TimeFac() * functint(vi));
        const int fvi = vi * my::numdofpernode_ + k;

        for (unsigned ui = 0; ui < nen_; ++ui)
        {
          const int fui = ui * my::numdofpernode_ + j;

          emat(fvi, fui) += v * my::funct_(ui);
        }
      }

      if (my::use2ndderiv_)
      {
        // diffusive stabilization of reactive term
        for (unsigned vi = 0; vi < nen_; ++vi)
        {
          const double v = my::scatrapara_->USFEMGLSFac() * timetaufac_reac * diff(vi);
          const int fvi = vi * my::numdofpernode_ + k;

          for (unsigned ui = 0; ui < nen_; ++ui)
          {
            const int fui = ui * my::numdofpernode_ + j;

            emat(fvi, fui) -= v * my::funct_(ui);
          }
        }
      }

      //----------------------------------------------------------------
      // reactive stabilization
      //----------------------------------------------------------------
      densreataufac = my::scatrapara_->USFEMGLSFac() * timetaufac_reac * densnp;

      // reactive stabilization of convective (in convective form) and reactive term
      for (unsigned vi = 0; vi < nen_; ++vi)
      {
        const double v = densreataufac * functint(vi);
        const int fvi = vi * my::numdofpernode_ + k;

        for (unsigned ui = 0; ui < nen_; ++ui)
        {
          const int fui = ui * my::numdofpernode_ + j;

          emat(fvi, fui) += v * (conv(ui) + remanager->GetReaCoeff(k) * my::funct_(ui));
        }
      }

      if (my::use2ndderiv_)
      {
        // reactive stabilization of diffusive term
        for (unsigned vi = 0; vi < nen_; ++vi)
        {
          const double v = my::scatrapara_->USFEMGLSFac() * timetaufac_reac * my::funct_(vi);
          const int fvi = vi * my::numdofpernode_ + k;

          for (unsigned ui = 0; ui < nen_; ++ui)
          {
            const int fui = ui * my::numdofpernode_ + j;

            emat(fvi, fui) -= v * diff(ui);
          }
        }
      }


      if (not my::scatraparatimint_->IsStationary())
      {
        // reactive stabilization of transient term
        for (unsigned vi = 0; vi < nen_; ++vi)
        {
          const double v = my::scatrapara_->USFEMGLSFac() * taufac * densnp *
                           remanager->GetReaCoeff(k) * densnp * functint(vi);
          const int fvi = vi * my::numdofpernode_ + k;

          for (unsigned ui = 0; ui < nen_; ++ui)
          {
            const int fui = ui * my::numdofpernode_ + j;

            emat(fvi, fui) += v * my::funct_(ui);
          }
        }

        if (my::use2ndderiv_ and remanager->GetReaCoeff(k) != 0.0)
          FOUR_C_THROW("Second order reactive stabilization is not fully implemented!! ");
      }
    }
  }  // end for
  return;
}


/*-------------------------------------------------------------------------------*
 |  Set advanced reaction terms and derivatives                       thon 09/14 |
 *-------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::ELEMENTS::ScaTraEleCalcAdvReac<distype, probdim>::set_advanced_reaction_terms(
    const int k,                                            //!< index of current scalar
    const Teuchos::RCP<Mat::MatListReactions> matreaclist,  //!< index of current scalar
    const double* gpcoord                                   //!< current Gauss-point coordinates
)
{
  const Teuchos::RCP<ScaTraEleReaManagerAdvReac> remanager = rea_manager();

  remanager->AddToReaBodyForce(
      matreaclist->calc_rea_body_force_term(k, my::scatravarmanager_->Phinp(), gpcoord), k);

  matreaclist->calc_rea_body_force_deriv_matrix(
      k, remanager->get_rea_body_force_deriv_vector(k), my::scatravarmanager_->Phinp(), gpcoord);
}

/*----------------------------------------------------------------------*
 | evaluate shape functions and derivatives at ele. center   jhoer 11/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
double Discret::ELEMENTS::ScaTraEleCalcAdvReac<distype,
    probdim>::eval_shape_func_and_derivs_at_ele_center()
{
  const double vol = my::eval_shape_func_and_derivs_at_ele_center();

  // shape function at element center
  funct_elementcenter_ = my::funct_;

  return vol;

}  // ScaTraImpl::eval_shape_func_and_derivs_at_ele_center

/*------------------------------------------------------------------------------*
 | set internal variables                                          vuong 11/14  |
 *------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::ELEMENTS::ScaTraEleCalcAdvReac<distype,
    probdim>::set_internal_variables_for_mat_and_rhs()
{
  my::set_internal_variables_for_mat_and_rhs();

  // calculate current Gauss-point coordinates from node coordinates and shape functions
  for (unsigned i = 0; i < nsd_; ++i)
  {
    gpcoord_[i] = 0.0;
    for (unsigned k = 0; k < nen_; ++k)
    {
      gpcoord_[i] += my::xyze_(i, k) * my::funct_(k);
    }
  }

  return;
}


// template classes

// 1D elements
template class Discret::ELEMENTS::ScaTraEleCalcAdvReac<Core::FE::CellType::line2, 1>;
template class Discret::ELEMENTS::ScaTraEleCalcAdvReac<Core::FE::CellType::line2, 2>;
template class Discret::ELEMENTS::ScaTraEleCalcAdvReac<Core::FE::CellType::line2, 3>;
template class Discret::ELEMENTS::ScaTraEleCalcAdvReac<Core::FE::CellType::line3, 1>;

// 2D elements
template class Discret::ELEMENTS::ScaTraEleCalcAdvReac<Core::FE::CellType::tri3, 2>;
template class Discret::ELEMENTS::ScaTraEleCalcAdvReac<Core::FE::CellType::tri3, 3>;
template class Discret::ELEMENTS::ScaTraEleCalcAdvReac<Core::FE::CellType::tri6, 2>;
template class Discret::ELEMENTS::ScaTraEleCalcAdvReac<Core::FE::CellType::quad4, 2>;
template class Discret::ELEMENTS::ScaTraEleCalcAdvReac<Core::FE::CellType::quad4, 3>;
// template class Discret::ELEMENTS::ScaTraEleCalcAdvReac<Core::FE::CellType::quad8>;
template class Discret::ELEMENTS::ScaTraEleCalcAdvReac<Core::FE::CellType::quad9, 2>;
template class Discret::ELEMENTS::ScaTraEleCalcAdvReac<Core::FE::CellType::nurbs9, 2>;

// 3D elements
template class Discret::ELEMENTS::ScaTraEleCalcAdvReac<Core::FE::CellType::hex8, 3>;
// template class Discret::ELEMENTS::ScaTraEleCalcAdvReac<Core::FE::CellType::hex20>;
template class Discret::ELEMENTS::ScaTraEleCalcAdvReac<Core::FE::CellType::hex27, 3>;
template class Discret::ELEMENTS::ScaTraEleCalcAdvReac<Core::FE::CellType::tet4, 3>;
template class Discret::ELEMENTS::ScaTraEleCalcAdvReac<Core::FE::CellType::tet10, 3>;
// template class Discret::ELEMENTS::ScaTraEleCalcAdvReac<Core::FE::CellType::wedge6>;
template class Discret::ELEMENTS::ScaTraEleCalcAdvReac<Core::FE::CellType::pyramid5, 3>;
// template class Discret::ELEMENTS::ScaTraEleCalcAdvReac<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE