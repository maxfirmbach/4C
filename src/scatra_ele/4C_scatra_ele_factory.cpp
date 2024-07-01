/*--------------------------------------------------------------------------*/
/*! \file

\brief Factory of scatra elements

\level 1

*/
/*--------------------------------------------------------------------------*/

#include "4C_scatra_ele_factory.hpp"

#include "4C_global_data.hpp"
#include "4C_scatra_ele_calc_advanced_reaction.hpp"
#include "4C_scatra_ele_calc_aniso.hpp"
#include "4C_scatra_ele_calc_artery.hpp"
#include "4C_scatra_ele_calc_cardiac_monodomain.hpp"
#include "4C_scatra_ele_calc_cardiac_monodomain_hdg.hpp"
#include "4C_scatra_ele_calc_chemo.hpp"
#include "4C_scatra_ele_calc_chemo_reac.hpp"
#include "4C_scatra_ele_calc_elch_diffcond.hpp"
#include "4C_scatra_ele_calc_elch_diffcond_multiscale.hpp"
#include "4C_scatra_ele_calc_elch_diffcond_sti_thermo.hpp"
#include "4C_scatra_ele_calc_elch_electrode.hpp"
#include "4C_scatra_ele_calc_elch_electrode_sti_thermo.hpp"
#include "4C_scatra_ele_calc_elch_NP.hpp"
#include "4C_scatra_ele_calc_elch_scl.hpp"
#include "4C_scatra_ele_calc_hdg.hpp"
#include "4C_scatra_ele_calc_loma.hpp"
#include "4C_scatra_ele_calc_ls.hpp"
#include "4C_scatra_ele_calc_lsreinit.hpp"
#include "4C_scatra_ele_calc_multiporo_reac.hpp"
#include "4C_scatra_ele_calc_no_physics.hpp"
#include "4C_scatra_ele_calc_poro.hpp"
#include "4C_scatra_ele_calc_poro_reac.hpp"
#include "4C_scatra_ele_calc_poro_reac_ECM.hpp"
#include "4C_scatra_ele_calc_refconc_reac.hpp"
#include "4C_scatra_ele_calc_std.hpp"
#include "4C_scatra_ele_calc_sti_diffcond.hpp"
#include "4C_scatra_ele_calc_sti_electrode.hpp"
#include "4C_scatra_ele_calc_utils.hpp"
#include "4C_scatra_ele_parameter_std.hpp"

FOUR_C_NAMESPACE_OPEN


/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
Discret::ELEMENTS::ScaTraEleInterface* Discret::ELEMENTS::ScaTraFactory::ProvideImpl(
    Core::FE::CellType distype, Inpar::ScaTra::ImplType problem, const int numdofpernode,
    const int numscal, const std::string& disname)
{
  // number of space dimensions
  const int ndim = disname != "scatra_micro"
                       ? Global::Problem::Instance(
                             Discret::ELEMENTS::ScaTraEleParameterStd::Instance(disname)->ProbNum())
                             ->NDim()
                       : 1;

  switch (distype)
  {
    case Core::FE::CellType::hex8:
    {
      if (ndim == 3)
        return define_problem_type<Core::FE::CellType::hex8, 3>(
            problem, numdofpernode, numscal, disname);
      else
        FOUR_C_THROW("invalid problem dimension for HEX8 transport element!");
      break;
    }
    case Core::FE::CellType::hex27:
    {
      if (ndim == 3)
        return define_problem_type<Core::FE::CellType::hex27, 3>(
            problem, numdofpernode, numscal, disname);
      else
        FOUR_C_THROW("invalid problem dimension for HEX27 transport element!");
      break;
    }
    case Core::FE::CellType::tet4:
    {
      if (ndim == 3)
        return define_problem_type<Core::FE::CellType::tet4, 3>(
            problem, numdofpernode, numscal, disname);
      else
        FOUR_C_THROW("invalid problem dimension for TET4 transport element!");
      break;
    }
    case Core::FE::CellType::tet10:
    {
      if (ndim == 3)
        return define_problem_type<Core::FE::CellType::tet10, 3>(
            problem, numdofpernode, numscal, disname);
      else
        FOUR_C_THROW("invalid problem dimension for TET10 transport element!");
      break;
    }
    case Core::FE::CellType::pyramid5:
    {
      if (ndim == 3)
      {
        return define_problem_type<Core::FE::CellType::pyramid5, 3>(
            problem, numdofpernode, numscal, disname);
      }
      else
        FOUR_C_THROW("invalid problem dimension for PYRAMID5 transport element!");
      break;
    }
    case Core::FE::CellType::quad4:
    {
      if (ndim == 2)
        return define_problem_type<Core::FE::CellType::quad4, 2>(
            problem, numdofpernode, numscal, disname);
      else if (ndim == 3)
        return define_problem_type<Core::FE::CellType::quad4, 3>(
            problem, numdofpernode, numscal, disname);
      else
        FOUR_C_THROW("invalid problem dimension for quad4 transport element!");
      break;
    }
    case Core::FE::CellType::quad9:
    {
      if (ndim == 2)
        return define_problem_type<Core::FE::CellType::quad9, 2>(
            problem, numdofpernode, numscal, disname);
      else
      {
        FOUR_C_THROW(
            "QUAD9 transport element not implemented as part of %i-dimensional problem. Just do it",
            ndim);
      }
      break;
    }
    case Core::FE::CellType::nurbs9:
    {
      if (ndim == 2)
        return define_problem_type<Core::FE::CellType::nurbs9, 2>(
            problem, numdofpernode, numscal, disname);
      else
      {
        FOUR_C_THROW(
            "NURBS9 transport element not implemented as part of %i-dimensional problem. Just do "
            "it",
            ndim);
      }
      break;
    }
    case Core::FE::CellType::tri3:
    {
      if (ndim == 2)
        return define_problem_type<Core::FE::CellType::tri3, 2>(
            problem, numdofpernode, numscal, disname);
      else if (ndim == 3)
        return define_problem_type<Core::FE::CellType::tri3, 3>(
            problem, numdofpernode, numscal, disname);
      else
        FOUR_C_THROW("invalid problem dimension for tri3 transport element!");
      break;
    }
    case Core::FE::CellType::tri6:
    {
      if (ndim == 2)
        return define_problem_type<Core::FE::CellType::tri6, 2>(
            problem, numdofpernode, numscal, disname);
      else
      {
        FOUR_C_THROW(
            "TRI6 transport element not implemented as part of %i-dimensional problem. Just do it",
            ndim);
      }
      break;
    }
    case Core::FE::CellType::line2:
    {
      if (ndim == 1)
        return define_problem_type<Core::FE::CellType::line2, 1>(
            problem, numdofpernode, numscal, disname);
      else if (ndim == 2)
        return define_problem_type<Core::FE::CellType::line2, 2>(
            problem, numdofpernode, numscal, disname);
      else if (ndim == 3)
        return define_problem_type<Core::FE::CellType::line2, 3>(
            problem, numdofpernode, numscal, disname);
      else
        FOUR_C_THROW("invalid problem dimension for LINE2 transport element!");
      break;
    }
    case Core::FE::CellType::line3:
    {
      if (ndim != 1)
      {
        FOUR_C_THROW(
            "LINE3 transport element not implemented as part of %i-dimensional problem. Just do it",
            ndim);
      }
      else
        return define_problem_type<Core::FE::CellType::line3, 1>(
            problem, numdofpernode, numscal, disname);
      break;
    }
    default:
      FOUR_C_THROW("Element shape %s not activated. Just do it.",
          Core::FE::CellTypeToString(distype).c_str());
      break;
  }
  return nullptr;
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
Discret::ELEMENTS::ScaTraEleInterface* Discret::ELEMENTS::ScaTraFactory::ProvideImplHDG(
    Core::FE::CellType distype, Inpar::ScaTra::ImplType problem, const int numdofpernode,
    const int numscal, const std::string& disname)
{
  // -------------------------------------- number of degrees of freedom
  // number of degrees of freedom
  static const int ndim = Global::Problem::Instance()->NDim();

  switch (distype)
  {
    case Core::FE::CellType::hex8:
    {
      if (ndim == 3)
      {
        return define_problem_type_hdg<Core::FE::CellType::hex8, 3>(
            problem, numdofpernode, numscal, disname);
      }
      else
        FOUR_C_THROW("invalid problem dimension for HEX8 transport element!");
      break;
    }
    case Core::FE::CellType::tet4:
    {
      if (ndim == 3)
      {
        return define_problem_type_hdg<Core::FE::CellType::tet4, 3>(
            problem, numdofpernode, numscal, disname);
      }
      else
        FOUR_C_THROW("invalid problem dimension for TET4 transport element!");
      break;
    }
    case Core::FE::CellType::tet10:
    {
      if (ndim == 3)
      {
        return define_problem_type_hdg<Core::FE::CellType::tet10, 3>(
            problem, numdofpernode, numscal, disname);
      }
      else
        FOUR_C_THROW("invalid problem dimension for TET10 transport element!");
      break;
    }
    case Core::FE::CellType::pyramid5:
    {
      if (ndim == 3)
      {
        return define_problem_type_hdg<Core::FE::CellType::pyramid5, 3>(
            problem, numdofpernode, numscal, disname);
      }
      else
        FOUR_C_THROW("invalid problem dimension for PYRAMID5 transport element!");
      break;
    }
    case Core::FE::CellType::quad4:
    {
      if (ndim == 2)
      {
        return define_problem_type_hdg<Core::FE::CellType::quad4, 2>(
            problem, numdofpernode, numscal, disname);
      }
      else if (ndim == 3)
      {
        return define_problem_type_hdg<Core::FE::CellType::quad4, 3>(
            problem, numdofpernode, numscal, disname);
      }
      else
        FOUR_C_THROW("invalid problem dimension for quad4 transport element!");
      break;
    }
    case Core::FE::CellType::tri3:
    {
      return define_problem_type_hdg<Core::FE::CellType::tri3, 2>(
          problem, numdofpernode, numscal, disname);
    }
    default:
      FOUR_C_THROW("Element shape %s not activated. Just do it.",
          Core::FE::CellTypeToString(distype).c_str());
      break;
  }
  return nullptr;
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::ELEMENTS::ScaTraEleInterface* Discret::ELEMENTS::ScaTraFactory::define_problem_type(
    Inpar::ScaTra::ImplType problem, const int numdofpernode, const int numscal,
    const std::string& disname)
{
  if ((probdim - Core::FE::dim<distype>) == 1)
  {
    if (problem != Inpar::ScaTra::impltype_std and
        problem != Inpar::ScaTra::impltype_cardiac_monodomain and
        problem != Inpar::ScaTra::impltype_advreac and
        problem != Inpar::ScaTra::impltype_lsreinit and
        problem != Inpar::ScaTra::impltype_one_d_artery and
        problem != Inpar::ScaTra::impltype_no_physics and
        problem != Inpar::ScaTra::impltype_elch_electrode and
        problem != Inpar::ScaTra::impltype_elch_diffcond)
      FOUR_C_THROW("ImplType '%s' not implemented for transport on manifolds!",
          ScaTra::ImplTypeToString(problem).c_str());
  }

  switch (problem)
  {
    case Inpar::ScaTra::impltype_std:
    {
      return Discret::ELEMENTS::ScaTraEleCalcStd<distype, probdim>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_thermo_elch_electrode:
    {
      return Discret::ELEMENTS::ScaTraEleCalcSTIElectrode<distype>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_thermo_elch_diffcond:
    {
      return Discret::ELEMENTS::ScaTraEleCalcSTIDiffCond<distype>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_levelset:
    {
      return Discret::ELEMENTS::ScaTraEleCalcLS<distype>::Instance(numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_lsreinit:
    {
      return Discret::ELEMENTS::ScaTraEleCalcLsReinit<distype, probdim>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_loma:
    {
      return Discret::ELEMENTS::ScaTraEleCalcLoma<distype>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_elch_NP:
    {
      return Discret::ELEMENTS::ScaTraEleCalcElchNP<distype>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_elch_electrode:
    case Inpar::ScaTra::impltype_elch_electrode_growth:
    {
      return Discret::ELEMENTS::ScaTraEleCalcElchElectrode<distype, probdim>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_elch_electrode_thermo:
    {
      return Discret::ELEMENTS::ScaTraEleCalcElchElectrodeSTIThermo<distype>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_elch_diffcond:
    {
      return Discret::ELEMENTS::ScaTraEleCalcElchDiffCond<distype, probdim>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_elch_diffcond_multiscale:
    {
      return Discret::ELEMENTS::ScaTraEleCalcElchDiffCondMultiScale<distype, probdim>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_elch_diffcond_thermo:
    {
      return Discret::ELEMENTS::ScaTraEleCalcElchDiffCondSTIThermo<distype>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_elch_scl:
    {
      return Discret::ELEMENTS::ScaTraEleCalcElchScl<distype, probdim>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_poro:
    {
      return Discret::ELEMENTS::ScaTraEleCalcPoro<distype>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_advreac:
    {
      return Discret::ELEMENTS::ScaTraEleCalcAdvReac<distype, probdim>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_refconcreac:
    {
      return Discret::ELEMENTS::ScaTraEleCalcRefConcReac<distype>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_chemo:
    {
      return Discret::ELEMENTS::ScaTraEleCalcChemo<distype>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_chemoreac:
    {
      return Discret::ELEMENTS::ScaTraEleCalcChemoReac<distype>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_multipororeac:
    {
      return Discret::ELEMENTS::ScaTraEleCalcMultiPoroReac<distype>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_pororeac:
    {
      return Discret::ELEMENTS::ScaTraEleCalcPoroReac<distype>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_pororeacECM:
    {
      return Discret::ELEMENTS::ScaTraEleCalcPoroReacECM<distype>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_aniso:
    {
      return Discret::ELEMENTS::ScaTraEleCalcAniso<distype, probdim>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_cardiac_monodomain:
    {
      return Discret::ELEMENTS::ScaTraEleCalcCardiacMonodomain<distype, probdim>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_one_d_artery:
    {
      return Discret::ELEMENTS::ScaTraEleCalcArtery<distype, probdim>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_no_physics:
      return Discret::ELEMENTS::ScaTraEleCalcNoPhysics<distype, probdim>::Instance(
          numdofpernode, numscal, disname);

    default:
    {
      FOUR_C_THROW("Defined problem type does not exist!!");
      break;
    }
  }

  return nullptr;
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::ELEMENTS::ScaTraEleInterface* Discret::ELEMENTS::ScaTraFactory::define_problem_type_hdg(
    Inpar::ScaTra::ImplType problem, const int numdofpernode, const int numscal,
    const std::string& disname)
{
  if (Core::FE::dim<distype> != probdim)
  {
    if (problem != Inpar::ScaTra::impltype_std and
        problem != Inpar::ScaTra::impltype_cardiac_monodomain)
      FOUR_C_THROW("ImplType '%s' not implemented for transport on manifolds!",
          ScaTra::ImplTypeToString(problem).c_str());
  }

  switch (problem)
  {
    case Inpar::ScaTra::impltype_std_hdg:
    {
      return Discret::ELEMENTS::ScaTraEleCalcHDG<distype, probdim>::Instance(
          numdofpernode, numscal, disname);
    }
    case Inpar::ScaTra::impltype_cardiac_monodomain_hdg:
    {
      return Discret::ELEMENTS::ScaTraEleCalcHDGCardiacMonodomain<distype, probdim>::Instance(
          numdofpernode, numscal, disname);
    }
    default:
    {
      FOUR_C_THROW("Defined problem type does not exist!!");
      break;
    }
  }

  return nullptr;
}

FOUR_C_NAMESPACE_CLOSE