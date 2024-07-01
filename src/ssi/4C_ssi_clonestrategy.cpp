/*----------------------------------------------------------------------*/
/*! \file
\brief mesh clone strategy for scatra-structure-interaction problems

\level 2


*/
/*----------------------------------------------------------------------*/

#include "4C_ssi_clonestrategy.hpp"

#include "4C_adapter_structure_scatra_ele.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_ssi.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_scatra_ele.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::map<std::string, std::string> SSI::ScatraStructureCloneStrategy::conditions_to_copy() const
{
  return {{"TransportDirichlet", "Dirichlet"}, {"TransportPointNeumann", "PointNeumann"},
      {"TransportLineNeumann", "LineNeumann"}, {"TransportSurfaceNeumann", "SurfaceNeumann"},
      {"TransportVolumeNeumann", "VolumeNeumann"},
      {"TransportNeumannInflow", "TransportNeumannInflow"}, {"FSICoupling", "FSICoupling"},
      {"ScaTraCoupling", "ScaTraCoupling"}, {"ScaTraFluxCalc", "ScaTraFluxCalc"},
      {"Initfield", "Initfield"}, {"ssi_interface_meshtying", "S2IMeshtying"},
      {"SSIInterfaceContact", "S2INoEvaluation"}, {"S2IKinetics", "S2IKinetics"},
      {"ScatraPartitioning", "ScatraPartitioning"}, {"ElectrodeSOC", "ElectrodeSOC"},
      {"CellVoltage", "CellVoltage"}, {"CellVoltagePoint", "CellVoltagePoint"},
      {"TotalAndMeanScalar", "TotalAndMeanScalar"}, {"CCCVCycling", "CCCVCycling"},
      {"CCCVHalfCycle", "CCCVHalfCycle"},
      {"TransportThermoConvections", "TransportThermoConvections"},
      {"SSIMeshtying3DomainIntersection", "Meshtying3DomainIntersection"},
      {"SSISurfaceManifold", "SSISurfaceManifold"},
      {"SSISurfaceManifoldKinetics", "SSISurfaceManifoldKinetics"}};
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::map<std::string, std::string> SSI::ScatraStructureCloneStrategyManifold::conditions_to_copy()
    const
{
  return {};
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Inpar::ScaTra::ImplType SSI::ScatraStructureCloneStrategy::GetImplType(Core::Elements::Element* ele)
{
  return Adapter::GetScaTraImplType(ele);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ScatraStructureCloneStrategy::check_material_type(const int matid)
{
  // We take the material with the ID specified by the user
  // Here we check first, whether this material is of admissible type
  Core::Materials::MaterialType mtype =
      Global::Problem::Instance()->Materials()->ParameterById(matid)->Type();
  if ((mtype != Core::Materials::m_scatra) && (mtype != Core::Materials::m_elchmat) &&
      (mtype != Core::Materials::m_electrode) && (mtype != Core::Materials::m_matlist) &&
      (mtype != Core::Materials::m_matlist_reactions) && (mtype != Core::Materials::m_myocard) &&
      (mtype != Core::Materials::m_thermostvenant))
    FOUR_C_THROW("Material with ID %d is not admissible for scalar transport elements", matid);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ScatraStructureCloneStrategy::set_element_data(
    Teuchos::RCP<Core::Elements::Element> newele, Core::Elements::Element* oldele, const int matid,
    const bool isnurbsdis)
{
  // We need to set material and possibly other things to complete element setup.
  // This is again really ugly as we have to extract the actual
  // element type in order to access the material property

  // note: SetMaterial() was reimplemented by the transport element!
  auto* trans = dynamic_cast<Discret::ELEMENTS::Transport*>(newele.get());
  if (trans != nullptr)
  {
    // set distype as well
    trans->SetDisType(oldele->Shape());

    // now check whether ImplType is reasonable and if set the ImplType
    Inpar::ScaTra::ImplType impltype = SSI::ScatraStructureCloneStrategy::GetImplType(oldele);
    if (impltype == Inpar::ScaTra::impltype_undefined)
    {
      FOUR_C_THROW(
          "ScatraStructureCloneStrategy copies scatra discretization from structure "
          "discretization, but the STRUCTURE elements that are defined in the .dat file are "
          "either not meant to be copied to scatra elements or the ImplType is set 'Undefined' "
          "which is not meaningful for the created scatra discretization! Use SOLIDSCATRA, "
          "WALLSCATRA, SHELLSCATRA or TRUSS3SCATRA elements with meaningful ImplType instead!");
    }
    else
      trans->SetImplType(impltype);

    // set material
    trans->SetMaterial(matid, oldele);
  }
  else
  {
    FOUR_C_THROW("unsupported element type '%s'", typeid(*newele).name());
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ScatraStructureCloneStrategyManifold::set_element_data(
    Teuchos::RCP<Core::Elements::Element> newele, Core::Elements::Element* oldele, const int matid,
    const bool isnurbsdis)
{
  // determine impl type from manifold condition by identifying the condition for this element
  auto struct_dis = Global::Problem::Instance()->GetDis("structure");

  std::vector<Core::Conditions::Condition*> conditions;
  struct_dis->GetCondition("SSISurfaceManifold", conditions);

  auto impltype = Inpar::ScaTra::impltype_undefined;
  for (auto* condition : conditions)
  {
    auto cond_eles = condition->Geometry();
    if (cond_eles.find(oldele->Id()) != cond_eles.end())
    {
      impltype = static_cast<Inpar::ScaTra::ImplType>(condition->parameters().get<int>("ImplType"));
      continue;
    }
  }

  if (impltype != Inpar::ScaTra::impltype_elch_electrode and
      impltype != Inpar::ScaTra::impltype_elch_diffcond and impltype != Inpar::ScaTra::impltype_std)
    FOUR_C_THROW("Scatra Impltype not supported for SSI with transport on manifolds");

  auto* trans = dynamic_cast<Discret::ELEMENTS::Transport*>(newele.get());
  if (trans == nullptr or oldele->ElementType().Name() != "StructuralSurfaceType")
    FOUR_C_THROW("element type not supported");

  // set distype
  trans->SetDisType(oldele->Shape());
  // set impltype
  trans->SetImplType(impltype);
  // set material
  trans->SetMaterial(matid, oldele);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool SSI::ScatraStructureCloneStrategy::determine_ele_type(
    Core::Elements::Element* actele, const bool ismyele, std::vector<std::string>& eletype)
{
  // note: ismyele, actele remain unused here! Used only for ALE creation

  // we only support transport elements here
  eletype.emplace_back("TRANSP");

  return true;  // yes, we copy EVERY element (no submeshes)
}

FOUR_C_NAMESPACE_CLOSE