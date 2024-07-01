/*----------------------------------------------------------------------*/
/*! \file

\brief utility functions for tsi problems

\level 2


*/


/*----------------------------------------------------------------------*
 | definitions                                               dano 12/09 |
 *----------------------------------------------------------------------*/
#include "4C_tsi_utils.hpp"

#include "4C_coupling_volmortar_utils.hpp"
#include "4C_fem_condition_periodic.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_dofset.hpp"
#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_fem_general_element_center.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_global_data.hpp"
#include "4C_so3_plast_ssn.hpp"
#include "4C_so3_thermo.hpp"
#include "4C_thermo_ele_impl_utils.hpp"
#include "4C_thermo_element.hpp"

#include <Epetra_MpiComm.h>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | remove flag thermo from condition                         dano 12/11 |
 *----------------------------------------------------------------------*/
std::map<std::string, std::string> TSI::UTILS::ThermoStructureCloneStrategy::conditions_to_copy()
    const
{
  return {{"ThermoDirichlet", "Dirichlet"}, {"ThermoPointNeumann", "PointNeumann"},
      {"ThermoLineNeumann", "LineNeumann"}, {"ThermoSurfaceNeumann", "SurfaceNeumann"},
      {"ThermoVolumeNeumann", "VolumeNeumann"}, {"ThermoConvections", "ThermoConvections"},
      {"LinePeriodic", "LinePeriodic"}, {"SurfacePeriodic", "SurfacePeriodic"},
      {"ThermoInitfield", "Initfield"}, {"MortarMulti", "MortarMulti"}};
}


/*----------------------------------------------------------------------*
 | check material of cloned element                          dano 12/11 |
 *----------------------------------------------------------------------*/
void TSI::UTILS::ThermoStructureCloneStrategy::check_material_type(const int matid)
{
  // We take the material with the ID specified by the user
  // Here we check first, whether this material is of admissible type
  //  Core::Materials::MaterialType mtype =
  //  Global::Problem::Instance()->Materials()->ParameterById(matid)->Type(); if ((mtype !=
  //  Core::Materials::m_th_fourier_iso))
  //    FOUR_C_THROW("Material with ID %d is not admissible for thermo elements",matid);

}  // check_material_type()


/*----------------------------------------------------------------------*
 | set element data for cloned element                       dano 12/11 |
 *----------------------------------------------------------------------*/
void TSI::UTILS::ThermoStructureCloneStrategy::set_element_data(
    Teuchos::RCP<Core::Elements::Element> newele, Core::Elements::Element* oldele, const int matid,
    const bool isnurbs)
{
  // We need to set material and possibly other things to complete element setup.
  // This is again really ugly as we have to extract the actual
  // element type in order to access the material property

  // initialise kinematic type to geo_linear.
  // kintype is passed to the cloned thermo element
  Inpar::STR::KinemType kintype = Inpar::STR::KinemType::linear;
  // if oldele is a so3_base element or a so3_Plast element
  Discret::ELEMENTS::SoBase* so_base = dynamic_cast<Discret::ELEMENTS::SoBase*>(oldele);
  if (so_base != nullptr)
    kintype = so_base->KinematicType();
  else
    FOUR_C_THROW("oldele is neither a So_base element!");

  // note: SetMaterial() was reimplemented by the thermo element!

  Teuchos::RCP<Discret::ELEMENTS::Thermo> therm =
      Teuchos::rcp_dynamic_cast<Discret::ELEMENTS::Thermo>(newele);
  if (therm != Teuchos::null)
  {
    // cloning to same material id -> use the same material instance
    if (so_base->Material()->Parameter()->Id() == matid)
      therm->SetMaterial(0, so_base->Material());
    else
      therm->SetMaterial(0, Mat::Factory(matid));
    therm->SetDisType(oldele->Shape());  // set distype as well!
    therm->SetKinematicType(kintype);    // set kintype in cloned thermal element
  }
  else
  {
    FOUR_C_THROW("unsupported element type '%s'", typeid(*newele).name());
  }
  return;
}  // set_element_data()


/*----------------------------------------------------------------------*
 | cloned element has to be a THERMO element                 dano 12/11 |
 *----------------------------------------------------------------------*/
bool TSI::UTILS::ThermoStructureCloneStrategy::determine_ele_type(
    Core::Elements::Element* actele, const bool ismyele, std::vector<std::string>& eletype)
{
  // we only support thermo elements here
  eletype.push_back("THERMO");

  return true;  // yes, we copy EVERY element (no submeshes)
}  // determine_ele_type()


/*----------------------------------------------------------------------*
 | setup TSI                                                 dano 12/11 |
 *----------------------------------------------------------------------*/
void TSI::UTILS::SetupTSI(const Epetra_Comm& comm)
{
  // access the structure discretization, make sure it is filled
  Teuchos::RCP<Core::FE::Discretization> structdis;
  structdis = Global::Problem::Instance()->GetDis("structure");
  // set degrees of freedom in the discretization
  if (!structdis->Filled() or !structdis->HaveDofs())
  {
    structdis->fill_complete();
    Epetra_Map nc = *(structdis->NodeColMap());
    Epetra_Map nr = *(structdis->NodeRowMap());
    structdis->Redistribute(nr, nc);
  }

  // access the thermo discretization
  Teuchos::RCP<Core::FE::Discretization> thermdis;
  thermdis = Global::Problem::Instance()->GetDis("thermo");
  if (!thermdis->Filled()) thermdis->fill_complete();

  // access the problem-specific parameter list
  const Teuchos::ParameterList& tsidyn = Global::Problem::Instance()->TSIDynamicParams();

  bool matchinggrid = Core::UTILS::IntegralValue<bool>(tsidyn, "MATCHINGGRID");

  // we use the structure discretization as layout for the temperature discretization
  if (structdis->NumGlobalNodes() == 0) FOUR_C_THROW("Structure discretization is empty!");

  // create thermo elements if the temperature discretization is empty
  if (thermdis->NumGlobalNodes() == 0)
  {
    if (!matchinggrid)
      FOUR_C_THROW(
          "MATCHINGGRID is set to 'no' in TSI DYNAMIC section, but thermo discretization is "
          "empty!");

    Core::FE::CloneDiscretization<TSI::UTILS::ThermoStructureCloneStrategy>(
        structdis, thermdis, Global::Problem::Instance()->CloningMaterialMap());
    thermdis->fill_complete();

    // connect degrees of freedom for periodic boundary conditions
    {
      Core::Conditions::PeriodicBoundaryConditions pbc_struct(structdis);

      if (pbc_struct.HasPBC())
      {
        pbc_struct.update_dofs_for_periodic_boundary_conditions();
      }
    }

    // connect degrees of freedom for periodic boundary conditions
    {
      Core::Conditions::PeriodicBoundaryConditions pbc(thermdis);

      if (pbc.HasPBC())
      {
        pbc.update_dofs_for_periodic_boundary_conditions();
      }
    }

    // TSI must know the other discretization
    // build a proxy of the structure discretization for the temperature field
    Teuchos::RCP<Core::DOFSets::DofSetInterface> structdofset = structdis->GetDofSetProxy();
    // build a proxy of the temperature discretization for the structure field
    Teuchos::RCP<Core::DOFSets::DofSetInterface> thermodofset = thermdis->GetDofSetProxy();

    // check if ThermoField has 2 discretizations, so that coupling is possible
    if (thermdis->AddDofSet(structdofset) != 1) FOUR_C_THROW("unexpected dof sets in thermo field");
    if (structdis->AddDofSet(thermodofset) != 1)
      FOUR_C_THROW("unexpected dof sets in structure field");

    structdis->fill_complete(true, true, true);
    thermdis->fill_complete(true, true, true);

    TSI::UTILS::SetMaterialPointersMatchingGrid(structdis, thermdis);
  }
  else
  {
    if (matchinggrid)
      FOUR_C_THROW(
          "MATCHINGGRID is set to 'yes' in TSI DYNAMIC section, but thermo discretization is not "
          "empty!");

    // first call fill_complete for single discretizations.
    // This way the physical dofs are numbered successively
    structdis->fill_complete();
    thermdis->fill_complete();

    // build auxiliary dofsets, i.e. pseudo dofs on each discretization
    const int ndofpernode_thermo = 1;
    const int ndofperelement_thermo = 0;
    const int ndofpernode_struct = Global::Problem::Instance()->NDim();
    const int ndofperelement_struct = 0;
    Teuchos::RCP<Core::DOFSets::DofSetInterface> dofsetaux;
    dofsetaux = Teuchos::rcp(new Core::DOFSets::DofSetPredefinedDoFNumber(
        ndofpernode_thermo, ndofperelement_thermo, 0, true));
    if (structdis->AddDofSet(dofsetaux) != 1)
      FOUR_C_THROW("unexpected dof sets in structure field");
    dofsetaux = Teuchos::rcp(new Core::DOFSets::DofSetPredefinedDoFNumber(
        ndofpernode_struct, ndofperelement_struct, 0, true));
    if (thermdis->AddDofSet(dofsetaux) != 1) FOUR_C_THROW("unexpected dof sets in thermo field");

    // call assign_degrees_of_freedom also for auxiliary dofsets
    // note: the order of fill_complete() calls determines the gid numbering!
    // 1. structure dofs
    // 2. thermo dofs
    // 3. structure auxiliary dofs
    // 4. thermo auxiliary dofs
    structdis->fill_complete(true, false, false);
    thermdis->fill_complete(true, false, false);
  }

}  // SetupTSI()


/*----------------------------------------------------------------------*
 | print TSI-logo                                            dano 03/10 |
 *----------------------------------------------------------------------*/
void TSI::UTILS::SetMaterialPointersMatchingGrid(
    Teuchos::RCP<const Core::FE::Discretization> sourcedis,
    Teuchos::RCP<const Core::FE::Discretization> targetdis)
{
  const int numelements = targetdis->NumMyColElements();

  for (int i = 0; i < numelements; ++i)
  {
    Core::Elements::Element* targetele = targetdis->lColElement(i);
    const int gid = targetele->Id();

    Core::Elements::Element* sourceele = sourcedis->gElement(gid);

    // for coupling we add the source material to the target element and vice versa
    targetele->AddMaterial(sourceele->Material());
    sourceele->AddMaterial(targetele->Material());
  }
}

/*----------------------------------------------------------------------*
 |  assign material to discretization A                       vuong 09/14|
 *----------------------------------------------------------------------*/
void TSI::UTILS::TSIMaterialStrategy::AssignMaterial2To1(
    const Core::VolMortar::VolMortarCoupl* volmortar, Core::Elements::Element* ele1,
    const std::vector<int>& ids_2, Teuchos::RCP<Core::FE::Discretization> dis1,
    Teuchos::RCP<Core::FE::Discretization> dis2)
{
  // call default assignment
  Core::VolMortar::UTILS::DefaultMaterialStrategy::AssignMaterial2To1(
      volmortar, ele1, ids_2, dis1, dis2);

  // done
  return;
};


/*----------------------------------------------------------------------*
|  assign material to discretization B                       vuong 09/14|
 *----------------------------------------------------------------------*/
void TSI::UTILS::TSIMaterialStrategy::AssignMaterial1To2(
    const Core::VolMortar::VolMortarCoupl* volmortar, Core::Elements::Element* ele2,
    const std::vector<int>& ids_1, Teuchos::RCP<Core::FE::Discretization> dis1,
    Teuchos::RCP<Core::FE::Discretization> dis2)
{
  // if no corresponding element found -> leave
  if (ids_1.empty()) return;

  // call default assignment
  Core::VolMortar::UTILS::DefaultMaterialStrategy::AssignMaterial1To2(
      volmortar, ele2, ids_1, dis1, dis2);

  // initialise kinematic type to geo_linear.
  // kintype is passed to the corresponding thermo element
  Inpar::STR::KinemType kintype = Inpar::STR::KinemType::linear;

  // default strategy: take material of element with closest center in reference coordinates
  Core::Elements::Element* ele1 = nullptr;
  double mindistance = 1e10;
  {
    std::vector<double> centercoords2 = Core::FE::element_center_refe_coords(*ele2);

    for (unsigned i = 0; i < ids_1.size(); ++i)
    {
      Core::Elements::Element* actele1 = dis1->gElement(ids_1[i]);
      std::vector<double> centercoords1 = Core::FE::element_center_refe_coords(*actele1);

      Core::LinAlg::Matrix<3, 1> diffcoords(true);

      for (int j = 0; j < 3; ++j) diffcoords(j, 0) = centercoords1[j] - centercoords2[j];

      if (diffcoords.norm2() - mindistance < 1e-16)
      {
        mindistance = diffcoords.norm2();
        ele1 = actele1;
      }
    }
  }

  // if Aele is a so3_base element
  Discret::ELEMENTS::SoBase* so_base = dynamic_cast<Discret::ELEMENTS::SoBase*>(ele1);
  if (so_base != nullptr)
    kintype = so_base->KinematicType();
  else
    FOUR_C_THROW("ele1 is not a so3_thermo element!");

  Discret::ELEMENTS::Thermo* therm = dynamic_cast<Discret::ELEMENTS::Thermo*>(ele2);
  if (therm != nullptr)
  {
    therm->SetKinematicType(kintype);  // set kintype in cloned thermal element
  }

  // done
  return;
}


/*----------------------------------------------------------------------*
 | print TSI-logo                                            dano 03/10 |
 *----------------------------------------------------------------------*/
void TSI::printlogo()
{
  // more at http://www.ascii-art.de under entry "rockets"
  std::cout << "Welcome to Thermo-Structure-Interaction " << std::endl;
  std::cout << "         !\n"
            << "         !\n"
            << "         ^\n"
            << "        / \\\n"
            << "       /___\\\n"
            << "      |=   =|\n"
            << "      |     |\n"
            << "      |     |\n"
            << "      |     |\n"
            << "      |     |\n"
            << "      |     |\n"
            << "      | TSI |\n"
            << "      |     |\n"
            << "     /|##!##|\\\n"
            << "    / |##!##| \\\n"
            << "   /  |##!##|  \\\n"
            << "  |  / ^ | ^ \\  |\n"
            << "  | /  ( | )  \\ |\n"
            << "  |/   ( | )   \\|\n"
            << "      ((   ))\n"
            << "     ((  :  ))\n"
            << "     ((  :  ))\n"
            << "      ((   ))\n"
            << "       (( ))\n"
            << "        ( )\n"
            << "         .\n"
            << "         .\n"
            << "         .\n"
            << "\n"
            << std::endl;
}  // printlogo()


/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE