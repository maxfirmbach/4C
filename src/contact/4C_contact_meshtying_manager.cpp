/*----------------------------------------------------------------------*/
/*! \file
\level 1


\brief 4C implementation of main class to control all solid meshtying
*/
/*----------------------------------------------------------------------*/

#include "4C_contact_meshtying_manager.hpp"

#include "4C_contact_meshtying_defines.hpp"
#include "4C_contact_meshtying_lagrange_strategy.hpp"
#include "4C_contact_meshtying_penalty_strategy.hpp"
#include "4C_contact_meshtying_poro_lagrange_strategy.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_contact.hpp"
#include "4C_inpar_mortar.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_mortar_defines.hpp"
#include "4C_mortar_element.hpp"
#include "4C_mortar_interface.hpp"
#include "4C_mortar_node.hpp"
#include "4C_mortar_utils.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor (public)                                             popp 03/08|
 *----------------------------------------------------------------------*/
CONTACT::MtManager::MtManager(Core::FE::Discretization& discret, double alphaf)
    : Mortar::ManagerBase()
{
  // overwrite base class communicator
  comm_ = Teuchos::rcp(discret.Comm().Clone());

  // create some local variables (later to be stored in strategy)
  const int spatialDim = Global::Problem::Instance()->NDim();
  if (spatialDim != 2 && spatialDim != 3) FOUR_C_THROW("Meshtying problem must be 2D or 3D.");

  std::vector<Teuchos::RCP<Mortar::Interface>> interfaces;
  Teuchos::ParameterList mtparams;

  // read and check meshtying input parameters
  if (Comm().MyPID() == 0)
    std::cout << "Checking meshtying input parameters..........." << std::endl;

  read_and_check_input(mtparams, discret);
  if (Comm().MyPID() == 0) std::cout << "done!" << std::endl;

  // check for fill_complete of discretization
  if (!discret.Filled()) FOUR_C_THROW("discretization of underlying problem is not fillcomplete.");

  // let's check for meshtying boundary conditions in discret
  // and detect groups of matching conditions
  // for each group, create a contact interface and store it
  if (Comm().MyPID() == 0)
    std::cout << "Building meshtying interface(s)..............." << std::endl;

  std::vector<Core::Conditions::Condition*> contactconditions(0);
  discret.GetCondition("Mortar", contactconditions);

  // there must be more than one meshtying condition
  if (contactconditions.size() < 2) FOUR_C_THROW("Not enough contact conditions in discretization");

  // find all pairs of matching meshtying conditions
  // there is a maximum of (conditions / 2) groups
  std::vector<int> foundgroups(0);
  int numgroupsfound = 0;

  // get nurbs information
  const bool nurbs = mtparams.get<bool>("NURBS");

  // maximum dof number in discretization
  // later we want to create NEW Lagrange multiplier degrees of
  // freedom, which of course must not overlap with displacement dofs
  const int maxdof = discret.dof_row_map()->MaxAllGID();

  for (unsigned i = 0; i < contactconditions.size(); ++i)
  {
    // initialize vector for current group of conditions and temp condition
    std::vector<Core::Conditions::Condition*> currentgroup(0);
    Core::Conditions::Condition* tempcond = nullptr;

    // try to build meshtying group around this condition
    currentgroup.push_back(contactconditions[i]);
    const auto groupid1 = currentgroup[0]->parameters().get<int>("Interface ID");
    bool foundit = false;

    for (unsigned j = 0; j < contactconditions.size(); ++j)
    {
      if (j == i) continue;  // do not detect contactconditions[i] again
      tempcond = contactconditions[j];
      const auto groupid2 = tempcond->parameters().get<int>("Interface ID");
      if (groupid1 != groupid2) continue;  // not in the group
      foundit = true;                      // found a group entry
      currentgroup.push_back(tempcond);    // store it in currentgroup
    }

    // now we should have found a group of conds
    if (!foundit) FOUR_C_THROW("Cannot find matching contact condition for id %d", groupid1);

    // see whether we found this group before
    bool foundbefore = false;
    for (int j = 0; j < numgroupsfound; ++j)
    {
      if (groupid1 == foundgroups[j])
      {
        foundbefore = true;
        break;
      }
    }

    // if we have processed this group before, do nothing
    if (foundbefore) continue;

    // we have not found this group before, process it
    foundgroups.push_back(groupid1);
    ++numgroupsfound;

    // find out which sides are Master and Slave
    bool hasslave = false;
    bool hasmaster = false;
    std::vector<const std::string*> sides(currentgroup.size());
    std::vector<bool> isslave(currentgroup.size());

    for (unsigned j = 0; j < sides.size(); ++j)
    {
      sides[j] = &currentgroup[j]->parameters().get<std::string>("Side");
      if (*sides[j] == "Slave")
      {
        hasslave = true;
        isslave[j] = true;
      }
      else if (*sides[j] == "Master")
      {
        hasmaster = true;
        isslave[j] = false;
      }
      else
      {
        FOUR_C_THROW("MtManager: Unknown mortar side qualifier!");
      }
    }

    if (!hasslave) FOUR_C_THROW("Slave side missing in contact condition group!");
    if (!hasmaster) FOUR_C_THROW("Master side missing in contact condition group!");

    // find out which sides are initialized as Active
    std::vector<const std::string*> active(currentgroup.size());
    std::vector<bool> isactive(currentgroup.size());

    for (unsigned j = 0; j < sides.size(); ++j)
    {
      active[j] = &currentgroup[j]->parameters().get<std::string>("Initialization");
      if (*sides[j] == "Slave")
      {
        // slave sides must be initialized as "Active"
        if (*active[j] == "Active")
          isactive[j] = true;
        else if (*active[j] == "Inactive")
          FOUR_C_THROW(" Slave side must be active for meshtying!");
        else
          FOUR_C_THROW(
              "Unknown initialization qualifier for slave side of mortar meshtying interface!");
      }
      else if (*sides[j] == "Master")
      {
        // master sides must NOT be initialized as "Active" as this makes no sense
        if (*active[j] == "Active")
          FOUR_C_THROW("Master side cannot be active!");
        else if (*active[j] == "Inactive")
          isactive[j] = false;
        else
          FOUR_C_THROW(
              "Unknown initialization qualifier for master side of mortar meshtying interface!");
      }
      else
      {
        FOUR_C_THROW("MtManager: Unknown contact side qualifier!");
      }
    }

    // create an empty meshtying interface and store it in this Manager
    interfaces.push_back(Mortar::Interface::Create(groupid1, Comm(), spatialDim, mtparams));

    // get it again
    Teuchos::RCP<Mortar::Interface> interface = interfaces.back();

    // note that the nodal IDs are unique because they come from
    // one global problem discretization containing all nodes of the
    // contact interface
    // We rely on this fact, therefore it is not possible to
    // do meshtying between two distinct discretizations here

    //-------------------------------------------------- process nodes
    for (unsigned j = 0; j < currentgroup.size(); ++j)
    {
      // get all nodes and add them
      const std::vector<int>* nodeids = currentgroup[j]->GetNodes();
      if (!nodeids) FOUR_C_THROW("Condition does not have Node Ids");
      for (unsigned k = 0; k < nodeids->size(); ++k)
      {
        int gid = (*nodeids)[k];
        // do only nodes that I have in my discretization
        if (!discret.NodeColMap()->MyGID(gid)) continue;
        Core::Nodes::Node* node = discret.gNode(gid);
        if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);

        // create Node object
        Teuchos::RCP<Mortar::Node> mtnode = Teuchos::rcp(new Mortar::Node(
            node->Id(), node->X(), node->Owner(), discret.Dof(0, node), isslave[j]));
        //-------------------
        // get nurbs weight!
        if (nurbs) Mortar::UTILS::prepare_nurbs_node(node, mtnode);

        // get edge and corner information:
        std::vector<Core::Conditions::Condition*> contactcornercond(0);
        discret.GetCondition("mrtrcorner", contactcornercond);
        for (unsigned j = 0; j < contactcornercond.size(); j++)
        {
          if (contactcornercond.at(j)->ContainsNode(node->Id()))
          {
            mtnode->SetOnCorner() = true;
          }
        }
        std::vector<Core::Conditions::Condition*> contactedgecond(0);
        discret.GetCondition("mrtredge", contactedgecond);
        for (unsigned j = 0; j < contactedgecond.size(); j++)
        {
          if (contactedgecond.at(j)->ContainsNode(node->Id()))
          {
            mtnode->SetOnEdge() = true;
          }
        }

        // Check, if this node (and, in case, which dofs) are in the contact symmetry condition
        std::vector<Core::Conditions::Condition*> contactSymconditions(0);
        discret.GetCondition("mrtrsym", contactSymconditions);

        for (unsigned j = 0; j < contactSymconditions.size(); j++)
          if (contactSymconditions.at(j)->ContainsNode(node->Id()))
          {
            const std::vector<int>& onoff =
                contactSymconditions.at(j)->parameters().get<std::vector<int>>("onoff");
            for (unsigned k = 0; k < onoff.size(); k++)
              if (onoff.at(k) == 1) mtnode->DbcDofs()[k] = true;
          }

        // note that we do not have to worry about double entries
        // as the AddNode function can deal with this case!
        interface->AddMortarNode(mtnode);
      }
    }

    //----------------------------------------------- process elements
    int ggsize = 0;
    for (unsigned j = 0; j < currentgroup.size(); ++j)
    {
      // get elements from condition j of current group
      std::map<int, Teuchos::RCP<Core::Elements::Element>>& currele = currentgroup[j]->Geometry();

      // elements in a boundary condition have a unique id
      // but ids are not unique among 2 distinct conditions
      // due to the way elements in conditions are build.
      // We therefore have to give the second, third,... set of elements
      // different ids. ids do not have to be continous, we just add a large
      // enough number ggsize to all elements of cond2, cond3,... so they are
      // different from those in cond1!!!
      // note that elements in ele1/ele2 already are in column (overlapping) map
      int lsize = static_cast<int>(currele.size());
      int gsize = 0;
      Comm().SumAll(&lsize, &gsize, 1);


      for (const auto& element : currele)
      {
        Teuchos::RCP<Core::Elements::Element> ele = element.second;
        Teuchos::RCP<Mortar::Element> mtele = Teuchos::rcp(new Mortar::Element(ele->Id() + ggsize,
            ele->Owner(), ele->Shape(), ele->num_node(), ele->NodeIds(), isslave[j], nurbs));
        //------------------------------------------------------------------
        // get knotvector, normal factor and zero-size information for nurbs
        if (nurbs) Mortar::UTILS::prepare_nurbs_element(discret, ele, mtele, spatialDim);

        interface->AddMortarElement(mtele);
      }

      ggsize += gsize;  // update global element counter
    }

    /* -------------------- finalize the meshtying interface construction
     *
     * If this is the final parallel distribution, we need to assign degrees of freedom during
     * during fill_complete(). If parallel redistribution is enabled, there will be another call to
     * fill_complete(), so we skip this expensive operation here and do it later. DOFs have to be
     * assigned only once!
     */
    {
      const Inpar::Mortar::ParallelRedist parallelRedist =
          Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(
              mtparams.sublist("PARALLEL REDISTRIBUTION"), "PARALLEL_REDIST");
      bool isFinalDistribution = false;
      if (parallelRedist == Inpar::Mortar::ParallelRedist::redist_none or comm_->NumProc() == 1)
        isFinalDistribution = true;

      interface->fill_complete(isFinalDistribution, maxdof);
    }
  }
  if (Comm().MyPID() == 0) std::cout << "done!" << std::endl;

  //**********************************************************************
  // create the solver strategy object
  // and pass all necessary data to it
  if (Comm().MyPID() == 0)
    std::cout << "Building meshtying strategy object............" << std::endl;

  const Core::ProblemType problemtype = Global::Problem::Instance()->GetProblemType();

  Inpar::CONTACT::SolvingStrategy stype =
      Core::UTILS::IntegralValue<Inpar::CONTACT::SolvingStrategy>(mtparams, "STRATEGY");
  if (stype == Inpar::CONTACT::solution_lagmult)
  {
    // finally we should use another criteria to decide which strategy
    if (problemtype != Core::ProblemType::poroelast && problemtype != Core::ProblemType::fpsi &&
        problemtype != Core::ProblemType::fpsi_xfem && problemtype != Core::ProblemType::fps3i)
    {
      strategy_ = Teuchos::rcp(new MtLagrangeStrategy(discret.dof_row_map(), discret.NodeRowMap(),
          mtparams, interfaces, spatialDim, comm_, alphaf, maxdof));
    }
    else
    {
      strategy_ = Teuchos::rcp(new PoroMtLagrangeStrategy(discret.dof_row_map(),
          discret.NodeRowMap(), mtparams, interfaces, spatialDim, comm_, alphaf, maxdof));
    }
  }
  else if (stype == Inpar::CONTACT::solution_penalty or stype == Inpar::CONTACT::solution_uzawa)
    strategy_ = Teuchos::rcp(new MtPenaltyStrategy(discret.dof_row_map(), discret.NodeRowMap(),
        mtparams, interfaces, spatialDim, comm_, alphaf, maxdof));
  else
    FOUR_C_THROW("Unrecognized strategy");

  if (Comm().MyPID() == 0) std::cout << "done!" << std::endl;
  //**********************************************************************

  //**********************************************************************
  // parallel redistribution of all interfaces
  GetStrategy().redistribute_meshtying();
  //**********************************************************************

  // create binary search tree
  for (auto& interface : interfaces) interface->CreateSearchTree();

  return;
}


/*----------------------------------------------------------------------*
 |  read and check input parameters (public)                  popp 04/08|
 *----------------------------------------------------------------------*/
bool CONTACT::MtManager::read_and_check_input(
    Teuchos::ParameterList& mtparams, const Core::FE::Discretization& discret)
{
  // read parameter lists from Global::Problem
  const Teuchos::ParameterList& mortar = Global::Problem::Instance()->mortar_coupling_params();
  const Teuchos::ParameterList& meshtying = Global::Problem::Instance()->contact_dynamic_params();
  const Teuchos::ParameterList& wearlist = Global::Problem::Instance()->WearParams();

  // read Problem Type and Problem Dimension from Global::Problem
  const Core::ProblemType problemtype = Global::Problem::Instance()->GetProblemType();
  const int spatialDim = Global::Problem::Instance()->NDim();
  Core::FE::ShapeFunctionType distype = Global::Problem::Instance()->spatial_approximation_type();

  // get mortar information
  std::vector<Core::Conditions::Condition*> mtcond(0);
  std::vector<Core::Conditions::Condition*> ccond(0);

  discret.GetCondition("Mortar", mtcond);
  discret.GetCondition("Contact", ccond);

  bool onlymeshtying = false;
  bool meshtyingandcontact = false;

  // check for case
  if (mtcond.size() != 0 and ccond.size() != 0) meshtyingandcontact = true;

  if (mtcond.size() != 0 and ccond.size() == 0) onlymeshtying = true;

  // *********************************************************************
  // invalid parallel strategies
  // *********************************************************************
  const Teuchos::ParameterList& mortarParallelRedistParams =
      mortar.sublist("PARALLEL REDISTRIBUTION");

  if (Teuchos::getIntegralValue<Inpar::Mortar::ExtendGhosting>(mortarParallelRedistParams,
          "GHOSTING_STRATEGY") == Inpar::Mortar::ExtendGhosting::roundrobin)
    FOUR_C_THROW(
        "Extending the ghosting via a Round-Robin loop is not implemented for mortar meshtying.");

  // *********************************************************************
  // invalid parameter combinations
  // *********************************************************************
  if (Core::UTILS::IntegralValue<Inpar::CONTACT::SolvingStrategy>(meshtying, "STRATEGY") ==
          Inpar::CONTACT::solution_penalty &&
      meshtying.get<double>("PENALTYPARAM") <= 0.0)
    FOUR_C_THROW("Penalty parameter eps <= 0, must be greater than 0");

  if (Core::UTILS::IntegralValue<Inpar::CONTACT::SolvingStrategy>(meshtying, "STRATEGY") ==
          Inpar::CONTACT::solution_uzawa &&
      meshtying.get<double>("PENALTYPARAM") <= 0.0)
    FOUR_C_THROW("Penalty parameter eps <= 0, must be greater than 0");

  if (Core::UTILS::IntegralValue<Inpar::CONTACT::SolvingStrategy>(meshtying, "STRATEGY") ==
          Inpar::CONTACT::solution_uzawa &&
      meshtying.get<int>("UZAWAMAXSTEPS") < 2)
    FOUR_C_THROW("Maximum number of Uzawa / Augmentation steps must be at least 2");

  if (Core::UTILS::IntegralValue<Inpar::CONTACT::SolvingStrategy>(meshtying, "STRATEGY") ==
          Inpar::CONTACT::solution_uzawa &&
      meshtying.get<double>("UZAWACONSTRTOL") <= 0.0)
    FOUR_C_THROW("Constraint tolerance for Uzawa / Augmentation scheme must be greater than 0");

  if (onlymeshtying && Core::UTILS::IntegralValue<Inpar::CONTACT::FrictionType>(
                           meshtying, "FRICTION") != Inpar::CONTACT::friction_none)
    FOUR_C_THROW("Friction law supplied for mortar meshtying");

  if (Core::UTILS::IntegralValue<Inpar::CONTACT::SolvingStrategy>(meshtying, "STRATEGY") ==
          Inpar::CONTACT::solution_lagmult &&
      Core::UTILS::IntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") ==
          Inpar::Mortar::shape_standard &&
      (Core::UTILS::IntegralValue<Inpar::CONTACT::SystemType>(meshtying, "SYSTEM") ==
              Inpar::CONTACT::system_condensed ||
          Core::UTILS::IntegralValue<Inpar::CONTACT::SystemType>(meshtying, "SYSTEM") ==
              Inpar::CONTACT::system_condensed_lagmult))
    FOUR_C_THROW("Condensation of linear system only possible for dual Lagrange multipliers");

  if (Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(mortarParallelRedistParams,
          "PARALLEL_REDIST") == Inpar::Mortar::ParallelRedist::redist_dynamic and
      onlymeshtying)
    FOUR_C_THROW("Dynamic parallel redistribution not possible for meshtying");

  if (Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(mortarParallelRedistParams,
          "PARALLEL_REDIST") != Inpar::Mortar::ParallelRedist::redist_none &&
      mortarParallelRedistParams.get<int>("MIN_ELEPROC") < 0)
    FOUR_C_THROW(
        "ERROR: Minimum number of elements per processor for parallel redistribution must be >= 0");

  if (Core::UTILS::IntegralValue<Inpar::Mortar::ConsistentDualType>(mortar, "LM_DUAL_CONSISTENT") !=
          Inpar::Mortar::consistent_none &&
      Core::UTILS::IntegralValue<Inpar::CONTACT::SolvingStrategy>(meshtying, "STRATEGY") !=
          Inpar::CONTACT::solution_lagmult &&
      Core::UTILS::IntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") !=
          Inpar::Mortar::shape_standard)
    FOUR_C_THROW(
        "ERROR: Consistent dual shape functions in boundary elements only for Lagrange multiplier "
        "strategy.");

  if (Core::UTILS::IntegralValue<Inpar::Mortar::ConsistentDualType>(mortar, "LM_DUAL_CONSISTENT") !=
          Inpar::Mortar::consistent_none &&
      Core::UTILS::IntegralValue<Inpar::Mortar::IntType>(mortar, "INTTYPE") ==
          Inpar::Mortar::inttype_elements &&
      (Core::UTILS::IntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") ==
              Inpar::Mortar::shape_dual ||
          Core::UTILS::IntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") ==
              Inpar::Mortar::shape_petrovgalerkin))

    // *********************************************************************
    // not (yet) implemented combinations
    // *********************************************************************
    if (Core::UTILS::IntegralValue<int>(mortar, "CROSSPOINTS") == true && spatialDim == 3)
      FOUR_C_THROW("Crosspoints / edge node modification not yet implemented for 3D");

  if (Core::UTILS::IntegralValue<int>(mortar, "CROSSPOINTS") == true &&
      Core::UTILS::IntegralValue<Inpar::Mortar::LagMultQuad>(mortar, "LM_QUAD") ==
          Inpar::Mortar::lagmult_lin)
    FOUR_C_THROW("Crosspoints and linear LM interpolation for quadratic FE not yet compatible");

  if (Core::UTILS::IntegralValue<int>(mortar, "CROSSPOINTS") == true &&
      Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(mortarParallelRedistParams,
          "PARALLEL_REDIST") != Inpar::Mortar::ParallelRedist::redist_none)
    FOUR_C_THROW("Crosspoints and parallel redistribution not yet compatible");

  if (Core::UTILS::IntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") ==
          Inpar::Mortar::shape_petrovgalerkin and
      onlymeshtying)
    FOUR_C_THROW("Petrov-Galerkin approach makes no sense for meshtying");

  // *********************************************************************
  // 3D quadratic mortar (choice of interpolation and testing fcts.)
  // *********************************************************************
  if (Core::UTILS::IntegralValue<Inpar::Mortar::LagMultQuad>(mortar, "LM_QUAD") ==
          Inpar::Mortar::lagmult_pwlin &&
      Core::UTILS::IntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") ==
          Inpar::Mortar::shape_dual)
    FOUR_C_THROW(
        "ERROR: No pwlin approach (for LM) implemented for quadratic meshtying with DUAL shape "
        "fct.");

  // *********************************************************************
  // element-based vs. segment-based mortar integration
  // *********************************************************************
  Inpar::Mortar::IntType inttype =
      Core::UTILS::IntegralValue<Inpar::Mortar::IntType>(mortar, "INTTYPE");

  if (inttype == Inpar::Mortar::inttype_elements && mortar.get<int>("NUMGP_PER_DIM") <= 0)
    FOUR_C_THROW("Invalid Gauss point number NUMGP_PER_DIM for element-based integration.");

  if (inttype == Inpar::Mortar::inttype_elements_BS && mortar.get<int>("NUMGP_PER_DIM") <= 0)
    FOUR_C_THROW(
        "ERROR: Invalid Gauss point number NUMGP_PER_DIM for element-based integration with "
        "boundary segmentation."
        "\nPlease note that the value you have to provide only applies to the element-based "
        "integration"
        "\ndomain, while pre-defined default values will be used in the segment-based boundary "
        "domain.");

  if ((inttype == Inpar::Mortar::inttype_elements ||
          inttype == Inpar::Mortar::inttype_elements_BS) &&
      mortar.get<int>("NUMGP_PER_DIM") <= 1)
    FOUR_C_THROW("Invalid Gauss point number NUMGP_PER_DIM for element-based integration.");

  // *********************************************************************
  // warnings
  // *********************************************************************
  if (mortar.get<double>("SEARCH_PARAM") == 0.0 && Comm().MyPID() == 0)
    std::cout << ("Warning: Meshtying search called without inflation of bounding volumes\n")
              << std::endl;

  // get parameter lists
  mtparams.setParameters(mortar);
  mtparams.setParameters(meshtying);
  mtparams.setParameters(wearlist);

  // *********************************************************************
  // predefined params for meshtying and contact
  // *********************************************************************
  if (meshtyingandcontact)
  {
    // set options for mortar coupling
    mtparams.set<std::string>("SEARCH_ALGORITHM", "Binarytree");
    mtparams.set<double>("SEARCH_PARAM", 0.3);
    mtparams.set<std::string>("SEARCH_USE_AUX_POS", "no");
    mtparams.set<std::string>("LM_SHAPEFCN", "dual");
    mtparams.set<std::string>("SYSTEM", "condensed");
    mtparams.set<bool>("NURBS", false);
    mtparams.set<int>("NUMGP_PER_DIM", -1);
    mtparams.set<std::string>("STRATEGY", "LagrangianMultipliers");
    mtparams.set<std::string>("INTTYPE", "segments");
    mtparams.sublist("PARALLEL REDISTRIBUTION").set<std::string>("REDUNDANT_STORAGE", "Master");
    mtparams.sublist("PARALLEL REDISTRIBUTION").set<std::string>("PARALLEL_REDIST", "static");
  }
  // *********************************************************************
  // smooth interfaces
  // *********************************************************************
  // NURBS PROBLEM?
  switch (distype)
  {
    case Core::FE::ShapeFunctionType::nurbs:
    {
      mtparams.set<bool>("NURBS", true);
      break;
    }
    default:
    {
      mtparams.set<bool>("NURBS", false);
      break;
    }
  }

  // *********************************************************************
  // poroelastic meshtying
  // *********************************************************************
  if ((problemtype == Core::ProblemType::poroelast || problemtype == Core::ProblemType::fpsi ||
          problemtype == Core::ProblemType::fpsi_xfem) &&
      (Core::UTILS::IntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") !=
              Inpar::Mortar::shape_dual &&
          Core::UTILS::IntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") !=
              Inpar::Mortar::shape_petrovgalerkin))
    FOUR_C_THROW("POROCONTACT: Only dual and petrovgalerkin shape functions implemented yet!");

  if ((problemtype == Core::ProblemType::poroelast || problemtype == Core::ProblemType::fpsi ||
          problemtype == Core::ProblemType::fpsi_xfem) &&
      Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(mortarParallelRedistParams,
          "PARALLEL_REDIST") != Inpar::Mortar::ParallelRedist::redist_none)
    FOUR_C_THROW(
        "POROCONTACT: Parallel Redistribution not implemented yet!");  // Since we use Pointers to
                                                                       // Parent Elements, which are
                                                                       // not copied to other procs!

  if ((problemtype == Core::ProblemType::poroelast || problemtype == Core::ProblemType::fpsi ||
          problemtype == Core::ProblemType::fpsi_xfem) &&
      Core::UTILS::IntegralValue<Inpar::CONTACT::SolvingStrategy>(meshtying, "STRATEGY") !=
          Inpar::CONTACT::solution_lagmult)
    FOUR_C_THROW("POROCONTACT: Use Lagrangean Strategy for poro meshtying!");

  if ((problemtype == Core::ProblemType::poroelast || problemtype == Core::ProblemType::fpsi ||
          problemtype == Core::ProblemType::fpsi_xfem) &&
      Core::UTILS::IntegralValue<Inpar::CONTACT::SystemType>(meshtying, "SYSTEM") !=
          Inpar::CONTACT::system_condensed_lagmult)
    FOUR_C_THROW("POROCONTACT: Just lagrange multiplier should be condensed for poro meshtying!");

  if ((problemtype == Core::ProblemType::poroelast || problemtype == Core::ProblemType::fpsi ||
          problemtype == Core::ProblemType::fpsi_xfem) &&
      (spatialDim != 3) && (spatialDim != 2))
  {
    const Teuchos::ParameterList& porodyn = Global::Problem::Instance()->poroelast_dynamic_params();
    if (Core::UTILS::IntegralValue<int>(porodyn, "CONTACTNOPEN"))
      FOUR_C_THROW("POROCONTACT: PoroMeshtying with no penetration just tested for 3d (and 2d)!");
  }

  mtparams.setName("CONTACT DYNAMIC / MORTAR COUPLING");

  // no parallel redistribution in the serial case
  if (Comm().NumProc() == 1)
    mtparams.sublist("PARALLEL REDISTRIBUTION").set<std::string>("PARALLEL_REDIST", "None");

  return true;
}

/*----------------------------------------------------------------------*
 |  write restart information for meshtying (public)          popp 03/08|
 *----------------------------------------------------------------------*/
void CONTACT::MtManager::write_restart(Core::IO::DiscretizationWriter& output, bool forcedrestart)
{
  output.write_vector("mt_lagrmultold", GetStrategy().lagrange_multiplier_old());

  return;
}

/*----------------------------------------------------------------------*
 |  read restart information for meshtying (public)           popp 03/08|
 *----------------------------------------------------------------------*/
void CONTACT::MtManager::read_restart(Core::IO::DiscretizationReader& reader,
    Teuchos::RCP<Epetra_Vector> dis, Teuchos::RCP<Epetra_Vector> zero)
{
  // this is meshtying, thus we need zeros for restart
  // let strategy object do all the work
  GetStrategy().DoReadRestart(reader, zero);

  return;
}

/*----------------------------------------------------------------------*
 |  write interface tractions for postprocessing (public)     popp 03/08|
 *----------------------------------------------------------------------*/
void CONTACT::MtManager::postprocess_quantities(Core::IO::DiscretizationWriter& output)
{
  // evaluate interface tractions
  Teuchos::RCP<Epetra_Map> problem = GetStrategy().ProblemDofs();
  Teuchos::RCP<Epetra_Vector> traction =
      Teuchos::rcp(new Epetra_Vector(*(GetStrategy().lagrange_multiplier_old())));
  Teuchos::RCP<Epetra_Vector> tractionexp = Teuchos::rcp(new Epetra_Vector(*problem));
  Core::LinAlg::Export(*traction, *tractionexp);

  // evaluate slave and master forces
  Teuchos::RCP<Epetra_Vector> fcslave =
      Teuchos::rcp(new Epetra_Vector(GetStrategy().d_matrix()->RowMap()));
  Teuchos::RCP<Epetra_Vector> fcmaster =
      Teuchos::rcp(new Epetra_Vector(GetStrategy().m_matrix()->DomainMap()));
  Teuchos::RCP<Epetra_Vector> fcslaveexp = Teuchos::rcp(new Epetra_Vector(*problem));
  Teuchos::RCP<Epetra_Vector> fcmasterexp = Teuchos::rcp(new Epetra_Vector(*problem));
  GetStrategy().d_matrix()->Multiply(true, *traction, *fcslave);
  GetStrategy().m_matrix()->Multiply(true, *traction, *fcmaster);
  Core::LinAlg::Export(*fcslave, *fcslaveexp);
  Core::LinAlg::Export(*fcmaster, *fcmasterexp);

  // write to output
  output.write_vector("interfacetraction", tractionexp);
  output.write_vector("slaveforces", fcslaveexp);
  output.write_vector("masterforces", fcmasterexp);

  return;
}

/*-----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::MtManager::postprocess_quantities_per_interface(
    Teuchos::RCP<Teuchos::ParameterList> outputParams)
{
  GetStrategy().postprocess_quantities_per_interface(outputParams);
}

FOUR_C_NAMESPACE_CLOSE