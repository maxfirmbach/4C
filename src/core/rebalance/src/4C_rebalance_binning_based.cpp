/*---------------------------------------------------------------------*/
/*! \file

\brief A collection of functions for parallel redistribution

\level 3

*/
/*---------------------------------------------------------------------*/


#include "4C_rebalance_binning_based.hpp"

#include "4C_binstrategy.hpp"
#include "4C_coupling_matchingoctree.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_rebalance_print.hpp"
#include "4C_utils_exceptions.hpp"

#include <Epetra_IntVector.h>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Rebalance using BinningStrategy                         rauch 08/16 |
 *----------------------------------------------------------------------*/
void Core::Rebalance::RebalanceDiscretizationsByBinning(
    const Teuchos::ParameterList& binning_params,
    Teuchos::RCP<Core::IO::OutputControl> output_control,
    const std::vector<Teuchos::RCP<Core::FE::Discretization>>& vector_of_discretizations,
    std::function<Core::Binstrategy::Utils::SpecialElement(const Core::Elements::Element* element)>
        element_filter,
    std::function<double(const Core::Elements::Element* element)> rigid_sphere_radius,
    std::function<Core::Nodes::Node const*(Core::Nodes::Node const* node)> correct_beam_center_node,
    bool revertextendedghosting)
{
  // safety check
  if (vector_of_discretizations.size() == 0)
    FOUR_C_THROW("No discretizations provided for binning !");

  // get communicator
  const Epetra_Comm& comm = vector_of_discretizations[0]->Comm();

  // rebalance discr. with help of binning strategy
  if (comm.NumProc() > 1)
  {
    if (comm.MyPID() == 0)
    {
      Core::IO::cout(Core::IO::verbose)
          << "+---------------------------------------------------------------" << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "| Rebalance discretizations using Binning Strategy ...          " << Core::IO::endl;
      for (const auto& curr_dis : vector_of_discretizations)
      {
        if (!curr_dis->Filled()) FOUR_C_THROW("fill_complete(false,false,false) was not called");
        Core::IO::cout(Core::IO::verbose)
            << "| Rebalance discretization " << std::setw(11) << curr_dis->Name() << Core::IO::endl;
      }
      Core::IO::cout(Core::IO::verbose)
          << "+---------------------------------------------------------------" << Core::IO::endl;
    }

    std::vector<Teuchos::RCP<Epetra_Map>> stdelecolmap;
    std::vector<Teuchos::RCP<Epetra_Map>> stdnodecolmap;

    // binning strategy is created and parallel redistribution is performed
    auto binningstrategy = Teuchos::rcp(new Core::Binstrategy::BinningStrategy(binning_params,
        output_control, vector_of_discretizations[0]->Comm(),
        vector_of_discretizations[0]->Comm().MyPID(), element_filter, rigid_sphere_radius,
        correct_beam_center_node, vector_of_discretizations));

    binningstrategy
        ->do_weighted_partitioning_of_bins_and_extend_ghosting_of_discret_to_one_bin_layer(
            vector_of_discretizations, stdelecolmap, stdnodecolmap);

    // revert extended ghosting if requested
    if (revertextendedghosting)
      binningstrategy->revert_extended_ghosting(
          vector_of_discretizations, stdelecolmap, stdnodecolmap);
  }  // if more than 1 proc
  else
    for (const auto& curr_dis : vector_of_discretizations) curr_dis->fill_complete();

}  // Core::Rebalance::RebalanceDiscretizationsByBinning

/*----------------------------------------------------------------------*
 |  Ghost input discr. redundantly on all procs             rauch 09/16 |
 *----------------------------------------------------------------------*/
void Core::Rebalance::GhostDiscretizationOnAllProcs(
    const Teuchos::RCP<Core::FE::Discretization> distobeghosted)
{
  // clone communicator of target discretization
  Teuchos::RCP<Epetra_Comm> com = Teuchos::rcp(distobeghosted->Comm().Clone());

  if (com->MyPID() == 0)
  {
    Core::IO::cout(Core::IO::verbose)
        << "+-----------------------------------------------------------------------+"
        << Core::IO::endl;
    Core::IO::cout(Core::IO::verbose)
        << "|   Ghost discretization " << std::setw(11) << distobeghosted->Name()
        << " redundantly on all procs ...       |" << Core::IO::endl;
    Core::IO::cout(Core::IO::verbose)
        << "+-----------------------------------------------------------------------+"
        << Core::IO::endl;
  }

  std::vector<int> allproc(com->NumProc());
  for (int i = 0; i < com->NumProc(); ++i) allproc[i] = i;

  // fill my own row node ids
  const Epetra_Map* noderowmap = distobeghosted->NodeRowMap();
  std::vector<int> sdata;
  for (int lid = 0; lid < noderowmap->NumMyElements(); ++lid)
  {
    int gid = noderowmap->GID(lid);
    sdata.push_back(gid);
  }

  // gather all master row node gids redundantly in rdata
  std::vector<int> rdata;
  Core::LinAlg::Gather<int>(sdata, rdata, (int)allproc.size(), allproc.data(), *com);

  // build new node column map (on ALL processors)
  Teuchos::RCP<Epetra_Map> newnodecolmap =
      Teuchos::rcp(new Epetra_Map(-1, (int)rdata.size(), rdata.data(), 0, *com));
  sdata.clear();
  rdata.clear();

  // fill my own row element ids
  const Epetra_Map* elerowmap = distobeghosted->ElementRowMap();
  sdata.resize(0);
  for (int i = 0; i < elerowmap->NumMyElements(); ++i)
  {
    int gid = elerowmap->GID(i);
    sdata.push_back(gid);
  }

  // gather all gids of elements redundantly
  rdata.resize(0);
  Core::LinAlg::Gather<int>(sdata, rdata, (int)allproc.size(), allproc.data(), *com);

  // build new element column map (on ALL processors)
  Teuchos::RCP<Epetra_Map> newelecolmap =
      Teuchos::rcp(new Epetra_Map(-1, (int)rdata.size(), rdata.data(), 0, *com));
  sdata.clear();
  rdata.clear();
  allproc.clear();

  // rebalance the nodes and elements of the discr. according to the
  // new node / element column layout (i.e. master = full overlap)
  distobeghosted->ExportColumnNodes(*newnodecolmap);
  distobeghosted->export_column_elements(*newelecolmap);

  // Safety checks in DEBUG
#ifdef FOUR_C_ENABLE_ASSERTIONS
  int nummycolnodes = newnodecolmap->NumMyElements();
  std::vector<int> sizelist(com->NumProc());
  com->GatherAll(&nummycolnodes, sizelist.data(), 1);
  com->Barrier();
  for (int k = 1; k < com->NumProc(); ++k)
  {
    if (sizelist[k - 1] != nummycolnodes)
      FOUR_C_THROW(
          "Since whole dis. is ghosted every processor should have the same number of colnodes.\n"
          "This is not the case."
          "Fix this!");
  }
#endif
}  // Core::Rebalance::GhostDiscretizationOnAllProcs

/*---------------------------------------------------------------------*
|  Rebalance Nodes Matching Template discretization     rauch 09/16    |
*----------------------------------------------------------------------*/
void Core::Rebalance::MatchNodalDistributionOfMatchingDiscretizations(
    Core::FE::Discretization& dis_template, Core::FE::Discretization& dis_to_rebalance)
{
  // clone communicator of target discretization
  Teuchos::RCP<Epetra_Comm> com = Teuchos::rcp(dis_template.Comm().Clone());
  if (com->NumProc() > 1)
  {
    // print to screen
    if (com->MyPID() == 0)
    {
      Core::IO::cout(Core::IO::verbose)
          << "+---------------------------------------------------------------------------+"
          << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "|   Match nodal distribution of discr. " << std::setw(11) << dis_to_rebalance.Name()
          << "to discr. " << std::setw(11) << dis_template.Name() << " ... |" << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "+---------------------------------------------------------------------------+"
          << Core::IO::endl;
    }

    ////////////////////////////////////////
    // MATCH NODES
    ////////////////////////////////////////
    std::vector<int> rebalance_nodegid_vec(0);
    std::vector<int> rebalance_colnodegid_vec(0);

    // match nodes to be rebalanced to template nodes and fill vectors
    // with desired row and col gids for redistribution.
    MatchNodalRowColDistribution(
        dis_template, dis_to_rebalance, rebalance_nodegid_vec, rebalance_colnodegid_vec);

    // construct rebalanced node row map
    Teuchos::RCP<Epetra_Map> rebalanced_noderowmap = Teuchos::rcp(
        new Epetra_Map(-1, rebalance_nodegid_vec.size(), rebalance_nodegid_vec.data(), 0, *com));

    // construct rebalanced node col map
    Teuchos::RCP<Epetra_Map> rebalanced_nodecolmap = Teuchos::rcp(new Epetra_Map(
        -1, rebalance_colnodegid_vec.size(), rebalance_colnodegid_vec.data(), 0, *com));

    // we finally rebalance.
    // fill_complete(...) inside.
    dis_to_rebalance.Redistribute(*rebalanced_noderowmap, *rebalanced_nodecolmap,
        false,  // assigndegreesoffreedom
        false,  // initelements
        false,  // doboundaryconditions
        true,   // killdofs
        true    // killcond
    );

    // print to screen
    Core::Rebalance::UTILS::print_parallel_distribution(dis_to_rebalance);
  }  // if more than one proc
}  // MatchDistributionOfMatchingDiscretizations


/*---------------------------------------------------------------------*
|  Rebalance Elements Matching Template discretization     rauch 09/16 |
*----------------------------------------------------------------------*/
void Core::Rebalance::MatchElementDistributionOfMatchingDiscretizations(
    Core::FE::Discretization& dis_template, Core::FE::Discretization& dis_to_rebalance)
{
  // clone communicator of target discretization
  Teuchos::RCP<Epetra_Comm> com = Teuchos::rcp(dis_template.Comm().Clone());
  if (com->NumProc() > 1)
  {
    // print to screen
    if (com->MyPID() == 0)
    {
      Core::IO::cout(Core::IO::verbose)
          << "+-----------------------------------------------------------------------------+"
          << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "|   Match element distribution of discr. " << std::setw(11) << dis_to_rebalance.Name()
          << "to discr. " << std::setw(11) << dis_template.Name() << " ... |" << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "+-----------------------------------------------------------------------------+"
          << Core::IO::endl;
    }

    ////////////////////////////////////////
    // MATCH ELEMENTS
    ////////////////////////////////////////
    std::vector<int> rebalance_rowelegid_vec(0);
    std::vector<int> rebalance_colelegid_vec(0);

    // match elements to be rebalanced to template elements and fill vectors
    // with desired row and col gids for redistribution.
    MatchElementRowColDistribution(
        dis_template, dis_to_rebalance, rebalance_rowelegid_vec, rebalance_colelegid_vec);

    // construct rebalanced element row map
    Teuchos::RCP<Epetra_Map> rebalanced_elerowmap = Teuchos::rcp(new Epetra_Map(
        -1, rebalance_rowelegid_vec.size(), rebalance_rowelegid_vec.data(), 0, *com));

    // construct rebalanced element col map
    Teuchos::RCP<Epetra_Map> rebalanced_elecolmap = Teuchos::rcp(new Epetra_Map(
        -1, rebalance_colelegid_vec.size(), rebalance_colelegid_vec.data(), 0, *com));

    ////////////////////////////////////////
    // MATCH NODES
    ////////////////////////////////////////
    std::vector<int> rebalance_nodegid_vec(0);
    std::vector<int> rebalance_colnodegid_vec(0);

    // match nodes to be rebalanced to template nodes and fill vectors
    // with desired row and col gids for redistribution.
    MatchNodalRowColDistribution(
        dis_template, dis_to_rebalance, rebalance_nodegid_vec, rebalance_colnodegid_vec);

    // construct rebalanced node row map
    Teuchos::RCP<Epetra_Map> rebalanced_noderowmap = Teuchos::rcp(
        new Epetra_Map(-1, rebalance_nodegid_vec.size(), rebalance_nodegid_vec.data(), 0, *com));

    // construct rebalanced node col map
    Teuchos::RCP<Epetra_Map> rebalanced_nodecolmap = Teuchos::rcp(new Epetra_Map(
        -1, rebalance_colnodegid_vec.size(), rebalance_colnodegid_vec.data(), 0, *com));

    ////////////////////////////////////////
    // REBALANCE
    ////////////////////////////////////////
    // export the nodes
    dis_to_rebalance.ExportRowNodes(*rebalanced_noderowmap, false, false);
    dis_to_rebalance.ExportColumnNodes(*rebalanced_nodecolmap, false, false);
    // export the elements
    dis_to_rebalance.ExportRowElements(*rebalanced_elerowmap, false, false);
    dis_to_rebalance.export_column_elements(*rebalanced_elecolmap, false, false);

    ////////////////////////////////////////
    // FINISH
    ////////////////////////////////////////
    int err = dis_to_rebalance.fill_complete(false, false, false);

    if (err) FOUR_C_THROW("fill_complete() returned err=%d", err);

    // print to screen
    Core::Rebalance::UTILS::print_parallel_distribution(dis_to_rebalance);
  }  // if more than one proc
}  // Core::Rebalance::MatchElementDistributionOfMatchingDiscretizations


/*---------------------------------------------------------------------*
|  Rebalance Conditioned Elements Matching Template        rauch 10/16 |
*----------------------------------------------------------------------*/
void Core::Rebalance::MatchElementDistributionOfMatchingConditionedElements(
    Core::FE::Discretization& dis_template, Core::FE::Discretization& dis_to_rebalance,
    const std::string& condname_template, const std::string& condname_rebalance, const bool print)
{
  // clone communicator of target discretization
  Teuchos::RCP<Epetra_Comm> com = Teuchos::rcp(dis_template.Comm().Clone());
  if (com->NumProc() > 1)
  {
    // print to screen
    if (com->MyPID() == 0)
    {
      Core::IO::cout(Core::IO::verbose)
          << "+-----------------------------------------------------------------------------+"
          << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "|   Match element distribution of discr. " << std::setw(11) << dis_to_rebalance.Name()
          << "                          |" << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose) << "|   Condition : " << std::setw(35) << condname_rebalance
                                        << "                           |" << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "|   to template discr. " << std::setw(11) << dis_template.Name()
          << "                                            |" << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose) << "|   Condition : " << std::setw(35) << condname_template
                                        << "                           |" << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "+-----------------------------------------------------------------------------+"
          << Core::IO::endl;
    }

    // create vectors for element matching
    std::vector<int> my_template_colelegid_vec(0);
    std::vector<int> rebalance_rowelegid_vec(0);

    // create vectors for node matching
    std::vector<int> my_template_nodegid_vec(0);
    std::vector<int> rebalance_rownodegid_vec(0);
    std::vector<int> rebalance_colnodegid_vec(0);

    // geometry iterator
    std::map<int, Teuchos::RCP<Core::Elements::Element>>::iterator geom_it;

    // fill condition discretization by cloning scatra discretization
    Teuchos::RCP<Core::FE::Discretization> dis_from_template_condition;

    Teuchos::RCP<Core::FE::DiscretizationCreatorBase> discreator =
        Teuchos::rcp(new Core::FE::DiscretizationCreatorBase());
    std::vector<std::string> conditions_to_copy(0);
    dis_from_template_condition = discreator->create_matching_discretization_from_condition(
        dis_template,        ///< discretization with condition
        condname_template,   ///< name of the condition, by which the derived discretization is
                             ///< identified
        "aux_dis",           ///< name of the new discretization
        "TRANSP",            ///< name/type of the elements to be created
        conditions_to_copy,  ///< list of conditions that will be copied to the new discretization
        -1  ///< coupling id, only elements conditioned with this coupling id are considered
    );

    // get element col map of conditioned template dis
    const Epetra_Map* template_cond_dis_elecolmap = dis_from_template_condition->ElementColMap();
    // get element row map of dis to be rebalanced
    const Epetra_Map* rebalance_elerowmap = dis_to_rebalance.ElementRowMap();


    ////////////////////////////////////////
    // MATCH CONDITIONED ELEMENTS
    ////////////////////////////////////////
    // fill element gid vectors
    for (int lid = 0; lid < template_cond_dis_elecolmap->NumMyElements(); lid++)
      my_template_colelegid_vec.push_back(template_cond_dis_elecolmap->GID(lid));

    for (int lid = 0; lid < rebalance_elerowmap->NumMyElements(); lid++)
      rebalance_rowelegid_vec.push_back(rebalance_elerowmap->GID(lid));


    // initialize search tree for matching with template (source,master) elements
    auto elementmatchingtree = Core::COUPLING::ElementMatchingOctree();
    elementmatchingtree.init(*dis_from_template_condition, my_template_colelegid_vec, 150, 1e-06);
    elementmatchingtree.setup();

    // map that will be filled with matched elements.
    // mapping: redistr. ele gid to (template ele gid, dist.).
    // note: 'fill_slave_to_master_gid_mapping' loops over all
    //        template eles and finds corresponding redistr. eles.
    std::map<int, std::vector<double>> matched_ele_map;
    // match target (slave) elements to source (master) elements using octtree
    elementmatchingtree.fill_slave_to_master_gid_mapping(
        dis_to_rebalance, rebalance_rowelegid_vec, matched_ele_map);

    // now we have a map matching the geometry ids of slave elements
    // to the geometry id of master elements (always starting from 0).
    // for redistribution we need to translate the geometry ids to the
    // actual element gids.
    // fill vectors with row and col gids for new distribution
    std::vector<int> rebalance_colelegid_vec;
    rebalance_rowelegid_vec.clear();
    for (const auto& it : matched_ele_map)
    {
      // if this proc owns the template element we also want to own
      // the element of the rebalanced discretization.
      // we also want to own all nodes of this element.
      if (static_cast<int>((it.second)[2]) == 1) rebalance_rowelegid_vec.push_back(it.first);

      rebalance_colelegid_vec.push_back(it.first);
    }

    if (print)
    {
      dis_to_rebalance.Comm().Barrier();
      for (const auto& it : matched_ele_map)
      {
        std::cout << "ELEMENT : " << it.first << " ->  ( " << it.second[0] << ", " << it.second[1]
                  << ", " << it.second[2] << " )"
                  << " on PROC " << dis_to_rebalance.Comm().MyPID()
                  << " map size = " << matched_ele_map.size() << std::endl;
      }
    }


    ////////////////////////////////////////
    // ALSO APPEND UNCONDITIONED ELEMENTS
    ////////////////////////////////////////
    // add row elements
    for (int lid = 0; lid < dis_to_rebalance.ElementColMap()->NumMyElements(); lid++)
    {
      bool conditionedele = false;
      Core::Elements::Element* ele =
          dis_to_rebalance.gElement(dis_to_rebalance.ElementColMap()->GID(lid));
      Core::Nodes::Node** nodes = ele->Nodes();
      for (int node = 0; node < ele->num_node(); node++)
      {
        Core::Conditions::Condition* nodal_cond = nodes[node]->GetCondition(condname_rebalance);
        if (nodal_cond != nullptr)
        {
          conditionedele = true;
          break;
        }
      }  // loop over nodes

      if (not conditionedele)
      {
        // append unconditioned ele id to col gid vec
        rebalance_colelegid_vec.push_back(ele->Id());

        // append unconditioned ele id to row gid vec
        if (ele->Owner() == com->MyPID()) rebalance_rowelegid_vec.push_back(ele->Id());
      }

    }  // loop over col elements


    // construct rebalanced element row map
    Teuchos::RCP<Epetra_Map> rebalanced_elerowmap = Teuchos::rcp(new Epetra_Map(
        -1, rebalance_rowelegid_vec.size(), rebalance_rowelegid_vec.data(), 0, *com));

    // construct rebalanced element col map
    Teuchos::RCP<Epetra_Map> rebalanced_elecolmap = Teuchos::rcp(new Epetra_Map(
        -1, rebalance_colelegid_vec.size(), rebalance_colelegid_vec.data(), 0, *com));


    ////////////////////////////////////////
    // MATCH CONDITIONED NODES
    ////////////////////////////////////////
    // fill vector with processor local conditioned node gids for template dis
    for (int lid = 0; lid < dis_template.NodeColMap()->NumMyElements(); ++lid)
    {
      if (dis_template.gNode(dis_template.NodeColMap()->GID(lid))
              ->GetCondition(condname_template) != nullptr)
        my_template_nodegid_vec.push_back(dis_template.NodeColMap()->GID(lid));
    }

    // fill vec with processor local node gids of dis to be rebalanced
    std::vector<Core::Conditions::Condition*> rebalance_conds;
    dis_to_rebalance.GetCondition(condname_rebalance, rebalance_conds);

    for (auto* const rebalance_cond : rebalance_conds)
    {
      const std::vector<int>* rebalance_cond_nodes = rebalance_cond->GetNodes();
      for (int rebalance_cond_node : *rebalance_cond_nodes)
      {
        if (dis_to_rebalance.HaveGlobalNode(rebalance_cond_node))
          if (dis_to_rebalance.gNode(rebalance_cond_node)->Owner() == com->MyPID())
            rebalance_rownodegid_vec.push_back(rebalance_cond_node);
      }
    }

    // initialize search tree for matching with template (source) nodes
    auto nodematchingtree = Core::COUPLING::NodeMatchingOctree();
    nodematchingtree.init(dis_template, my_template_nodegid_vec, 150, 1e-06);
    nodematchingtree.setup();

    // map that will be filled with matched nodes.
    // mapping: redistr. node gid to (template node gid, dist.).
    // note: FindMatch loops over all template nodes
    //       and finds corresponding redistr. nodes.
    std::map<int, std::vector<double>> matched_node_map;
    // match target nodes to source nodes using octtree
    nodematchingtree.fill_slave_to_master_gid_mapping(
        dis_to_rebalance, rebalance_rownodegid_vec, matched_node_map);

    // fill vectors with row gids for new distribution
    rebalance_rownodegid_vec.clear();
    // std::vector<int> rebalance_colnodegid_vec;
    for (const auto& it : matched_node_map)
    {
      // if this proc owns the template node we also want to own
      // the node of the rebalanced discretization
      if (static_cast<int>((it.second)[2]) == 1) rebalance_rownodegid_vec.push_back(it.first);

      rebalance_colnodegid_vec.push_back(it.first);
    }
    if (print)
    {
      dis_to_rebalance.Comm().Barrier();
      for (const auto& it : matched_node_map)
      {
        std::cout << "NODE : " << it.first << " ->  ( " << it.second[0] << ", " << it.second[1]
                  << ", " << it.second[2] << " )"
                  << " on PROC " << dis_to_rebalance.Comm().MyPID()
                  << " map size = " << matched_node_map.size() << std::endl;
      }
    }


    ////////////////////////////////////////
    // ALSO APPEND UNCONDITIONED  NODES
    ////////////////////////////////////////
    // add row nodes
    for (int lid = 0; lid < dis_to_rebalance.NodeRowMap()->NumMyElements(); lid++)
    {
      Core::Conditions::Condition* testcond =
          dis_to_rebalance.gNode(dis_to_rebalance.NodeRowMap()->GID(lid))
              ->GetCondition(condname_rebalance);
      if (testcond == nullptr)
        rebalance_rownodegid_vec.push_back(
            dis_to_rebalance.gNode(dis_to_rebalance.NodeRowMap()->GID(lid))->Id());
    }
    // add col nodes
    for (int lid = 0; lid < dis_to_rebalance.NodeColMap()->NumMyElements(); lid++)
    {
      Core::Conditions::Condition* testcond =
          dis_to_rebalance.gNode(dis_to_rebalance.NodeColMap()->GID(lid))
              ->GetCondition(condname_rebalance);
      if (testcond == nullptr)
        rebalance_colnodegid_vec.push_back(
            dis_to_rebalance.gNode(dis_to_rebalance.NodeColMap()->GID(lid))->Id());
    }

    // construct rebalanced node row map
    Teuchos::RCP<Epetra_Map> rebalanced_noderowmap = Teuchos::rcp(new Epetra_Map(
        -1, rebalance_rownodegid_vec.size(), rebalance_rownodegid_vec.data(), 0, *com));

    // construct rebalanced node col map
    Teuchos::RCP<Epetra_Map> rebalanced_nodecolmap = Teuchos::rcp(new Epetra_Map(
        -1, rebalance_colnodegid_vec.size(), rebalance_colnodegid_vec.data(), 0, *com));


    ////////////////////////////////////////
    // REBALANCE
    ////////////////////////////////////////
    // export the nodes
    dis_to_rebalance.ExportRowNodes(*rebalanced_noderowmap, false, false);
    dis_to_rebalance.ExportColumnNodes(*rebalanced_nodecolmap, false, false);
    // export the elements
    dis_to_rebalance.ExportRowElements(*rebalanced_elerowmap, false, false);
    dis_to_rebalance.export_column_elements(*rebalanced_elecolmap, false, false);


    ////////////////////////////////////////
    // FINISH
    ////////////////////////////////////////
    int err = dis_to_rebalance.fill_complete(false, false, false);

    if (err) FOUR_C_THROW("fill_complete() returned err=%d", err);

    // print to screen
    Core::Rebalance::UTILS::print_parallel_distribution(dis_to_rebalance);

  }  // if more than one proc
}  // MatchElementDistributionOfMatchingConditionedElements

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> Core::Rebalance::GetColVersionOfRowVector(
    const Teuchos::RCP<const Core::FE::Discretization> dis,
    const Teuchos::RCP<const Epetra_Vector> state, const int nds)
{
  // note that this routine has the same functionality as set_state,
  // although here we do not store the new vector anywhere
  // maybe this routine can be used in set_state or become a member function of the discretization
  // class

  if (!dis->HaveDofs()) FOUR_C_THROW("fill_complete() was not called");
  const Epetra_Map* colmap = dis->DofColMap(nds);
  const Epetra_BlockMap& vecmap = state->Map();

  // if it's already in column map just set a reference
  // This is a rought test, but it might be ok at this place. It is an
  // error anyway to hand in a vector that is not related to our dof
  // maps.
  if (vecmap.PointSameAs(*colmap)) return state;
  // if it's not in column map export and allocate
  else
  {
    Teuchos::RCP<Epetra_Vector> tmp = Core::LinAlg::CreateVector(*colmap, false);
    Core::LinAlg::Export(*state, *tmp);
    return tmp;
  }
}  // GetColVersionOfRowVector

/*----------------------------------------------------------------------*
 |(private)                                                   tk 06/10  |
 |recompute nodecolmap of standard discretization to include all        |
 |nodes as of subdicretization                                          |
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Map> Core::Rebalance::ComputeNodeColMap(
    const Teuchos::RCP<Core::FE::Discretization>
        sourcedis,  ///< standard discretization we want to rebalance
    const Teuchos::RCP<Core::FE::Discretization> subdis  ///< subdiscretization prescribing ghosting
)
{
  const Epetra_Map* oldcolnodemap = sourcedis->NodeColMap();

  std::vector<int> mycolnodes(oldcolnodemap->NumMyElements());
  oldcolnodemap->MyGlobalElements(mycolnodes.data());
  for (int inode = 0; inode != subdis->NumMyColNodes(); ++inode)
  {
    const Core::Nodes::Node* newnode = subdis->lColNode(inode);
    const int gid = newnode->Id();
    if (!(sourcedis->HaveGlobalNode(gid)))
    {
      mycolnodes.push_back(gid);
    }
  }

  // now reconstruct the extended colmap
  Teuchos::RCP<Epetra_Map> newcolnodemap =
      Teuchos::rcp(new Epetra_Map(-1, mycolnodes.size(), mycolnodes.data(), 0, sourcedis->Comm()));
  return newcolnodemap;
}  // Core::Rebalance::ComputeNodeColMap

/*----------------------------------------------------------------------*
 *                                                          rauch 10/16 |
 *----------------------------------------------------------------------*/
void Core::Rebalance::MatchElementRowColDistribution(const Core::FE::Discretization& dis_template,
    const Core::FE::Discretization& dis_to_rebalance, std::vector<int>& row_id_vec_to_fill,
    std::vector<int>& col_id_vec_to_fill)
{
  // preliminary work
  const Epetra_Map* rebalance_elerowmap = dis_to_rebalance.ElementRowMap();
  const Epetra_Map* template_elecolmap = dis_template.ElementColMap();
  std::vector<int> my_template_elegid_vec(template_elecolmap->NumMyElements());
  std::vector<int> my_rebalance_elegid_vec(0);

  // fill vector with processor local ele gids for template dis
  for (int lid = 0; lid < template_elecolmap->NumMyElements(); ++lid)
    my_template_elegid_vec[lid] = template_elecolmap->GID(lid);

  // fill vec with processor local ele gids of dis to be rebalanced
  for (int lid = 0; lid < rebalance_elerowmap->NumMyElements(); ++lid)
    my_rebalance_elegid_vec.push_back(rebalance_elerowmap->GID(lid));

  // initialize search tree for matching with template (source,master) elements
  auto elementmatchingtree = Core::COUPLING::ElementMatchingOctree();
  elementmatchingtree.init(dis_template, my_template_elegid_vec, 150, 1e-07);
  elementmatchingtree.setup();

  // map that will be filled with matched elements.
  // mapping: redistr. ele gid to (template ele gid, dist.).
  // note: 'fill_slave_to_master_gid_mapping' loops over all
  //        template eles and finds corresponding redistr. eles.
  std::map<int, std::vector<double>> matched_ele_map;
  // match target (slave) nodes to source (master) nodes using octtree
  elementmatchingtree.fill_slave_to_master_gid_mapping(
      dis_to_rebalance, my_rebalance_elegid_vec, matched_ele_map);

  // declare iterator
  std::map<int, std::vector<double>>::iterator it;

  // fill vectors with row and col gids for new distribution
  for (it = matched_ele_map.begin(); it != matched_ele_map.end(); ++it)
  {
    // if this proc owns the template element we also want to own
    // the element of the rebalanced discretization.
    // we also want to own all nodes of this element.
    if (static_cast<int>((it->second)[2]) == 1) row_id_vec_to_fill.push_back(it->first);

    col_id_vec_to_fill.push_back(it->first);
  }
}  // Core::Rebalance::MatchElementRowColDistribution

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::Rebalance::MatchNodalRowColDistribution(const Core::FE::Discretization& dis_template,
    const Core::FE::Discretization& dis_to_rebalance, std::vector<int>& row_id_vec_to_fill,
    std::vector<int>& col_id_vec_to_fill)
{
  // temp sets
  std::set<int> temprowset;
  std::set<int> tempcolset;

  for (int& row_id_to_fill : row_id_vec_to_fill) temprowset.insert(row_id_to_fill);

  for (int& col_id_to_fill : col_id_vec_to_fill) tempcolset.insert(col_id_to_fill);

  // preliminary work
  const Epetra_Map* rebalance_noderowmap = dis_to_rebalance.NodeRowMap();
  const Epetra_Map* template_nodecolmap = dis_template.NodeColMap();
  std::vector<int> my_template_nodegid_vec(template_nodecolmap->NumMyElements());
  std::vector<int> my_rebalance_nodegid_vec(0);

  // fill vector with processor local node gids for template dis
  for (int lid = 0; lid < template_nodecolmap->NumMyElements(); ++lid)
    my_template_nodegid_vec[lid] = template_nodecolmap->GID(lid);

  // fill vec with processor local node gids of dis to be rebalanced
  for (int lid = 0; lid < rebalance_noderowmap->NumMyElements(); ++lid)
    my_rebalance_nodegid_vec.push_back(rebalance_noderowmap->GID(lid));

  // initialize search tree for matching with template (source) nodes
  auto nodematchingtree = Core::COUPLING::NodeMatchingOctree();
  nodematchingtree.init(dis_template, my_template_nodegid_vec, 150, 1e-07);
  nodematchingtree.setup();

  // map that will be filled with matched nodes.
  // mapping: redistr. node gid to (template node gid, dist.).
  // note: FindMatch loops over all template nodes
  //       and finds corresponding redistr. nodes.
  std::map<int, std::vector<double>> matched_node_map;
  // match target nodes to source nodes using octtree
  nodematchingtree.fill_slave_to_master_gid_mapping(
      dis_to_rebalance, my_rebalance_nodegid_vec, matched_node_map);

  // declare iterator
  std::map<int, std::vector<double>>::iterator it;

  // fill vectors with row gids for new distribution
  // std::vector<int> rebalance_colnodegid_vec;
  for (it = matched_node_map.begin(); it != matched_node_map.end(); ++it)
  {
    // if this proc owns the template node we also want to own
    // the node of the rebalanced discretization
    if (static_cast<int>((it->second)[2]) == 1) temprowset.insert(it->first);

    tempcolset.insert(it->first);
  }

  // assign temporary sets to vectors
  row_id_vec_to_fill.clear();
  row_id_vec_to_fill.reserve(temprowset.size());
  row_id_vec_to_fill.assign(temprowset.begin(), temprowset.end());

  col_id_vec_to_fill.clear();
  col_id_vec_to_fill.reserve(tempcolset.size());
  col_id_vec_to_fill.assign(tempcolset.begin(), tempcolset.end());
}  // Core::Rebalance::MatchNodalRowColDistribution

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Map> Core::Rebalance::RebalanceInAccordanceWithReference(
    const Epetra_Map& ref_red_map, const Epetra_Map& unred_map)
{
  Teuchos::RCP<Epetra_Map> red_map = Teuchos::null;
  RebalanceInAccordanceWithReference(ref_red_map, unred_map, red_map);
  return red_map;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::Rebalance::RebalanceInAccordanceWithReference(
    const Epetra_Map& ref_red_map, const Epetra_Map& unred_map, Teuchos::RCP<Epetra_Map>& red_map)
{
  const Epetra_Comm& comm = unred_map.Comm();

  Epetra_IntVector source(unred_map);
  {
    const int num_myentries = unred_map.NumMyElements();
    const int* mygids = unred_map.MyGlobalElements();
    int* vals = source.Values();
    std::copy(mygids, mygids + num_myentries, vals);
  }

  Epetra_Import importer(ref_red_map, unred_map);

  Epetra_IntVector target(ref_red_map);
  target.PutValue(-1);

  const int err = target.Import(source, importer, Insert);
  if (err) FOUR_C_THROW("Import failed with error %d!", err);

  std::vector<int> my_red_gids;
  {
    const int num_myentries = ref_red_map.NumMyElements();
    const int* my_received_gids = target.Values();
    my_red_gids.reserve(num_myentries);

    for (int i = 0; i < num_myentries; ++i)
    {
      if (my_received_gids[i] != -1) my_red_gids.push_back(my_received_gids[i]);
    }
  }

  // create new map
  red_map = Teuchos::rcp(new Epetra_Map(-1, my_red_gids.size(), my_red_gids.data(), 0, comm));

  //  red_map->print( std::cout );
}
FOUR_C_NAMESPACE_CLOSE