/*----------------------------------------------------------------------------*/
/** \file

\brief MultiFieldMapExtractor class to handle different discretizations
       with joint interfaces


\level 3

*/
/*----------------------------------------------------------------------------*/


#include "4C_xfem_multi_field_mapextractor.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_comm_parobject.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_matrixtransform.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_xfem_discretization.hpp"
#include "4C_xfem_xfield_field_coupling.hpp"
#include "4C_xfem_xfield_field_coupling_dofset.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
XFEM::MultiFieldMapExtractor::MultiFieldMapExtractor()
    : isinit_(false),
      issetup_(false),
      max_num_reserved_dofs_per_node_(0),
      comm_(Teuchos::null),
      slave_discret_vec_(0),
      element_map_extractor_(Teuchos::null),
      idiscret_(Teuchos::null),
      icoupl_dofset_(Teuchos::null)
{
  // intentionally left blank
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::MultiFieldMapExtractor::reset(unsigned num_dis, bool full)
{
  // set the flag to false (just to be sure)
  issetup_ = false;

  // --------------------------------------------------------------------------
  // reset the slave sided map extractors
  slave_map_extractors_.clear();
  slave_map_extractors_.resize(num_dis,
      std::vector<Teuchos::RCP<Core::LinAlg::MultiMapExtractor>>(NUM_MAP_TYPES, Teuchos::null));
  // loop over the number of discretizations
  for (unsigned i = 0; i < slave_map_extractors_.size(); ++i)
    // loop over the map types (0: DoF's, 1: Nodes)
    for (unsigned j = 0; j < NUM_MAP_TYPES; ++j)
      slave_map_extractors_[i][j] = Teuchos::rcp(new Core::LinAlg::MultiMapExtractor());

  // --------------------------------------------------------------------------
  // reset the master sided map extractor
  master_map_extractor_.clear();
  master_map_extractor_.resize(NUM_MAP_TYPES, Teuchos::null);
  // loop over the two map extractor types (0: DoF's, 1: Nodes)
  std::vector<Teuchos::RCP<Core::LinAlg::MultiMapExtractor>>::iterator it;
  for (it = master_map_extractor_.begin(); it != master_map_extractor_.end(); ++it)
    (*it) = Teuchos::RCP<Core::LinAlg::MultiMapExtractor>(new Core::LinAlg::MultiMapExtractor());

  // --------------------------------------------------------------------------
  // reset the interface coupling objects
  interface_couplings_.clear();
  interface_couplings_.resize(num_dis, Teuchos::null);
  std::vector<Teuchos::RCP<XFEM::XFieldField::Coupling>>::iterator iit;
  for (iit = interface_couplings_.begin(); iit != interface_couplings_.end(); ++iit)
    (*iit) = Teuchos::rcp(new XFEM::XFieldField::Coupling());

  // --------------------------------------------------------------------------
  // reset the element map extractor
  element_map_extractor_ = Teuchos::null;
  element_map_extractor_ =
      Teuchos::rcp<Core::LinAlg::MultiMapExtractor>(new Core::LinAlg::MultiMapExtractor());

  // clear these variables only if a full reset is desired!
  if (full)
  {
    // set the flag to false (just to be sure)
    isinit_ = false;

    xfem_dis_ids_.clear();

    slave_discret_vec_.clear();
    slave_discret_id_map_.clear();

    idiscret_ = Teuchos::null;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::MultiFieldMapExtractor::init(const XDisVec& dis_vec, int max_num_reserved_dofs_per_node)
{
  // reset flags
  isinit_ = false;
  issetup_ = false;

  // sanity check
  if (dis_vec.size() < 2)
    FOUR_C_THROW(
        "You gave less than 2 discretizations. What are you planning "
        "to do? Seems as you are wrong here...");

  // get the communicator (supposed to be the same for all discretizations)
  comm_ = Teuchos::rcpFromRef(dis_vec[0]->Comm());

  max_num_reserved_dofs_per_node_ = max_num_reserved_dofs_per_node;

  // reset member variables
  reset(dis_vec.size());

  // save the slave discretization vector
  slave_discret_vec_ = dis_vec;

  // build the slave discret id map
  build_slave_discret_id_map();

  // --------------------------------------------------------------------------
  // looking for xFEM discretizations
  // --------------------------------------------------------------------------
  XDisVec::const_iterator cit_dis;
  int d = 0;
  for (cit_dis = sl_dis_vec().begin(); cit_dis != sl_dis_vec().end(); ++cit_dis)
  {
    if (Teuchos::rcp_dynamic_cast<const XFEM::DiscretizationXFEM>(*cit_dis).is_null())
      xfem_dis_ids_.insert(d++);
  }

  // --------------------------------------------------------------------------
  // get a std::set holding all interface node GIDs
  // (this has to be done only once, since it is independent of any
  //  redistribution)
  // --------------------------------------------------------------------------
  build_global_interface_node_gid_set();

  // ------------------------------------------------------------------------
  // create an auxiliary master interface discretization
  // ------------------------------------------------------------------------
  idiscret_ = Teuchos::rcp(new Core::FE::Discretization("multifield_interface",
      Teuchos::rcp<Epetra_Comm>(comm().Clone()), Global::Problem::Instance()->NDim()));

  // ------------------------------------------------------------------------
  // (1) create a list of coupling discretizations per node on this proc and
  //     pack it
  // ------------------------------------------------------------------------
  std::map<int, std::set<int>> my_coupled_sl_dis;
  std::map<int, std::set<int>> g_coupled_sl_dis;
  std::map<int, std::set<int>>::const_iterator cit_map;
  std::set<int>::const_iterator cit_set;

  for (std::set<int>::const_iterator ngid = g_interface_node_gid_set().begin();
       ngid != g_interface_node_gid_set().end(); ++ngid)
  {
    for (unsigned d = 0; d < sl_dis_vec().size(); ++d)
    {
      if (sl_dis_vec()[d]->NodeRowMap()->MyGID(*ngid))
      {
        std::pair<std::set<int>::iterator, bool> is_unique = my_coupled_sl_dis[*ngid].insert(d);
        if (not is_unique.second)
          FOUR_C_THROW(
              "That is impossible, since this row node cannot be part of the "
              "same discretization on one single processor more than once!");
      }
    }
  }
  // collect the information over all processors
  std::vector<int> sendsize(2);
  std::vector<int> sendgid;
  std::vector<char> sendset;
  sendgid.reserve(my_coupled_sl_dis.size());

  std::vector<int> receivedsize(2);
  std::vector<int> receivedgid;
  std::vector<char> receivedset;

  Core::Communication::PackBuffer data;
  // --------------------------------------------------------------------------
  // pack gids and std::set's separately
  // --------------------------------------------------------------------------

  // --- pack
  for (cit_map = my_coupled_sl_dis.begin(); cit_map != my_coupled_sl_dis.end(); ++cit_map)
  {
    sendgid.push_back(cit_map->first);
    Core::Communication::ParObject::add_to_pack(data, cit_map->second);
  }

  // swap into std::vector<char>
  swap(sendset, data());

  // get the send information (how many messages will be received)
  sendsize[0] = sendgid.size();
  sendsize[1] = sendset.size();

  // ------------------------------------------------------------------------
  // (2) round robin loop
  // ------------------------------------------------------------------------
  // create an exporter for point to point communication
  Core::Communication::Exporter exporter(comm());
  const int numprocs = comm().NumProc();

  for (int p = 0; p < numprocs; ++p)
  {
    // Send block to next proc. Receive a block from the last proc
    if (p > 0)
    {
      int myrank = comm().MyPID();
      int tag = myrank;

      int frompid = myrank;
      int topid = (myrank + 1) % numprocs;

      MPI_Request sizerequest;
      exporter.i_send(frompid, topid, sendsize.data(), 2, tag, sizerequest);

      // send gid information
      MPI_Request gidrequest;
      exporter.i_send(frompid, topid, sendgid.data(), sendgid.size(), tag * 10, gidrequest);

      // send set data
      MPI_Request setrequest;
      exporter.i_send(frompid, topid, sendset.data(), sendset.size(), tag * 100, setrequest);

      // make sure that you do not think you received something if
      // you didn't
      if (not receivedset.empty() or not receivedgid.empty() or not receivedsize.empty())
        FOUR_C_THROW("Received data objects are not empty!");

      // receive from predecessor
      frompid = (myrank + numprocs - 1) % numprocs;

      // receive size information
      int length = 0;
      exporter.ReceiveAny(frompid, tag, receivedsize, length);
      if (length != 2 or tag != frompid)
        FOUR_C_THROW(
            "Size information got mixed up!\n"
            "Received length = %d, Expected length = %d \n"
            "Received tag    = %d, Expected tag    = %d",
            length, 2, tag, frompid);

      exporter.Wait(sizerequest);

      // receive the gids
      exporter.ReceiveAny(frompid, tag, receivedgid, length);
      if (length != receivedsize[0] or tag != frompid * 10)
        FOUR_C_THROW(
            "GID information got mixed up! \n"
            "Received length = %d, Expected length = %d \n"
            "Received tag    = %d, Expected tag    = %d",
            length, receivedsize[0], tag, frompid * 10);

      exporter.Wait(gidrequest);

      // receive the sets
      exporter.ReceiveAny(frompid, tag, receivedset, length);
      if (length != receivedsize[1] or tag != frompid * 100)
        FOUR_C_THROW(
            "Set information got mixed up! \n"
            "Received length = %d, Expected length = %d \n"
            "Received tag    = %d, Expected tag    = %d",
            length, receivedsize[1], tag, frompid * 100);

      exporter.Wait(setrequest);
    }
    // in the first step, we keep all nodes on the owning proc
    else
    {
      // no need to communicate in the first step
      swap(receivedsize, sendsize);
      swap(receivedgid, sendgid);
      swap(receivedset, sendset);
    }

    // ------------------------------------------------------------------------
    // (3) unpack received block
    // ------------------------------------------------------------------------
    std::vector<char>::size_type index = 0;
    int j = 0;
    while (index < receivedset.size())
    {
      int gid = receivedgid[j];
      // the set gets cleared at the beginning of the extract_from_pack routine!
      std::set<int> rs;
      Core::Communication::ParObject::extract_from_pack(index, receivedset, rs);
      g_coupled_sl_dis[gid].insert(rs.begin(), rs.end());
      ++j;
    }

    // the received data will be sent to the next proc
    swap(receivedsize, sendsize);
    swap(receivedgid, sendgid);
    swap(receivedset, sendset);

    // we need a new receive buffer
    receivedsize.clear();
    receivedgid.clear();
    receivedset.clear();
  }

  // ID's for the master/slave coupling maps
  std::vector<std::vector<int>> my_master_inode_gids(sl_dis_vec().size(), std::vector<int>(0));

  // loop over all (global) coupling nodes
  for (cit_map = g_coupled_sl_dis.begin(); cit_map != g_coupled_sl_dis.end(); ++cit_map)
  {
    // current considered nodal GID
    int ngid = cit_map->first;

    // ------------------------------------------------------------------------
    // find the slave discretization ID we want to copy from
    // ------------------------------------------------------------------------
    int sl_dis_id_to_copy_from = -1;
    for (cit_set = cit_map->second.begin(); cit_set != cit_map->second.end(); ++cit_set)
    {
      // prefer and choose the first XFEM discretization (if there are more than
      // one joint at this node)
      if (is_x_fem_dis(*cit_set))
      {
        sl_dis_id_to_copy_from = *cit_set;
        break;
      }
    }
    /* If there is no XFEM discretization, we choose the first one. Since the
     * the coupled discretization set is ordered (by default), this should be
     * still deterministic. */
    if (sl_dis_id_to_copy_from == -1) sl_dis_id_to_copy_from = *cit_map->second.begin();

    // add the node on the owning processor
    if (sl_dis_vec()[sl_dis_id_to_copy_from]->NodeRowMap()->MyGID(ngid))
    {
      // clone the node, thus it becomes independent of any redistribution
      Core::Nodes::Node* node = sl_dis_vec()[sl_dis_id_to_copy_from]->gNode(ngid);
      Teuchos::RCP<Core::Nodes::Node> inode = Teuchos::rcp(node->Clone());
      idiscret_->AddNode(inode);
      // store the id for the master/slave coupling maps
      for (cit_set = cit_map->second.begin(); cit_set != cit_map->second.end(); ++cit_set)
        my_master_inode_gids[*cit_set].push_back(inode->Id());
    }
  }

  // build the master interface coupling node row maps
  build_master_interface_node_maps(my_master_inode_gids);

  // --------------------------------------------------------------------------
  // set the interface matrix transformation objects
  // --------------------------------------------------------------------------
  interface_matrix_row_transformers_.resize(num_sl_dis(), Teuchos::null);
  interface_matrix_col_transformers_.resize(num_sl_dis(), Teuchos::null);
  interface_matrix_row_col_transformers_.resize(num_sl_dis(), Teuchos::null);
  for (unsigned i = 0; i < num_sl_dis(); ++i)
  {
    interface_matrix_row_transformers_[i] = Teuchos::rcp(new Core::LinAlg::MatrixRowTransform());
    interface_matrix_col_transformers_[i] = Teuchos::rcp(new Core::LinAlg::MatrixColTransform());
    interface_matrix_row_col_transformers_[i] =
        Teuchos::rcp(new Core::LinAlg::MatrixRowColTransform());
  }
  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::MultiFieldMapExtractor::setup()
{
  check_init();

  // reset the flag
  issetup_ = false;

  // first call reset (no full reset!)
  reset(num_sl_dis(), false);

  // build the slave node map extractor objects
  build_slave_node_map_extractors();

  // build the slave dof map extractor objects
  build_slave_dof_map_extractors();

  /* build the interface coupling dof set and set it in the auxiliary interface
   * discretization */
  build_interface_coupling_dof_set();

  // build the master (i.e. auxiliary interface) node map extractor object
  build_master_node_map_extractor();

  // build the master (i.e. auxiliary interface) dof map extractor object
  build_master_dof_map_extractor();

  // build the interface coupling adapters
  build_interface_coupling_adapters();

  // build element map extractor
  build_element_map_extractor();

  // Everything is done. Set the flag.
  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::MultiFieldMapExtractor::build_slave_node_map_extractors()
{
  XDisVec::const_iterator cit_dis;
  std::vector<Teuchos::RCP<const Epetra_Map>> partial_maps(2, Teuchos::null);

  // ------------------------------------------------------------------------
  /* build the interface row node GID maps of each slave discretization
   * on each single proc */
  // ------------------------------------------------------------------------
  unsigned dis_count = 0;
  for (cit_dis = sl_dis_vec().begin(); cit_dis != sl_dis_vec().end(); ++cit_dis)
  {
    std::vector<int> my_interface_row_node_gids(0);
    std::vector<int> my_non_interface_row_node_gids(0);

    const int num_my_rnodes = (*cit_dis)->NumMyRowNodes();
    int* my_row_node_gids = (*cit_dis)->NodeRowMap()->MyGlobalElements();

    for (unsigned nlid = 0; nlid < static_cast<unsigned>(num_my_rnodes); ++nlid)
    {
      int ngid = my_row_node_gids[nlid];

      // find the interface gids
      if (IsInterfaceNode(ngid))
        my_interface_row_node_gids.push_back(ngid);
      else
        my_non_interface_row_node_gids.push_back(ngid);
    }

    // slave sided interface node maps
    partial_maps[MultiField::block_interface] = Teuchos::null;
    partial_maps[MultiField::block_interface] =
        Teuchos::rcp(new Epetra_Map(-1, static_cast<int>(my_interface_row_node_gids.size()),
            my_interface_row_node_gids.data(), 0, comm()));

    // slave sided non-interface node maps
    partial_maps[MultiField::block_non_interface] = Teuchos::null;
    partial_maps[MultiField::block_non_interface] =
        Teuchos::rcp(new Epetra_Map(-1, static_cast<int>(my_non_interface_row_node_gids.size()),
            my_non_interface_row_node_gids.data(), 0, comm()));

    // setup node map extractor
    slave_map_extractors_[dis_count++][map_nodes]->setup(*((*cit_dis)->NodeRowMap()), partial_maps);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::MultiFieldMapExtractor::build_slave_dof_map_extractors()
{
  std::vector<int> my_sl_interface_dofs(0);
  std::vector<int> my_sl_non_interface_dofs(0);

  std::vector<Teuchos::RCP<const Epetra_Map>> partial_maps(2, Teuchos::null);

  // loop over all slave discretizations
  XDisVec::const_iterator cit_dis;
  unsigned dis_count = 0;
  for (cit_dis = sl_dis_vec().begin(); cit_dis != sl_dis_vec().end(); ++cit_dis)
  {
    int* my_node_gids = (*cit_dis)->NodeRowMap()->MyGlobalElements();

    // loop over my nodes
    for (int nlid = 0; nlid < (*cit_dis)->NodeRowMap()->NumMyElements(); ++nlid)
    {
      int ngid = my_node_gids[nlid];
      // ----------------------------------------------------------------------
      // interface DoF's
      // ----------------------------------------------------------------------
      if (slave_node_row_map(dis_count, MultiField::block_interface).MyGID(ngid))
      {
        const Core::Nodes::Node* node = (*cit_dis)->lRowNode(nlid);
        const unsigned numdof = (*cit_dis)->NumDof(node);

        for (unsigned i = 0; i < numdof; ++i)
          my_sl_interface_dofs.push_back((*cit_dis)->Dof(node, i));
      }
      // ----------------------------------------------------------------------
      // non-interface DoF's
      // ----------------------------------------------------------------------
      else
      {
        const Core::Nodes::Node* node = (*cit_dis)->lRowNode(nlid);
        const unsigned numdof = (*cit_dis)->NumDof(node);

        for (unsigned i = 0; i < numdof; ++i)
          my_sl_non_interface_dofs.push_back((*cit_dis)->Dof(node, i));
      }
    }
    // create slave interface dof row map
    partial_maps[MultiField::block_interface] = Teuchos::null;
    partial_maps[MultiField::block_interface] = Teuchos::rcp(new Epetra_Map(
        -1, static_cast<int>(my_sl_interface_dofs.size()), my_sl_interface_dofs.data(), 0, comm()));

    // create slave non-interface dof row map
    partial_maps[MultiField::block_non_interface] = Teuchos::null;
    partial_maps[MultiField::block_non_interface] =
        Teuchos::rcp(new Epetra_Map(-1, static_cast<int>(my_sl_non_interface_dofs.size()),
            my_sl_non_interface_dofs.data(), 0, comm()));

    // setup dof map extractor
    slave_map_extractors_[dis_count++][map_dofs]->setup(*((*cit_dis)->dof_row_map()), partial_maps);

    // clear the vectors for a new round
    my_sl_interface_dofs.clear();
    my_sl_non_interface_dofs.clear();
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::MultiFieldMapExtractor::build_interface_coupling_dof_set()
{
  std::map<int, int> sl_max_num_dof_per_inode;
  std::map<int, int> ma_max_num_dof_per_inode;
  int g_num_std_dof = -1;
  for (unsigned i = 0; i < num_sl_dis(); ++i)
  {
    const Epetra_Map& sl_inodemap = slave_node_row_map(i, MultiField::block_interface);
    const Epetra_Map& ma_inodemap = master_interface_node_row_map(i);

    int nnodes = sl_inodemap.NumMyElements();
    int* ngids = sl_inodemap.MyGlobalElements();

    int my_num_std_dof = -1;

    for (int j = 0; j < nnodes; ++j)
    {
      const Core::Nodes::Node* node = sl_discret(i).gNode(ngids[j]);
      const int numdof = sl_discret(i).NumDof(node);
      my_num_std_dof = sl_discret(i).NumStandardDof(0, node);
      sl_max_num_dof_per_inode[ngids[j]] = numdof;
    }

    Core::Communication::Exporter export_max_dof_num(
        sl_inodemap, ma_inodemap, sl_discret(i).Comm());
    export_max_dof_num.Export(sl_max_num_dof_per_inode);

    // communicate the number of standard DoF's
    // Supposed to be the same value on all discretizations and all nodes.
    if (g_num_std_dof == -1) comm().MaxAll(&my_num_std_dof, &g_num_std_dof, 1);

    if (my_num_std_dof != -1 and g_num_std_dof != my_num_std_dof)
      FOUR_C_THROW(
          "The number of standard DoF's is not equal on all procs"
          "and discretizations!");

    std::map<int, int>::const_iterator cit;
    for (cit = sl_max_num_dof_per_inode.begin(); cit != sl_max_num_dof_per_inode.end(); ++cit)
    {
      std::map<int, int>::iterator ma_pos = ma_max_num_dof_per_inode.find(cit->first);
      // If the nodal GID was not yet added, we insert the number of DoF's.
      if (ma_pos == ma_max_num_dof_per_inode.end())
      {
        ma_max_num_dof_per_inode[cit->first] = cit->second;
      }
      // If the nodal GID was already added, we calculate the maximum value.
      else
      {
        ma_pos->second = std::max(ma_pos->second, cit->second);
      }
    }

    // clear the slave maximum number of DoF's map for a new round
    sl_max_num_dof_per_inode.clear();
  }

  // get the node index range over all procs
  if (g_interface_node_gid_set_.empty()) FOUR_C_THROW("set of global nodes is empty?!");
  const int min_all_gid = *(g_interface_node_gid_set_.begin());
  const int max_all_gid = *(g_interface_node_gid_set_.rbegin());

  const int g_node_index_range = max_all_gid - min_all_gid + 1;

  // create a new xfield/field coupling DoF set
  icoupl_dofset_ =
      Teuchos::rcp(new XFEM::XFieldField::CouplingDofSet(max_num_reserved_dofs_per_node_,
          g_node_index_range, g_num_std_dof, ma_max_num_dof_per_inode));

  // set the new dof-set and finish the interface discretization
  idiscret_->ReplaceDofSet(0, icoupl_dofset_, true);

  idiscret_->fill_complete(true, false, false);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::MultiFieldMapExtractor::build_master_node_map_extractor()
{
  Teuchos::RCP<Epetra_Map> fullmap = Teuchos::null;
  std::vector<Teuchos::RCP<const Epetra_Map>> partial_maps(2 * num_sl_dis(), Teuchos::null);

  // --------------------------------------------------------------------------
  // interface DoF's
  // --------------------------------------------------------------------------
  for (unsigned i = 0; i < num_sl_dis(); ++i) partial_maps.at(i) = master_interface_node_maps_[i];

  // --------------------------------------------------------------------------
  // non-interface DoF's
  // --------------------------------------------------------------------------
  for (unsigned i = 0; i < num_sl_dis(); ++i)
    partial_maps.at(num_sl_dis() + i) =
        sl_map_extractor(i, map_nodes).Map(MultiField::block_non_interface);

  // --------------------------------------------------------------------------
  // create non-overlapping full dof map
  // --------------------------------------------------------------------------
  fullmap = Teuchos::null;
  // add interface nodes
  fullmap = Teuchos::rcp<Epetra_Map>(new Epetra_Map(*i_discret().NodeRowMap()));

  // merge non-interface nodes into the full map
  for (unsigned i = num_sl_dis(); i < partial_maps.size(); ++i)
    fullmap = Core::LinAlg::MergeMap(*fullmap, *partial_maps[i], false);

  // setup map extractor
  master_map_extractor_[map_nodes]->setup(*fullmap, partial_maps);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::MultiFieldMapExtractor::build_master_dof_map_extractor()
{
  std::vector<int> my_ma_interface_dofs(0);

  /* the 1-st num_dis partial maps are the master interface dof maps,
   * the 2-nd num_dis partial maps are the master non-interface dof maps
   * (which correspond to the slave_non_interface_dof_row_maps) */
  Teuchos::RCP<Epetra_Map> fullmap = Teuchos::null;
  std::vector<Teuchos::RCP<const Epetra_Map>> partial_maps(2 * num_sl_dis(), Teuchos::null);

  // --------------------------------------------------------------------------
  // interface DoF's
  // --------------------------------------------------------------------------
  for (unsigned i = 0; i < num_sl_dis(); ++i)
  {
    my_ma_interface_dofs.clear();

    const int num_my_inodes = master_interface_node_row_map(i).NumMyElements();
    int* inode_gids = master_interface_node_row_map(i).MyGlobalElements();

    // get the dofs of the master interface coupling nodes
    for (int nlid = 0; nlid < num_my_inodes; ++nlid)
    {
      int ngid = inode_gids[nlid];
      const Core::Nodes::Node* inode = i_discret().gNode(ngid);
      const unsigned numdof = i_discret().NumDof(inode);
      for (unsigned j = 0; j < numdof; ++j)
        my_ma_interface_dofs.push_back(i_discret().Dof(inode, j));
    }
    partial_maps.at(i) = Teuchos::rcp<const Epetra_Map>(new Epetra_Map(
        -1, static_cast<int>(my_ma_interface_dofs.size()), my_ma_interface_dofs.data(), 0, comm()));
  }

  // --------------------------------------------------------------------------
  // non-interface DoF's
  // --------------------------------------------------------------------------
  for (unsigned i = 0; i < num_sl_dis(); ++i)
    partial_maps.at(num_sl_dis() + i) =
        sl_map_extractor(i, map_dofs).Map(MultiField::block_non_interface);

  // --------------------------------------------------------------------------
  // create non-overlapping full dof map
  // --------------------------------------------------------------------------
  fullmap = Teuchos::null;

  // add interface DoF's
  fullmap = Teuchos::rcp<Epetra_Map>(new Epetra_Map(*i_discret().dof_row_map()));

  // merge non-interface DoF's into the full map
  for (unsigned i = num_sl_dis(); i < partial_maps.size(); ++i)
  {
    fullmap = Core::LinAlg::MergeMap(*fullmap, *partial_maps[i], false);
  }

  // setup map extractor
  master_map_extractor_[map_dofs]->setup(*fullmap, partial_maps);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::MultiFieldMapExtractor::build_interface_coupling_adapters()
{
  check_init();

  std::vector<Teuchos::RCP<XFEM::XFieldField::Coupling>>::iterator it;
  unsigned i = 0;
  for (it = interface_couplings_.begin(); it != interface_couplings_.end(); ++it)
  {
    /* Set the slave discretization to the discretization with minimum number
     * of DoF's at each interface node. This is true by construction. */
    (*it)->init(XFEM::XFieldField::Coupling::min_dof_slave);

    /* Setup the interface coupling objects. The interface discretization
     * is always the master discretization. Since the GID's at the interface
     * coincide in the coupling interface maps, the master interface map
     * becomes the permuted slave interface map. */
    (*it)->setup_coupling(i_discret(), sl_discret(i), master_interface_node_row_map(i),
        slave_node_row_map(i, MultiField::block_interface), master_interface_node_row_map(i), -1);

    ++i;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::MultiFieldMapExtractor::build_element_map_extractor()
{
  check_init();

  Teuchos::RCP<Epetra_Map> fullmap = Teuchos::null;
  std::vector<Teuchos::RCP<const Epetra_Map>> partial_maps(num_sl_dis(), Teuchos::null);
  unsigned d = 0;
  XDisVec::const_iterator cit;
  for (cit = sl_dis_vec().begin(); cit != sl_dis_vec().end(); ++cit)
  {
    // get the element row map of each wrapped discretization
    partial_maps[d] = Teuchos::rcp((*cit)->ElementRowMap(), false);

    // merge the partial maps to the full map
    fullmap = Core::LinAlg::MergeMap(fullmap, partial_maps[d], false);

    // increase discretization counter
    ++d;
  }
  // setup the element multi map extractor
  element_map_extractor_->setup(*fullmap, partial_maps);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> XFEM::MultiFieldMapExtractor::extract_vector(
    const Epetra_Vector& full, enum FieldName field, enum MapType map_type) const
{
  const int dis_id = slave_id(field);

  /* the partial map is equivalent to the full map (of desired type) from
   * the field slave map extractor */
  const Teuchos::RCP<const Epetra_Map>& sl_full_map = sl_map_extractor(dis_id, map_type).FullMap();

  if (sl_full_map.is_null())
    FOUR_C_THROW("null full map for field %s", FieldName2String(field).c_str());

  // create a new vector
  Teuchos::RCP<Epetra_Vector> vec = Teuchos::rcp(new Epetra_Vector(*sl_full_map));

  // extract the actual vector and return it
  extract_vector(full, dis_id, *vec, map_type);

  return vec;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Epetra_MultiVector> XFEM::MultiFieldMapExtractor::extract_vector(
    const Epetra_MultiVector& full, enum FieldName field, enum MapType map_type) const
{
  const int dis_id = slave_id(field);

  /* the partial map is equivalent to the full map (of desired type) from
   * the field slave map extractor */
  const Teuchos::RCP<const Epetra_Map>& sl_full_map = sl_map_extractor(dis_id, map_type).FullMap();

  if (sl_full_map.is_null())
    FOUR_C_THROW("null full map for field %s", FieldName2String(field).c_str());

  // create a new multi vector
  Teuchos::RCP<Epetra_MultiVector> vec =
      Teuchos::rcp(new Epetra_MultiVector(*sl_full_map, full.NumVectors()));

  // extract the actual vector and return it
  extract_vector(full, dis_id, *vec, map_type);

  return vec;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::MultiFieldMapExtractor::extract_vector(const Epetra_MultiVector& full, int block,
    Epetra_MultiVector& partial, enum MapType map_type) const
{
  // --------------------------------------------------------------------------
  // extract the non-interface part
  // --------------------------------------------------------------------------
  Teuchos::RCP<Epetra_MultiVector> partial_non_interface =
      ma_map_extractor(map_type).extract_vector(full, num_sl_dis() + block);
  sl_map_extractor(block, map_type)
      .insert_vector(*partial_non_interface, MultiField::block_non_interface, partial);

  // --------------------------------------------------------------------------
  // extract the interface part
  // --------------------------------------------------------------------------
  Teuchos::RCP<Epetra_MultiVector> partial_ma_interface =
      ma_map_extractor(map_type).extract_vector(full, block);
  Teuchos::RCP<Epetra_MultiVector> partial_sl_interface =
      i_coupling(block).MasterToSlave(partial_ma_interface, map_type);
  sl_map_extractor(block, map_type)
      .insert_vector(*partial_sl_interface, MultiField::block_interface, partial);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::MultiFieldMapExtractor::extract_element_vector(
    const Epetra_MultiVector& full, int block, Epetra_MultiVector& partial) const
{
  element_map_extractor_->extract_vector(full, block, partial);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::MultiFieldMapExtractor::InsertElementVector(
    const Epetra_MultiVector& partial, int block, Epetra_MultiVector& full) const
{
  element_map_extractor_->insert_vector(partial, block, full);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::MultiFieldMapExtractor::AddElementVector(
    const Epetra_MultiVector& partial, int block, Epetra_MultiVector& full, double scale) const
{
  element_map_extractor_->add_vector(partial, block, full, scale);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> XFEM::MultiFieldMapExtractor::insert_vector(
    const Epetra_Vector& partial, enum FieldName field, enum MapType map_type) const
{
  const int dis_id = slave_id(field);
  Teuchos::RCP<Epetra_Vector> full = Teuchos::rcp(new Epetra_Vector(*FullMap(map_type)));
  insert_vector(partial, dis_id, *full, map_type);
  return full;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Epetra_MultiVector> XFEM::MultiFieldMapExtractor::insert_vector(
    const Epetra_MultiVector& partial, enum FieldName field, enum MapType map_type) const
{
  const int dis_id = slave_id(field);

  Teuchos::RCP<Epetra_MultiVector> full =
      Teuchos::rcp(new Epetra_MultiVector(*FullMap(map_type), partial.NumVectors()));

  insert_vector(partial, dis_id, *full, map_type);
  return full;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::MultiFieldMapExtractor::insert_vector(const Epetra_MultiVector& partial, int block,
    Epetra_MultiVector& full, enum MapType map_type) const
{
  // --------------------------------------------------------------------------
  // insert the non_interface part
  // --------------------------------------------------------------------------
  Teuchos::RCP<Epetra_MultiVector> partial_non_interface =
      sl_map_extractor(block, map_type).extract_vector(partial, MultiField::block_non_interface);

  ma_map_extractor(map_type).insert_vector(*partial_non_interface, num_sl_dis() + block, full);

  // --------------------------------------------------------------------------
  // insert the interface part
  // --------------------------------------------------------------------------
  Teuchos::RCP<Epetra_MultiVector> partial_sl_interface =
      sl_map_extractor(block, map_type).extract_vector(partial, MultiField::block_interface);

  Teuchos::RCP<Epetra_MultiVector> partial_ma_interface =
      i_coupling(block).SlaveToMaster(partial_sl_interface, map_type);

  ma_map_extractor(map_type).insert_vector(*partial_ma_interface, block, full);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::MultiFieldMapExtractor::add_vector(const Epetra_MultiVector& partial, int block,
    Epetra_MultiVector& full, double scale, enum MapType map_type) const
{
  // --------------------------------------------------------------------------
  // insert the non_interface part
  // --------------------------------------------------------------------------
  Teuchos::RCP<Epetra_MultiVector> partial_non_interface =
      sl_map_extractor(block, map_type).extract_vector(partial, MultiField::block_non_interface);

  ma_map_extractor(map_type).add_vector(*partial_non_interface, num_sl_dis() + block, full, scale);

  // --------------------------------------------------------------------------
  // insert the interface part
  // --------------------------------------------------------------------------
  Teuchos::RCP<Epetra_MultiVector> partial_sl_interface =
      sl_map_extractor(block, map_type).extract_vector(partial, MultiField::block_interface);

  Teuchos::RCP<Epetra_MultiVector> partial_ma_interface =
      i_coupling(block).SlaveToMaster(partial_sl_interface, map_type);

  ma_map_extractor(map_type).add_vector(*partial_ma_interface, block, full, scale);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::MultiFieldMapExtractor::build_slave_discret_id_map()
{
  XDisVec::const_iterator cit_dis;
  int dis_count = 0;
  for (cit_dis = sl_dis_vec().begin(); cit_dis != sl_dis_vec().end(); ++cit_dis)
  {
    const std::string& name = (*cit_dis)->Name();
    if (name == "structure")
      slave_discret_id_map_[structure] = dis_count;
    else if (name == "xstructure")
      slave_discret_id_map_[xstructure] = dis_count;
    else
      FOUR_C_THROW("Unknown field discretization name \"%s\"!", name.c_str());
    // increase counter
    ++dis_count;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool XFEM::MultiFieldMapExtractor::IsInterfaceNode(const int& ngid) const
{
  return (g_interface_node_gid_set().find(ngid) != g_interface_node_gid_set().end());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool XFEM::MultiFieldMapExtractor::is_x_fem_dis(int dis_id) const
{
  return (xfem_dis_ids_.find(dis_id) != xfem_dis_ids_.end());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int XFEM::MultiFieldMapExtractor::slave_id(enum FieldName field) const
{
  std::map<enum FieldName, int>::const_iterator cit = slave_discret_id_map_.find(field);
  if (cit == slave_discret_id_map_.end())
    FOUR_C_THROW("The slave field \"%s\" could not be found!", FieldName2String(field).c_str());

  return cit->second;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::MultiFieldMapExtractor::build_global_interface_node_gid_set()
{
  comm().Barrier();
  std::set<int> g_unique_row_node_gid_set;
  g_interface_node_gid_set_.clear();

  // loop over all proc's
  for (unsigned p = 0; p < static_cast<unsigned>(comm().NumProc()); ++p)
  {
    int num_my_unique_row_nodes = 0;
    std::vector<int> my_unique_row_node_gid_vec(0);

    int num_my_interface_row_nodes = 0;
    std::vector<int> my_interface_row_node_gid_vec(0);

    if (p == static_cast<unsigned>(comm().MyPID()))
    {
      std::set<int> my_unique_row_node_gid_set;
      std::set<int> my_interface_row_node_gid_set;
      for (unsigned j = 0; j < slave_discret_vec_.size(); ++j)
      {
        for (unsigned i = 0; i < static_cast<unsigned>(slave_discret_vec_[j]->NumMyRowNodes()); ++i)
        {
          int gid = slave_discret_vec_[j]->NodeRowMap()->GID(i);
          // insert the gid and check if it is unique on the current processor
          std::pair<std::set<int>::iterator, bool> is_unique =
              my_unique_row_node_gid_set.insert(gid);
          if (not is_unique.second) my_interface_row_node_gid_set.insert(gid);
        }
      }
      // copy the unique set into a vector
      num_my_unique_row_nodes = my_unique_row_node_gid_set.size();
      my_unique_row_node_gid_vec.resize(num_my_unique_row_nodes);

      std::copy(my_unique_row_node_gid_set.begin(), my_unique_row_node_gid_set.end(),
          my_unique_row_node_gid_vec.begin());

      // copy the interface set into a vector
      num_my_interface_row_nodes = my_interface_row_node_gid_set.size();
      my_interface_row_node_gid_vec.resize(num_my_interface_row_nodes);

      std::copy(my_interface_row_node_gid_set.begin(), my_interface_row_node_gid_set.end(),
          my_interface_row_node_gid_vec.begin());
    }
    // wait since only one proc did all the work
    comm().Barrier();
    // ------------------------------------------------------------------------
    // send the unique row node GID vector from processor p to all proc's
    // ------------------------------------------------------------------------
    comm().Broadcast(&num_my_unique_row_nodes, 1, p);
    if (num_my_unique_row_nodes == 0) continue;
    my_unique_row_node_gid_vec.resize(num_my_unique_row_nodes, -1);
    comm().Broadcast(my_unique_row_node_gid_vec.data(), num_my_unique_row_nodes, p);

    // ------------------------------------------------------------------------
    // send the interface row node GID vector from processor p to all proc's
    // ------------------------------------------------------------------------
    comm().Broadcast(&num_my_interface_row_nodes, 1, p);
    if (num_my_interface_row_nodes > 0)
    {
      my_interface_row_node_gid_vec.resize(num_my_interface_row_nodes, -1);
      comm().Broadcast(my_interface_row_node_gid_vec.data(), num_my_interface_row_nodes, p);
      // create/extend the global interface row node gid set
      g_interface_node_gid_set_.insert(
          my_interface_row_node_gid_vec.begin(), my_interface_row_node_gid_vec.end());
    }
    // ------------------------------------------------------------------------
    /* Insert the local unique row node GIDs into a global set. If the GID was
     * already inserted by another proc, we have to extend our global interface
     * row node set as well. */
    // ------------------------------------------------------------------------
    for (std::vector<int>::const_iterator cit = my_unique_row_node_gid_vec.begin();
         cit != my_unique_row_node_gid_vec.end(); ++cit)
    {
      std::pair<std::set<int>::iterator, bool> is_unique = g_unique_row_node_gid_set.insert(*cit);
      if (not is_unique.second) g_interface_node_gid_set_.insert(*cit);
    }
  }  // end: loop over all proc's
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::MultiFieldMapExtractor::build_master_interface_node_maps(
    const std::vector<std::vector<int>>& my_master_interface_node_gids)
{
  for (unsigned i = 0; i < my_master_interface_node_gids.size(); ++i)
  {
    master_interface_node_maps_.push_back(
        Teuchos::rcp(new Epetra_Map(-1, static_cast<int>(my_master_interface_node_gids[i].size()),
            my_master_interface_node_gids[i].data(), 0, comm())));
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> XFEM::MultiFieldMapExtractor::NodeRowMap(
    enum FieldName field, enum MultiField::BlockType block) const
{
  switch (block)
  {
    case MultiField::block_interface:
    {
      return Teuchos::rcpFromRef<const Epetra_Map>(master_interface_node_row_map(field));
      break;
    }
    case MultiField::block_non_interface:
    {
      return Teuchos::rcpFromRef<const Epetra_Map>(
          slave_node_row_map(field, MultiField::block_non_interface));
      break;
    }
    default:
      FOUR_C_THROW("Unknown block type!");
      exit(EXIT_FAILURE);
  }
  // hoops, shouldn't happen ...
  exit(EXIT_FAILURE);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::MultiFieldMapExtractor::AddMatrix(const Core::LinAlg::SparseOperator& partial_mat,
    int block, Core::LinAlg::SparseOperator& full_mat, double scale)
{
  const Core::LinAlg::BlockSparseMatrixBase* block_mat =
      dynamic_cast<const Core::LinAlg::BlockSparseMatrixBase*>(&partial_mat);
  if (not block_mat) FOUR_C_THROW("The partial matrix must be a  Core::LinAlg::BlockSparseMatrix!");
  if (block_mat->Rows() != 2 or block_mat->Cols() != 2)
    FOUR_C_THROW("We support only 2x2 block matrices!");

  Core::LinAlg::SparseMatrix* sp_mat = dynamic_cast<Core::LinAlg::SparseMatrix*>(&full_mat);
  if (not sp_mat) FOUR_C_THROW("The full matrix must be a Core::LinAlg::SparseMatrix!");

  AddMatrix(*block_mat, block, *sp_mat, scale);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::MultiFieldMapExtractor::AddMatrix(const Core::LinAlg::BlockSparseMatrixBase& partial_mat,
    int block, Core::LinAlg::SparseMatrix& full_mat, double scale)
{
  check_init_setup();
  // --------------------------------------------------------------------------
  // non-interface DoF's
  // --------------------------------------------------------------------------
  // Add block non_interface/non_interface. Here is no communication necessary.
  /* ToDo Maybe there is a way to circumvent the add command by using some kind
   *      of direct assignment for the non-interface DoF's? The current
   *      implementation makes it necessary to allocate almost the double
   *      amount of memory!                                        hiermeier */
  full_mat.Add(partial_mat.Matrix(MultiField::block_non_interface, MultiField::block_non_interface),
      false, scale, 1.0);

  // --------------------------------------------------------------------------
  // interface DoF's
  // --------------------------------------------------------------------------
  // (0) Add block non_interface/interface
  const Core::LinAlg::SparseMatrix& src_ni =
      partial_mat.Matrix(MultiField::block_non_interface, MultiField::block_interface);
  i_mat_col_transform(block)(partial_mat.FullRowMap(), partial_mat.FullColMap(), src_ni, scale,
      Core::Adapter::CouplingSlaveConverter(i_coupling(block)), full_mat, false, true);

  // (1) Add block interface/non_interface
  const Core::LinAlg::SparseMatrix& src_in =
      partial_mat.Matrix(MultiField::block_interface, MultiField::block_non_interface);
  i_mat_row_transform(block)(
      src_in, scale, Core::Adapter::CouplingSlaveConverter(i_coupling(block)), full_mat, true);

  // (2) Add block interface/interface
  const Core::LinAlg::SparseMatrix& src_ii =
      partial_mat.Matrix(MultiField::block_interface, MultiField::block_interface);
  i_mat_row_col_transform(block)(src_ii, scale,
      Core::Adapter::CouplingSlaveConverter(i_coupling(block)),
      Core::Adapter::CouplingSlaveConverter(i_coupling(block)), full_mat, false, true);

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Teuchos::RCP<const Epetra_Map>& XFEM::MultiFieldMapExtractor::FullMap(
    enum MapType map_type) const
{
  return ma_map_extractor(map_type).FullMap();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::Nodes::Node* XFEM::MultiFieldMapExtractor::gINode(const int& gid) const
{
  return i_discret().gNode(gid);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Epetra_Map* XFEM::MultiFieldMapExtractor::INodeRowMap() const
{
  return i_discret().NodeRowMap();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int XFEM::MultiFieldMapExtractor::INumDof(const Core::Nodes::Node* inode) const
{
  return i_discret().NumDof(0, inode);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int XFEM::MultiFieldMapExtractor::INumStandardDof() const
{
  return icoupl_dofset_->num_standard_dof_per_node();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int XFEM::MultiFieldMapExtractor::IDof(const Core::Nodes::Node* inode, int dof) const
{
  return i_discret().Dof(0, inode, dof);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::MultiFieldMapExtractor::IDof(
    const Core::Nodes::Node* inode, std::vector<int>& dofs) const
{
  i_discret().Dof(static_cast<unsigned>(0), inode, dofs);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::MultiFieldMapExtractor::IDof(std::vector<int>& dof, Core::Nodes::Node* inode,
    unsigned nodaldofset_id, const Core::Elements::Element* element) const
{
  i_discret().Dof(dof, inode, 0, nodaldofset_id, element);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Epetra_Map& XFEM::MultiFieldMapExtractor::slave_node_row_map(
    unsigned dis_id, enum MultiField::BlockType btype) const
{
  check_init();
  return *(sl_map_extractor(dis_id, map_nodes).Map(btype));
}

FOUR_C_NAMESPACE_CLOSE