/*----------------------------------------------------------------------*/
/*! \file

 \brief porous medium algorithm with block matrices for splitting and condensation

\level 2

 *----------------------------------------------------------------------*/

#include "4C_poroelast_monolithicsplit.hpp"

#include "4C_adapter_fld_base_algorithm.hpp"
#include "4C_adapter_fld_poro.hpp"
#include "4C_adapter_str_fpsiwrapper.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_structure_aux.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

PoroElast::MonolithicSplit::MonolithicSplit(const Epetra_Comm& comm,
    const Teuchos::ParameterList& timeparams,
    Teuchos::RCP<Core::LinAlg::MapExtractor> porosity_splitter)
    : Monolithic(comm, timeparams, porosity_splitter)
{
  icoupfs_ = Teuchos::rcp(new Core::Adapter::Coupling());

  evaluateinterface_ = false;

  fgcur_ = Teuchos::null;
  ddiinc_ = Teuchos::null;
  solipre_ = Teuchos::null;
  ddginc_ = Teuchos::null;
  solgpre_ = Teuchos::null;
  duiinc_ = Teuchos::null;
  solivelpre_ = Teuchos::null;
  duginc_ = Teuchos::null;
  solgvelpre_ = Teuchos::null;

  fsibcmap_ = Teuchos::null;
  fsibcextractor_ = Teuchos::null;

  ddi_ = Teuchos::null;
}

void PoroElast::MonolithicSplit::prepare_time_step()
{
  // call base class
  PoroElast::Monolithic::prepare_time_step();

  //  // call the predictor
  //  structure_field()->prepare_time_step();
  //  fluid_field()->prepare_time_step();

  if (evaluateinterface_)
  {
    // here we account for DBCs and preconditioning on the FSI-Interface. In both cases the
    // structure field decides, what to do (I don't think this is the best solution, but at least
    // the
    // easiest one)

    double timescale = fluid_field()->TimeScaling();

    Teuchos::RCP<const Epetra_Vector> idispnp =
        structure_field()->Interface()->extract_fsi_cond_vector(structure_field()->Dispnp());
    Teuchos::RCP<const Epetra_Vector> idispn =
        structure_field()->Interface()->extract_fsi_cond_vector(structure_field()->Dispn());
    Teuchos::RCP<const Epetra_Vector> ivelnp =
        structure_field()->Interface()->extract_fsi_cond_vector(structure_field()->Velnp());
    Teuchos::RCP<Epetra_Vector> ifvelnp = fluid_field()->extract_interface_velnp();
    Teuchos::RCP<Epetra_Vector> ifveln = fluid_field()->extract_interface_veln();

    ddi_->Update(1.0, *idispnp, -1.0, *idispn, 0.0);
    ddi_->Update(-1.0, *ifveln, timescale);

    if (fsibcmap_->NumGlobalElements())
    {
      // if there are DBCs on FSI conditioned nodes, they have to be treated seperately

      Teuchos::RCP<Epetra_Vector> ibcveln =
          fsibcextractor_->extract_cond_vector(structure_to_fluid_at_interface(ivelnp));
      Teuchos::RCP<Epetra_Vector> inobcveln =
          fsibcextractor_->extract_other_vector(structure_to_fluid_at_interface(ddi_));

      // DBCs at FSI-Interface
      fsibcextractor_->insert_cond_vector(ibcveln, ifvelnp);
      // any preconditioned values at the FSI-Interface
      fsibcextractor_->insert_other_vector(inobcveln, ifvelnp);
    }
    else
      // no DBCs on FSI interface -> just make preconditioners consistent (structure decides)
      ifvelnp = structure_to_fluid_at_interface(ddi_);

    fluid_field()->apply_interface_velocities(ifvelnp);
  }
}

Teuchos::RCP<Epetra_Vector> PoroElast::MonolithicSplit::structure_to_fluid_at_interface(
    Teuchos::RCP<const Epetra_Vector> iv) const
{
  return icoupfs_->MasterToSlave(iv);
}

Teuchos::RCP<Epetra_Vector> PoroElast::MonolithicSplit::fluid_to_structure_at_interface(
    Teuchos::RCP<const Epetra_Vector> iv) const
{
  return icoupfs_->SlaveToMaster(iv);
}

Teuchos::RCP<Epetra_Map> PoroElast::MonolithicSplit::fsidbc_map()
{
  TEUCHOS_FUNC_TIME_MONITOR("PoroElast::MonolithicSplit::FSIDBCMap");

  // get interface map and DBC map of fluid
  std::vector<Teuchos::RCP<const Epetra_Map>> fluidmaps;
  fluidmaps.push_back(fluid_field()->Interface()->fsi_cond_map());
  fluidmaps.push_back(fluid_field()->get_dbc_map_extractor()->cond_map());

  // build vector of dbc and fsi coupling of fluid field
  Teuchos::RCP<Epetra_Map> fluidfsibcmap =
      Core::LinAlg::MultiMapExtractor::intersect_maps(fluidmaps);

  if (fluidfsibcmap->NumMyElements())
    FOUR_C_THROW("Dirichlet boundary conditions on fluid and FSI interface not supported!!");

  // get interface map and DBC map of structure
  std::vector<Teuchos::RCP<const Epetra_Map>> structmaps;
  structmaps.push_back(structure_field()->Interface()->fsi_cond_map());
  structmaps.push_back(structure_field()->get_dbc_map_extractor()->cond_map());

  // vector of dbc and fsi coupling of structure field
  Teuchos::RCP<Epetra_Map> structfsibcmap =
      Core::LinAlg::MultiMapExtractor::intersect_maps(structmaps);

  Teuchos::RCP<Epetra_Vector> gidmarker_struct =
      Teuchos::rcp(new Epetra_Vector(*structure_field()->Interface()->fsi_cond_map(), true));

  // Todo this is ugly, fix it!
  const int mylength = structfsibcmap->NumMyElements();  // on each processor (lids)
  const int* mygids = structfsibcmap->MyGlobalElements();

  // mark gids with fsi and DBC Condition
  for (int i = 0; i < mylength; ++i)
  {
    int gid = mygids[i];
    // FOUR_C_ASSERT(slavemastermap.count(gid),"master gid not found on slave side");
    int err = gidmarker_struct->ReplaceGlobalValue(gid, 0, 1.0);
    if (err) FOUR_C_THROW("ReplaceMyValue failed for gid %i error code %d", gid, err);
  }

  // transfer to fluid side
  Teuchos::RCP<Epetra_Vector> gidmarker_fluid = structure_to_fluid_at_interface(gidmarker_struct);

  std::vector<int> structfsidbcvector;
  const int numgids = gidmarker_fluid->MyLength();  // on each processor (lids)
  double* mygids_fluid = gidmarker_fluid->Values();
  const int* fluidmap = gidmarker_fluid->Map().MyGlobalElements();
  for (int i = 0; i < numgids; ++i)
  {
    double val = mygids_fluid[i];
    if (val == 1.0) structfsidbcvector.push_back(fluidmap[i]);
  }

  Teuchos::RCP<Epetra_Map> structfsidbcmap = Teuchos::rcp(
      new Epetra_Map(-1, structfsidbcvector.size(), structfsidbcvector.data(), 0, Comm()));
  // FOUR_C_ASSERT(fluidfsidbcmap->UniqueGIDs(),"fsidbcmap is not unique!");

  return structfsidbcmap;
}

void PoroElast::MonolithicSplit::setup_coupling_and_matrices()
{
  const int ndim = Global::Problem::Instance()->NDim();
  icoupfs_->setup_condition_coupling(*structure_field()->discretization(),
      structure_field()->Interface()->fsi_cond_map(), *fluid_field()->discretization(),
      fluid_field()->Interface()->fsi_cond_map(), "FSICoupling", ndim);

  fsibcmap_ = fsidbc_map();

  evaluateinterface_ = structure_field()->Interface()->fsi_cond_relevant();

  if (evaluateinterface_)
  {
    if (fsibcmap_->NumGlobalElements())
    {
      const Teuchos::RCP<Adapter::FluidPoro>& fluidfield =
          Teuchos::rcp_dynamic_cast<Adapter::FluidPoro>(fluid_field());
      fluidfield->add_dirich_cond(fsibcmap_);

      fsibcextractor_ = Teuchos::rcp(
          new Core::LinAlg::MapExtractor(*fluid_field()->Interface()->fsi_cond_map(), fsibcmap_));
    }

    Teuchos::RCP<const Epetra_Vector> idispnp =
        structure_field()->Interface()->extract_fsi_cond_vector(structure_field()->Dispnp());
    ddi_ = Teuchos::rcp(new Epetra_Vector(idispnp->Map(), true));
  }

  // initialize Poroelasticity-systemmatrix_
  systemmatrix_ =
      Teuchos::rcp(new Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          *Extractor(), *Extractor(), 81, false, true));

  // initialize coupling matrices
  k_fs_ =
      Teuchos::rcp(new Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          *(structure_field()->Interface()), *(fluid_field()->Interface()), 81, false, true));

  k_sf_ =
      Teuchos::rcp(new Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          *(fluid_field()->Interface()), *(structure_field()->Interface()), 81, false, true));
}

void PoroElast::MonolithicSplit::build_combined_dbc_map()
{
  TEUCHOS_FUNC_TIME_MONITOR("PoroElast::MonolithicSplit::combined_dbc_map");

  // first, get DBC-maps from structure and fluid field and merge them to one map
  const Teuchos::RCP<const Epetra_Map> scondmap =
      structure_field()->get_dbc_map_extractor()->cond_map();
  const Teuchos::RCP<const Epetra_Map> fcondmap =
      fluid_field()->get_dbc_map_extractor()->cond_map();

  std::vector<Teuchos::RCP<const Epetra_Map>> vectoroverallfsimaps;
  vectoroverallfsimaps.push_back(scondmap);
  vectoroverallfsimaps.push_back(fcondmap);

  Teuchos::RCP<Epetra_Map> overallfsidbcmaps =
      Core::LinAlg::MultiMapExtractor::merge_maps(vectoroverallfsimaps);

  // now we intersect the global dof map with the DBC map to get all dofs with DBS applied, which
  // are in the global
  // system, i.e. are not condensed
  std::vector<Teuchos::RCP<const Epetra_Map>> vectordbcmaps;
  vectordbcmaps.emplace_back(overallfsidbcmaps);
  vectordbcmaps.emplace_back(fullmap_);

  combinedDBCMap_ = Core::LinAlg::MultiMapExtractor::intersect_maps(vectordbcmaps);
}

void PoroElast::MonolithicSplit::Solve()
{
  // solve monolithic system by newton iteration
  Monolithic::Solve();

  // recover Lagrange multiplier \lambda_\Gamma at the interface at the end of each time step
  // (i.e. condensed forces onto the structure) needed for rhs in next time step
  recover_lagrange_multiplier_after_time_step();
}

FOUR_C_NAMESPACE_CLOSE