/*----------------------------------------------------------------------*/
/*! \file
\brief Class for monolithic fluid-fluid-FSI using XFEM

\level 3

*----------------------------------------------------------------------*/

#include "4C_fsi_fluidfluidmonolithic_fluidsplit_nonox.hpp"

#include "4C_adapter_ale_xffsi.hpp"
#include "4C_adapter_fld_fluid_fluid_fsi.hpp"
#include "4C_adapter_str_fsiwrapper.hpp"
#include "4C_ale_utils_mapextractor.hpp"
#include "4C_constraint_manager.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_fsi_debugwriter.hpp"
#include "4C_fsi_monolithic_linearsystem.hpp"
#include "4C_fsi_statustest.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_ale.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_inpar_xfem.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_matrixtransform.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_structure_aux.hpp"

FOUR_C_NAMESPACE_OPEN

FSI::FluidFluidMonolithicFluidSplitNoNOX::FluidFluidMonolithicFluidSplitNoNOX(
    const Epetra_Comm& comm, const Teuchos::ParameterList& timeparams)
    : MonolithicNoNOX(comm, timeparams)
{
  // Determine fluid (=slave) DOF on the FSI interface, for which a Dirichlet boundary
  // condition (DBC) has been prescribed
  std::vector<Teuchos::RCP<const Epetra_Map>> intersectionmaps;

  // Fluid DBC-DOFs
  intersectionmaps.push_back(fluid_field()->get_dbc_map_extractor()->cond_map());
  // Fluid interface DOF
  intersectionmaps.push_back(fluid_field()->Interface()->fsi_cond_map());

  // intersect
  Teuchos::RCP<Epetra_Map> intersectionmap =
      Core::LinAlg::MultiMapExtractor::intersect_maps(intersectionmaps);

  // It is not allowed, that slave DOFs at the interface hold a Dirichlet
  if (intersectionmap->NumGlobalElements() != 0)
  {
    std::stringstream errormsg;
    errormsg << "  "
                "+---------------------------------------------------------------------------------"
                "------------+"
             << std::endl
             << "  |                DIRICHLET BOUNDARY CONDITIONS ON SLAVE SIDE OF FSI INTERFACE   "
                "              |"
             << std::endl
             << "  "
                "+---------------------------------------------------------------------------------"
                "------------+"
             << std::endl
             << "  | NOTE: The slave side of the interface is not allowed to carry Dirichlet "
                "boundary conditions.|"
             << std::endl
             << "  |                                                                               "
                "              |"
             << std::endl
             << "  | This is a fluid split scheme. Hence, master and slave field are chosen as "
                "follows:          |"
             << std::endl
             << "  |     MASTER  = STRUCTURE                                                       "
                "              |"
             << std::endl
             << "  |     SLAVE   = FLUID                                                           "
                "              |"
             << std::endl
             << "  |                                                                               "
                "              |"
             << std::endl
             << "  | Dirichlet boundary conditions were detected on slave interface degrees of "
                "freedom. Please   |"
             << std::endl
             << "  | remove Dirichlet boundary conditions from the slave side of the FSI "
                "interface.              |"
             << std::endl
             << "  | Only the master side of the FSI interface is allowed to carry Dirichlet "
                "boundary conditions.|"
             << std::endl
             << "  "
                "+---------------------------------------------------------------------------------"
                "------------+"
             << std::endl;

    FOUR_C_THROW(errormsg.str());
  }

  // Initialization of row/column transformation objects
  // These are needed for the system matrix setup,
  // as matrices from 3 different fields (S,F,A) are set together.
  fggtransform_ = Teuchos::rcp(new Core::LinAlg::MatrixRowColTransform);
  fmggtransform_ = Teuchos::rcp(new Core::LinAlg::MatrixRowColTransform);

  fgitransform_ = Teuchos::rcp(new Core::LinAlg::MatrixRowTransform);
  figtransform_ = Teuchos::rcp(new Core::LinAlg::MatrixColTransform);
  fmiitransform_ = Teuchos::rcp(new Core::LinAlg::MatrixColTransform);
  fmgitransform_ = Teuchos::rcp(new Core::LinAlg::MatrixRowColTransform);
  aigtransform_ = Teuchos::rcp(new Core::LinAlg::MatrixColTransform);

  // Lagrange multiplier
  lambda_ = Teuchos::rcp(new Epetra_Vector(*fluid_field()->Interface()->fsi_cond_map(), true));

  // Storage for matrices from previous time steps
  fggcur_ = Teuchos::null;
  fgicur_ = Teuchos::null;
  fmggcur_ = Teuchos::null;
  fmgicur_ = Teuchos::null;

  // Structural predictor step, initially filled with zeros
  ddgpred_ = Teuchos::rcp(new Epetra_Vector(*structure_field()->Interface()->fsi_cond_map(), true));
}


/*----------------------------------------------------------------------
 * SetupSystem:
 *    - Setup field coupling
 *    - System matrix initialization
 *    - Build of combined DOF map, including all DOFs
 *      except from the fluid FSI DOFs
 *----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplitNoNOX::SetupSystem()
{
  FSI::MonolithicNoNOX::SetupSystem();

  // Create a combined map for Structure/Fluid/ALE-DOFs all in one
  create_combined_dof_row_map();

  // Tell fluid field to split the fluid system matrix
  // (indicate the fluid split)
  fluid_field()->use_block_matrix(true);

  // Build the ALE-matrix in split system
  ale_field()->create_system_matrix(ale_field()->Interface());

  // Initialize the global system matrix!
  systemmatrix_ =
      Teuchos::rcp(new Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          extractor(), extractor(), 81, false, true));
}

/*----------------------------------------------------------------------
 * setup_rhs:
 *   - Build the RHS-vector for the Newton-loop!
 *   - The RHS-vector is made up of 2 parts:
 *     the single-field RHS-contributions and special
 *     terms resulting from condensation of fluid-DOFs from the FSI interface
 *     and predictor steps. These terms are added at the first Newton
 *     step only!
 *----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplitNoNOX::setup_rhs(Epetra_Vector& f, bool firstcall)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (fluid_field()->RHS() == Teuchos::null) FOUR_C_THROW("empty fluid residual");
#endif

  // Get the contributions from the field residuals
  setup_vector(f, structure_field()->RHS(), fluid_field()->RHS(), ale_field()->RHS(),
      fluid_field()->residual_scaling());

  /*----------------------------------------------------------------------
  The following terms are added only at the first Newton iteration!
  ----------------------------------------------------------------------*/

  if (firstcall)
  {
    // Store fluid interface velocity:
    Teuchos::RCP<const Epetra_Vector> fveln = fluid_field()->extract_interface_veln();

    /*----------------------------------------------------------------------*/
    // Time integration parameters
    /*----------------------------------------------------------------------*/
    // Structure:
    // a*x_n+(1-a)*x_n+1
    // Fluid:
    // b*y_n+(1-b)*y_n+1
    // a: stimintparam
    // b: ftimintparam
    /*----------------------------------------------------------------------*/

    const double stimintparam = structure_field()->TimIntParam();
    const double ftimintparam = fluid_field()->TimIntParam();

    // Fluid time scaling parameter
    // fluidtimescale: \tau
    //  \tau = 1/\Delta t  for Backward-Euler;
    //  \tau = 2/\Delta t  for Trapezoidal Rule
    const double fluidtimescale = fluid_field()->TimeScaling();
    const double fluidresidualscale = fluid_field()->residual_scaling();

    // Get the fluid block system matrix "blockf"
    // shape derivative matrix (linearization
    // of Navier-Stokes with respect to mesh movement: moving mesh matrix "mmm")
    // ALE block matrix "blocka"
    // F_...

    // Fluid block system matrix
    const Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> blockf =
        fluid_field()->BlockSystemMatrix();
    if (blockf == Teuchos::null) FOUR_C_THROW("Expected fluid block matrix...");

    // F^G_...
    // Fluid shape derivative matrix
    const Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> mmm = fluid_field()->ShapeDerivatives();

    // A_...
    // ALE block system matrix
    const Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> blocka =
        ale_field()->BlockSystemMatrix();
    if (blocka == Teuchos::null) FOUR_C_THROW("Expected ALE block matrix...");

    // Extracting submatrices for fluid & ALE from block field matrices:
    // F_{\Gamma\Gamma}, F_{I\Gamma} and A_{I\Gamma}
    const Core::LinAlg::SparseMatrix& fgg = blockf->Matrix(1, 1);
    const Core::LinAlg::SparseMatrix& fig = blockf->Matrix(0, 1);
    const Core::LinAlg::SparseMatrix& aig = blocka->Matrix(0, 1);

    // Vector for storage of the temporary result for the RHS vector
    Teuchos::RCP<Epetra_Vector> rhs = Teuchos::null;

    /*----------------------------------------------------------------------*/
    // Starting the setup!
    /*----------------------------------------------------------------------*/

    // Step 1: Taking care of the  DOFs @ structural side
    /*
     * (1)  + (1-stintparam)/(1-flintparam)* \Delta t * timescale * F_{\Gamma\Gamma} *
     * u^{n}_{\Gamma}
     *
     * (2)  - (1-stintparam)/(1-flintparam) * F^{G}_{\Gamma\Gamma} * \Delta d_{\Gamma,p}
     *
     * (3)  - (1-stintparam)/(1-flintparam) * timescale * F_{\Gamma\Gamma} * \Delta d_{\Gamma,p}
     *
     *
     */

    // ------------------
    // Build term (1):
    // ------------------

    // Create zero-filled vector copy based on the row map of F_{\Gamma\Gamma}
    rhs = Teuchos::rcp(new Epetra_Vector(fgg.RangeMap(), true));

    // Compute F_{\Gamma\Gamma}*u^n_\Gamma
    // Write into rhs
    fgg.Apply(*fveln, *rhs);

    // Apply scaling
    rhs->Scale(
        fluidresidualscale * (1.0 - stimintparam) / (1.0 - ftimintparam) * Dt() * fluidtimescale);

    // Insert into structural side of the interface
    rhs = fluid_to_struct(rhs);
    rhs = structure_field()->Interface()->insert_fsi_cond_vector(rhs);

    // Add to the structure block (index 0) of the final RHS vector f
    extractor().add_vector(*rhs, 0, f);

    if (mmm != Teuchos::null)
    {
      // ------------------
      // Build term (2):
      // ------------------

      // Extract the matrix F^G_\Gamma_\Gamma
      const Core::LinAlg::SparseMatrix& fmgg = mmm->Matrix(1, 1);

      // Re-initialize rhs
      rhs = Teuchos::rcp(new Epetra_Vector(fmgg.RangeMap(), true));


      // Compute F^{G}_{\Gamma\Gamma} * \Delta d_{\Gamma,p}
      // Write into rhs
      fmgg.Apply(*struct_to_fluid(ddgpred_), *rhs);

      // Apply scaling
      rhs->Scale(-1.0 * (1.0 - stimintparam) / (1.0 - ftimintparam));

      // Insert into structure side of the interface
      rhs = fluid_to_struct(rhs);
      rhs = structure_field()->Interface()->insert_fsi_cond_vector(rhs);

      extractor().add_vector(*rhs, 0, f);
    }

    // ------------------
    // Build term (3):
    // ------------------

    // Re-initialize rhs
    rhs = Teuchos::rcp(new Epetra_Vector(fgg.RangeMap(), true));

    // Compute F_{\Gamma\Gamma} * \Delta d_{\Gamma,p}
    // Write into rhs
    fgg.Apply(*struct_to_fluid(ddgpred_), *rhs);

    // Apply scaling
    rhs->Scale(
        -1.0 * fluidresidualscale * (1.0 - stimintparam) / (1.0 - ftimintparam) * fluidtimescale);

    // Insert into structure side of the interface
    rhs = fluid_to_struct(rhs);
    rhs = structure_field()->Interface()->insert_fsi_cond_vector(rhs);

    extractor().add_vector(*rhs, 0, f);


    // Step 2: Fluid sided inner DOFs
    /*
     * (1) F_{I\Gamma} * \Delta t* timescale * u^{n}_{\Gamma}
     *
     * (2)  - timescale * F_{I\Gamma} * \Delta d_{\Gamma,p}
     *
     * (3) - F^{G}_{I\Gamma} * \Delta d_{\Gamma,p}
     *
     */

    // ------------------
    // Build term (1):
    // ------------------

    // Re-initialize rhs
    rhs = Teuchos::rcp(new Epetra_Vector(fig.RangeMap(), true));

    // Compute term F_{I\Gamma} *u^{n}_{\Gamma}
    // Write into rhs
    fig.Apply(*fveln, *rhs);

    // Apply scaling
    rhs->Scale(fluidtimescale * Dt());

    // Add to the final vector f, to the fluid block (index 1)
    extractor().add_vector(*rhs, 1, f);


    // ------------------
    // Build term (2):
    // ------------------

    // Re-initialize rhs
    rhs = Teuchos::rcp(new Epetra_Vector(fig.RangeMap(), true));

    // Compute term F_{I\Gamma} * \Delta d_{\Gamma,p}
    // Write into rhs
    fig.Apply(*struct_to_fluid(ddgpred_), *rhs);

    // Apply scaling
    rhs->Scale(-fluidtimescale);

    extractor().add_vector(*rhs, 1, f);

    if (mmm != Teuchos::null)
    {
      // ------------------
      // Build term (3):
      // ------------------

      // Extract the matrix F^G_I\Gamma
      const Core::LinAlg::SparseMatrix& fmig = mmm->Matrix(0, 1);

      // Re-initialize rhs
      rhs = Teuchos::rcp(new Epetra_Vector(fmig.RangeMap(), true));

      // Compute F^{G}_{I\Gamma} * \Delta d_{\Gamma,p}
      // Write into rhs
      fmig.Apply(*struct_to_fluid(ddgpred_), *rhs);

      // Apply scaling
      rhs->Scale(-1.0);

      extractor().add_vector(*rhs, 1, f);
    }

    // Step 3: Inner ALE DOFs
    //
    // Adding terms to ALE-Part of RHS-vector
    // -A_{I\Gamma} * \Delta d_{\Gamma,p}

    // Re-initialize rhs
    rhs = Teuchos::rcp(new Epetra_Vector(aig.RangeMap(), true));

    // Compute term A_{I\Gamma} * \Delta d_{\Gamma,p}
    // Write into rhs
    aig.Apply(*struct_to_ale(ddgpred_), *rhs);

    // Apply scaling
    rhs->Scale(-1.0);

    extractor().add_vector(*rhs, 2, f);
  }
}

/*----------------------------------------------------------------------
 * setup_system_matrix:
 *
 *   - Build the final system block matrix extracting, scaling
 *     & transforming the single field submatrices
 ----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplitNoNOX::setup_system_matrix()
{
  const Core::Adapter::Coupling& coupsf = structure_fluid_coupling();
  const Core::Adapter::Coupling& coupsa = structure_ale_coupling();
  const Core::Adapter::Coupling& coupfa = fluid_ale_coupling();

  // get single field block matrices
  Teuchos::RCP<Core::LinAlg::SparseMatrix> s =
      structure_field()->system_matrix();  // can't be 'const' --> is modified by STC
  const Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> f = fluid_field()->BlockSystemMatrix();
  const Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> a = ale_field()->BlockSystemMatrix();

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // check whether allocation was successful
  if (s == Teuchos::null)
  {
    FOUR_C_THROW("expect structure block matrix");
  }
  if (f == Teuchos::null)
  {
    FOUR_C_THROW("expect fluid block matrix");
  }
  if (a == Teuchos::null)
  {
    FOUR_C_THROW("expect ale block matrix");
  }
#endif

  // extract submatrices
  Core::LinAlg::SparseMatrix& fii = f->Matrix(0, 0);
  Core::LinAlg::SparseMatrix& fig = f->Matrix(0, 1);
  Core::LinAlg::SparseMatrix& fgi = f->Matrix(1, 0);
  Core::LinAlg::SparseMatrix& fgg = f->Matrix(1, 1);
  Core::LinAlg::SparseMatrix& aii = a->Matrix(0, 0);
  Core::LinAlg::SparseMatrix& aig = a->Matrix(0, 1);

  // scaling factors for fluid
  const double scale = fluid_field()->residual_scaling();
  const double timescale = fluid_field()->TimeScaling();

  // get time integration parameters of structure and fluid time integrators
  // to enable consistent time integration among the fields
  const double stiparam = structure_field()->TimIntParam();
  const double ftiparam = fluid_field()->TimIntParam();

  /*----------------------------------------------------------------------*/

  // build block matrix
  // The maps of the block matrix have to match the maps of the blocks we
  // insert here.

  // uncomplete because the fluid interface can have more connections than the
  // structural one. (Tet elements in fluid can cause this.) We should do
  // this just once...
  s->UnComplete();

  (*fggtransform_)(fgg, (1.0 - stiparam) / (1.0 - ftiparam) * scale * timescale,
      Core::Adapter::CouplingSlaveConverter(coupsf), Core::Adapter::CouplingSlaveConverter(coupsf),
      *s, true, true);

  Teuchos::RCP<Core::LinAlg::SparseMatrix> lfgi =
      Teuchos::rcp(new Core::LinAlg::SparseMatrix(s->RowMap(), 81, false));
  (*fgitransform_)(fgi, (1.0 - stiparam) / (1.0 - ftiparam) * scale,
      Core::Adapter::CouplingSlaveConverter(coupsf), *lfgi);

  lfgi->Complete(fgi.DomainMap(), s->RangeMap());

  systemmatrix_->Assign(0, 1, Core::LinAlg::View, *lfgi);

  Teuchos::RCP<Core::LinAlg::SparseMatrix> lfig =
      Teuchos::rcp(new Core::LinAlg::SparseMatrix(fig.RowMap(), 81, false));
  (*figtransform_)(f->FullRowMap(), f->FullColMap(), fig, timescale,
      Core::Adapter::CouplingSlaveConverter(coupsf), systemmatrix_->Matrix(1, 0));

  systemmatrix_->Assign(1, 1, Core::LinAlg::View, fii);

  (*aigtransform_)(a->FullRowMap(), a->FullColMap(), aig, 1.,
      Core::Adapter::CouplingSlaveConverter(coupsa), systemmatrix_->Matrix(2, 0));

  systemmatrix_->Assign(2, 2, Core::LinAlg::View, aii);

  /*----------------------------------------------------------------------*/
  // add optional blocks from fluid linearization with respect to mesh motion

  Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> mmm = fluid_field()->ShapeDerivatives();
  if (mmm != Teuchos::null)
  {
    Core::LinAlg::SparseMatrix& fmii = mmm->Matrix(0, 0);
    Core::LinAlg::SparseMatrix& fmgi = mmm->Matrix(1, 0);

    Core::LinAlg::SparseMatrix& fmig = mmm->Matrix(0, 1);
    Core::LinAlg::SparseMatrix& fmgg = mmm->Matrix(1, 1);

    // reuse transform objects to add shape derivative matrices to structural blocks
    (*figtransform_)(f->FullRowMap(), f->FullColMap(), fmig, 1.,
        Core::Adapter::CouplingSlaveConverter(coupsf), systemmatrix_->Matrix(1, 0), false, true);


    (*fmggtransform_)(fmgg, (1.0 - stiparam) / (1.0 - ftiparam) * scale,
        Core::Adapter::CouplingSlaveConverter(coupsf),
        Core::Adapter::CouplingSlaveConverter(coupsf), *s, false, true);

    // We cannot copy the pressure value. It is not used anyway. So no exact
    // match here.
    (*fmiitransform_)(mmm->FullRowMap(), mmm->FullColMap(), fmii, 1.,
        Core::Adapter::CouplingMasterConverter(coupfa), systemmatrix_->Matrix(1, 2), false);

    {
      Teuchos::RCP<Core::LinAlg::SparseMatrix> lfmgi =
          Teuchos::rcp(new Core::LinAlg::SparseMatrix(s->RowMap(), 81, false));
      (*fmgitransform_)(fmgi, (1.0 - stiparam) / (1.0 - ftiparam) * scale,
          Core::Adapter::CouplingSlaveConverter(coupsf),
          Core::Adapter::CouplingMasterConverter(coupfa), *lfmgi, false, false);

      lfmgi->Complete(aii.DomainMap(), s->RangeMap());

      systemmatrix_->Assign(0, 2, Core::LinAlg::View, *lfmgi);
    }
  }

  // finally assign structure block
  systemmatrix_->Matrix(0, 0).Assign(Core::LinAlg::View, *s);

  // done. make sure all blocks are filled.
  systemmatrix_->Complete();

  // Store some submatrices required for RHS-Setup
  // to know them for LM-recovery!
  // store parts of fluid matrix to know them in the next iteration as previous iteration matrices
  fgiprev_ = fgicur_;
  fggprev_ = fggcur_;
  fgicur_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(f->Matrix(1, 0)));
  fggcur_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(f->Matrix(1, 1)));

  // store parts of fluid shape derivative matrix to know them in the next iteration as previous
  // iteration matrices
  fmgiprev_ = fmgicur_;
  fmggprev_ = fmggcur_;
  if (mmm != Teuchos::null)
  {
    fmgicur_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(mmm->Matrix(1, 0)));
    fmggcur_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(mmm->Matrix(1, 1)));
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/

void FSI::FluidFluidMonolithicFluidSplitNoNOX::initial_guess(Teuchos::RCP<Epetra_Vector> ig)
{
  setup_vector(*ig, structure_field()->initial_guess(), fluid_field()->initial_guess(),
      ale_field()->initial_guess(), 0.0);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Map> FSI::FluidFluidMonolithicFluidSplitNoNOX::combined_dbc_map()
{
  // Create a combined map vector with the 3 field DBC maps
  std::vector<Teuchos::RCP<const Epetra_Map>> alldbcmaps;

  // structure DBC
  alldbcmaps.push_back(structure_field()->get_dbc_map_extractor()->cond_map());
  // fluid DBC
  alldbcmaps.push_back(fluid_field()->get_dbc_map_extractor()->cond_map());
  // ALE-DBC
  std::vector<Teuchos::RCP<const Epetra_Map>> aleintersectionmaps;
  aleintersectionmaps.push_back(ale_field()->get_dbc_map_extractor()->cond_map());
  aleintersectionmaps.push_back(ale_field()->Interface()->other_map());
  Teuchos::RCP<Epetra_Map> aleintersectionmap =
      Core::LinAlg::MultiMapExtractor::intersect_maps(aleintersectionmaps);
  alldbcmaps.push_back(aleintersectionmap);

  // Merge the maps
  Teuchos::RCP<Epetra_Map> alldbcmap = Core::LinAlg::MultiMapExtractor::merge_maps(alldbcmaps);

  return alldbcmap;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplitNoNOX::setup_vector(Epetra_Vector& f,
    Teuchos::RCP<const Epetra_Vector> sv, Teuchos::RCP<const Epetra_Vector> fv,
    Teuchos::RCP<const Epetra_Vector> av, double fluidscale)
{
  // Writes the following entries into f :

  // [
  // f^S_I
  // -----
  // f^S_{\Gamma}+
  // (1-stintparam_)/(1-flintparam)*fluidscale*f^F_{\Gamma}
  // lambda*(stintparam_-flintparam*(1-stintparam_)/(1-flintparam)
  // -----
  // f^ F_I
  // -----
  // 0
  // ]

  /*----------------------------------------------------------------------*/
  // Time integration parameters
  /*----------------------------------------------------------------------*/
  // Structure:
  // a*x_n+(1-a)*x_n+1
  // Fluid:
  // b*y_n+(1-b)*y_n+1
  // a: stimintparam
  // b: ftimintparam
  /*----------------------------------------------------------------------*/

  const double stimintparam = structure_field()->TimIntParam();
  const double ftimintparam = fluid_field()->TimIntParam();


  // Extract inner DOFs for ALE-field
  Teuchos::RCP<Epetra_Vector> aov = ale_field()->Interface()->extract_other_vector(av);

  // Get the FSI-interface RHS-vector for the fluid side!
  Teuchos::RCP<Epetra_Vector> fcv = fluid_field()->Interface()->extract_fsi_cond_vector(fv);

  // Convert previously extracted vector to structure !
  Teuchos::RCP<Epetra_Vector> modsv =
      structure_field()->Interface()->insert_fsi_cond_vector(fluid_to_struct(fcv));

  // Add the converted interface RHS-contributions (scaled) to the global structural RHS!
  int err = modsv->Update(1.0, *sv, (1.0 - stimintparam) / (1.0 - ftimintparam) * fluidscale);
  if (err) FOUR_C_THROW("Update of structural residual vector failed! Error code %i", err);

  // Add the previous Lagrange Multiplier
  if (lambda_ != Teuchos::null)
  {
    Teuchos::RCP<Epetra_Vector> lambdaglob =
        structure_field()->Interface()->insert_fsi_cond_vector(fluid_to_struct(lambda_));
    err = modsv->Update(stimintparam - ftimintparam * (1.0 - stimintparam) / (1.0 - ftimintparam),
        *lambdaglob, 1.0);
    if (err) FOUR_C_THROW("Update of structural residual vector failed! Error code %i", err);

    // Insert structural contribution
    extractor().insert_vector(*modsv, 0, f);
  }
  else
  {
    extractor().insert_vector(*sv, 0, f);
  }

  Teuchos::RCP<Epetra_Vector> fglobalv = fluid_field()->Interface()->extract_other_vector(fv);

  // Insert fluid contribution
  extractor().insert_vector(*fglobalv, 1, f);

  // Insert ALE contribution
  extractor().insert_vector(*aov, 2, f);

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplitNoNOX::extract_field_vectors(
    Teuchos::RCP<const Epetra_Vector> x, Teuchos::RCP<const Epetra_Vector>& sx,
    Teuchos::RCP<const Epetra_Vector>& fx, Teuchos::RCP<const Epetra_Vector>& ax)
{
  /*----------------------------------------------------------------------*/
  // Process structure unknowns
  /*----------------------------------------------------------------------*/
  // Extract whole structure field vector
  sx = extractor().extract_vector(x, 0);

  // Structural part of FSI interface
  Teuchos::RCP<Epetra_Vector> scx = structure_field()->Interface()->extract_fsi_cond_vector(sx);

  /*----------------------------------------------------------------------*/
  // Process ALE unknowns
  /*----------------------------------------------------------------------*/
  Teuchos::RCP<const Epetra_Vector> aox = extractor().extract_vector(x, 2);
  // Update interface part of structure vector with predictor increment
  scx->Update(1.0, *ddgpred_, 1.0);
  Teuchos::RCP<Epetra_Vector> acx = struct_to_ale(scx);

  Teuchos::RCP<Epetra_Vector> a = ale_field()->Interface()->insert_other_vector(aox);
  // Insert the FSI-DOF vector into full vector a
  ale_field()->Interface()->insert_fsi_cond_vector(acx, a);
  // Write a into passed argument ax
  ax = a;

  /*----------------------------------------------------------------------*/
  // Process fluid unknowns
  /*----------------------------------------------------------------------*/
  // Extract vector of fluid unknowns from x
  Teuchos::RCP<const Epetra_Vector> fox = extractor().extract_vector(x, 1);

  // Conversion ALE displacement to fluid field:
  Teuchos::RCP<Epetra_Vector> fcx = ale_to_fluid_interface(acx);
  fluid_field()->displacement_to_velocity(fcx);


  // The previously computed fluid interface values have to be inserted into the fluid field vector
  Teuchos::RCP<Epetra_Vector> f = fluid_field()->Interface()->insert_other_vector(fox);
  fluid_field()->Interface()->insert_fsi_cond_vector(fcx, f);

  fx = f;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplitNoNOX::read_restart(int step)
{
  // Read Lagrange Multiplier (associated with embedded fluid)
  {
    Teuchos::RCP<Epetra_Vector> lambdaemb = Teuchos::rcp(
        new Epetra_Vector(*(fluid_field()->x_fluid_fluid_map_extractor()->fluid_map()), true));
    Core::IO::DiscretizationReader reader = Core::IO::DiscretizationReader(
        fluid_field()->discretization(), Global::Problem::Instance()->InputControlFile(), step);
    reader.read_vector(lambdaemb, "fsilambda");
    // Insert into vector containing the whole merged fluid DOF
    Teuchos::RCP<Epetra_Vector> lambdafull =
        fluid_field()->x_fluid_fluid_map_extractor()->insert_fluid_vector(lambdaemb);
    lambda_ = fluid_field()->Interface()->extract_fsi_cond_vector(lambdafull);
  }

  structure_field()->read_restart(step);
  fluid_field()->read_restart(step);
  ale_field()->read_restart(step);

  SetTimeStep(fluid_field()->Time(), fluid_field()->Step());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplitNoNOX::output()
{
  structure_field()->output();
  fluid_field()->output();

  // output Lagrange multiplier
  {
    // the Lagrange multiplier lives on the FSI interface
    // for output, we want to insert lambda into a full vector, defined on the embedded fluid field
    // 1. insert into vector containing all fluid DOF
    Teuchos::RCP<Epetra_Vector> lambdafull =
        fluid_field()->Interface()->insert_fsi_cond_vector(lambda_);

    // 2. extract the embedded fluid part
    Teuchos::RCP<Epetra_Vector> lambdaemb =
        fluid_field()->x_fluid_fluid_map_extractor()->extract_fluid_vector(lambdafull);

    const Teuchos::ParameterList& fsidyn = Global::Problem::Instance()->FSIDynamicParams();
    const int uprestart = fsidyn.get<int>("RESTARTEVRY");
    const int upres = fsidyn.get<int>("RESULTSEVRY");
    if ((uprestart != 0 && fluid_field()->Step() % uprestart == 0) ||
        fluid_field()->Step() % upres == 0)
      fluid_field()->DiscWriter()->write_vector("fsilambda", lambdaemb);
  }
  ale_field()->output();

  if (structure_field()->get_constraint_manager()->HaveMonitor())
  {
    structure_field()->get_constraint_manager()->compute_monitor_values(
        structure_field()->Dispnp());
    if (Comm().MyPID() == 0) structure_field()->get_constraint_manager()->PrintMonitorValues();
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplitNoNOX::create_combined_dof_row_map()
{
  // Create a combined map for Structure/Fluid/ALE-DOFs all in one
  std::vector<Teuchos::RCP<const Epetra_Map>> vecSpaces;

  // Append the structural DOF map
  vecSpaces.push_back(structure_field()->dof_row_map());

  // Append the final fluid DOF map, free of FSI DOF
  vecSpaces.push_back(fluid_field()->Interface()->other_map());

  // Append ALE DOF map
  vecSpaces.push_back(ale_field()->Interface()->other_map());

  // If the non-FSI fluid maps are empty
  if (vecSpaces[1]->NumGlobalElements() == 0)
    FOUR_C_THROW("No inner fluid equations. Can't split!");

  // The vector is complete, fill the system's global BlockRowMap
  // with the maps previously set together!
  set_dof_row_maps(vecSpaces);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplitNoNOX::newton()
{
  /*----------------------------------------------------------------------
  Extract predictor increments
  ----------------------------------------------------------------------*/
  // Increment of structural interface displacement --> structural predictor!!
  ddgpred_ = Teuchos::rcp(new Epetra_Vector(*structure_field()->extract_interface_dispnp()));
  ddgpred_->Update(-1.0, *structure_field()->extract_interface_dispn(), 1.0);

  /*----------------------------------------------------------------------*/
  // Initialize the increment vectors, they are updated in evaluate(...)->extract_field_vectors(...)
  // at every Newton iteration!

  // Initialization for 1st Newton call
  // structural interface predictor
  ddginc_ = Teuchos::rcp(new Epetra_Vector(*ddgpred_));
  ddialeinc_ = Teuchos::rcp(new Epetra_Vector(*ale_field()->Interface()->other_map()), true);
  duiinc_ = Teuchos::rcp(new Epetra_Vector(*fluid_field()->Interface()->other_map(), true));

  FSI::MonolithicNoNOX::newton();

  /*----------------------------------------------------------------------*/
  // Compute the increments needed for recovery of Lagrange Multiplier!
  // After the last Newton iteration, the increments are not updated.
  // We need the last increment for the recovery of lambda.
  /*----------------------------------------------------------------------*/
  // Fluid
  duiinc_->Update(1.0, *extractor().extract_vector(iterinc_, 1), 0.0);
  // Structure
  Teuchos::RCP<Epetra_Vector> ddinc = extractor().extract_vector(iterinc_, 0);
  ddginc_->Update(1.0, *structure_field()->Interface()->extract_fsi_cond_vector(ddinc), 0.0);
  // ALE
  ddialeinc_->Update(1.0, *extractor().extract_vector(iterinc_, 2), 0.0);
}

/*----------------------------------------------------------------------
 * build_convergence_norms:
 *     - Calculate the residual and incremental norms required for
 *        the convergence test in Newton-loop
 *     - Implemented:
 *         The (Euclidian) L2-Norm
 ----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplitNoNOX::build_convergence_norms()
{
  /*----------------------------------------------------------------------
  Residual norms - L2 of:
  - global rhs
  - inner structural rhs
  - ale rhs
  - inner fluid-fluid velocity rhs
  - inner fluid-fluid pressure rhs
  - complete interface residual
  ----------------------------------------------------------------------*/
  // Norm of global RHS vector
  rhs_->Norm2(&normrhs_);

  // Inner structural RHS and interface RHS

  // RHS-vector from fluid_field() without FSI-DOFs
  Teuchos::RCP<const Epetra_Vector> innerfluidfluidrhs = extractor().extract_vector(rhs_, 1);
  // (Inner) ALE RHS
  Teuchos::RCP<const Epetra_Vector> alerhs = extractor().extract_vector(rhs_, 2);

  // Norm of inner structural residual forces
  Teuchos::RCP<const Epetra_Vector> structrhs = extractor().extract_vector(rhs_, 0);
  structure_field()->Interface()->extract_other_vector(structrhs)->Norm2(&normstrrhsL2_);
  structure_field()->Interface()->extract_other_vector(structrhs)->NormInf(&normstrrhsInf_);

  // Norm of ALE residual forces
  alerhs->Norm2(&normalerhsL2_);

  // Norm of fluid velocity residual
  // This requires an Epetra_Map of the inner fluid velocity DOFs first!
  Teuchos::RCP<const Epetra_Map> innerfluidvel = fluid_field()->InnerVelocityRowMap();

  // Merged FSI-free fluid maps
  Teuchos::RCP<const Epetra_Map> fluidmaps = fluid_field()->Interface()->other_map();
  // Create a MapExtractor to access the velocity DOFs from the FSI-free fluid map
  Teuchos::RCP<Core::LinAlg::MapExtractor> fluidvelextract =
      Teuchos::rcp(new Core::LinAlg::MapExtractor(*fluidmaps, innerfluidvel, true));

  // Finally, compute the fluid velocity RHS-norm
  fluidvelextract->extract_cond_vector(innerfluidfluidrhs)->Norm2(&normflvelrhsL2_);
  fluidvelextract->extract_cond_vector(innerfluidfluidrhs)->NormInf(&normflvelrhsInf_);

  // Norm of fluid pressure residual
  // This requires an Epetra_Map of the fluid pressure DOFs
  if (fluid_field()->PressureRowMap() == Teuchos::null) FOUR_C_THROW("Empty pressure row map!");

  // Finally, compute the fluid pressure RHS-norm
  fluidvelextract->extract_other_vector(innerfluidfluidrhs)->Norm2(&normflpresrhsL2_);
  fluidvelextract->extract_other_vector(innerfluidfluidrhs)->NormInf(&normflpresrhsInf_);

  // The true RHS for the FSI interface equation block consists of
  // more than just the structure residual, namely the scaled fluid interface residual and the
  // previous Lagrange multiplier. The first idea is, to test this whole term, which can be easily
  // extracted from rhs_. For a more strict testing, the L_inf-norm should be employed!
  Teuchos::RCP<Epetra_Vector> interfaceresidual =
      structure_field()->Interface()->extract_fsi_cond_vector(*structrhs);
  interfaceresidual->Norm2(&norminterfacerhsL2_);
  interfaceresidual->NormInf(&norminterfacerhsInf_);

  /*----------------------------------------------------------------------
  Incremental norms - L2
  - global inc
  - inner structural inc
  - complete interface inc
  - inner fluid-fluid velocity rhs
  - inner fluid-fluid pressure rhs
  ----------------------------------------------------------------------*/
  // Norm of global increment vector
  iterinc_->Norm2(&norminc_);

  Teuchos::RCP<const Epetra_Vector> structinc = extractor().extract_vector(iterinc_, 0);
  Teuchos::RCP<const Epetra_Vector> fluidinc = extractor().extract_vector(iterinc_, 1);

  // Norm of inner structural increment vector
  structure_field()->Interface()->extract_other_vector(structinc)->Norm2(&normstrincL2_);
  structure_field()->Interface()->extract_other_vector(structinc)->NormInf(&normstrincInf_);

  // Norm of interface increment vector
  structure_field()->Interface()->extract_fsi_cond_vector(structinc)->Norm2(&norminterfaceincL2_);
  structure_field()->Interface()->extract_fsi_cond_vector(structinc)->NormInf(
      &norminterfaceincInf_);

  // Norm of fluid velocity increment
  fluidvelextract->extract_cond_vector(fluidinc)->Norm2(&normflvelincL2_);
  fluidvelextract->extract_cond_vector(fluidinc)->NormInf(&normflvelincInf_);

  // Norm of fluid pressure increment
  fluidvelextract->extract_other_vector(fluidinc)->Norm2(&normflpresincL2_);
  fluidvelextract->extract_other_vector(fluidinc)->NormInf(&normflpresincInf_);

  // Norm of ALE increment vector
  extractor().extract_vector(iterinc_, 2)->Norm2(&normaleincL2_);

  // get length of the structural, fluid and ale vector
  ni_ = (structure_field()->Interface()->extract_fsi_cond_vector(*structrhs))->GlobalLength();
  ns_ = (structure_field()->Interface()->extract_other_vector(structrhs))->GlobalLength();
  nf_ = innerfluidfluidrhs->GlobalLength();
  nfv_ = fluidvelextract->extract_cond_vector(fluidinc)->GlobalLength();
  nfp_ = fluidvelextract->extract_other_vector(fluidinc)->GlobalLength();
  na_ = alerhs->GlobalLength();
  nall_ = rhs_->GlobalLength();
}

/*----------------------------------------------------------------------
 * recover_lagrange_multiplier:
 *     - Compute the Lagrange multiplier at the FSI-interface
 ----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplitNoNOX::recover_lagrange_multiplier()
{
  /*----------------------------------------------------------------------*/
  // Time integration parameter
  /*----------------------------------------------------------------------*/
  // Fluid
  // b*y_n+(1-b)*y_n+1
  // b:flintparam
  /*----------------------------------------------------------------------*/
  const double ftimintparam = fluid_field()->TimIntParam();

  // Fluid time scaling parameter
  // fluidtimescale: \tau
  // \tau = 1/\Delta t  for Backward-Euler;
  // \tau = 2/\Delta t  for Trapezoidal Rule
  const double fluidtimescale = fluid_field()->TimeScaling();

  // Scaling factor for different fluid/structural units
  const double fluidresidualscale = fluid_field()->residual_scaling();

  /*----------------------------------------------------------------------
    Lagrange Multiplier Setup
  ------------------------------------------------------------------------
   The Langrange Multiplier is updated as follows:

   lambda_^{n+1}=

       1/(1-flintparam)*(

   (1)  -flintparam*lambda_^n

   (2)  -r_{\Gamma}^{F,n+1}

   (3)  - 1/Tau *(F_{\Gamma\Gamma)})* \Delta d_{\Gamma}{S,n+1}

   (4)  -F^{G}_{\Gamma\Gamma} * \Delta d_{\Gamma}{S,n+1}

   (5)  -F_{\Gamma I} * \Delta u_{I}^{F,n+1}

   (6)  -F^{G}_{\Gamma I} * \Delta d_{I,n+1}^{G,n+1}

      Only at first Newton iteration:
   (7)  +dt / Tau * F_{\Gamma\Gamma} * u_{\Gamma}^n}
          )
    + tau: time scaling factor for interface time integration (tau = 1/fluid_field()->TimeScaling())
    + the terms 2 to 7 are first saved in a tmpvec which will be added to lambda
   ----------------------------------------------------------------------*/

  // creating & initializing the storage vectors for the last four terms
  Teuchos::RCP<Epetra_Vector> fggddg = Teuchos::null;
  Teuchos::RCP<Epetra_Vector> fmggddg = Teuchos::null;
  Teuchos::RCP<Epetra_Vector> fgidui = Teuchos::null;
  Teuchos::RCP<Epetra_Vector> fmgiddia = Teuchos::null;

  // stores intermediate result of terms (3)-(7)
  Teuchos::RCP<Epetra_Vector> tmpvec = Teuchos::null;

  // ---------Addressing term (2)
  // store f^F_{\Gamma}! As Recover-LM is called after the Newton loop, the RHS will have changed!
  Teuchos::RCP<Epetra_Vector> fluidresidual =
      fluid_field()->Interface()->extract_fsi_cond_vector(fluid_field()->RHS());

  // ---------Addressing term (1)
  lambda_->Update(ftimintparam, *lambda_, 0.0);

  // ---------Addressing term (2)
  tmpvec = Teuchos::rcp(new Epetra_Vector(*fluidresidual));
  tmpvec->Scale(-1.0);


  // ---------Addressing term (3)
  if (fggprev_ != Teuchos::null)
  {
    fggddg = Teuchos::rcp(new Epetra_Vector(fggprev_->RangeMap(), true));
    fggprev_->Apply(*struct_to_fluid(ddginc_), *fggddg);
    tmpvec->Update(fluidtimescale, *fggddg, 1.0);
  }

  //(4)
  if (fmggprev_ != Teuchos::null)
  {
    Teuchos::RCP<Epetra_Vector> fmggddg =
        Teuchos::rcp(new Epetra_Vector(fmggprev_->RangeMap(), true));
    fmggprev_->Apply(*struct_to_fluid(ddginc_), *fmggddg);
    tmpvec->Update(1.0, *fmggddg, 1.0);
  }

  //(5)
  if (fgiprev_ != Teuchos::null)
  {
    Teuchos::RCP<Epetra_Vector> fgidui =
        Teuchos::rcp(new Epetra_Vector(fgiprev_->RangeMap(), true));
    fgiprev_->Apply(*duiinc_, *fgidui);
    tmpvec->Update(1.0, *fgidui, 1.0);
  }

  //(6)
  if (fmgiprev_ != Teuchos::null)
  {
    // The domain map of matrix fmgiprev_ contains inner velocity and pressure DOFs!
    // AleToFluid converts the inner ALE displacement increments to inner Fluid velocity DOFs.
    // The underlying map of this vector has to match the domain map!
    // Hence, the missing pressure DOFs have to be appended.
    Teuchos::RCP<Epetra_Vector> fmgiddia =
        Teuchos::rcp(new Epetra_Vector(fmgiprev_->RangeMap(), true));

    std::vector<Teuchos::RCP<const Epetra_Map>> fluidpresmaps;
    // Merged fluid pressure DOF map
    fluidpresmaps.push_back(fluid_field()->PressureRowMap());
    // Embedded fluid DOF map
    fluidpresmaps.push_back(fluid_field()->x_fluid_fluid_map_extractor()->fluid_map());

    // Embedded fluid pressure map
    Teuchos::RCP<const Epetra_Map> innerfluidpresmap =
        Core::LinAlg::MultiMapExtractor::intersect_maps(fluidpresmaps);

    // To keep merged fluid velocity and pressure DOF apart
    Teuchos::RCP<Core::LinAlg::MapExtractor> innerfluidvelextractor =
        Teuchos::rcp(new Core::LinAlg::MapExtractor(
            *fluid_field()->x_fluid_fluid_map_extractor()->fluid_map(), innerfluidpresmap, false));

    // Get the ALE-displacements, convert to inner fluid DOF. Still mapped to the embedded fluid.
    Teuchos::RCP<Epetra_Vector> aux =
        ale_to_fluid(ale_field()->Interface()->insert_other_vector(ddialeinc_));
    // Add the pressure DOF as zeros
    aux = innerfluidvelextractor->insert_cond_vector(aux);
    aux = fluid_field()->x_fluid_fluid_map_extractor()->insert_fluid_vector(aux);
    // Remove FSI DOF
    Teuchos::RCP<Epetra_Vector> tmp = fluid_field()->Interface()->extract_other_vector(aux);
    fmgiprev_->Apply(*tmp, *fmgiddia);
    tmpvec->Update(1.0, *fmgiddia, 1.0);
  }

  //(7)
  if (firstcall_)
  {
    if (fggprev_ != Teuchos::null)
    {
      Teuchos::RCP<Epetra_Vector> tmp = Teuchos::rcp(new Epetra_Vector(fggprev_->RangeMap(), true));
      Teuchos::RCP<Epetra_Vector> fveln = fluid_field()->extract_interface_veln();
      fggprev_->Apply(*fveln, *tmp);
      tmpvec->Update(Dt() * fluidtimescale, *tmp, 1.0);
    }
  }

  // ---------Adding tmpvec to lambda_
  lambda_->Update(
      fluidresidualscale, *tmpvec, 1.0);  // scale with residual_scaling() to get [N/m^2]

  // Scaling everything with -1/(1-flintparam_)
  lambda_->Scale(-1.0 / (1.0 - ftimintparam));

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidFluidMonolithicFluidSplitNoNOX::handle_fluid_dof_map_change_in_newton()
{
  if (Comm().MyPID() == 0) Core::IO::cout << "New Map!" << Core::IO::endl;

  //  Save old sum of increments
  Teuchos::RCP<Epetra_Vector> x_sum_n = Core::LinAlg::CreateVector(*dof_row_map(), true);
  *x_sum_n = *x_sum_;
  //  Extract structural increment sum
  Teuchos::RCP<const Epetra_Vector> sx_n;
  sx_n = extractor().extract_vector(x_sum_n, 0);
  //  Extract ALE increment sum
  Teuchos::RCP<const Epetra_Vector> ax_n;
  ax_n = extractor().extract_vector(x_sum_n, 2);

  create_combined_dof_row_map();

  // Initialize the global system matrix!
  systemmatrix_ =
      Teuchos::rcp(new Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          extractor(), extractor(), 81, false, true));

  iterinc_ = Core::LinAlg::CreateVector(*dof_row_map(), true);
  rhs_ = Core::LinAlg::CreateVector(*dof_row_map(), true);
  zeros_ = Core::LinAlg::CreateVector(*dof_row_map(), true);
  x_sum_ = Core::LinAlg::CreateVector(*dof_row_map(), true);

  //  Set the new increment sum x_sum_ together

  extractor().insert_vector(sx_n, 0, x_sum_);

  Teuchos::RCP<Epetra_Vector> ff_stepinc =
      fluid_field()->Interface()->extract_other_vector(fluid_field()->Stepinc());
  extractor().insert_vector(ff_stepinc, 1, x_sum_);

  extractor().insert_vector(ax_n, 2, x_sum_);

  //  The fluid length may have changed
  nf_ = fluid_field()->RHS()->GlobalLength();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool FSI::FluidFluidMonolithicFluidSplitNoNOX::has_fluid_dof_map_changed(
    const Epetra_BlockMap& fluidincrementmap)
{
  bool isoldmap = fluidincrementmap.SameAs(*fluid_field()->Interface()->other_map());
  return !isoldmap;
}

FOUR_C_NAMESPACE_CLOSE