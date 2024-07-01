/*----------------------------------------------------------------------*/
/*! \file

\brief Fluid field adapter for XFSI. Can only be used in conjunction with XFluid!

\level 1


*/
/*----------------------------------------------------------------------*/

#include "4C_adapter_fld_fluid_xfsi.hpp"

#include "4C_adapter_fld_fluid.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_fluid_xfluid.hpp"
#include "4C_fluid_xfluid_fluid.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_xfem_condition_manager.hpp"
#include "4C_xfem_discretization.hpp"

#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Teuchos_RCP.hpp>

#include <set>
#include <vector>

FOUR_C_NAMESPACE_OPEN

/*======================================================================*/
/* constructor */
Adapter::XFluidFSI::XFluidFSI(Teuchos::RCP<Fluid> fluid,  // the XFluid object
    const std::string coupling_name,                      // name of the FSI coupling condition
    Teuchos::RCP<Core::LinAlg::Solver> solver, Teuchos::RCP<Teuchos::ParameterList> params,
    Teuchos::RCP<Core::IO::DiscretizationWriter> output)
    : FluidWrapper(fluid),  // the XFluid object is set as fluid_ in the FluidWrapper
      fpsiinterface_(Teuchos::rcp(new FLD::UTILS::MapExtractor())),
      coupling_name_(coupling_name),
      solver_(solver),
      params_(params)
{
  // make sure
  if (fluid_ == Teuchos::null) FOUR_C_THROW("Failed to create the underlying fluid adapter");
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::XFluidFSI::init()
{
  // call base class init
  FluidWrapper::init();

  // cast fluid to fluidimplicit
  xfluid_ = Teuchos::rcp_dynamic_cast<FLD::XFluid>(fluid_);
  if (xfluid_ == Teuchos::null) FOUR_C_THROW("Failed to cast Adapter::Fluid to FLD::XFluid.");

  // NOTE: currently we are using the XFluidFSI adapter also for pure ALE-fluid problems with
  // level-set boundary in this case no mesh coupling object is available and no interface objects
  // can be created
  Teuchos::RCP<XFEM::MeshCoupling> mc = xfluid_->GetMeshCoupling(coupling_name_);

  if (mc != Teuchos::null)  // classical mesh coupling case for FSI
  {
    // get the mesh coupling object
    mesh_coupling_fsi_ = Teuchos::rcp_dynamic_cast<XFEM::MeshCouplingFSI>(mc, true);

    structinterface_ = Teuchos::rcp(new FLD::UTILS::MapExtractor());

    // the solid mesh has to match the interface mesh
    // so we have to compute a interface true residual vector itrueresidual_
    structinterface_->setup(*mesh_coupling_fsi_->GetCutterDis());
  }

  interface_ = Teuchos::rcp(new FLD::UTILS::MapExtractor());

  interface_->setup(
      *xfluid_->discretization(), false, true);  // Always Create overlapping FSI/FPSI Interface

  fpsiinterface_->setup(
      *xfluid_->discretization(), true, true);  // Always Create overlapping FSI/FPSI Interface

  meshmap_ = Teuchos::rcp(new Core::LinAlg::MapExtractor());
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Adapter::XFluidFSI::TimeScaling() const
{
  // second order (OST(0.5) except for the first starting step, otherwise 1st order BackwardEuler
  if (params_->get<bool>("interface second order"))
    return 2. / xfluid_->Dt();
  else
    return 1. / xfluid_->Dt();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Core::FE::Discretization> Adapter::XFluidFSI::boundary_discretization()
{
  return mesh_coupling_fsi_->GetCutterDis();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> Adapter::XFluidFSI::extract_struct_interface_forces()
{
  // the trueresidual vector has to match the solid dis
  // it contains the forces acting on the structural surface
  return structinterface_->extract_fsi_cond_vector(mesh_coupling_fsi_->ITrueResidual());
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> Adapter::XFluidFSI::extract_struct_interface_veln()
{
  // it depends, when this method is called, and when velnp is updated
  // the FSI algorithm expects first an time update and then asks for the old time step velocity
  // meaning that it gets the velocity from the new time step
  // not clear? exactly! thats why the FSI time update should be more clear about it
  // needs discussion with the FSI people
  return structinterface_->extract_fsi_cond_vector(mesh_coupling_fsi_->IVeln());
}


/*----------------------------------------------------------------------*/
// apply the interface velocities to the fluid
/*----------------------------------------------------------------------*/
void Adapter::XFluidFSI::apply_struct_interface_velocities(Teuchos::RCP<Epetra_Vector> ivel)
{
  structinterface_->insert_fsi_cond_vector(ivel, mesh_coupling_fsi_->IVelnp());
}


/*----------------------------------------------------------------------*/
//  apply the interface displacements to the fluid
/*----------------------------------------------------------------------*/
void Adapter::XFluidFSI::apply_struct_mesh_displacement(
    Teuchos::RCP<const Epetra_Vector> interface_disp)
{
  // update last increment, before we set new idispnp
  mesh_coupling_fsi_->update_displacement_iteration_vectors();

  // set new idispnp
  structinterface_->insert_fsi_cond_vector(interface_disp, mesh_coupling_fsi_->IDispnp());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Adapter::XFluidFSI::SetMeshMap(Teuchos::RCP<const Epetra_Map> mm, const int nds_master)
{
  // check nds_master
  if (nds_master != 0) FOUR_C_THROW("nds_master is supposed to be 0 here");

  meshmap_->setup(*xfluid_->DiscretisationXFEM()->InitialDofRowMap(), mm,
      Core::LinAlg::SplitMap(*xfluid_->DiscretisationXFEM()->InitialDofRowMap(), *mm));
}

/*----------------------------------------------------------------------*/
//  apply the ale displacements to the fluid
/*----------------------------------------------------------------------*/
void Adapter::XFluidFSI::apply_mesh_displacement(Teuchos::RCP<const Epetra_Vector> fluiddisp)
{
  meshmap_->insert_cond_vector(fluiddisp, xfluid_->WriteAccessDispnp());

  // new grid velocity
  xfluid_->UpdateGridv();
}


/*----------------------------------------------------------------------*
 * convert increment of displacement to increment in velocity
 * Delta d = d^(n+1,i+1)-d^n is converted to the interface velocity increment
 * Delta u = u^(n+1,i+1)-u^n
 * via first order or second order OST-discretization of d/dt d(t) = u(t)
 *----------------------------------------------------------------------*/
void Adapter::XFluidFSI::displacement_to_velocity(
    Teuchos::RCP<Epetra_Vector> fcx  /// Delta d = d^(n+1,i+1)-d^n
)
{
  // get interface velocity at t(n)
  const Teuchos::RCP<const Epetra_Vector> veln =
      structinterface_->extract_fsi_cond_vector(mesh_coupling_fsi_->IVeln());

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // check, whether maps are the same
  if (!fcx->Map().PointSameAs(veln->Map()))
  {
    FOUR_C_THROW("Maps do not match, but they have to.");
  }
#endif

  /*
   * Delta u(n+1,i+1) = fac * (Delta d(n+1,i+1) - dt * u(n))
   *
   *             / = 2 / dt   if interface time integration is second order
   * with fac = |
   *             \ = 1 / dt   if interface time integration is first order
   */
  const double timescale = TimeScaling();
  fcx->Update(-timescale * xfluid_->Dt(), *veln, timescale);
}


/// return xfluid coupling matrix between structure and fluid as sparse matrices
Teuchos::RCP<Core::LinAlg::SparseMatrix> Adapter::XFluidFSI::c_struct_fluid_matrix()
{
  return xfluid_->C_sx_Matrix(coupling_name_);
}

/// return xfluid coupling matrix between fluid and structure as sparse matrices
Teuchos::RCP<Core::LinAlg::SparseMatrix> Adapter::XFluidFSI::c_fluid_struct_matrix()
{
  return xfluid_->C_xs_Matrix(coupling_name_);
}

/// return xfluid coupling matrix between structure and structure as sparse matrices
Teuchos::RCP<Core::LinAlg::SparseMatrix> Adapter::XFluidFSI::c_struct_struct_matrix()
{
  return xfluid_->C_ss_Matrix(coupling_name_);
}

/// return xfluid coupling matrix between structure and structure as sparse matrices
Teuchos::RCP<const Epetra_Vector> Adapter::XFluidFSI::RHS_Struct_Vec()
{
  return xfluid_->RHS_s_Vec(coupling_name_);
}

/// GmshOutput for background mesh and cut mesh
void Adapter::XFluidFSI::GmshOutput(const std::string& name,  ///< name for output file
    const int step,                                           ///< step number
    const int count,                  ///< counter for iterations within a global time step
    Teuchos::RCP<Epetra_Vector> vel,  ///< vector holding velocity and pressure dofs
    Teuchos::RCP<Epetra_Vector> acc   ///< vector holding accelerations
)
{
  // TODO (kruse): find a substitute!
  // xfluid_->GmshOutput(name, step, count, vel, acc);
  FOUR_C_THROW("Gmsh output for XFSI during Newton currently not available.");
}

FOUR_C_NAMESPACE_CLOSE