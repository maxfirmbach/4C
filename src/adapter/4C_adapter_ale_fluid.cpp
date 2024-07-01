/*----------------------------------------------------------------------------*/
/*! \file

\brief Wrapper for the ALE time integration for fluid problems with moving boundaries

\level 2

 */
/*----------------------------------------------------------------------------*/



/*----------------------------------------------------------------------------*/
/* header inclusions */
#include "4C_adapter_ale_fluid.hpp"

#include "4C_ale_utils_mapextractor.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Adapter::AleFluidWrapper::AleFluidWrapper(Teuchos::RCP<Ale> ale) : AleWrapper(ale)
{
  // create the FSI interface
  interface_ = Teuchos::rcp(new ALE::UTILS::MapExtractor);
  interface_->setup(*discretization());
  // extend dirichlet map by the dof
  if (interface_->fsi_cond_relevant())
    SetupDBCMapEx(ALE::UTILS::MapExtractor::dbc_set_part_fsi, interface_);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Teuchos::RCP<const ALE::UTILS::MapExtractor> Adapter::AleFluidWrapper::Interface() const
{
  return interface_;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
int Adapter::AleFluidWrapper::Solve()
{
  if (interface_->fsi_cond_relevant())
    evaluate(Teuchos::null, ALE::UTILS::MapExtractor::dbc_set_part_fsi);
  else
    evaluate();

  int err = AleWrapper::Solve();
  UpdateIter();

  return err;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Adapter::AleFluidWrapper::apply_free_surface_displacements(
    Teuchos::RCP<const Epetra_Vector> fsdisp)
{
  interface_->insert_fs_cond_vector(fsdisp, WriteAccessDispnp());
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Adapter::AleFluidWrapper::apply_ale_update_displacements(
    Teuchos::RCP<const Epetra_Vector> audisp)
{
  interface_->insert_au_cond_vector(audisp, WriteAccessDispnp());
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Adapter::AleFluidWrapper::apply_interface_displacements(
    Teuchos::RCP<const Epetra_Vector> idisp)
{
  interface_->insert_fsi_cond_vector(idisp, WriteAccessDispnp());
}

FOUR_C_NAMESPACE_CLOSE