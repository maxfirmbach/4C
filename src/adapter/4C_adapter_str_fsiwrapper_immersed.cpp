/*----------------------------------------------------------------------*/
/*! \file

\brief Structural adapter for Immersed and Immersed + ALE FSI problems containing the interface
       and methods dependent on the interface

\level 2


*/


#include "4C_adapter_str_fsiwrapper_immersed.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fsi_str_model_evaluator_partitioned.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_structure_aux.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_timint_implicit.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Adapter::FSIStructureWrapperImmersed::FSIStructureWrapperImmersed(Teuchos::RCP<Structure> structure)
    : FPSIStructureWrapper(structure)
{
  // immersed_ale fsi part
  std::vector<Teuchos::RCP<const Epetra_Map>> vecSpaces;

  vecSpaces.push_back(interface_->fsi_cond_map());       // fsi
  vecSpaces.push_back(interface_->immersed_cond_map());  // immersed

  combinedmap_ = Core::LinAlg::MultiMapExtractor::merge_maps(vecSpaces);

  // full blockmap
  Core::LinAlg::MultiMapExtractor blockrowdofmap;
  blockrowdofmap.setup(*combinedmap_, vecSpaces);

  combinedinterface_ = Teuchos::rcp(new Core::LinAlg::MapExtractor(
      *combinedmap_, interface_->fsi_cond_map(), interface_->immersed_cond_map()));
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FSIStructureWrapperImmersed::apply_immersed_interface_forces(
    Teuchos::RCP<Epetra_Vector> iforce_fsi, Teuchos::RCP<Epetra_Vector> iforce_immersed)
{
  fsi_model_evaluator()->get_interface_force_np_ptr()->PutScalar(0.0);

  if (iforce_fsi != Teuchos::null)
    interface_->add_fsi_cond_vector(
        iforce_fsi, fsi_model_evaluator()->get_interface_force_np_ptr());
  if (iforce_immersed != Teuchos::null)
    interface_->add_immersed_cond_vector(
        iforce_immersed, fsi_model_evaluator()->get_interface_force_np_ptr());

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector>
Adapter::FSIStructureWrapperImmersed::extract_immersed_interface_dispnp()
{
  FOUR_C_ASSERT(interface_->FullMap()->PointSameAs(Dispnp()->Map()),
      "Full map of map extractor and Dispnp() do not match.");

  return interface_->extract_immersed_cond_vector(Dispnp());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> Adapter::FSIStructureWrapperImmersed::extract_full_interface_dispnp()
{
  FOUR_C_ASSERT(interface_->FullMap()->PointSameAs(Dispnp()->Map()),
      "Full map of map extractor and Dispnp() do not match.");

  Teuchos::RCP<Epetra_Vector> fullvec =
      Teuchos::rcp(new Epetra_Vector(*combinedinterface_->FullMap(), true));

  // CondVector is FSI vector
  combinedinterface_->add_cond_vector(interface_->extract_fsi_cond_vector(Dispnp()), fullvec);
  // OtherVector is IMMERSED vector
  combinedinterface_->add_other_vector(interface_->extract_immersed_cond_vector(Dispnp()), fullvec);

  return fullvec;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector>
Adapter::FSIStructureWrapperImmersed::predict_immersed_interface_dispnp()
{
  Teuchos::RCP<Epetra_Vector> idis;

  switch (predictor_)
  {
    case 1:
    {
      idis = interface_->extract_immersed_cond_vector(Dispn());

      break;
    }
    case 2:
      // d(n)+dt*(1.5*v(n)-0.5*v(n-1))
      FOUR_C_THROW("interface velocity v(n-1) not available");
      break;
    case 3:
    {
      // d(n)+dt*v(n)
      double dt = Dt();

      idis = interface_->extract_immersed_cond_vector(Dispn());
      Teuchos::RCP<Epetra_Vector> ivel = interface_->extract_immersed_cond_vector(Veln());

      idis->Update(dt, *ivel, 1.0);
      break;
    }
    case 4:
    {
      // d(n)+dt*v(n)+0.5*dt^2*a(n)
      double dt = Dt();

      idis = interface_->extract_immersed_cond_vector(Dispn());
      Teuchos::RCP<Epetra_Vector> ivel = interface_->extract_immersed_cond_vector(Veln());
      Teuchos::RCP<Epetra_Vector> iacc = interface_->extract_immersed_cond_vector(Accn());

      idis->Update(dt, *ivel, 0.5 * dt * dt, *iacc, 1.0);
      break;
    }
    default:
      FOUR_C_THROW(
          "unknown interface displacement predictor '%s'", Global::Problem::Instance()
                                                               ->FSIDynamicParams()
                                                               .sublist("PARTITIONED SOLVER")
                                                               .get<std::string>("PREDICTOR")
                                                               .c_str());
      break;
  }

  return idis;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> Adapter::FSIStructureWrapperImmersed::predict_full_interface_dispnp()
{
  Teuchos::RCP<Epetra_Vector> idis =
      Teuchos::rcp(new Epetra_Vector(*combinedinterface_->FullMap(), true));

  switch (predictor_)
  {
    case 1:
    {
      // CondVector is FSI vector
      combinedinterface_->add_cond_vector(interface_->extract_fsi_cond_vector(Dispn()), idis);
      // OtherVector is IMMERSED vector
      combinedinterface_->add_other_vector(interface_->extract_immersed_cond_vector(Dispn()), idis);

      break;
    }
    case 2:
      // d(n)+dt*(1.5*v(n)-0.5*v(n-1))
      FOUR_C_THROW("interface velocity v(n-1) not available");
      break;
    case 3:
    {
      // d(n)+dt*v(n)
      double dt = Dt();

      // CondVector is FSI vector
      combinedinterface_->add_cond_vector(interface_->extract_fsi_cond_vector(Dispn()), idis);
      // OtherVector is IMMERSED vector
      combinedinterface_->add_other_vector(interface_->extract_immersed_cond_vector(Dispn()), idis);

      Teuchos::RCP<Epetra_Vector> ivel =
          Teuchos::rcp(new Epetra_Vector(*combinedinterface_->FullMap(), true));

      // CondVector is FSI vector
      combinedinterface_->add_cond_vector(interface_->extract_fsi_cond_vector(Veln()), ivel);
      // OtherVector is IMMERSED vector
      combinedinterface_->add_other_vector(interface_->extract_immersed_cond_vector(Veln()), ivel);

      idis->Update(dt, *ivel, 1.0);
      break;
    }
    case 4:
    {
      // d(n)+dt*v(n)+0.5*dt^2*a(n)
      double dt = Dt();

      // CondVector is FSI vector
      combinedinterface_->add_cond_vector(interface_->extract_fsi_cond_vector(Dispn()), idis);
      // OtherVector is IMMERSED vector
      combinedinterface_->add_other_vector(interface_->extract_immersed_cond_vector(Dispn()), idis);

      Teuchos::RCP<Epetra_Vector> ivel =
          Teuchos::rcp(new Epetra_Vector(*combinedinterface_->FullMap(), true));

      // CondVector is FSI vector
      combinedinterface_->add_cond_vector(interface_->extract_fsi_cond_vector(Veln()), ivel);
      // OtherVector is IMMERSED vector
      combinedinterface_->add_other_vector(interface_->extract_immersed_cond_vector(Veln()), ivel);

      Teuchos::RCP<Epetra_Vector> iacc =
          Teuchos::rcp(new Epetra_Vector(*combinedinterface_->FullMap(), true));

      // CondVector is FSI vector
      combinedinterface_->add_cond_vector(interface_->extract_fsi_cond_vector(Accn()), iacc);
      // OtherVector is IMMERSED vector
      combinedinterface_->add_other_vector(interface_->extract_immersed_cond_vector(Accn()), iacc);

      idis->Update(dt, *ivel, 0.5 * dt * dt, *iacc, 1.0);
      break;
    }
    default:
      FOUR_C_THROW(
          "unknown interface displacement predictor '%s'", Global::Problem::Instance()
                                                               ->FSIDynamicParams()
                                                               .sublist("PARTITIONED SOLVER")
                                                               .get<std::string>("PREDICTOR")
                                                               .c_str());
      break;
  }

  return idis;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FSIStructureWrapperImmersed::output(
    bool forced_writerestart, const int step, const double time)
{
  // always write velocity and displacement for extra output
  bool writevelacc_ = true;

  // write standard output if no arguments are provided (default -1)
  if (step == -1 and time == -1.0) structure_->output(forced_writerestart);
  // write extra output for specified step and time
  else
  {
    if (structure_->discretization()->Comm().MyPID() == 0)
      std::cout << "\n   Write EXTRA STRUCTURE Output Step=" << step << " Time=" << time
                << " ...   \n"
                << std::endl;


    structure_->disc_writer()->new_step(step, time);
    structure_->disc_writer()->write_vector("displacement", structure_->Dispnp());

    // for visualization of vel and acc do not forget to comment in corresponding lines in
    // StructureEnsightWriter
    if (writevelacc_)
    {
      structure_->disc_writer()->write_vector("velocity", structure_->Velnp());
      structure_->disc_writer()->write_vector("acceleration", structure_->Accnp());
    }
  }  // write extra output
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
STR::Dbc& Adapter::FSIStructureWrapperImmersed::GetDBC()
{
  return Teuchos::rcp_dynamic_cast<STR::TimeInt::Base>(structure_, true)->get_dbc();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FSIStructureWrapperImmersed::AddDirichDofs(
    const Teuchos::RCP<const Epetra_Map> maptoadd)
{
  GetDBC().AddDirichDofs(maptoadd);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FSIStructureWrapperImmersed::RemoveDirichDofs(
    const Teuchos::RCP<const Epetra_Map> maptoremove)
{
  GetDBC().RemoveDirichDofs(maptoremove);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FSIStructureWrapperImmersed::set_state(const Teuchos::RCP<Epetra_Vector>& x)
{
  return Teuchos::rcp_dynamic_cast<STR::TimeInt::Implicit>(structure_, true)->set_state(x);
}

FOUR_C_NAMESPACE_CLOSE