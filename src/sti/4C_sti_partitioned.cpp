/*----------------------------------------------------------------------*/
/*! \file

\brief partitioned coupling algorithm for scatra-thermo interaction

\level 2

*/
/*----------------------------------------------------------------------*/
#include "4C_sti_partitioned.hpp"

#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_scatra_timint_implicit.hpp"

FOUR_C_NAMESPACE_OPEN

/*--------------------------------------------------------------------------------*
 | constructor                                                         fang 09/17 |
 *--------------------------------------------------------------------------------*/
STI::Partitioned::Partitioned(const Epetra_Comm& comm,  //! communicator
    const Teuchos::ParameterList& stidyn,  //! parameter list for scatra-thermo interaction
    const Teuchos::ParameterList&
        scatradyn,  //! scalar transport parameter list for scatra and thermo fields
    const Teuchos::ParameterList& solverparams_scatra,  //! solver parameter list for scatra field
    const Teuchos::ParameterList& solverparams_thermo   //! solver parameter list for thermo field
    )
    :  // instantiate base class
      Algorithm(comm, stidyn, scatradyn, solverparams_scatra, solverparams_thermo),
      couplingtype_(Teuchos::getIntegralValue<Inpar::STI::CouplingType>(stidyn, "COUPLINGTYPE")),
      omegamax_(stidyn.sublist("PARTITIONED").get<double>("OMEGAMAX"))
{
  // set control parameters for outer coupling iteration loop
  itermax_ = fieldparameters_->sublist("NONLINEAR").get<int>("ITEMAX_OUTER");
  itertol_ = fieldparameters_->sublist("NONLINEAR").get<double>("CONVTOL_OUTER");

  // initialize vectors for outer coupling iteration loop
  switch (couplingtype_)
  {
    case Inpar::STI::CouplingType::oneway_scatratothermo:
    case Inpar::STI::CouplingType::oneway_thermotoscatra:
      // do nothing
      break;

    case Inpar::STI::CouplingType::twoway_scatratothermo:
    case Inpar::STI::CouplingType::twoway_scatratothermo_aitken:
    case Inpar::STI::CouplingType::twoway_scatratothermo_aitken_dofsplit:
    case Inpar::STI::CouplingType::twoway_thermotoscatra:
    case Inpar::STI::CouplingType::twoway_thermotoscatra_aitken:
    {
      // initialize increment vectors
      ScaTraField()->PhinpInc() =
          Core::LinAlg::CreateVector(*ScaTraField()->discretization()->dof_row_map(), true);
      ThermoField()->PhinpInc() =
          Core::LinAlg::CreateVector(*ThermoField()->discretization()->dof_row_map(), true);

      // initialize old increment vectors
      if (couplingtype_ == Inpar::STI::CouplingType::twoway_scatratothermo_aitken or
          couplingtype_ == Inpar::STI::CouplingType::twoway_scatratothermo_aitken_dofsplit)
        ScaTraField()->PhinpIncOld() =
            Core::LinAlg::CreateVector(*ScaTraField()->discretization()->dof_row_map(), true);
      else if (couplingtype_ == Inpar::STI::CouplingType::twoway_thermotoscatra_aitken)
        ThermoField()->PhinpIncOld() =
            Core::LinAlg::CreateVector(*ThermoField()->discretization()->dof_row_map(), true);

      // initialize relaxation parameter
      if (couplingtype_ == Inpar::STI::CouplingType::twoway_scatratothermo)
        ScaTraField()->Omega().resize(1, stidyn.sublist("PARTITIONED").get<double>("OMEGA"));
      else if (couplingtype_ == Inpar::STI::CouplingType::twoway_scatratothermo_aitken)
        ScaTraField()->Omega().resize(1, 1.);
      else if (couplingtype_ == Inpar::STI::CouplingType::twoway_scatratothermo_aitken_dofsplit)
        ScaTraField()->Omega().resize(ScaTraField()->NumDofPerNode(), 1.);
      else if (couplingtype_ == Inpar::STI::CouplingType::twoway_thermotoscatra)
        ThermoField()->Omega().resize(1, stidyn.sublist("PARTITIONED").get<double>("OMEGA"));
      else if (couplingtype_ == Inpar::STI::CouplingType::twoway_thermotoscatra_aitken)
        ThermoField()->Omega().resize(1, 1.);
      else
        ThermoField()->Omega().resize(ScaTraField()->NumDofPerNode(), 1.);

      break;
    }

    default:
    {
      FOUR_C_THROW("What the hell?!");
      break;
    }
  }

  return;
}  // STI::Partitioned::Partitioned


/*--------------------------------------------------------------------*
 | convergence check for outer iteration loop              fang 09/17 |
 *--------------------------------------------------------------------*/
bool STI::Partitioned::exit_outer_coupling() const
{
  // extract processor ID
  const int mypid = Comm().MyPID();

  // compute vector norms
  double L2_scatra(0.), L2_scatra_inc(0.), L2_thermo(0.), L2_thermo_inc(0.);
  ScaTraField()->Phinp()->Norm2(&L2_scatra);
  ScaTraField()->PhinpInc()->Norm2(&L2_scatra_inc);
  ThermoField()->Phinp()->Norm2(&L2_thermo);
  ThermoField()->PhinpInc()->Norm2(&L2_thermo_inc);
  if (L2_scatra < 1.e-10) L2_scatra = 1.;
  if (L2_thermo < 1.e-10) L2_thermo = 1.;

  // print convergence status
  if (mypid == 0)
  {
    std::cout << std::endl;
    std::cout << "+------------+-------------------+--------------+--------------+" << std::endl;
    std::cout << "|                       OUTER ITERATION                        |" << std::endl;
    std::cout << "+------------+-------------------+--------------+--------------+" << std::endl;
    std::cout << "|- step/max -|- tol      [norm] -|- scatra-inc -|- thermo-inc -|" << std::endl;
    std::cout << "|  " << std::setw(3) << iter_ << "/" << std::setw(3) << itermax_ << "   | "
              << std::setw(10) << std::setprecision(3) << std::scientific << itertol_
              << "[L_2 ]  | " << std::setw(10) << std::setprecision(3) << std::scientific
              << L2_scatra_inc / L2_scatra << "   | " << std::setw(10) << std::setprecision(3)
              << std::scientific << L2_thermo_inc / L2_thermo << "   |" << std::endl;
    std::cout << "+------------+-------------------+--------------+--------------+" << std::endl;
  }

  // convergence check
  if (L2_scatra_inc / L2_scatra <= itertol_ and L2_thermo_inc / L2_thermo <= itertol_)
  {
    if (mypid == 0)
    {
      std::cout << "|   OUTER ITERATION LOOP CONVERGED AFTER ITERATION " << std::setw(3) << iter_
                << "/" << std::setw(3) << itermax_ << " !   |" << std::endl;
      std::cout << "+------------+-------------------+--------------+--------------+" << std::endl;
    }

    return true;
  }

  // throw error in case maximum number of iteration steps is reached without convergence
  else if (iter_ == itermax_)
  {
    if (mypid == 0)
    {
      std::cout << "| >>>> not converged within maximum number of iteration steps! |" << std::endl;
      std::cout << "+------------+-------------------+--------------+--------------+" << std::endl;
    }

    FOUR_C_THROW("Outer iteration did not converge within maximum number of iteration steps!");

    return true;
  }

  // proceed with next outer iteration step
  return false;
}  // STI::Partitioned::exit_outer_coupling()


/*--------------------------------------------------------------------------------*
 | evaluate time step using outer coupling iteration                   fang 09/17 |
 *--------------------------------------------------------------------------------*/
void STI::Partitioned::solve()
{
  switch (couplingtype_)
  {
    case Inpar::STI::CouplingType::oneway_scatratothermo:
    case Inpar::STI::CouplingType::oneway_thermotoscatra:
    {
      solve_one_way();
      break;
    }

    case Inpar::STI::CouplingType::twoway_scatratothermo:
    case Inpar::STI::CouplingType::twoway_scatratothermo_aitken:
    case Inpar::STI::CouplingType::twoway_scatratothermo_aitken_dofsplit:
    case Inpar::STI::CouplingType::twoway_thermotoscatra:
    case Inpar::STI::CouplingType::twoway_thermotoscatra_aitken:
    {
      solve_two_way();
      break;
    }

    default:
    {
      FOUR_C_THROW("Dude... what's wrong with you?!");
      break;
    }
  }

  return;
}  // STI::Partitioned::Solve()


/*--------------------------------------------------------------------------------*
 | evaluate time step using one-way coupling iteration                 fang 09/17 |
 *--------------------------------------------------------------------------------*/
void STI::Partitioned::solve_one_way() const
{
  switch (couplingtype_)
  {
    case Inpar::STI::CouplingType::oneway_scatratothermo:
    {
      // pass thermo degrees of freedom to scatra discretization
      transfer_thermo_to_scatra(ThermoField()->Phiafnp());

      // solve scatra field
      ScaTraField()->Solve();

      // pass scatra degrees of freedom to thermo discretization
      transfer_scatra_to_thermo(ScaTraField()->Phiafnp());

      // solve thermo field
      ThermoField()->Solve();

      break;
    }

    case Inpar::STI::CouplingType::oneway_thermotoscatra:
    {
      // pass scatra degrees of freedom to thermo discretization
      transfer_scatra_to_thermo(ScaTraField()->Phiafnp());

      // solve thermo field
      ThermoField()->Solve();

      // pass thermo degrees of freedom to scatra discretization
      transfer_thermo_to_scatra(ThermoField()->Phiafnp());

      // solve scatra field
      ScaTraField()->Solve();

      break;
    }

    default:
    {
      FOUR_C_THROW("No, no, noooooooo...!");
      break;
    }
  }
}  // STI::Partitioned::solve_one_way()


/*----------------------------------------------------------------------*
 | evaluate time step using two-way coupling iteration       fang 09/17 |
 *----------------------------------------------------------------------*/
void STI::Partitioned::solve_two_way()
{
  // reset number of outer iterations
  iter_ = 0;

  switch (couplingtype_)
  {
    case Inpar::STI::CouplingType::twoway_scatratothermo:
    case Inpar::STI::CouplingType::twoway_scatratothermo_aitken:
    case Inpar::STI::CouplingType::twoway_scatratothermo_aitken_dofsplit:
    {
      // initialize relaxed scatra state vector
      const Teuchos::RCP<Epetra_Vector> scatra_relaxed =
          Teuchos::rcp(new Epetra_Vector(*ScaTraField()->Phiafnp()));

      // begin outer iteration loop
      while (true)
      {
        // increment iteration number
        iter_++;

        // pass relaxed scatra degrees of freedom to thermo discretization
        transfer_scatra_to_thermo(scatra_relaxed);

        // store current thermo state vector
        if (ThermoField()->PhinpInc()->Update(1., *ThermoField()->Phiafnp(), 0.))
          FOUR_C_THROW("Update failed!");

        // solve thermo field
        ThermoField()->Solve();

        // compute increment of thermo state vector
        if (ThermoField()->PhinpInc()->Update(1., *ThermoField()->Phiafnp(), -1.))
          FOUR_C_THROW("Update failed!");

        // pass thermo degrees of freedom to scatra discretization
        transfer_thermo_to_scatra(ThermoField()->Phiafnp());

        // store current scatra state vector
        if (ScaTraField()->PhinpInc()->Update(1., *scatra_relaxed, 0.))
          FOUR_C_THROW("Update failed!");

        // solve scatra field
        ScaTraField()->Solve();

        // compute increment of scatra state vector
        if (ScaTraField()->PhinpInc()->Update(1., *ScaTraField()->Phiafnp(), -1.))
          FOUR_C_THROW("Update failed!");

        // convergence check
        if (exit_outer_coupling()) break;

        // perform static relaxation
        if (couplingtype_ == Inpar::STI::CouplingType::twoway_scatratothermo)
        {
          if (scatra_relaxed->Update(ScaTraField()->Omega()[0], *ScaTraField()->PhinpInc(), 1.))
            FOUR_C_THROW("Update failed!");
        }

        // perform dynamic relaxation
        else
        {
          // compute difference between current and previous increments of scatra state vector
          Epetra_Vector scatra_inc_diff(*ScaTraField()->PhinpInc());
          if (scatra_inc_diff.Update(-1., *ScaTraField()->PhinpIncOld(), 1.))
            FOUR_C_THROW("Update failed!");

          if (couplingtype_ == Inpar::STI::CouplingType::twoway_scatratothermo_aitken)
          {
            // compute L2 norm of difference between current and previous increments of scatra state
            // vector
            double scatra_inc_diff_L2(0.);
            scatra_inc_diff.Norm2(&scatra_inc_diff_L2);

            // compute dot product between increment of scatra state vector and difference between
            // current and previous increments of scatra state vector
            double scatra_inc_dot_scatra_inc_diff(0.);
            if (scatra_inc_diff.Dot(*ScaTraField()->PhinpInc(), &scatra_inc_dot_scatra_inc_diff))
              FOUR_C_THROW("Couldn't compute dot product!");

            // compute Aitken relaxation factor
            if (iter_ > 1 and scatra_inc_diff_L2 > 1.e-12)
              ScaTraField()->Omega()[0] *=
                  1 - scatra_inc_dot_scatra_inc_diff / (scatra_inc_diff_L2 * scatra_inc_diff_L2);

            // restrict Aitken relaxation factor if necessary
            if (omegamax_ > 0. and ScaTraField()->Omega()[0] > omegamax_)
              ScaTraField()->Omega()[0] = omegamax_;

            // perform Aitken relaxation
            if (scatra_relaxed->Update(ScaTraField()->Omega()[0], *ScaTraField()->PhinpInc(), 1.))
              FOUR_C_THROW("Update failed!");
          }

          else
          {
            // safety check
            if (ScaTraField()->Splitter() == Teuchos::null)
              FOUR_C_THROW("Map extractor was not initialized!");

            // loop over all degrees of freedom
            for (int idof = 0; idof < ScaTraField()->Splitter()->num_maps(); ++idof)
            {
              // extract subvectors associated with current degree of freedom
              const Teuchos::RCP<const Epetra_Vector> scatra_inc_dof =
                  ScaTraField()->Splitter()->extract_vector(*ScaTraField()->PhinpInc(), idof);
              const Teuchos::RCP<const Epetra_Vector> scatra_inc_diff_dof =
                  ScaTraField()->Splitter()->extract_vector(scatra_inc_diff, idof);

              // compute L2 norm of difference between current and previous increments of current
              // degree of freedom
              double scatra_inc_diff_L2(0.);
              scatra_inc_diff_dof->Norm2(&scatra_inc_diff_L2);

              // compute dot product between increment of current degree of freedom and difference
              // between current and previous increments of current degree of freedom
              double scatra_inc_dot_scatra_inc_diff(0.);
              if (scatra_inc_diff_dof->Dot(*scatra_inc_dof, &scatra_inc_dot_scatra_inc_diff))
                FOUR_C_THROW("Couldn't compute dot product!");

              // compute Aitken relaxation factor for current degree of freedom
              if (iter_ > 1 and scatra_inc_diff_L2 > 1.e-12)
                ScaTraField()->Omega()[idof] *=
                    1 - scatra_inc_dot_scatra_inc_diff / (scatra_inc_diff_L2 * scatra_inc_diff_L2);

              // restrict Aitken relaxation factor if necessary
              if (omegamax_ > 0. and ScaTraField()->Omega()[idof] > omegamax_)
                ScaTraField()->Omega()[idof] = omegamax_;

              // perform Aitken relaxation for current degree of freedom
              ScaTraField()->Splitter()->add_vector(
                  *scatra_inc_dof, idof, *scatra_relaxed, ScaTraField()->Omega()[idof]);
            }
          }

          // update increment of scatra state vector
          if (ScaTraField()->PhinpIncOld()->Update(1., *ScaTraField()->PhinpInc(), 0.))
            FOUR_C_THROW("Update failed!");
        }
      }

      break;
    }

    case Inpar::STI::CouplingType::twoway_thermotoscatra:
    case Inpar::STI::CouplingType::twoway_thermotoscatra_aitken:
    {
      // initialize relaxed thermo state vector
      const Teuchos::RCP<Epetra_Vector> thermo_relaxed =
          Teuchos::rcp(new Epetra_Vector(*ThermoField()->Phiafnp()));

      // begin outer iteration loop
      while (true)
      {
        // increment iteration number
        iter_++;

        // pass relaxed thermo degrees of freedom to scatra discretization
        transfer_thermo_to_scatra(thermo_relaxed);

        // store current scatra state vector
        if (ScaTraField()->PhinpInc()->Update(1., *ScaTraField()->Phiafnp(), 0.))
          FOUR_C_THROW("Update failed!");

        // solve scatra field
        ScaTraField()->Solve();

        // compute increment of scatra state vector
        if (ScaTraField()->PhinpInc()->Update(1., *ScaTraField()->Phiafnp(), -1.))
          FOUR_C_THROW("Update failed!");

        // pass scatra degrees of freedom to thermo discretization
        transfer_scatra_to_thermo(ScaTraField()->Phiafnp());

        // store current thermo state vector
        if (ThermoField()->PhinpInc()->Update(1., *thermo_relaxed, 0.))
          FOUR_C_THROW("Update failed!");

        // solve thermo field
        ThermoField()->Solve();

        // compute increment of thermo state vector
        if (ThermoField()->PhinpInc()->Update(1., *ThermoField()->Phiafnp(), -1.))
          FOUR_C_THROW("Update failed!");

        // convergence check
        if (exit_outer_coupling()) break;

        if (couplingtype_ == Inpar::STI::CouplingType::twoway_thermotoscatra_aitken)
        {
          // compute difference between current and previous increments of thermo state vector
          Epetra_Vector thermo_inc_diff(*ThermoField()->PhinpInc());
          if (thermo_inc_diff.Update(-1., *ThermoField()->PhinpIncOld(), 1.))
            FOUR_C_THROW("Update failed!");

          // compute L2 norm of difference between current and previous increments of thermo state
          // vector
          double thermo_inc_diff_L2(0.);
          thermo_inc_diff.Norm2(&thermo_inc_diff_L2);

          // compute dot product between increment of thermo state vector and difference between
          // current and previous increments of thermo state vector
          double thermo_inc_dot_thermo_inc_diff(0.);
          if (thermo_inc_diff.Dot(*ThermoField()->PhinpInc(), &thermo_inc_dot_thermo_inc_diff))
            FOUR_C_THROW("Couldn't compute dot product!");

          // compute Aitken relaxation factor
          if (iter_ > 1 and thermo_inc_diff_L2 > 1.e-12)
            ThermoField()->Omega()[0] *=
                1 - thermo_inc_dot_thermo_inc_diff / (thermo_inc_diff_L2 * thermo_inc_diff_L2);

          // restrict Aitken relaxation factor if necessary
          if (omegamax_ > 0. and ThermoField()->Omega()[0] > omegamax_)
            ThermoField()->Omega()[0] = omegamax_;

          // update increment of thermo state vector
          if (ThermoField()->PhinpIncOld()->Update(1., *ThermoField()->PhinpInc(), 0.))
            FOUR_C_THROW("Update failed!");
        }

        // perform relaxation
        if (thermo_relaxed->Update(ThermoField()->Omega()[0], *ThermoField()->PhinpInc(), 1.))
          FOUR_C_THROW("Update failed!");
      }

      break;
    }

    default:
    {
      FOUR_C_THROW("Please stop coding a bunch of crap...");
      break;
    }
  }

  return;
}  // STI::Partitioned::solve_two_way()

FOUR_C_NAMESPACE_CLOSE