/*----------------------------------------------------------------------*/
/*! \file

\brief one-step theta time integration scheme for level-set problems

\level 2


*----------------------------------------------------------------------*/

#ifndef FOUR_C_LEVELSET_TIMINT_OST_HPP
#define FOUR_C_LEVELSET_TIMINT_OST_HPP

#include "4C_config.hpp"

#include "4C_levelset_algorithm.hpp"
#include "4C_scatra_timint_ost.hpp"

FOUR_C_NAMESPACE_OPEN


namespace ScaTra
{
  class LevelSetTimIntOneStepTheta : public LevelSetAlgorithm, public TimIntOneStepTheta
  {
   public:
    /// standard Constructor
    LevelSetTimIntOneStepTheta(Teuchos::RCP<Core::FE::Discretization> dis,
        Teuchos::RCP<Core::LinAlg::Solver> solver, Teuchos::RCP<Teuchos::ParameterList> params,
        Teuchos::RCP<Teuchos::ParameterList> sctratimintparams,
        Teuchos::RCP<Teuchos::ParameterList> extraparams,
        Teuchos::RCP<Core::IO::DiscretizationWriter> output);


    /// initialize time-integration scheme
    void init() override;

    /// setup time-integration scheme
    void setup() override;

    /// read restart data
    void read_restart(
        const int step, Teuchos::RCP<Core::IO::InputControl> input = Teuchos::null) override;

    /// redistribute the scatra discretization and vectors according to nodegraph
    void Redistribute(const Teuchos::RCP<Epetra_CrsGraph>& nodegraph);

    /// interpolate phi to intermediate time n+theta with 0<theta<1
    Teuchos::RCP<Epetra_Vector> Phinptheta(const double theta_inter);

    /// interpolate phidt to intermediate time n+theta with 0<theta<1
    Teuchos::RCP<Epetra_Vector> Phidtnptheta(const double theta_inter);

   protected:
    /// Print information about current time step to screen (reimplementation for OST)
    void print_time_step_info() override;

    /// calculate consistent initial scalar time derivatives in compliance with initial scalar field
    void calc_initial_time_derivative() override;

    /// additional predictor not intended for level-set methods
    void explicit_predictor() const override { return; };

    /// Set the part of the righthandside belonging to the last timestep.
    void set_old_part_of_righthandside() override;

    /// update state vectors
    /// current solution becomes old solution of next time step
    void update_state() override;

    /// update the solution after Solve()
    /// extended version for coupled level-set problems including reinitialization
    void update() override;

    /// update phi within the reinitialization loop
    void update_reinit() override;

   private:
  };  // class LevelSetTimIntOneStepTheta

}  // namespace ScaTra

FOUR_C_NAMESPACE_CLOSE

#endif