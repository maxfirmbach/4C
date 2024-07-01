/*-----------------------------------------------------------*/
/*! \file

\brief Generic class for all explicit time integrators


\level 3

*/
/*-----------------------------------------------------------*/

#include "4C_structure_new_expl_generic.hpp"

#include "4C_global_data.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_solver_nonlin_nox_aux.hpp"
#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_solver_nonlin_nox_group_prepostoperator.hpp"
#include "4C_structure_new_dbc.hpp"
#include "4C_structure_new_model_evaluator.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"
#include "4C_structure_new_timint_basedatasdyn.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::EXPLICIT::Generic::setup()
{
  check_init();

  // call base class first
  STR::Integrator::setup();

  // ---------------------------------------------------------------------------
  // set the new pre/post operator for the nox nln group in the parameter list
  // ---------------------------------------------------------------------------
  Teuchos::ParameterList& p_grp_opt = sdyn().get_nox_params().sublist("Group Options");

  // create the new generic pre/post operator
  Teuchos::RCP<NOX::Nln::Abstract::PrePostOperator> prepost_generic_ptr =
      Teuchos::rcp(new NOX::Nln::PrePostOp::EXPLICIT::Generic(*this));

  // Get the current map. If there is no map, return a new empty one. (reference)
  NOX::Nln::GROUP::PrePostOperator::Map& prepostgroup_map =
      NOX::Nln::GROUP::PrePostOp::GetMap(p_grp_opt);

  // insert/replace the old pointer in the map
  prepostgroup_map[NOX::Nln::GROUP::prepost_impl_generic] = prepost_generic_ptr;

  // ---------------------------------------------------------------------------
  // set the new pre/post operator for the nox nln solver in the parameter list
  // ---------------------------------------------------------------------------
  Teuchos::ParameterList& p_sol_opt = sdyn().get_nox_params().sublist("Solver Options");

  NOX::Nln::Aux::AddToPrePostOpVector(p_sol_opt, prepost_generic_ptr);

  // No issetup_ = true, since the setup() functions of the derived classes
  // have to be called and finished first!
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::EXPLICIT::Generic::apply_force(const Epetra_Vector& x, Epetra_Vector& f)
{
  check_init_setup();

  // ---------------------------------------------------------------------------
  // evaluate the different model types (static case) at t_{n+1}^{i}
  // ---------------------------------------------------------------------------
  // set the time step dependent parameters for the element evaluation
  reset_eval_params();
  bool ok = model_eval().apply_force(x, f, 1.0);
  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::EXPLICIT::Generic::apply_stiff(const Epetra_Vector& x, Core::LinAlg::SparseOperator& jac)
{
  check_init_setup();

  // ---------------------------------------------------------------------------
  // evaluate the different model types (static case) at t_{n+1}^{i}
  // ---------------------------------------------------------------------------
  // set the time step dependent parameters for the element evaluation
  reset_eval_params();
  const bool ok = model_eval().apply_stiff(x, jac, 1.0);

  if (not ok) return ok;

  jac.Complete();

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::EXPLICIT::Generic::apply_force_stiff(
    const Epetra_Vector& x, Epetra_Vector& f, Core::LinAlg::SparseOperator& jac)
{
  check_init_setup();
  // ---------------------------------------------------------------------------
  // evaluate the different model types (static case) at t_{n+1}^{i}
  // ---------------------------------------------------------------------------
  // set the time step dependent parameters for the element evaluation
  reset_eval_params();
  const bool ok = model_eval().apply_force_stiff(x, f, jac, 1.0);

  if (not ok) return ok;

  jac.Complete();

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double STR::EXPLICIT::Generic::calc_ref_norm_force(
    const enum ::NOX::Abstract::Vector::NormType& type) const
{
  FOUR_C_THROW("%s is not yet implemented", __FUNCTION__);
  return 0.0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::EXPLICIT::Generic::compute_jacobian_contributions_from_element_level_for_ptc(
    Teuchos::RCP<Core::LinAlg::SparseMatrix>& scalingMatrixOpPtr)
{
  FOUR_C_THROW("%s is not yet implemented", __FUNCTION__);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::EXPLICIT::Generic::assemble_force(
    Epetra_Vector& f, const std::vector<Inpar::STR::ModelType>* without_these_models) const
{
  FOUR_C_THROW("%s is not yet implemented", __FUNCTION__);
  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::EXPLICIT::Generic::remove_condensed_contributions_from_rhs(Epetra_Vector& rhs) const
{
  model_eval().remove_condensed_contributions_from_rhs(rhs);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::EXPLICIT::Generic::update_step_element()
{
  check_init_setup();
  model_eval().update_step_element();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::EXPLICIT::Generic::update_constant_state_contributions() {}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::EXPLICIT::Generic::post_update() { update_constant_state_contributions(); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double STR::EXPLICIT::Generic::get_default_step_length() const
{
  // default: return a step length of 1.0
  return 1.0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::EXPLICIT::Generic::reset_eval_params()
{
  // set the time step dependent parameters for the element evaluation
  eval_data().set_total_time(global_state().get_time_np());
  eval_data().set_delta_time((*global_state().get_delta_time())[0]);
  eval_data().set_is_tolerate_error(true);
  eval_data().set_function_manager(Global::Problem::Instance()->FunctionManager());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::PrePostOp::EXPLICIT::Generic::runPreIterate(const ::NOX::Solver::Generic& nlnSolver)
{
  // For explicit integration this action simply does nothing.
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::PrePostOp::EXPLICIT::Generic::runPreSolve(const ::NOX::Solver::Generic& nlnSolver)
{
  // For explicit integration this action simply does nothing.
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::PrePostOp::EXPLICIT::Generic::runPreComputeX(const NOX::Nln::Group& input_grp,
    const Epetra_Vector& dir, const double& step, const NOX::Nln::Group& curr_grp)
{
  // For explicit integration this action simply does nothing.
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::PrePostOp::EXPLICIT::Generic::runPostComputeX(const NOX::Nln::Group& input_grp,
    const Epetra_Vector& dir, const double& step, const NOX::Nln::Group& curr_grp)
{
  // For explicit integration this action simply does nothing.
}

FOUR_C_NAMESPACE_CLOSE