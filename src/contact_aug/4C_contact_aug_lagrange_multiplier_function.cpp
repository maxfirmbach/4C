/*----------------------------------------------------------------------------*/
/** \file
  \brief Lagrange multiplier function: solve a least squares problem to compute
  the Lagrange multiplier value dependent on the current displacement state

  \level 3
*/
/*----------------------------------------------------------------------------*/

#include "4C_contact_aug_lagrange_multiplier_function.hpp"

#include "4C_contact_aug_interface.hpp"
#include "4C_contact_aug_strategy.hpp"
#include "4C_contact_paramsinterface.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_multiply.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_mortar_matrix_transform.hpp"
#include "4C_structure_new_model_evaluator_contact.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
CONTACT::Aug::LagrangeMultiplierFunction::LagrangeMultiplierFunction()
    : isinit_(false),
      issetup_(false),
      strategy_(nullptr),
      interfaces_(0),
      data_(Teuchos::null),
      lin_solver_type_(Core::LinearSolver::SolverType::undefined),
      lin_solver_(Teuchos::null),
      bmat_(Teuchos::null)
{
  /* empty */
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::Aug::LagrangeMultiplierFunction::init(
    const Strategy* const strategy, CONTACT::Aug::DataContainer& data)
{
  issetup_ = false;

  strategy_ = strategy;
  interfaces_.reserve(strategy->contact_interfaces().size());
  std::copy(strategy->contact_interfaces().begin(), strategy->contact_interfaces().end(),
      std::back_inserter(interfaces_));

  data_ = Teuchos::rcpFromRef(data);

  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::Aug::LagrangeMultiplierFunction::setup()
{
  check_init();

  const Teuchos::ParameterList& p_lm_func =
      strategy_->Params().sublist("AUGMENTED").sublist("LAGRANGE_MULTIPLIER_FUNCTION");

  lin_solver_ = create_linear_solver(
      p_lm_func.get<int>("LINEAR_SOLVER"), strategy_->Comm(), lin_solver_type_);

  Redistribute();

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::Aug::LagrangeMultiplierFunction::Redistribute()
{
  const Epetra_Map& slMaDofRowMap = *data_->global_slave_master_dof_row_map_ptr();
  bmat_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(slMaDofRowMap, 100, false, false));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::Solver> CONTACT::Aug::LagrangeMultiplierFunction::create_linear_solver(
    const int lin_sol_id, const Epetra_Comm& comm,
    enum Core::LinearSolver::SolverType& solver_type) const
{
  if (lin_sol_id == -1) FOUR_C_THROW("You must specify a meaningful LINEAR_SOLVER!");

  // get solver parameter list of linear solver
  const Teuchos::ParameterList& solverparams =
      Global::Problem::Instance()->SolverParams(lin_sol_id);
  solver_type = Teuchos::getIntegralValue<Core::LinearSolver::SolverType>(solverparams, "SOLVER");

  Teuchos::RCP<Core::LinAlg::Solver> solver =
      Teuchos::rcp(new Core::LinAlg::Solver(solverparams, comm, nullptr,
          Core::UTILS::IntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::Instance()->IOParams(), "VERBOSITY")));

  if (solver_type != Core::LinearSolver::SolverType::umfpack and
      solver_type != Core::LinearSolver::SolverType::superlu)
    FOUR_C_THROW("Currently only direct linear solvers are supported!");

  return solver;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::Aug::LagrangeMultiplierFunction::lin_solve(
    Core::LinAlg::SparseOperator& mat, Epetra_MultiVector& rhs, Epetra_MultiVector& sol)
{
  if (rhs.NumVectors() > 1 or sol.NumVectors() > 1)
    FOUR_C_THROW("MultiVector support is not yet implemented!");

  Core::LinAlg::SolverParams solver_params;
  solver_params.refactor = true;
  solver_params.reset = true;
  int err = lin_solver_->Solve(
      mat.EpetraOperator(), Teuchos::rcpFromRef(sol), Teuchos::rcpFromRef(rhs), solver_params);

  if (err) FOUR_C_THROW("lin_solve failed with err = %d", err);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> CONTACT::Aug::LagrangeMultiplierFunction::Compute(
    const CONTACT::ParamsInterface& cparams)
{
  TEUCHOS_FUNC_TIME_MONITOR(CONTACT_FUNC_NAME);

  check_init_setup();

  Teuchos::RCP<Epetra_Vector> lmn_vec =
      Teuchos::rcp(new Epetra_Vector(data_->global_active_n_dof_row_map()));

  Teuchos::RCP<Epetra_Vector> str_gradient = get_structure_gradient(cparams);

  create_b_matrix();

  Epetra_Vector str_gradient_exp(*data_->global_slave_master_dof_row_map_ptr(), true);
  Core::LinAlg::Export(*str_gradient, str_gradient_exp);

  Epetra_Vector rhs(data_->global_active_n_dof_row_map(), true);
  bmat_->Multiply(true, str_gradient_exp, rhs);

  Teuchos::RCP<Core::LinAlg::SparseMatrix> bbmat =
      Core::LinAlg::MLMultiply(*bmat_, true, *bmat_, false, false, false, true);

  lin_solve(*bbmat, rhs, *lmn_vec);

  return lmn_vec;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> CONTACT::Aug::LagrangeMultiplierFunction::get_structure_gradient(
    const CONTACT::ParamsInterface& cparams) const
{
  const STR::MODELEVALUATOR::Generic& model = cparams.get_model_evaluator();
  const STR::MODELEVALUATOR::Contact& cmodel =
      dynamic_cast<const STR::MODELEVALUATOR::Contact&>(model);

  const std::vector<Inpar::STR::ModelType> without_contact_model(1, model.Type());

  Teuchos::RCP<Epetra_Vector> str_gradient =
      cmodel.assemble_force_of_models(&without_contact_model, true);

  return str_gradient;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::Aug::LagrangeMultiplierFunction::create_b_matrix()
{
  bmat_->reset();

  bmat_->Add(data_->DMatrix(), true, 1.0, 0.0);
  bmat_->Add(data_->MMatrix(), true, 1.0, 1.0);
  //  bmat_->Add( data_->DLmNWGapLinMatrix(), true, 1.0, 0.0 );

  bmat_->Complete(
      data_->global_active_n_dof_row_map(), *data_->global_slave_master_dof_row_map_ptr());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> CONTACT::Aug::LagrangeMultiplierFunction::FirstOrderDirection(
    const CONTACT::ParamsInterface& cparams, const Epetra_Vector& dincr)
{
  TEUCHOS_FUNC_TIME_MONITOR(CONTACT_FUNC_NAME);

  Epetra_Vector rhs(data_->global_active_n_dof_row_map(), true);

  const STR::MODELEVALUATOR::Generic& model = cparams.get_model_evaluator();
  const STR::MODELEVALUATOR::Contact& cmodel =
      dynamic_cast<const STR::MODELEVALUATOR::Contact&>(model);

  // access the full stiffness matrix
  Core::LinAlg::SparseMatrix full_stiff(
      *cmodel.get_jacobian_block(STR::MatBlockType::displ_displ), Core::LinAlg::Copy);

  Teuchos::RCP<Core::LinAlg::SparseMatrix> kdd_ptr =
      strategy_->get_matrix_block_ptr(CONTACT::MatBlockType::displ_displ);

  // undo matrix contributions
  full_stiff.Add(*kdd_ptr, false, -1.0, 1.0);

  // --- first summand
  Teuchos::RCP<Epetra_Vector> tmp_vec = Core::LinAlg::CreateVector(full_stiff.RangeMap(), true);

  int err = full_stiff.Multiply(false, dincr, *tmp_vec);
  if (err) FOUR_C_THROW("Multiply failed with err = %d", err);

  Teuchos::RCP<Epetra_Vector> tmp_vec_exp =
      Core::LinAlg::CreateVector(*data_->global_slave_master_dof_row_map_ptr(), true);

  // build necessary exporter
  Epetra_Export exporter(tmp_vec_exp->Map(), tmp_vec->Map());
  err = tmp_vec_exp->Import(*tmp_vec, exporter, Insert);
  if (err) FOUR_C_THROW("Import failed with err = %d", err);

  create_b_matrix();

  bmat_->Multiply(true, *tmp_vec_exp, rhs);

  // --- second summand
  tmp_vec = get_structure_gradient(cparams);
  tmp_vec_exp->PutScalar(0.0);
  tmp_vec_exp->Import(*tmp_vec, exporter, Insert);

  Teuchos::RCP<Epetra_Vector> dincr_exp =
      Core::LinAlg::CreateVector(*data_->global_slave_master_dof_row_map_ptr(), true);
  err = dincr_exp->Import(dincr, exporter, Insert);
  if (err) FOUR_C_THROW("Import failed with err = %d", err);

  assemble_gradient_b_matrix_contribution(*dincr_exp, *tmp_vec_exp, rhs);

  // --- 3rd summand
  assemble_gradient_bb_matrix_contribution(*dincr_exp, data_->LmN(), rhs);

  Teuchos::RCP<Epetra_Vector> lmincr =
      Core::LinAlg::CreateVector(data_->global_active_n_dof_row_map(), true);

  Teuchos::RCP<Core::LinAlg::SparseMatrix> bbmat =
      Core::LinAlg::MLMultiply(*bmat_, true, *bmat_, false, false, false, true);

  lin_solve(*bbmat, rhs, *lmincr);

  return lmincr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::Aug::LagrangeMultiplierFunction::assemble_gradient_bb_matrix_contribution(
    const Epetra_Vector& dincr, const Epetra_Vector& lm, Epetra_Vector& lmincr) const
{
  if (not lmincr.Map().SameAs(lm.Map())) FOUR_C_THROW("The maps must be identical!");

  for (plain_interface_set::const_iterator cit = interfaces_.begin(); cit != interfaces_.end();
       ++cit)
  {
    const CONTACT::Aug::Interface& interface = dynamic_cast<const CONTACT::Aug::Interface&>(**cit);

    interface.assemble_gradient_bb_matrix_contribution(dincr, lm, lmincr);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::Aug::LagrangeMultiplierFunction::assemble_gradient_b_matrix_contribution(
    const Epetra_Vector& dincr, const Epetra_Vector& str_grad, Epetra_Vector& lmincr) const
{
  if (not dincr.Map().SameAs(str_grad.Map())) FOUR_C_THROW("The maps must be identical!");

  for (plain_interface_set::const_iterator cit = interfaces_.begin(); cit != interfaces_.end();
       ++cit)
  {
    const CONTACT::Aug::Interface& interface = dynamic_cast<const CONTACT::Aug::Interface&>(**cit);

    interface.assemble_gradient_b_matrix_contribution(dincr, str_grad, lmincr);
  }
}

FOUR_C_NAMESPACE_CLOSE