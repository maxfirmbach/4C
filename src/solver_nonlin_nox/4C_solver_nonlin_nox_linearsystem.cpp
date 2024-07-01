/*-----------------------------------------------------------*/
/*! \file

\brief %NOX::NLN extension of the %::NOX::Epetra::LinearSystem.



\level 3

*/
/*-----------------------------------------------------------*/

#include "4C_solver_nonlin_nox_linearsystem.hpp"

#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_solver_nonlin_nox_aux.hpp"
#include "4C_solver_nonlin_nox_interface_jacobian.hpp"
#include "4C_solver_nonlin_nox_interface_required.hpp"
#include "4C_solver_nonlin_nox_linearsystem_prepostoperator.hpp"
#include "4C_solver_nonlin_nox_solver_ptc.hpp"
#include "4C_structure_new_nln_linearsystem_scaling.hpp"
#include "4C_utils_epetra_exceptions.hpp"

#include <Epetra_LinearProblem.h>
#include <Epetra_Vector.h>
#include <NOX_Epetra_Interface_Preconditioner.H>
#include <NOX_Epetra_Scaling.H>
#include <Teuchos_LAPACK.hpp>
#include <Teuchos_ParameterList.hpp>

#include <unordered_map>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams,
    const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers,
    const Teuchos::RCP<::NOX::Epetra::Interface::Required>& iReq,
    const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& jacobian_op,
    const Teuchos::RCP<::NOX::Epetra::Interface::Preconditioner>& iPrec,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& preconditioner,
    const ::NOX::Epetra::Vector& cloneVector,
    const Teuchos::RCP<::NOX::Epetra::Scaling> scalingObject)
    : utils_(printParams),
      solvers_(solvers),
      reqInterfacePtr_(iReq),
      jacInterfacePtr_(iJac),
      jacType_(NOX::Nln::LinSystem::LinalgSparseOperator),
      precInterfacePtr_(iPrec),
      precType_(NOX::Nln::LinSystem::LinalgSparseOperator),
      precPtr_(preconditioner),
      precMatrixSource_(SeparateMatrix),
      scaling_(scalingObject),
      conditionNumberEstimate_(0.0),
      timer_("", true),
      timeCreatePreconditioner_(0.0),
      timeApplyJacbianInverse_(0.0),
      resNorm2_(0.0),
      prePostOperatorPtr_(Teuchos::null),
      jac_ptr_(jacobian_op)
{
  // Jacobian operator is supplied
  jacType_ = NOX::Nln::Aux::GetOperatorType(jacobian());

  reset(linearSolverParams);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams,
    const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers,
    const Teuchos::RCP<::NOX::Epetra::Interface::Required>& iReq,
    const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& jacobian_op,
    const Teuchos::RCP<::NOX::Epetra::Interface::Preconditioner>& iPrec,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& preconditioner,
    const ::NOX::Epetra::Vector& cloneVector)
    : utils_(printParams),
      solvers_(solvers),
      reqInterfacePtr_(iReq),
      jacInterfacePtr_(iJac),
      jacType_(NOX::Nln::LinSystem::LinalgSparseOperator),
      precInterfacePtr_(iPrec),
      precType_(NOX::Nln::LinSystem::LinalgSparseOperator),
      precPtr_(preconditioner),
      precMatrixSource_(SeparateMatrix),
      scaling_(Teuchos::null),
      conditionNumberEstimate_(0.0),
      timer_("", true),
      timeCreatePreconditioner_(0.0),
      timeApplyJacbianInverse_(0.0),
      resNorm2_(0.0),
      prePostOperatorPtr_(Teuchos::null),
      jac_ptr_(jacobian_op)
{
  // Jacobian operator is supplied
  jacType_ = NOX::Nln::Aux::GetOperatorType(jacobian());

  reset(linearSolverParams);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams,
    const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers,
    const Teuchos::RCP<::NOX::Epetra::Interface::Required>& iReq,
    const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& jacobian_op,
    const ::NOX::Epetra::Vector& cloneVector,
    const Teuchos::RCP<::NOX::Epetra::Scaling> scalingObject)
    : utils_(printParams),
      solvers_(solvers),
      reqInterfacePtr_(iReq),
      jacInterfacePtr_(iJac),
      jacType_(NOX::Nln::LinSystem::LinalgSparseOperator),
      precInterfacePtr_(Teuchos::null),
      precType_(NOX::Nln::LinSystem::LinalgSparseOperator),
      precPtr_(Teuchos::null),
      precMatrixSource_(SeparateMatrix),
      scaling_(scalingObject),
      conditionNumberEstimate_(0.0),
      timer_("", true),
      timeCreatePreconditioner_(0.0),
      timeApplyJacbianInverse_(0.0),
      resNorm2_(0.0),
      prePostOperatorPtr_(Teuchos::null),
      jac_ptr_(jacobian_op)
{
  // Jacobian operator is supplied
  jacType_ = NOX::Nln::Aux::GetOperatorType(jacobian());

  reset(linearSolverParams);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams,
    const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers,
    const Teuchos::RCP<::NOX::Epetra::Interface::Required>& iReq,
    const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& jacobian_op,
    const ::NOX::Epetra::Vector& cloneVector)
    : utils_(printParams),
      solvers_(solvers),
      reqInterfacePtr_(iReq),
      jacInterfacePtr_(iJac),
      jacType_(NOX::Nln::LinSystem::LinalgSparseOperator),
      precInterfacePtr_(Teuchos::null),
      precType_(NOX::Nln::LinSystem::LinalgSparseOperator),
      precPtr_(Teuchos::null),
      precMatrixSource_(SeparateMatrix),
      scaling_(Teuchos::null),
      conditionNumberEstimate_(0.0),
      timer_("", true),
      timeCreatePreconditioner_(0.0),
      timeApplyJacbianInverse_(0.0),
      resNorm2_(0.0),
      prePostOperatorPtr_(Teuchos::null),
      jac_ptr_(jacobian_op)
{
  // Jacobian operator is supplied
  jacType_ = NOX::Nln::Aux::GetOperatorType(jacobian());

  reset(linearSolverParams);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::LinearSystem::reset(Teuchos::ParameterList& p)
{
  zeroInitialGuess_ = p.get<bool>("Zero Initial Guess", false);

  manualScaling_ = p.get<bool>("Compute Scaling Manually", true);

  // Place linear solver details in the "Output" sublist of the
  // "Linear Solver" parameter list
  outputSolveDetails_ = p.get<bool>("Output Solver Details", true);

  // set the pre/post-operator
  reset_pre_post_operator(p);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::LinearSystem::reset_pre_post_operator(Teuchos::ParameterList& p)
{
  if (prePostOperatorPtr_.is_null())
    prePostOperatorPtr_ = Teuchos::rcp(new NOX::Nln::LinSystem::PrePostOperator(p));
  else
    prePostOperatorPtr_->reset(p);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::Nln::LinearSystem::applyJacobianBlock(const ::NOX::Epetra::Vector& input,
    Teuchos::RCP<::NOX::Epetra::Vector>& result, unsigned rbid, unsigned cbid) const
{
  const Core::LinAlg::SparseMatrix& blockc = getJacobianBlock(rbid, cbid);
  Core::LinAlg::SparseMatrix& block = const_cast<Core::LinAlg::SparseMatrix&>(blockc);
  const Epetra_Map& domainmap = block.DomainMap();
  const Epetra_Map& rangemap = block.RangeMap();

  const Epetra_Vector& input_epetra = input.getEpetraVector();
  Teuchos::RCP<const Epetra_Vector> input_apply = Teuchos::null;

  if (not input_epetra.Map().SameAs(domainmap))
  {
    input_apply = Core::LinAlg::ExtractMyVector(input_epetra, domainmap);
  }
  else
  {
    input_apply = Teuchos::rcpFromRef(input_epetra);
  }

  Teuchos::RCP<Epetra_Vector> result_apply = Teuchos::rcp(new Epetra_Vector(rangemap, true));

  block.SetUseTranspose(false);
  int status = block.Apply(*input_apply, *result_apply);

  result = Teuchos::rcp(new ::NOX::Epetra::Vector(result_apply, ::NOX::Epetra::Vector::CreateView));

  return (status == 0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::Nln::LinearSystem::applyJacobian(
    const ::NOX::Epetra::Vector& input, ::NOX::Epetra::Vector& result) const
{
  jacobian().SetUseTranspose(false);
  int status = jacobian().Apply(input.getEpetraVector(), result.getEpetraVector());
  return (status == 0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::Nln::LinearSystem::applyJacobianTranspose(
    const ::NOX::Epetra::Vector& input, ::NOX::Epetra::Vector& result) const
{
  // Apply the Jacobian
  jacobian().SetUseTranspose(true);
  int status = jacobian().Apply(input.getEpetraVector(), result.getEpetraVector());
  jacobian().SetUseTranspose(false);

  return (status == 0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::LinearSystem::set_linear_problem_for_solve(Epetra_LinearProblem& linear_problem,
    Core::LinAlg::SparseOperator& jac, Epetra_Vector& lhs, Epetra_Vector& rhs) const
{
  linear_problem.SetOperator(jac.EpetraOperator().get());
  linear_problem.SetLHS(&lhs);
  linear_problem.SetRHS(&rhs);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::LinearSystem::complete_solution_after_solve(
    const Epetra_LinearProblem& linProblem, Epetra_Vector& lhs) const
{
  /* nothing to do in the default case */
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::Nln::LinearSystem::applyJacobianInverse(Teuchos::ParameterList& linearSolverParams,
    const ::NOX::Epetra::Vector& input, ::NOX::Epetra::Vector& result)
{
  /* Need non-const version of the input vector
   * Epetra_LinearProblem requires non-const versions so we can perform
   * scaling of the linear problem.
   * Same is valid for the prePostOperator. We want to have the
   * possibility to change the linear system. */
  ::NOX::Epetra::Vector& nonConstInput = const_cast<::NOX::Epetra::Vector&>(input);

  prePostOperatorPtr_->run_pre_apply_jacobian_inverse(nonConstInput, jacobian(), *this);

  double startTime = timer_.wallTime();

  // calculate the residual norm
  resNorm2_ = nonConstInput.norm(::NOX::Abstract::Vector::TwoNorm);

  // Zero out the delta X of the linear problem if requested by user.
  if (zeroInitialGuess_) result.init(0.0);

  // Create Epetra linear problem object for the linear solve
  /* Note: We switch from LINALG_objects to pure Epetra_objects.
   * This is necessary for the linear solver.
   *     Core::LinAlg::SparseMatrix ---> Epetra_CrsMatrix */
  Epetra_LinearProblem linProblem;
  set_linear_problem_for_solve(
      linProblem, jacobian(), result.getEpetraVector(), nonConstInput.getEpetraVector());

  // ************* Begin linear system scaling *****************
  if (!Teuchos::is_null(scaling_))
  {
    if (!manualScaling_) scaling_->computeScaling(linProblem);

    scaling_->scaleLinearSystem(linProblem);

    if (utils_.isPrintType(::NOX::Utils::Details)) utils_.out() << *scaling_ << std::endl;
  }
  // ************* End linear system scaling *******************

  // get current linear solver from the std_map
  Teuchos::RCP<Core::LinAlg::Solver> currSolver;
  NOX::Nln::SolutionType solType = get_active_lin_solver(solvers_, currSolver);

  // set solver options if necessary
  auto solver_params = set_solver_options(linearSolverParams, currSolver, solType);

  // solve
  int iter = linearSolverParams.get<int>("Number of Nonlinear Iterations", -10);
  if (iter == -10)
    throw_error("applyJacobianInverse", "\"Number of Nonlinear Iterations\" was not specified");

  solver_params.refactor = true;
  solver_params.reset = iter == 0;
  const int linsol_status = currSolver->NoxSolve(linProblem, solver_params);
  if (linsol_status)
  {
    if (utils_.isPrintType(::NOX::Utils::Warning))
      utils_.out() << "NOX::Nln::LinearSystem::applyJacobianInverse -- "
                      "linear solve failed (err = "
                   << linsol_status << ")\n";
  }

  // ************* Begin linear system unscaling *************
  if (!Teuchos::is_null(scaling_)) scaling_->unscaleLinearSystem(linProblem);
  // ************* End linear system unscaling ***************

  complete_solution_after_solve(linProblem, result.getEpetraVector());

  double endTime = timer_.wallTime();
  timeApplyJacbianInverse_ += (endTime - startTime);

  prePostOperatorPtr_->run_post_apply_jacobian_inverse(result, nonConstInput, jacobian(), *this);

  return (linsol_status == 0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::Nln::LinearSystem::applyRightPreconditioning(bool useTranspose,
    Teuchos::ParameterList& params, const ::NOX::Epetra::Vector& input,
    ::NOX::Epetra::Vector& result) const
{
  if (&result != &input) result = input;
  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::Nln::LinearSystem::computeJacobian(const ::NOX::Epetra::Vector& x)
{
  prePostOperatorPtr_->run_pre_compute_jacobian(jacobian(), x.getEpetraVector(), *this);

  bool success = jacInterfacePtr_->computeJacobian(x.getEpetraVector(), jacobian());

  prePostOperatorPtr_->run_post_compute_jacobian(jacobian(), x.getEpetraVector(), *this);
  return success;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::Nln::LinearSystem::computeFandJacobian(
    const ::NOX::Epetra::Vector& x, ::NOX::Epetra::Vector& rhs)
{
  prePostOperatorPtr_->run_pre_compute_fand_jacobian(
      rhs.getEpetraVector(), jacobian(), x.getEpetraVector(), *this);

  const bool success =
      Teuchos::rcp_dynamic_cast<NOX::Nln::Interface::Jacobian>(jacInterfacePtr_, true)
          ->computeFandJacobian(x.getEpetraVector(), rhs.getEpetraVector(), jacobian());

  prePostOperatorPtr_->run_post_compute_fand_jacobian(
      rhs.getEpetraVector(), jacobian(), x.getEpetraVector(), *this);
  return success;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::Nln::LinearSystem::compute_correction_system(const enum CorrectionType type,
    const ::NOX::Abstract::Group& grp, const ::NOX::Epetra::Vector& x, ::NOX::Epetra::Vector& rhs)
{
  prePostOperatorPtr_->run_pre_compute_fand_jacobian(
      rhs.getEpetraVector(), jacobian(), x.getEpetraVector(), *this);

  const bool success =
      Teuchos::rcp_dynamic_cast<NOX::Nln::Interface::Jacobian>(jacInterfacePtr_, true)
          ->compute_correction_system(
              type, grp, x.getEpetraVector(), rhs.getEpetraVector(), jacobian());

  prePostOperatorPtr_->run_post_compute_fand_jacobian(
      rhs.getEpetraVector(), jacobian(), x.getEpetraVector(), *this);

  return success;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<::NOX::Epetra::Scaling> NOX::Nln::LinearSystem::getScaling() { return scaling_; }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::LinearSystem::resetScaling(const Teuchos::RCP<::NOX::Epetra::Scaling>& scalingObject)
{
  scaling_ = scalingObject;
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::LinearSystem::adjust_pseudo_time_step(double& delta, const double& stepSize,
    const ::NOX::Epetra::Vector& dir, const ::NOX::Epetra::Vector& rhs,
    const NOX::Nln::Solver::PseudoTransient& ptcsolver)
{
  const Epetra_Vector& scalingDiagOp = ptcsolver.get_scaling_diag_operator();
  // ---------------------------------------------------------------------
  // first undo the modification of the jacobian
  // ---------------------------------------------------------------------
  Teuchos::RCP<Epetra_Vector> v = Teuchos::rcp(new Epetra_Vector(scalingDiagOp));
  v->Scale(ptcsolver.get_inverse_pseudo_time_step());
  Teuchos::RCP<Core::LinAlg::SparseMatrix> jac =
      Teuchos::rcp_dynamic_cast<Core::LinAlg::SparseMatrix>(jacobian_ptr());
  if (jac.is_null())
    throw_error("adjust_pseudo_time_step()", "Cast to Core::LinAlg::SparseMatrix failed!");
  // get the diagonal terms of the jacobian
  Teuchos::RCP<Epetra_Vector> diag = Core::LinAlg::CreateVector(jac->RowMap(), false);
  jac->ExtractDiagonalCopy(*diag);
  diag->Update(-1.0, *v, 1.0);
  // Finally undo the changes
  jac->replace_diagonal_values(*diag);

  // ---------------------------------------------------------------------
  // calculate the least squares approximated corrected pseudo time step
  // ---------------------------------------------------------------------
  /* evaluate the first vector:
   *    eta^{-1} F_{n-1} + (\nabla_{x} F_{n-1})^{T} d_{n-1}             */
  double stepSizeInv = 1.0 / stepSize;
  Teuchos::RCP<Epetra_Vector> vec_1 = Core::LinAlg::CreateVector(jac->RowMap(), true);
  Teuchos::RCP<Epetra_Vector> vec_2 = Teuchos::rcp(new Epetra_Vector(rhs.getEpetraVector()));
  jac->Multiply(false, dir.getEpetraVector(), *vec_1);
  vec_2->Scale(stepSizeInv);
  vec_1->Update(1.0, *vec_2, 1.0);
  /* evaluate the second vector:              d^{T} V                   */
  vec_2->Multiply(1.0, scalingDiagOp, dir.getEpetraVector(), 0.0);

  // finally evaluate the scalar product
  double numerator = 0.0;
  double denominator = 0.0;
  vec_2->Dot(*vec_1, &numerator);
  vec_1->Dot(*vec_1, &denominator);

  // ---------------------------------------------------------------------
  // show the error (L2-norm)
  // ---------------------------------------------------------------------
  Teuchos::RCP<Epetra_Vector> vec_err = Core::LinAlg::CreateVector(jac->RowMap(), true);
  vec_err->Update(delta, *vec_1, 1.0, *vec_2, 0.0);
  double error_start = 0.0;
  vec_err->Norm2(&error_start);

  delta = -numerator / denominator;

  // ---------------------------------------------------------------------
  // show the actual remaining error (L2-norm)
  // ---------------------------------------------------------------------
  vec_err->Update(delta, *vec_1, 1.0, *vec_2, 0.0);
  double error_end = 0.0;
  vec_err->Norm2(&error_end);
  if (utils_.isPrintType(::NOX::Utils::Details))
  {
    utils_.out() << "| Error: " << std::setw(5) << std::setprecision(3) << std::scientific
                 << error_start << " -> " << std::setw(5) << std::setprecision(3) << std::scientific
                 << error_end << " |" << std::endl;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::LinearSystem::throw_error(
    const std::string& functionName, const std::string& errorMsg) const
{
  if (utils_.isPrintType(::NOX::Utils::Error))
  {
    utils_.out() << "NOX::Nln::LinearSystem::" << functionName << " - " << errorMsg << std::endl;
  }
  throw "NOX Error";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const ::NOX::Epetra::Interface::Required>
NOX::Nln::LinearSystem::get_required_interface() const
{
  return reqInterfacePtr_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const ::NOX::Epetra::Interface::Jacobian>
NOX::Nln::LinearSystem::get_jacobian_interface() const
{
  return jacInterfacePtr_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const ::NOX::Epetra::Interface::Preconditioner>
NOX::Nln::LinearSystem::getPrecInterface() const
{
  return precInterfacePtr_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Operator> NOX::Nln::LinearSystem::getJacobianOperator() const
{
  return jacobian_ptr();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Operator> NOX::Nln::LinearSystem::getJacobianOperator()
{
  return jacobian_ptr();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const enum NOX::Nln::LinSystem::OperatorType& NOX::Nln::LinearSystem::get_jacobian_operator_type()
    const
{
  return jacType_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::LinearSystem::setJacobianOperatorForSolve(
    const Teuchos::RCP<const Epetra_Operator>& solveJacOp)
{
  const Teuchos::RCP<const Core::LinAlg::SparseOperator>& linalgSprOp =
      Teuchos::rcp_dynamic_cast<const Core::LinAlg::SparseOperator>(solveJacOp);
  if (linalgSprOp.is_null())
    throw_error("setJacobianOperatorForSolve", "dynamic_cast to LINALG_SparseOperator failed!");

  set_jacobian_operator_for_solve(linalgSprOp);
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::LinearSystem::set_jacobian_operator_for_solve(
    const Teuchos::RCP<const Core::LinAlg::SparseOperator>& solveJacOp)
{
  if (jacType_ != NOX::Nln::Aux::GetOperatorType(*solveJacOp))
    throw_error("set_jacobian_operator_for_solve", "wrong operator type!");

  jac_ptr_ = Teuchos::rcp_const_cast<Core::LinAlg::SparseOperator>(solveJacOp);
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::Nln::LinearSystem::createPreconditioner(const ::NOX::Epetra::Vector& x,
    Teuchos::ParameterList& linearSolverParams, bool recomputeGraph) const
{
  return false;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::Nln::LinearSystem::destroyPreconditioner() const { return false; }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::Nln::LinearSystem::recomputePreconditioner(
    const ::NOX::Epetra::Vector& x, Teuchos::ParameterList& linearSolverParams) const
{
  return false;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
::NOX::Epetra::LinearSystem::PreconditionerReusePolicyType
NOX::Nln::LinearSystem::getPreconditionerPolicy(bool advanceReuseCounter)
{
  return PRPT_REBUILD;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::Nln::LinearSystem::isPreconditionerConstructed() const { return false; }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::Nln::LinearSystem::hasPreconditioner() const { return false; }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Operator> NOX::Nln::LinearSystem::getGeneratedPrecOperator() const
{
  return Teuchos::null;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Operator> NOX::Nln::LinearSystem::getGeneratedPrecOperator()
{
  return Teuchos::null;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::LinearSystem::setPrecOperatorForSolve(
    const Teuchos::RCP<const Epetra_Operator>& solvePrecOp)
{
  throw_error("setPrecOperatorForSolve", "no preconditioner supported");
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::Nln::LinearSystem::DestroyJacobian()
{
  jac_ptr_ = Teuchos::null;
  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::LinAlg::SparseMatrix& NOX::Nln::LinearSystem::getJacobianBlock(
    unsigned rbid, unsigned cbid) const
{
  switch (jacType_)
  {
    case LinSystem::LinalgBlockSparseMatrix:
    {
      typedef Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy> linalg_bsm;
      const linalg_bsm& jac_block = dynamic_cast<const linalg_bsm&>(jacobian());

      if (rbid >= static_cast<unsigned>(jac_block.Rows()) or
          cbid >= static_cast<unsigned>(jac_block.Cols()))
        FOUR_C_THROW(
            "The given row/column block ids exceed the block dimension of "
            "the jacobian matrix.");

      return jac_block.Matrix(rbid, cbid);
    }
    case LinSystem::LinalgSparseMatrix:
    {
      if (rbid != 0 or cbid != 0)
        FOUR_C_THROW("There is only one block for a Core::LinAlg::SparseMatrix!");

      const Core::LinAlg::SparseMatrix& jac_sm =
          dynamic_cast<const Core::LinAlg::SparseMatrix&>(jacobian());

      return jac_sm;
    }
    default:
    {
      FOUR_C_THROW("Unsupported LinSystem::OperatorType: %d | %s", jacType_,
          NOX::Nln::LinSystem::OperatorType2String(jacType_).c_str());
      exit(EXIT_FAILURE);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Epetra_Map& NOX::Nln::LinearSystem::getJacobianRangeMap(unsigned rbid, unsigned cbid) const
{
  return getJacobianBlock(rbid, cbid).RangeMap();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> NOX::Nln::LinearSystem::get_diagonal_of_jacobian(
    unsigned diag_bid) const
{
  const Core::LinAlg::SparseMatrix& diag_block = getJacobianBlock(diag_bid, diag_bid);
  const Epetra_Map& rmap = diag_block.RangeMap();

  Teuchos::RCP<Epetra_Vector> diag_copy = Teuchos::rcp(new Epetra_Vector(rmap, true));

  diag_block.ExtractDiagonalCopy(*diag_copy);

  return diag_copy;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::LinearSystem::replace_diagonal_of_jacobian(
    const Epetra_Vector& new_diag, unsigned diag_bid)
{
  const Core::LinAlg::SparseMatrix& diag_block = getJacobianBlock(diag_bid, diag_bid);
  Core::LinAlg::SparseMatrix& mod_diag_block = const_cast<Core::LinAlg::SparseMatrix&>(diag_block);

  CATCH_EPETRA_ERROR(mod_diag_block.replace_diagonal_values(new_diag));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double NOX::Nln::LinearSystem::compute_serial_condition_number_of_jacobian(
    const LinSystem::ConditionNumber condnum_type) const
{
  if (jacobian().Comm().NumProc() > 1) FOUR_C_THROW("Currently only one processor is supported!");

  Core::LinAlg::SerialDenseMatrix dense_jac;
  convert_jacobian_to_dense_matrix(dense_jac);

  Teuchos::LAPACK<int, double> lapack;

  const int N = dense_jac.numCols();
  const int M = dense_jac.numRows();
  double* A = dense_jac.values();
  const int LDA = dense_jac.stride();
  double anorm = 0.0;
  char ctype = char();
  switch (condnum_type)
  {
    case LinSystem::ConditionNumber::one_norm:
      anorm = dense_jac.normOne();
      ctype = '1';
      break;
    case LinSystem::ConditionNumber::inf_norm:
      anorm = dense_jac.normInf();
      ctype = 'I';
      break;
    default:
      FOUR_C_THROW("Unsupported");
      exit(EXIT_FAILURE);
  }
  double rcond = 0.0;

  std::vector<double> work(4 * N);
  std::vector<int> iwork(N);
  int info = 0;

  // pre-step: compute the factorization of the matrix A (LU-decomposition)
  std::vector<int> ipiv(std::min(M, N));
  lapack.GETRF(M, N, A, LDA, ipiv.data(), &info);

  if (info) FOUR_C_THROW("Error detected in LAPACK::GETRF");

  // compute the condition number
  lapack.GECON(ctype, N, A, LDA, anorm, &rcond, work.data(), iwork.data(), &info);

  if (info) FOUR_C_THROW("Error detected in LAPACK::GECON");

  if (rcond < std::numeric_limits<double>::min())
    FOUR_C_THROW(
        "The reciprocal condition number is zero. The jacobian"
        " seems to be singular.");

  return 1.0 / rcond;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::LinearSystem::compute_serial_eigenvalues_of_jacobian(
    Core::LinAlg::SerialDenseVector& reigenvalues,
    Core::LinAlg::SerialDenseVector& ieigenvalues) const
{
  if (jacobian().Comm().NumProc() > 1) FOUR_C_THROW("Currently only one processor is supported!");

  Core::LinAlg::SerialDenseMatrix dense_jac;
  convert_jacobian_to_dense_matrix(dense_jac);

  throw_if_zero_row(dense_jac);
  solve_non_symm_eigen_value_problem(dense_jac, reigenvalues, ieigenvalues);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::LinearSystem::prepare_block_dense_matrix(
    const Core::LinAlg::BlockSparseMatrixBase& block_sparse,
    Core::LinAlg::SerialDenseMatrix& block_dense) const
{
  const int grows = block_sparse.FullRangeMap().NumGlobalElements();
  const int gcols = block_sparse.FullDomainMap().NumGlobalElements();

  block_dense.reshape(grows, gcols);
  if (block_dense.numCols() != block_dense.numRows())
    FOUR_C_THROW("The complete block dense matrix is not quadratic!");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::LinearSystem::throw_if_zero_row(
    const Core::LinAlg::SerialDenseMatrix& block_dense) const
{
  for (int r = 0; r < block_dense.numCols(); ++r)
  {
    double csum = 0.0;
    for (int c = 0; c < block_dense.numRows(); ++c) csum += std::abs(block_dense(r, c));

    if (std::abs(csum) < std::numeric_limits<double>::epsilon()) FOUR_C_THROW("Zero row detected!");
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::LinearSystem::convert_jacobian_to_dense_matrix(
    Core::LinAlg::SerialDenseMatrix& dense_jac) const
{
  switch (get_jacobian_operator_type())
  {
    case NOX::Nln::LinSystem::LinalgBlockSparseMatrix:
    {
      typedef Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>
          linalg_blocksparsematrix;
      const linalg_blocksparsematrix& block_sparse =
          dynamic_cast<const linalg_blocksparsematrix&>(jacobian());

      const Epetra_Map& full_rangemap = block_sparse.FullRangeMap();
      const Epetra_Map& full_domainmap = block_sparse.FullDomainMap();

      prepare_block_dense_matrix(block_sparse, dense_jac);

      for (int r = 0; r < block_sparse.Rows(); ++r)
      {
        for (int c = 0; c < block_sparse.Cols(); ++c)
        {
          const Core::LinAlg::SparseMatrix& block = block_sparse.Matrix(r, c);
          convert_sparse_to_dense_matrix(block, dense_jac, full_rangemap, full_domainmap);
        }
      }

      break;
    }
    case NOX::Nln::LinSystem::LinalgSparseMatrix:
    {
      const Core::LinAlg::SparseMatrix& jac =
          dynamic_cast<const Core::LinAlg::SparseMatrix&>(jacobian());

      convert_sparse_to_dense_matrix(jac, dense_jac, jac.RangeMap(), jac.DomainMap());

      break;
    }
    default:
    {
      FOUR_C_THROW("Unsupported jacobian operator type!");
      exit(EXIT_FAILURE);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::LinearSystem::convert_sparse_to_dense_matrix(
    const Core::LinAlg::SparseMatrix& sparse, Core::LinAlg::SerialDenseMatrix& dense,
    const Epetra_Map& full_rangemap, const Epetra_Map& full_domainmap) const
{
  if (not sparse.Filled()) FOUR_C_THROW("The sparse matrix must be filled!");
  Teuchos::RCP<const Epetra_CrsMatrix> crs_mat = sparse.EpetraMatrix();

  if (dense.numCols() == 0 or dense.numRows() == 0)
  {
    const int numgrows = sparse.RowMap().NumGlobalElements();
    const int numgcols = sparse.ColMap().NumGlobalElements();
    dense.reshape(numgrows, numgcols);
    dense.putScalar(0.0);
  }

  const int num_myrows = sparse.RowMap().NumMyElements();
  const int* rgids = sparse.RowMap().MyGlobalElements();

  for (int rlid = 0; rlid < num_myrows; ++rlid)
  {
    // number of entries in this row
    int numentries = 0;
    // values of the entries in this row
    double* rvals = nullptr;
    // local indices
    int* indices = nullptr;

    crs_mat->ExtractMyRowView(rlid, numentries, rvals, indices);

    const int rgid = rgids[rlid];
    const int full_rlid = full_rangemap.LID(rgid);
    if (full_rlid == -1)
      FOUR_C_THROW("Row/Range: Couldn't find the corresponding LID to GID %d", rgid);

    for (int i = 0; i < numentries; ++i)
    {
      const int cgid = sparse.ColMap().GID(indices[i]);
      if (cgid == -1)
        FOUR_C_THROW("Column/Domain: Couldn't find the corresponding GID to LID %d", indices[i]);

      const int full_clid = full_domainmap.LID(cgid);
      if (full_clid == -1)
        FOUR_C_THROW("Column/Domain: Couldn't find the corresponding LID to GID %d", cgid);

      dense(full_rlid, full_clid) = rvals[i];
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::LinearSystem::solve_non_symm_eigen_value_problem(
    Core::LinAlg::SerialDenseMatrix& mat, Core::LinAlg::SerialDenseVector& reigenvalues,
    Core::LinAlg::SerialDenseVector& ieigenvalues) const
{
  // start debugging
  //  std::ofstream of("/home/hiermeier/Workspace/o/test/mat.csv",std::ios_base::out);
  //  const int num_cols = mat.numRows();
  //  for ( int r=0; r<mat.numCols(); ++r )
  //    for ( int c=0; c<num_cols; ++c )
  //      of << std::setprecision(16) << std::scientific << mat(r,c)
  //         << ( c != num_cols-1 ? "," : "\n" );
  //  of.close();
  // end debugging

  reigenvalues.size(mat.numCols());
  ieigenvalues.size(mat.numCols());

  call_geev(mat, reigenvalues, ieigenvalues);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::LinearSystem::call_ggev(Core::LinAlg::SerialDenseMatrix& mat,
    Core::LinAlg::SerialDenseVector& reigenvalues,
    Core::LinAlg::SerialDenseVector& ieigenvalues) const
{
  int info = 0;
  int lwork = 8.0 * mat.numCols();
  std::vector<double> work(lwork);

  if (mat.numCols() != mat.numRows())
    FOUR_C_THROW("A N-by-N real non-symmetric matrix is expected by GGEV!");

  // create dummy B matrix
  static bool first_execution = true;
  static Core::LinAlg::SerialDenseMatrix bmat;
  if (first_execution or bmat.numCols() != mat.numCols())
  {
    bmat.reshape(mat.numCols(), mat.numCols());
    for (int r = 0; r < bmat.numCols(); ++r) bmat(r, r) = 1.0;
  }

  Core::LinAlg::SerialDenseVector beta(reigenvalues);

  Teuchos::LAPACK<int, double> lapack;
  lapack.GGEV('N', 'N', mat.numCols(), mat.values(), mat.stride(), bmat.values(), bmat.stride(),
      reigenvalues.values(), ieigenvalues.values(), beta.values(), nullptr, 1, nullptr, 1,
      work.data(), lwork, &info);

  // postprocess the actual eigenvalues
  for (int i = 0; i < reigenvalues.numCols(); ++i)
  {
    if (beta(i) == 0.0)
      FOUR_C_THROW(
          "The BETA factor is equal to zero! The eigenvalues can not be"
          " computed.");
    reigenvalues(i) /= beta(i);
    ieigenvalues(i) /= beta(i);
  }

  if (info) FOUR_C_THROW("GGEV failed! (info = %d)", info);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::LinearSystem::call_geev(Core::LinAlg::SerialDenseMatrix& mat,
    Core::LinAlg::SerialDenseVector& reigenvalues,
    Core::LinAlg::SerialDenseVector& ieigenvalues) const
{
  if (mat.numCols() != mat.numRows())
    FOUR_C_THROW("A N-by-N real non-symmetric matrix is expected by GEEV!");

  int info = 0;
  int lwork = 3.0 * mat.numCols();
  std::vector<double> work(lwork);

  Teuchos::LAPACK<int, double> lapack;
  lapack.GEEV('N', 'N', mat.numCols(), mat.values(), mat.stride(), reigenvalues.values(),
      ieigenvalues.values(), nullptr, 1, nullptr, 1, work.data(), lwork, &info);

  if (info) FOUR_C_THROW("GEEV failed!");
}

FOUR_C_NAMESPACE_CLOSE