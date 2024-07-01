/*-----------------------------------------------------------*/
/*! \file

\brief Utility methods for the structural time integration.


\level 3

*/
/*-----------------------------------------------------------*/

#include "4C_structure_new_utils.hpp"

#include "4C_constraint_lagpenconstraint_noxinterface.hpp"
#include "4C_constraint_manager.hpp"
#include "4C_contact_abstract_strategy.hpp"
#include "4C_contact_meshtying_abstract_strategy.hpp"
#include "4C_contact_meshtying_noxinterface.hpp"
#include "4C_contact_noxinterface.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_solver_nonlin_nox_aux.hpp"
#include "4C_solver_nonlin_nox_constraint_interface_preconditioner.hpp"
#include "4C_solver_nonlin_nox_constraint_interface_required.hpp"
#include "4C_structure_new_impl_genalpha.hpp"
#include "4C_structure_new_integrator.hpp"
#include "4C_structure_new_model_evaluator_contact.hpp"
#include "4C_structure_new_model_evaluator_lagpenconstraint.hpp"
#include "4C_structure_new_model_evaluator_meshtying.hpp"
#include "4C_structure_new_nln_linearsystem_scaling.hpp"
#include "4C_structure_new_timint_basedatasdyn.hpp"
#include "4C_utils_exceptions.hpp"

#include <Epetra_Vector.h>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::LinSystem::ConditionNumber STR::Nln::Convert2NoxConditionNumberType(
    const Inpar::STR::ConditionNumber stype)
{
  switch (stype)
  {
    case Inpar::STR::ConditionNumber::max_min_ev_ratio:
      return NOX::Nln::LinSystem::ConditionNumber::max_min_ev_ratio;
    case Inpar::STR::ConditionNumber::one_norm:
      return NOX::Nln::LinSystem::ConditionNumber::one_norm;
    case Inpar::STR::ConditionNumber::inf_norm:
      return NOX::Nln::LinSystem::ConditionNumber::inf_norm;
    default:
      FOUR_C_THROW("No known conversion.");
      exit(EXIT_FAILURE);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
enum ::NOX::Abstract::Vector::NormType STR::Nln::Convert2NoxNormType(
    const enum Inpar::STR::VectorNorm& normtype)
{
  enum ::NOX::Abstract::Vector::NormType nox_normtype = ::NOX::Abstract::Vector::TwoNorm;

  switch (normtype)
  {
    case Inpar::STR::norm_l2:
      nox_normtype = ::NOX::Abstract::Vector::TwoNorm;
      break;
    case Inpar::STR::norm_l1:
      nox_normtype = ::NOX::Abstract::Vector::OneNorm;
      break;
    case Inpar::STR::norm_inf:
      nox_normtype = ::NOX::Abstract::Vector::MaxNorm;
      break;
    case Inpar::STR::norm_rms:
    case Inpar::STR::norm_vague:
    default:
      FOUR_C_THROW("Unknown conversion for the given vector norm type: \" %s \"!",
          Inpar::STR::VectorNormString(normtype).c_str());
      break;
  }  // switch case normtype

  return nox_normtype;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::Nln::ConvertModelType2SolType(std::vector<enum NOX::Nln::SolutionType>& soltypes,
    std::map<enum NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& slinsolvers,
    const std::set<enum Inpar::STR::ModelType>& modeltypes,
    const std::map<enum Inpar::STR::ModelType, Teuchos::RCP<Core::LinAlg::Solver>>& mlinsolvers)
{
  // initialize the vector and/or force the length to zero
  if (soltypes.size() > 0)
  {
    soltypes.clear();
    slinsolvers.clear();
  }

  // pre-set the vector size
  soltypes.reserve(modeltypes.size());

  // The strings of the different enums have to fit!
  std::set<enum Inpar::STR::ModelType>::const_iterator mt_iter;
  for (mt_iter = modeltypes.begin(); mt_iter != modeltypes.end(); ++mt_iter)
  {
    const enum NOX::Nln::SolutionType soltype = ConvertModelType2SolType(*mt_iter);

    soltypes.push_back(soltype);
    // copy the linsolver pointers into the new map
    if (mlinsolvers.find(*mt_iter) != mlinsolvers.end())
      slinsolvers[soltype] = mlinsolvers.at(*mt_iter);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
enum NOX::Nln::SolutionType STR::Nln::ConvertModelType2SolType(
    const enum Inpar::STR::ModelType& modeltype, const bool& do_check)
{
  enum NOX::Nln::SolutionType soltype = NOX::Nln::sol_unknown;
  switch (modeltype)
  {
    case Inpar::STR::model_structure:
    case Inpar::STR::model_springdashpot:
    case Inpar::STR::model_basic_coupling:
    case Inpar::STR::model_monolithic_coupling:
    case Inpar::STR::model_partitioned_coupling:
    case Inpar::STR::model_beam_interaction_old:
    case Inpar::STR::model_browniandyn:
    case Inpar::STR::model_beaminteraction:
    case Inpar::STR::model_constraints:
      soltype = NOX::Nln::sol_structure;
      break;
    case Inpar::STR::model_contact:
      soltype = NOX::Nln::sol_contact;
      break;
    case Inpar::STR::model_meshtying:
      soltype = NOX::Nln::sol_meshtying;
      break;
    case Inpar::STR::model_cardiovascular0d:
      soltype = NOX::Nln::sol_cardiovascular0d;
      break;
    case Inpar::STR::model_lag_pen_constraint:
      soltype = NOX::Nln::sol_lag_pen_constraint;
      break;
    default:
      // check if the corresponding enum could be found.
      if (do_check)
        FOUR_C_THROW(
            "The corresponding solution-type was not found. "
            "Given string: %s",
            Inpar::STR::ModelTypeString(modeltype).c_str());
      break;
  }

  return soltype;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
enum Inpar::STR::ModelType STR::Nln::ConvertSolType2ModelType(
    const enum NOX::Nln::SolutionType& soltype, const bool& do_check)
{
  enum Inpar::STR::ModelType modeltype = Inpar::STR::model_vague;
  switch (soltype)
  {
    case NOX::Nln::sol_structure:
      modeltype = Inpar::STR::model_structure;
      break;
    case NOX::Nln::sol_contact:
      modeltype = Inpar::STR::model_contact;
      break;
    case NOX::Nln::sol_meshtying:
      modeltype = Inpar::STR::model_meshtying;
      break;
    case NOX::Nln::sol_cardiovascular0d:
      modeltype = Inpar::STR::model_cardiovascular0d;
      break;
    case NOX::Nln::sol_lag_pen_constraint:
      modeltype = Inpar::STR::model_lag_pen_constraint;
      break;
    default:
      // check if the corresponding enum could be found.
      if (do_check)
        FOUR_C_THROW(
            "The corresponding model-type was not found. "
            "Given string: %s",
            NOX::Nln::SolutionType2String(soltype).c_str());
      break;
  }

  return modeltype;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
enum Inpar::STR::ModelType STR::Nln::ConvertQuantityType2ModelType(
    const enum NOX::Nln::StatusTest::QuantityType& qtype, const bool& do_check)
{
  const NOX::Nln::SolutionType st = NOX::Nln::Aux::ConvertQuantityType2SolutionType(qtype);
  return ConvertSolType2ModelType(st, do_check);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
enum Inpar::STR::EleTech STR::Nln::ConvertQuantityType2EleTech(
    const enum NOX::Nln::StatusTest::QuantityType& qtype)
{
  enum Inpar::STR::EleTech eletech = Inpar::STR::EleTech::pressure;
  switch (qtype)
  {
    case NOX::Nln::StatusTest::quantity_pressure:
      eletech = Inpar::STR::EleTech::pressure;
      break;
    case NOX::Nln::StatusTest::quantity_plasticity:
      eletech = Inpar::STR::EleTech::plasticity;
      break;
    case NOX::Nln::StatusTest::quantity_eas:
      eletech = Inpar::STR::EleTech::eas;
      break;
    default:
      FOUR_C_THROW("Cannot convert QuantityType %s to EleTech.",
          NOX::Nln::StatusTest::QuantityType2String(qtype).c_str());
      break;
  }

  return eletech;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
enum NOX::Nln::OptimizationProblemType STR::Nln::OptimizationType(
    const std::vector<enum NOX::Nln::SolutionType>& soltypes)
{
  enum NOX::Nln::OptimizationProblemType opttype = NOX::Nln::opt_unconstrained;
  std::vector<enum NOX::Nln::SolutionType>::const_iterator st_iter;

  for (st_iter = soltypes.begin(); st_iter != soltypes.end(); ++st_iter)
  {
    switch (*st_iter)
    {
      // -----------------------------------
      // Inequality constraint
      // active set decision necessary
      // saddle point structure or condensed
      // -----------------------------------
      case NOX::Nln::sol_contact:
        return NOX::Nln::opt_inequality_constrained;
        break;
      // -----------------------------------
      // Equality constraint
      // no active set decision necessary
      // saddle point structure or condensed
      // -----------------------------------
      case NOX::Nln::sol_meshtying:
      case NOX::Nln::sol_lag_pen_constraint:
        opttype = NOX::Nln::opt_equality_constrained;
        break;
      // -----------------------------------
      // Unconstrained problem
      // pure structural problem
      // no saddle point structure
      // -----------------------------------
      case NOX::Nln::sol_structure:
      case NOX::Nln::sol_cardiovascular0d:
      default:
        break;
    }
  }

  return opttype;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::Nln::CreateConstraintInterfaces(NOX::Nln::CONSTRAINT::ReqInterfaceMap& iconstr,
    STR::Integrator& integrator, const std::vector<enum NOX::Nln::SolutionType>& soltypes)
{
  if (iconstr.size() > 0) iconstr.clear();

  std::vector<enum NOX::Nln::SolutionType>::const_iterator st_iter;
  for (st_iter = soltypes.begin(); st_iter != soltypes.end(); ++st_iter)
  {
    switch (*st_iter)
    {
      case NOX::Nln::sol_contact:
      {
        STR::MODELEVALUATOR::Generic& model = integrator.evaluator(Inpar::STR::model_contact);
        STR::MODELEVALUATOR::Contact& contact_model =
            dynamic_cast<STR::MODELEVALUATOR::Contact&>(model);
        iconstr[NOX::Nln::sol_contact] = contact_model.strategy_ptr()->nox_interface_ptr();
        break;
      }
      case NOX::Nln::sol_meshtying:
      {
        STR::MODELEVALUATOR::Generic& model = integrator.evaluator(Inpar::STR::model_meshtying);
        STR::MODELEVALUATOR::Meshtying& mt_model =
            dynamic_cast<STR::MODELEVALUATOR::Meshtying&>(model);
        iconstr[NOX::Nln::sol_meshtying] = mt_model.StrategyPtr()->nox_interface_ptr();
        break;
      }
      case NOX::Nln::sol_lag_pen_constraint:
      {
        STR::MODELEVALUATOR::Generic& model =
            integrator.evaluator(Inpar::STR::model_lag_pen_constraint);
        STR::MODELEVALUATOR::LagPenConstraint& lagpenconstraint_model =
            dynamic_cast<STR::MODELEVALUATOR::LagPenConstraint&>(model);
        iconstr[NOX::Nln::sol_lag_pen_constraint] = lagpenconstraint_model.nox_interface_ptr();
        break;
      }
      default:
        break;
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::Nln::CreateConstraintPreconditioner(NOX::Nln::CONSTRAINT::PrecInterfaceMap& iconstr_prec,
    STR::Integrator& integrator, const std::vector<enum NOX::Nln::SolutionType>& soltypes)
{
  if (iconstr_prec.size() > 0) iconstr_prec.clear();

  std::vector<enum NOX::Nln::SolutionType>::const_iterator st_iter;
  for (st_iter = soltypes.begin(); st_iter != soltypes.end(); ++st_iter)
  {
    switch (*st_iter)
    {
      case NOX::Nln::sol_contact:
      {
        STR::MODELEVALUATOR::Generic& model = integrator.evaluator(Inpar::STR::model_contact);
        STR::MODELEVALUATOR::Contact& contact_model =
            dynamic_cast<STR::MODELEVALUATOR::Contact&>(model);
        /* Actually we use the underlying Mortar::StrategyBase as Preconditioner
         * interface. Nevertheless, the implementations can differ for the
         * contact/meshtying cases. */
        iconstr_prec[NOX::Nln::sol_contact] = contact_model.strategy_ptr();
        break;
      }
      case NOX::Nln::sol_meshtying:
      {
        STR::MODELEVALUATOR::Generic& model = integrator.evaluator(Inpar::STR::model_meshtying);
        STR::MODELEVALUATOR::Meshtying& mt_model =
            dynamic_cast<STR::MODELEVALUATOR::Meshtying&>(model);
        iconstr_prec[NOX::Nln::sol_meshtying] = mt_model.StrategyPtr();
        break;
      }
      case NOX::Nln::sol_lag_pen_constraint:
      {
        STR::MODELEVALUATOR::Generic& model =
            integrator.evaluator(Inpar::STR::model_lag_pen_constraint);
        STR::MODELEVALUATOR::LagPenConstraint& lagpenconstraint_model =
            dynamic_cast<STR::MODELEVALUATOR::LagPenConstraint&>(model);
        iconstr_prec[NOX::Nln::sol_lag_pen_constraint] =
            lagpenconstraint_model.NoxInterfacePrecPtr();
        break;
      }
      default:
        // do nothing
        break;
    }  // switch (*st_iter)
  }
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::Nln::CreateScaling(Teuchos::RCP<::NOX::Epetra::Scaling>& iscale,
    const STR::TimeInt::BaseDataSDyn& DataSDyn, STR::TimeInt::BaseDataGlobalState& GState)
{
  if (DataSDyn.get_stc_algo_type() != Inpar::STR::stc_none)
    iscale = Teuchos::rcp(new STR::Nln::LinSystem::StcScaling(DataSDyn, GState));
}

void STR::ComputeGeneralizedAlphaParameters(STR::IMPLICIT::GenAlpha::Coefficients& coeffs)
{
  // ------ check if the user provide RHO_INF and any other parameters at the same time
  if (((coeffs.beta_ != -1.0) or (coeffs.gamma_ != -1.0) or (coeffs.alpham_ != -1.0) or
          (coeffs.alphaf_ != -1.0)) and
      (coeffs.rhoinf_ != -1.0))
    FOUR_C_THROW(
        "There are two ways to provide GenAlpha parameters:\n"
        "- You can choose to only provide RHO_INF as the spectral radius."
        "In this way, no other parameters are allowed.\n"
        "- You may also specify all the four parameters"
        "In this way, you MUST set RHO_INF as -1.0");

  // ------ rho_inf set to -1.0--> use the four parameters provided by the user -----------------
  else if (coeffs.rhoinf_ == -1.0)
  {
    if ((coeffs.alpham_ < 0.0) or (coeffs.alpham_ >= 1.0))
      FOUR_C_THROW("alpham out of range [0.0,1.0)");
    if ((coeffs.alphaf_ < 0.0) or (coeffs.alphaf_ >= 1.0))
      FOUR_C_THROW("alphaf out of range [0.0,1.0)");
    if ((coeffs.beta_ <= 0.0) or (coeffs.beta_ > 0.5)) FOUR_C_THROW("beta out of range (0.0,0.5]");
    if ((coeffs.gamma_ <= 0.0) or (coeffs.gamma_ > 1.0))
      FOUR_C_THROW("gamma out of range (0.0,1.0]");
  }

  // ------ rho_inf out of [0,1]--> report error
  else if ((coeffs.rhoinf_ < 0.0) or (coeffs.rhoinf_ > 1.0))
    FOUR_C_THROW("rho_inf out of range [0.0,1.0]");

  // ------ rho_inf specified --> calculate optimal parameters -----------------
  else
  {
    coeffs.alpham_ = (2.0 * coeffs.rhoinf_ - 1.0) / (coeffs.rhoinf_ + 1.0);
    coeffs.alphaf_ = coeffs.rhoinf_ / (coeffs.rhoinf_ + 1.0);
    coeffs.beta_ =
        0.25 * (1.0 - coeffs.alpham_ + coeffs.alphaf_) * (1.0 - coeffs.alpham_ + coeffs.alphaf_);
    coeffs.gamma_ = 0.5 - coeffs.alpham_ + coeffs.alphaf_;
  };
}

FOUR_C_NAMESPACE_CLOSE