/*-----------------------------------------------------------*/
/*!
\file str_model_evaluator_data.cpp

\brief Concrete implementation of the structural and all related
       parameter interfaces.

\maintainer Michael Hiermeier

\date Mar 24, 2016

\level 3

*/
/*-----------------------------------------------------------*/

#include "str_model_evaluator_data.H"
#include "str_timint_implicit.H"
#include "str_nln_solver_utils.H"
#include "str_nln_solver_nox.H"

#include "../solver_nonlin_nox/nox_nln_aux.H"
#include "../solver_nonlin_nox/nox_nln_statustest_normf.H"
#include "../solver_nonlin_nox/nox_nln_statustest_normwrms.H"
#include "../solver_nonlin_nox/nox_nln_statustest_normupdate.H"

#include <Epetra_Comm.h>

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
STR::MODELEVALUATOR::Data::Data()
    : isinit_(false),
      issetup_(false),
      isntmaps_filled_(false),
      ele_action_(DRT::ELEMENTS::none),
      ele_eval_error_flag_(STR::ELEMENTS::ele_error_none),
      is_tolerate_errors_(false),
      total_time_(-1.0),
      delta_time_(-1.0),
      step_length_(-1.0),
      is_default_step_(false),
      timintfactor_disp_(-1.0),
      timintfactor_vel_(-1.0),
      stressdata_ptr_(Teuchos::null),
      straindata_ptr_(Teuchos::null),
      plastic_straindata_ptr_(Teuchos::null),
      sdyn_ptr_(Teuchos::null),
      io_ptr_(Teuchos::null),
      gstate_ptr_(Teuchos::null),
      timint_ptr_(Teuchos::null),
      comm_ptr_(Teuchos::null)
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::Data::Init(
    const Teuchos::RCP<const STR::TIMINT::Base>& timint_ptr)
{
  sdyn_ptr_ = timint_ptr->GetDataSDynPtr();
  io_ptr_ = timint_ptr->GetDataIOPtr();
  gstate_ptr_ = timint_ptr->GetDataGlobalStatePtr();
  timint_ptr_ = timint_ptr;
  comm_ptr_ = timint_ptr->GetDataGlobalState().GetCommPtr();
  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::Data::Setup()
{
  CheckInit();

  const std::set<enum INPAR::STR::ModelType>& mt = sdyn_ptr_->GetModelTypes();
  std::set<enum INPAR::STR::ModelType>::const_iterator it;
  // setup model type specific data containers
  for (it=mt.begin();it!=mt.end();++it)
  {
    switch (*it)
    {
      case INPAR::STR::model_contact:
      {
        contact_data_ptr_ = Teuchos::rcp(new ContactData());
        contact_data_ptr_->Init(Teuchos::rcp(this,false));
        contact_data_ptr_->Setup();
        break;
      }
      default:
      {
        // nothing to do
        break;
      }
    }
  }

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::Data::FillNormTypeMaps()
{
  // we have to do all this only once...
  if (isntmaps_filled_)
    return;

  std::set<enum NOX::NLN::StatusTest::QuantityType> qtypes;
  STR::NLN::SOLVER::CreateQuantityTypes(qtypes,*sdyn_ptr_);

  // --- check if the nox nln solver is active ---------------------------------
  bool isnox = false;
  Teuchos::RCP<const STR::NLN::SOLVER::Nox> nox_nln_ptr = Teuchos::null;
  Teuchos::RCP<const STR::TIMINT::Implicit> timint_impl_ptr =
      Teuchos::rcp_dynamic_cast<const STR::TIMINT::Implicit>(timint_ptr_);
  if (not timint_impl_ptr.is_null())
  {
    nox_nln_ptr =
        Teuchos::rcp_dynamic_cast<const STR::NLN::SOLVER::Nox>(
            timint_impl_ptr->GetNlnSolverPtr());
    if (not nox_nln_ptr.is_null())
      isnox = true;
  }

  // --- get the normtypes for the different quantities -------------------------
  std::set<enum NOX::NLN::StatusTest::QuantityType>::const_iterator qiter;
  if (isnox)
  {
    const NOX::StatusTest::Generic& ostatus = nox_nln_ptr->GetOStatusTest();
    for (qiter=qtypes.begin();qiter!=qtypes.end();++qiter)
    {
      // fill the normtype_force map
      int inormtype =
          NOX::NLN::AUX::GetNormType<NOX::NLN::StatusTest::NormF>(ostatus,*qiter);
      if (inormtype != -100)
        normtype_force_[*qiter] = static_cast<NOX::Abstract::Vector::NormType>(inormtype);
      // fill the normtype_update map
      inormtype =
          NOX::NLN::AUX::GetNormType<NOX::NLN::StatusTest::NormUpdate>(ostatus,*qiter);
      if (inormtype != -100)
        normtype_update_[*qiter] = static_cast<NOX::Abstract::Vector::NormType>(inormtype);

      // check for the root mean square test (wrms)
      if (NOX::NLN::AUX::IsQuantity<NOX::NLN::StatusTest::NormWRMS>(ostatus,*qiter))
      {
        /* get the absolute and relative tolerances, since we have to use them
         * during the summation. */
        double atol = NOX::NLN::AUX::GetNormWRMSClassVariable(ostatus,*qiter,"ATOL");
        if (atol < 0.0)
          dserror("The absolute wrms tolerance of the quantity %s is missing.",
              NOX::NLN::StatusTest::QuantityType2String(*qiter).c_str());
        else
          atol_wrms_[*qiter] = atol;
        double rtol = NOX::NLN::AUX::GetNormWRMSClassVariable(ostatus,*qiter,"RTOL");
        if (rtol < 0.0)
          dserror("The relative wrms tolerance of the quantity %s is missing.",
              NOX::NLN::StatusTest::QuantityType2String(*qiter).c_str());
        else
          rtol_wrms_[*qiter] = rtol;
      }
    } // loop over all quantity types
  } // if (isnox)
  else
  {
    for (qiter=qtypes.begin();qiter!=qtypes.end();++qiter)
    {
      normtype_force_[*qiter] = sdyn_ptr_->GetNoxNormType();
      normtype_update_[*qiter] = sdyn_ptr_->GetNoxNormType();
    }
  }
  // do it only once!
  isntmaps_filled_ = true;
  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::Data::GetUpdateNormType(
    const enum NOX::NLN::StatusTest::QuantityType& qtype,
    enum NOX::Abstract::Vector::NormType& normtype)
{
  FillNormTypeMaps();
  // check if there is a normtype for the corresponding quantity type
  std::map<enum NOX::NLN::StatusTest::QuantityType,
      enum NOX::Abstract::Vector::NormType>::const_iterator miter;
  miter = normtype_update_.find(qtype);
  if (miter==normtype_update_.end())
    return false;
  // we found the corresponding type
  normtype = miter->second;
  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::Data::GetWRMSTolerances(
    const enum NOX::NLN::StatusTest::QuantityType& qtype,
    double& atol, double& rtol)
{
  FillNormTypeMaps();
  // check if there is a wrms test for the corresponding quantity type
  std::map<enum NOX::NLN::StatusTest::QuantityType,double>::const_iterator iter;
  iter = atol_wrms_.find(qtype);
  if (iter==atol_wrms_.end())
    return false;
  // we found the corrsponding type
  atol = iter->second;
  rtol = rtol_wrms_.at(qtype);
  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::Data::SumIntoMyUpdateNorm(
    const enum NOX::NLN::StatusTest::QuantityType& qtype,
    const int& numentries,
    const double* my_update_values,
    const double* my_new_sol_values,
    const double& step_length,
    const int& owner)
{
  if (owner!=comm_ptr_->MyPID())
    return;
  // --- standard update norms
  enum NOX::Abstract::Vector::NormType normtype =
      NOX::Abstract::Vector::TwoNorm;
  if (GetUpdateNormType(qtype,normtype))
    SumIntoMyNorm(numentries,my_update_values,normtype,
        step_length,my_update_norm_[qtype]);

  // --- weighted root mean square norms
  double atol = 0.0;
  double rtol = 0.0;
  if (GetWRMSTolerances(qtype,atol,rtol))
  {
    SumIntoMyRelativeMeanSquare(atol,rtol,step_length,numentries,
        my_update_values,my_new_sol_values,my_rms_norm_[qtype]);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::Data::SumIntoMyPreviousSolNorm(
    const enum NOX::NLN::StatusTest::QuantityType& qtype,
    const int& numentries,
    const double* my_old_sol_values,
    const int& owner)
{
  if (owner!=comm_ptr_->MyPID())
    return;

  enum NOX::Abstract::Vector::NormType normtype =
      NOX::Abstract::Vector::TwoNorm;
  if (not GetUpdateNormType(qtype,normtype))
    return;

  SumIntoMyNorm(numentries,my_old_sol_values,normtype,1.0,my_prev_sol_norm_[qtype]);
  // update the dof counter
  my_dof_number_[qtype] += numentries;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::Data::SumIntoMyRelativeMeanSquare(
    const double& atol,
    const double& rtol,
    const double& step_length,
    const int& numentries,
    const double* my_update_values,
    const double* my_new_sol_values,
    double& my_rms) const
{
  for (int i=0;i<numentries;++i)
  {
    // calculate v_i = x_{i}^{k-1} = x_{i}^{k} - sl* \Delta x_{i}^{k}
    double dx_i = step_length*my_update_values[i];
    double v_i = my_new_sol_values[i]- dx_i;
    // calculate the relative mean square sum:
    // my_rms_norm = \sum_{i} [(x_i^{k}-x_{i}^{k-1}) / (RTOL * |x_{i}^{k-1}| + ATOL)]^{2}
    v_i = dx_i / (rtol * std::abs(v_i) + atol);
    my_rms += v_i*v_i;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::Data::SumIntoMyNorm(
    const int& numentries,
    const double* my_values,
    const enum NOX::Abstract::Vector::NormType& normtype,
    const double& step_length,
    double& my_norm) const
{
  switch (normtype)
  {
    case NOX::Abstract::Vector::OneNorm:
    {
      for (int i=0;i<numentries;++i)
        my_norm +=std::abs(my_values[i]*step_length);
      break;
    }
    case NOX::Abstract::Vector::TwoNorm:
    {
      for (int i=0;i<numentries;++i)
        my_norm +=(my_values[i]*my_values[i])*(step_length*step_length);
      break;
    }
    case NOX::Abstract::Vector::MaxNorm:
    {
      for (int i=0;i<numentries;++i)
        my_norm = std::max(my_norm,std::abs(my_values[i]*step_length));
      break;
    }
  }
  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::Data::ResetMyNorms(
    const bool& isdefaultstep)
{
  CheckInitSetup();
  std::map<enum NOX::NLN::StatusTest::QuantityType,double>::iterator it;
  for (it=my_update_norm_.begin();it!=my_update_norm_.end();++it)
    it->second = 0.0;
  for (it=my_rms_norm_.begin();it!=my_rms_norm_.end();++it)
      it->second = 0.0;

  if (isdefaultstep)
  {
    // reset the map holding the previous solution norms of the last converged
    // Newton step
    for (it=my_prev_sol_norm_.begin();it!=my_prev_sol_norm_.end();++it)
      it->second = 0.0;
    // reset the dof number
    std::map<enum NOX::NLN::StatusTest::QuantityType,std::size_t>::iterator dit;
    for (dit=my_dof_number_.begin();
        dit!=my_dof_number_.end();++dit)
      dit->second = 0;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::Data::IsEleEvalError() const
{
  CheckInitSetup();
  return (ele_eval_error_flag_!=STR::ELEMENTS::ele_error_none);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
enum INPAR::STR::DampKind STR::MODELEVALUATOR::Data::GetDampingType() const
{
  CheckInitSetup();
  return sdyn_ptr_->GetDampingType();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<std::vector<char> >& STR::MODELEVALUATOR::Data::MutableStressDataPtr()
{
  CheckInitSetup();
  return stressdata_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const std::vector<char>& STR::MODELEVALUATOR::Data::StressData() const
{
  if (stressdata_ptr_.is_null())
    dserror("Undefined reference to the stress data!");
  return *stressdata_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<std::vector<char> >& STR::MODELEVALUATOR::Data::MutableStrainDataPtr()
{
  CheckInitSetup();
  return straindata_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const std::vector<char>& STR::MODELEVALUATOR::Data::StrainData() const
{
  if (straindata_ptr_.is_null())
    dserror("Undefined reference to the strain data!");
  return *straindata_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<std::vector<char> >& STR::MODELEVALUATOR::Data::MutablePlasticStrainDataPtr()
{
  CheckInitSetup();
  return plastic_straindata_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const std::vector<char>& STR::MODELEVALUATOR::Data::PlasticStrainData() const
{
  if (plastic_straindata_ptr_.is_null())
    dserror("Undefined reference to the plastic strain data!");
  return *plastic_straindata_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
enum INPAR::STR::StressType STR::MODELEVALUATOR::Data::GetStressOutputType() const
{
  CheckInitSetup();
  return io_ptr_->GetStressOutputType();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
enum INPAR::STR::StrainType STR::MODELEVALUATOR::Data::GetStrainOutputType() const
{
  CheckInitSetup();
  return io_ptr_->GetStrainOutputType();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
enum INPAR::STR::StrainType STR::MODELEVALUATOR::Data::GetPlasticStrainOutputType() const
{
  CheckInitSetup();
  return io_ptr_->GetPlasticStrainOutputType();
}