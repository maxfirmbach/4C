/*-----------------------------------------------------------------------------------------------*/
/*! \file

\brief input parameters related to VTP output at runtime for the structural (time) integration

\level 3

*/
/*-----------------------------------------------------------------------------------------------*/

#include "baci_structure_new_timint_basedataio_runtime_vtp_output.H"

#include "baci_beam3_discretization_runtime_vtu_output_params.H"
#include "baci_inpar_parameterlist_utils.H"
#include "baci_utils_exceptions.H"

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void STR::TIMINT::ParamsRuntimeVtpOutput::Init(
    const Teuchos::ParameterList& IO_vtp_structure_paramslist)
{
  // We have to call Setup() after Init()
  issetup_ = false;

  // initialize the parameter values
  output_data_format_ =
      DRT::INPUT::IntegralValue<INPAR::IO_RUNTIME_VTP_STRUCTURE::OutputDataFormat>(
          IO_vtp_structure_paramslist, "OUTPUT_DATA_FORMAT");

  output_interval_steps_ = IO_vtp_structure_paramslist.get<int>("INTERVAL_STEPS");

  output_step_offset_ = IO_vtp_structure_paramslist.get<int>("STEP_OFFSET");

  output_every_iteration_ =
      (bool)DRT::INPUT::IntegralValue<int>(IO_vtp_structure_paramslist, "EVERY_ITERATION");

  /*  output_displacement_state_ =
        (bool) DRT::INPUT::IntegralValue<int>(IO_vtp_structure_paramslist, "DISPLACEMENT");*/

  if (output_every_iteration_) dserror("not implemented yet!");

  output_owner_ = (bool)DRT::INPUT::IntegralValue<int>(IO_vtp_structure_paramslist, "OWNER");

  output_orientationandlength_ =
      (bool)DRT::INPUT::IntegralValue<int>(IO_vtp_structure_paramslist, "ORIENTATIONANDLENGTH");

  output_numberofbonds_ =
      (bool)DRT::INPUT::IntegralValue<int>(IO_vtp_structure_paramslist, "NUMBEROFBONDS");

  output_linkingforce_ =
      (bool)DRT::INPUT::IntegralValue<int>(IO_vtp_structure_paramslist, "LINKINGFORCE");


  isinit_ = true;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void STR::TIMINT::ParamsRuntimeVtpOutput::Setup()
{
  if (not IsInit()) dserror("Init() has not been called, yet!");

  // Nothing to do here at the moment

  issetup_ = true;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void STR::TIMINT::ParamsRuntimeVtpOutput::CheckInitSetup() const
{
  if (not IsInit() or not IsSetup()) dserror("Call Init() and Setup() first!");
}