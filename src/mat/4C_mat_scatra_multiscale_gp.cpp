/*----------------------------------------------------------------------*/
/*! \file
\brief submaterial associated with macro-scale Gauss point in multi-scale simulations of scalar
transport problems

\level 2

 */
/*----------------------------------------------------------------------*/
#include "4C_mat_scatra_multiscale_gp.hpp"

#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_scatra_timint_ost.hpp"
#include "4C_utils_parameter_list.hpp"

#include <filesystem>

FOUR_C_NAMESPACE_OPEN

// instantiate static maps
std::map<int, Teuchos::RCP<ScaTra::TimIntOneStepTheta>>
    Mat::ScatraMultiScaleGP::microdisnum_microtimint_map_;
std::map<int, int> Mat::ScatraMultiScaleGP::microdisnum_nummacrogp_map_;

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::ScatraMultiScaleGP::ScatraMultiScaleGP(
    const int ele_id, const int gp_id, const int microdisnum, const bool is_ale)
    : gp_id_(gp_id),
      ele_id_(ele_id),
      eleowner_(Global::Problem::Instance()->GetDis("scatra")->ElementRowMap()->MyGID(ele_id)),
      microdisnum_(microdisnum),
      step_(0),
      phin_(Teuchos::null),
      phinp_(Teuchos::null),
      phidtn_(Teuchos::null),
      phidtnp_(Teuchos::null),
      hist_(Teuchos::null),
      micro_output_(Teuchos::null),
      restartname_(""),
      det_fn_(1.0),
      det_fnp_(1.0),
      ddet_fdtn_(0.0),
      ddet_fdtnp_(0.0),
      is_ale_(is_ale)

{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::ScatraMultiScaleGP::~ScatraMultiScaleGP()
{
  // decrement number of macro-scale Gauss points associated with micro-scale time integrator
  --microdisnum_nummacrogp_map_[microdisnum_];

  // once all macro-scale Gauss point submaterials are removed, destruct micro-scale time integrator
  if (microdisnum_nummacrogp_map_[microdisnum_] == 0)
    microdisnum_microtimint_map_[microdisnum_] = Teuchos::null;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMultiScaleGP::init()
{
  // extract micro-scale problem
  Global::Problem* microproblem = Global::Problem::Instance(microdisnum_);

  // extract micro-scale discretization
  std::stringstream microdisname;
  microdisname << "scatra_multiscale_" << microdisnum_;
  Teuchos::RCP<Core::FE::Discretization> microdis = microproblem->GetDis(microdisname.str());

  // instantiate and initialize micro-scale state vectors
  phin_ = Core::LinAlg::CreateVector(*microdis->dof_row_map(), true);
  phinp_ = Core::LinAlg::CreateVector(*microdis->dof_row_map(), true);
  phidtn_ = Core::LinAlg::CreateVector(*microdis->dof_row_map(), true);
  phidtnp_ = Core::LinAlg::CreateVector(*microdis->dof_row_map(), true);
  hist_ = Core::LinAlg::CreateVector(*microdis->dof_row_map(), true);

  // set up micro-scale time integrator for micro-scale problem if not already done
  if (microdisnum_microtimint_map_.find(microdisnum_) == microdisnum_microtimint_map_.end() or
      microdisnum_microtimint_map_[microdisnum_] == Teuchos::null)
  {
    // extract macro-scale parameter list
    const Teuchos::ParameterList& sdyn_macro =
        Global::Problem::Instance()->scalar_transport_dynamic_params();

    // extract micro-scale parameter list and create deep copy
    Teuchos::RCP<Teuchos::ParameterList> sdyn_micro = Teuchos::rcp(new Teuchos::ParameterList(
        Global::Problem::Instance(microdisnum_)->scalar_transport_dynamic_params()));

    // preliminary safety check
    if (Global::Problem::Instance(microdisnum_)->NDim() != 1)
    {
      FOUR_C_THROW(
          "Must have one-dimensional micro scale in multi-scale simulations of scalar transport "
          "problems!");
    }
    if (Core::UTILS::IntegralValue<Inpar::ScaTra::TimeIntegrationScheme>(
            sdyn_macro, "TIMEINTEGR") != Inpar::ScaTra::timeint_one_step_theta or
        Core::UTILS::IntegralValue<Inpar::ScaTra::TimeIntegrationScheme>(
            *sdyn_micro, "TIMEINTEGR") != Inpar::ScaTra::timeint_one_step_theta)
    {
      FOUR_C_THROW(
          "Multi-scale calculations for scalar transport only implemented for one-step-theta time "
          "integration scheme!");
    }
    if (Core::UTILS::IntegralValue<bool>(sdyn_macro, "SKIPINITDER") !=
        Core::UTILS::IntegralValue<bool>(*sdyn_micro, "SKIPINITDER"))
      FOUR_C_THROW("Flag SKIPINITDER in input file must be equal on macro and micro scales!");
    if (sdyn_macro.get<double>("TIMESTEP") != sdyn_micro->get<double>("TIMESTEP"))
      FOUR_C_THROW("Must have identical time step size on macro and micro scales!");
    if (sdyn_macro.get<int>("NUMSTEP") != sdyn_micro->get<int>("NUMSTEP"))
      FOUR_C_THROW("Must have identical number of time steps on macro and micro scales!");
    if (sdyn_macro.get<double>("THETA") != sdyn_micro->get<double>("THETA"))
      FOUR_C_THROW(
          "Must have identical one-step-theta time integration factor on macro and micro scales!");
    if (microdis->NumGlobalElements() == 0)
      FOUR_C_THROW("No elements in TRANSPORT ELEMENTS section of micro-scale input file!");
    if (microdis->gNode(0)->X()[0] != 0.0)
    {
      FOUR_C_THROW(
          "Micro-scale domain must have one end at coordinate 0 and the other end at a coordinate "
          "> 0!");
    }

    // extract multi-scale coupling conditions from micro-scale discretization
    std::vector<Teuchos::RCP<Core::Conditions::Condition>> conditions;
    microdis->GetCondition("ScatraMultiScaleCoupling", conditions);

    // safety check
    if (conditions.size() == 0)
      FOUR_C_THROW(
          "Couldn't extract multi-scale coupling condition from micro-scale discretization!");

    // loop over all multi-scale coupling conditions
    for (auto& condition : conditions)
    {
      // extract nodal cloud
      const std::vector<int>* const nodeids = condition->GetNodes();
      if (nodeids == nullptr)
        FOUR_C_THROW("Multi-scale coupling condition does not have nodal cloud!");

      // loop over all nodes in nodal cloud
      for (int inode : *nodeids)
      {
        if (microdis->NodeRowMap()->MyGID(inode))
        {
          // extract node from micro-scale discretization
          Core::Nodes::Node* node = microdis->gNode(inode);

          // safety checks
          if (node == nullptr)
          {
            FOUR_C_THROW(
                "Cannot extract node with global ID %d from micro-scale discretization!", inode);
          }
          else if (node->X()[0] <= 0.0)
            FOUR_C_THROW(
                "Multi-scale coupling condition must be enforced on a node with coordinate > 0!");
        }
      }
    }

    // add proxy of velocity related degrees of freedom to scatra discretization
    Teuchos::RCP<Core::DOFSets::DofSetInterface> dofsetaux =
        Teuchos::rcp(new Core::DOFSets::DofSetPredefinedDoFNumber(
            Global::Problem::Instance(microdisnum_)->NDim() + 1, 0, 0, true));
    if (microdis->AddDofSet(dofsetaux) != 1)
      FOUR_C_THROW("Micro-scale discretization has illegal number of dofsets!");

    // finalize discretization
    microdis->fill_complete(true, false, false);

    // get solver number
    const int linsolvernumber = sdyn_micro->get<int>("LINEAR_SOLVER");

    // check solver number
    if (linsolvernumber < 0)
    {
      FOUR_C_THROW(
          "No linear solver defined for scalar field in input file for micro scale! Please set "
          "LINEAR_SOLVER in SCALAR TRANSPORT DYNAMIC to a valid number!");
    }

    // create solver
    Teuchos::RCP<Core::LinAlg::Solver> solver = Teuchos::rcp(new Core::LinAlg::Solver(
        Global::Problem::Instance(microdisnum_)->SolverParams(linsolvernumber), microdis->Comm(),
        Global::Problem::Instance()->solver_params_callback(),
        Core::UTILS::IntegralValue<Core::IO::Verbositylevel>(
            Global::Problem::Instance()->IOParams(), "VERBOSITY")));

    // provide solver with null space information if necessary
    microdis->compute_null_space_if_necessary(solver->Params());

    // supplementary parameter list
    Teuchos::RCP<Teuchos::ParameterList> extraparams = Teuchos::rcp(new Teuchos::ParameterList());
    extraparams->set<bool>("isale", false);
    extraparams->sublist("TURBULENT INFLOW") =
        Global::Problem::Instance(microdisnum_)->FluidDynamicParams().sublist("TURBULENT INFLOW");
    extraparams->sublist("TURBULENCE MODEL") =
        Global::Problem::Instance(microdisnum_)->FluidDynamicParams().sublist("TURBULENCE MODEL");

    // instantiate and initialize micro-scale time integrator
    microdisnum_microtimint_map_[microdisnum_] = Teuchos::rcp(new ScaTra::TimIntOneStepTheta(
        microdis, solver, sdyn_micro, extraparams, Teuchos::null, microdisnum_));
    microdisnum_microtimint_map_[microdisnum_]->init();
    microdisnum_microtimint_map_[microdisnum_]->set_number_of_dof_set_velocity(1);
    microdisnum_microtimint_map_[microdisnum_]->setup();

    // set initial velocity field
    microdisnum_microtimint_map_[microdisnum_]->set_velocity_field();

    // create counter for number of macro-scale Gauss points associated with micro-scale time
    // integrator
    microdisnum_nummacrogp_map_[microdisnum_] = 0;
  }

  // increment counter
  ++microdisnum_nummacrogp_map_[microdisnum_];

  // extract initial state vectors from micro-scale time integrator
  phin_->Scale(1., *microdisnum_microtimint_map_[microdisnum_]->Phin());
  phinp_->Scale(1., *microdisnum_microtimint_map_[microdisnum_]->Phinp());

  // create new result file
  new_result_file();
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMultiScaleGP::prepare_time_step(const std::vector<double>& phinp_macro)
{
  // extract micro-scale time integrator
  const Teuchos::RCP<ScaTra::TimIntOneStepTheta>& microtimint =
      microdisnum_microtimint_map_[microdisnum_];

  // set current state in micro-scale time integrator
  microtimint->set_state(phin_, phinp_, phidtn_, phidtnp_, hist_, micro_output_, phinp_macro, step_,
      Discret::ELEMENTS::ScaTraEleParameterTimInt::Instance("scatra")->Time());

  // prepare time step
  microtimint->prepare_time_step();

  // clear state in micro-scale time integrator
  microtimint->ClearState();

  // increment time step
  ++step_;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMultiScaleGP::evaluate(const std::vector<double>& phinp_macro, double& q_micro,
    std::vector<double>& dq_dphi_micro, const double detFnp, const bool solve)
{
  // extract micro-scale time integrator
  const Teuchos::RCP<ScaTra::TimIntOneStepTheta>& microtimint =
      microdisnum_microtimint_map_[microdisnum_];

  // set current state in micro-scale time integrator
  microtimint->set_state(phin_, phinp_, phidtn_, phidtnp_, hist_, micro_output_, phinp_macro, step_,
      Discret::ELEMENTS::ScaTraEleParameterTimInt::Instance("scatra")->Time());

  if (is_ale_)
  {
    // update determinant of deformation gradient
    det_fnp_ = detFnp;

    // calculate time derivative and pass to micro time integration as reaction coefficient
    CalculateDdetFDt(microtimint);
    microtimint->set_macro_micro_rea_coeff(ddet_fdtnp_);
  }

  if (step_ == 0 or !solve)
  {
    // only evaluate the micro-scale coupling quantities without solving the entire micro-scale
    // problem relevant for truly partitioned multi-scale simulations or for calculation of initial
    // time derivative of macro-scale state vector
    microtimint->evaluate_macro_micro_coupling();
  }
  else
  {
    // solve micro-scale problem
    // note that it is not necessary to transfer the final micro-scale state vectors back to the
    // Gauss-point submaterial due to RCP usage
    microtimint->Solve();
  }

  // transfer micro-scale coupling quantities to macro scale
  q_micro = -microtimint->Q();
  dq_dphi_micro = microtimint->DqDphi();
  for (double& dq_dphi_micro_component : dq_dphi_micro) dq_dphi_micro_component *= -1.0;

  // clear state in micro-scale time integrator
  microtimint->ClearState();
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::ScatraMultiScaleGP::evaluate_mean_concentration() const
{
  // extract micro-scale discretization
  Core::FE::Discretization& discret = *microdisnum_microtimint_map_[microdisnum_]->discretization();

  // set micro-scale state vector
  discret.ClearState();
  discret.set_state("phinp", phinp_);

  // set parameters for micro-scale elements
  Teuchos::ParameterList eleparams;
  Core::UTILS::AddEnumClassToParameterList<ScaTra::Action>(
      "action", ScaTra::Action::calc_total_and_mean_scalars, eleparams);
  eleparams.set("inverting", false);
  eleparams.set("calc_grad_phi", false);

  // initialize result vector: first component = concentration integral, second component = domain
  // integral
  const Teuchos::RCP<Core::LinAlg::SerialDenseVector> integrals =
      Teuchos::rcp(new Core::LinAlg::SerialDenseVector(2));

  // evaluate concentration and domain integrals on micro scale
  discret.EvaluateScalars(eleparams, integrals);

  // clear discretization
  discret.ClearState();

  // compute and return mean concentration on micro scale
  return (*integrals)[0] / (*integrals)[1];
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
double Mat::ScatraMultiScaleGP::evaluate_mean_concentration_time_derivative() const
{
  // extract micro-scale discretization
  Core::FE::Discretization& discret = *microdisnum_microtimint_map_[microdisnum_]->discretization();

  // set micro-scale state vector
  discret.ClearState();
  discret.set_state("phidtnp", phidtnp_);

  // set parameters for micro-scale elements
  Teuchos::ParameterList eleparams;
  Core::UTILS::AddEnumClassToParameterList<ScaTra::Action>(
      "action", ScaTra::Action::calc_mean_scalar_time_derivatives, eleparams);

  // initialize result vector: first component = integral of concentration time derivative, second
  // component = integral of domain
  const Teuchos::RCP<Core::LinAlg::SerialDenseVector> integrals =
      Teuchos::rcp(new Core::LinAlg::SerialDenseVector(2));

  // evaluate integrals of domain and time derivative of concentration on micro scale
  discret.EvaluateScalars(eleparams, integrals);

  // clear discretization
  discret.ClearState();

  // compute and return mean concentration time derivative on micro scale
  return (*integrals)[0] / (*integrals)[1];
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
void Mat::ScatraMultiScaleGP::update()
{
  if (is_ale_)
  {
    // Update detF
    det_fn_ = det_fnp_;
    ddet_fdtn_ = ddet_fdtnp_;
  }

  // extract micro-scale time integrator
  Teuchos::RCP<ScaTra::TimIntOneStepTheta> microtimint = microdisnum_microtimint_map_[microdisnum_];

  // set current state in micro-scale time integrator
  microtimint->set_state(phin_, phinp_, phidtn_, phidtnp_, hist_, micro_output_,
      std::vector<double>(0, 0.), step_,
      Discret::ELEMENTS::ScaTraEleParameterTimInt::Instance("scatra")->Time());

  // update micro-scale time integrator
  microtimint->update();

  // clear state in micro-scale time integrator
  microtimint->ClearState();
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMultiScaleGP::new_result_file()
{
  // get properties from macro scale
  Teuchos::RCP<Core::IO::OutputControl> macrocontrol =
      Global::Problem::Instance()->OutputControlFile();
  std::string microprefix = macrocontrol->restart_name();
  std::string micronewprefix = macrocontrol->new_output_file_name();

  // extract micro-scale problem and discretization
  Global::Problem* microproblem = Global::Problem::Instance(microdisnum_);
  std::stringstream microdisname;
  microdisname << "scatra_multiscale_" << microdisnum_;
  Teuchos::RCP<Core::FE::Discretization> microdis = microproblem->GetDis(microdisname.str());

  // figure out prefix of micro-scale restart files
  restartname_ = new_result_file_path(microprefix);

  // figure out new prefix for micro-scale output files
  const std::string newfilename = new_result_file_path(micronewprefix);

  if (eleowner_)
  {
    const int ndim = microproblem->NDim();
    const int restart = Global::Problem::Instance()->restart();
    bool adaptname = true;

    // in case of restart, the new output file name has already been adapted
    if (restart) adaptname = false;

    Teuchos::RCP<Core::IO::OutputControl> microcontrol = Teuchos::rcp(new Core::IO::OutputControl(
        microdis->Comm(), "Scalar_Transport", microproblem->spatial_approximation_type(),
        "micro-input-file-not-known", restartname_, newfilename, ndim, restart,
        Global::Problem::Instance(microdisnum_)->IOParams().get<int>("FILESTEPS"),
        Core::UTILS::IntegralValue<bool>(
            Global::Problem::Instance(microdisnum_)->IOParams(), "OUTPUT_BIN"),
        adaptname));

    micro_output_ = Teuchos::rcp(new Core::IO::DiscretizationWriter(
        microdis, microcontrol, microproblem->spatial_approximation_type()));
    micro_output_->set_output(microcontrol);
    micro_output_->write_mesh(
        step_, Discret::ELEMENTS::ScaTraEleParameterTimInt::Instance("scatra")->Time());
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
std::string Mat::ScatraMultiScaleGP::new_result_file_path(const std::string& newprefix)
{
  std::string newfilename;

  // create path from string to extract only filename prefix
  const std::filesystem::path path(newprefix);
  const std::string newfileprefix = path.filename().string();

  const size_t posn = newfileprefix.rfind('-');
  if (posn != std::string::npos)
  {
    const std::string number = newfileprefix.substr(posn + 1);
    const std::string prefix = newfileprefix.substr(0, posn);

    // recombine path and file
    const std::filesystem::path parent_path(path.parent_path());
    const std::filesystem::path filen_name(prefix);
    const std::filesystem::path recombined_path = parent_path / filen_name;

    std::ostringstream s;
    s << recombined_path.string() << "_microdis" << microdisnum_ << "_el" << ele_id_ << "_gp"
      << gp_id_ << "-" << number;
    newfilename = s.str();
  }
  else
  {
    std::ostringstream s;
    s << newprefix << "_microdis" << microdisnum_ << "_el" << ele_id_ << "_gp" << gp_id_;
    newfilename = s.str();
  }

  return newfilename;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMultiScaleGP::output()
{
  // skip ghosted macro-scale elements
  if (eleowner_)
  {
    // extract micro-scale time integrator
    Teuchos::RCP<ScaTra::TimIntOneStepTheta> microtimint =
        microdisnum_microtimint_map_[microdisnum_];

    // set current state in micro-scale time integrator
    microtimint->set_state(phin_, phinp_, phidtn_, phidtnp_, hist_, micro_output_,
        std::vector<double>(0, 0.), step_,
        Discret::ELEMENTS::ScaTraEleParameterTimInt::Instance("scatra")->Time());

    // output micro-scale results
    if (microtimint->IsResultStep()) microtimint->WriteResult();

    // output micro-scale restart information
    if (microtimint->IsRestartStep())
    {
      microtimint->write_restart();
      if (is_ale_)
      {
        microtimint->DiscWriter()->write_double("detFn", det_fn_);
        microtimint->DiscWriter()->write_double("ddetFdtn", ddet_fdtn_);
      }
    }

    // clear state in micro-scale time integrator
    microtimint->ClearState();
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMultiScaleGP::read_restart()
{
  // extract micro-scale time integrator
  Teuchos::RCP<ScaTra::TimIntOneStepTheta> microtimint = microdisnum_microtimint_map_[microdisnum_];

  // extract restart step
  step_ = Global::Problem::Instance()->restart();

  // set current state in micro-scale time integrator
  microtimint->set_state(phin_, phinp_, phidtn_, phidtnp_, hist_, micro_output_,
      std::vector<double>(0, 0.), step_,
      Discret::ELEMENTS::ScaTraEleParameterTimInt::Instance("scatra")->Time());

  // read restart on micro scale
  auto inputcontrol = Teuchos::rcp(new Core::IO::InputControl(restartname_, true));
  microtimint->read_restart(step_, inputcontrol);

  // safety check
  if (microtimint->Step() != step_) FOUR_C_THROW("Time step mismatch!");

  Teuchos::RCP<Core::IO::DiscretizationReader> reader(Teuchos::null);
  if (inputcontrol == Teuchos::null)
    reader = Teuchos::rcp(new Core::IO::DiscretizationReader(
        microtimint->discretization(), Global::Problem::Instance()->InputControlFile(), step_));
  else
    reader = Teuchos::rcp(
        new Core::IO::DiscretizationReader(microtimint->discretization(), inputcontrol, step_));

  if (is_ale_)
  {
    det_fn_ = reader->read_double("detFn");
    ddet_fdtn_ = reader->read_double("ddetFdtn");
  }

  // clear state in micro-scale time integrator
  microtimint->ClearState();
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMultiScaleGP::CalculateDdetFDt(Teuchos::RCP<ScaTra::TimIntOneStepTheta> microtimint)
{
  const double dt = microtimint->Dt();

  switch (microtimint->MethodName())
  {
    case Inpar::ScaTra::TimeIntegrationScheme::timeint_one_step_theta:
    {
      const double theta = microtimint->ScatraParameterList()->get<double>("THETA");

      const double part1 = (det_fnp_ - det_fn_) / dt;
      const double part2 = (1.0 - theta) * ddet_fdtn_;
      ddet_fdtnp_ = 1.0 / theta * (part1 - part2);

      break;
    }
    default:
    {
      FOUR_C_THROW("time integration scheme not supported to calculate d detF / d t.");
      break;
    }
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMultiScaleGP::SetTimeStepping(const double dt, const double time, const int step)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  FOUR_C_ASSERT(dt > 0.0, "Time step for micro scale must be positive.");
  FOUR_C_ASSERT(time >= 0.0, "Time for micro scale must be positive.");
  FOUR_C_ASSERT(step >= 0, "Number of step for micro scale must be positive.");
#endif

  Teuchos::RCP<ScaTra::TimIntOneStepTheta> microtimint = microdisnum_microtimint_map_[microdisnum_];
  microtimint->set_dt(dt);
  microtimint->SetTimeStep(time, step);
}
FOUR_C_NAMESPACE_CLOSE