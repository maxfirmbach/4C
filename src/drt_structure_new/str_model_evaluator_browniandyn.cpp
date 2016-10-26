/*-----------------------------------------------------------*/
/*!
\file str_model_evaluator_browniandyn.cpp

\brief model evaluator for brownian (stochastic and damping)
       forces

\maintainer Jonas Eichinger

\date May, 2016

\level 3

*/
/*-----------------------------------------------------------*/

#include "str_model_evaluator_browniandyn.H"

#include "str_model_evaluator_data.H"
#include "str_timint_base.H"
#include "str_utils.H"
#include "str_integrator.H"

#include <Epetra_Vector.h>
#include <Epetra_Time.h>
#include <Teuchos_ParameterList.hpp>

#include "../linalg/linalg_utils.H"
#include "../drt_lib/drt_discret.H"

#include "../drt_beam3/beam3_base.H"
#include "../drt_biopolynet/biopolynet_calc_utils.H"
#include "../drt_lib/drt_globalproblem.H"

#include "../drt_io/io.H"
#include "str_timint_databiopolynetdyn.H"


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
STR::MODELEVALUATOR::BrownianDyn::BrownianDyn():
  eval_statmech_ptr_(Teuchos::null),
  f_brown_np_ptr_(Teuchos::null),
  f_ext_np_ptr_(Teuchos::null),
  stiff_brownian_ptr_(Teuchos::null),
  maxrandnumelement_(0),
  rs_(DRT::Problem::Instance()->getParameterList()->sublist("PROBLEM TYP").get<int>("RANDSEED"))
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::Setup()
{
  CheckInit();
  // -------------------------------------------------------------------------
  // get pointer to biopol<mer network data
  // -------------------------------------------------------------------------
  eval_statmech_ptr_ = EvalData().StatMechPtr();
  // -------------------------------------------------------------------------
  // setup the brownian forces and the external force pointers
  // -------------------------------------------------------------------------
  f_brown_np_ptr_ =
      Teuchos::rcp(new Epetra_Vector(*GState().DofRowMap(),true));
  f_ext_np_ptr_   =
      Teuchos::rcp(new Epetra_Vector(*GState().DofRowMap(),true));
  // -------------------------------------------------------------------------
  // setup the brownian forces and the external force pointers
  // -------------------------------------------------------------------------
  stiff_brownian_ptr_ = Teuchos::rcp(new LINALG::SparseMatrix(
      *GState().DofRowMapView(), 81, true, true));
  // -------------------------------------------------------------------------
  // adapt displacement vector so that node positions are consistent with
  // periodic boundary condition. Note: Input file contains the unshifted
  // configuration so that we do not have to set up the elements twice.
  // Shifting to periodic boundary configuration is done here. From now on the
  // global displacement vector always contains the shifted configuration.
  // -------------------------------------------------------------------------
  STATMECH::UTILS::PeriodicBoundaryConsistentDis(
      GStatePtr()->GetMutableDisN(),                            // disn
      eval_statmech_ptr_->GetDataSMDynPtr()->PeriodLength(),
      DiscretPtr());
  STATMECH::UTILS::PeriodicBoundaryConsistentDis(
      GStatePtr()->GetMutableDisNp(),                           // disnp
      eval_statmech_ptr_->GetDataSMDynPtr()->PeriodLength(),
      DiscretPtr());
  // -------------------------------------------------------------------------
  // get maximal number of random numbers required by any element in the
  // discretization and store them in randomnumbersperelement_
  // -------------------------------------------------------------------------
  RandomNumbersPerElement();
  // -------------------------------------------------------------------------
  // seed random generator to generate the same random
  // numbers even if the simulation was interrupted by a restart
  // -------------------------------------------------------------------------
  SeedRandomGenerator();
  // -------------------------------------------------------------------------
  // Generate random forces for first time step
  // -------------------------------------------------------------------------
  /* multivector for stochastic forces evaluated by each element; the numbers of
   * vectors in the multivector equals the maximal number of random numbers
   * required by any element in the discretization per time step; therefore this
   * multivector is suitable for synchrinisation of these random numbers in
   *  parallel computing*/
  eval_statmech_ptr_->ResizeRandomForceMVector(DiscretPtr(), maxrandnumelement_);
  GenerateGaussianRandomNumbers();

  issetup_ = true;

  return;
} // Setup()

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::Reset(const Epetra_Vector& x)
{
  CheckInitSetup();
  // -------------------------------------------------------------------------
  // adapt displacement vector so that node positions are consistent with
  // periodic boundary condition.
  // -------------------------------------------------------------------------
  STATMECH::UTILS::PeriodicBoundaryConsistentDis(
      GStatePtr()->GetMutableDisNp(),                           // disnp
      eval_statmech_ptr_->GetDataSMDynPtr()->PeriodLength(),
      DiscretPtr());
  // -------------------------------------------------------------------------
  // reset brownian (stochastic and damping) forces
  // -------------------------------------------------------------------------
  f_brown_np_ptr_->PutScalar(0.0);
  // -------------------------------------------------------------------------
  // reset external forces
  // -------------------------------------------------------------------------
  f_ext_np_ptr_->PutScalar(0.0);
  // -------------------------------------------------------------------------
  // zero out statmech stiffness contributions
  // -------------------------------------------------------------------------
  stiff_brownian_ptr_->Zero();

  return;
} // Reset()

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::BrownianDyn::EvaluateForce()
{
  CheckInitSetup();
  bool ok = true;
  // ---------------------------------------
  // (1) EXTERNAL FORCES
  // ---------------------------------------
  ok = ApplyForceExternal();

  // ---------------------------------------
  // (2) INTERNAL FORCES
  // ---------------------------------------
  // ordinary internal force
  ok = (ok ? ApplyForceBrownian() : false);

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::BrownianDyn::EvaluateInitialForce()
{
  CheckInitSetup();
  bool ok = true;

  // do nothing here as we do not want initial stochastic forces in case of
  // ml_rotations = true;
  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::BrownianDyn::EvaluateStiff()
{
  CheckInitSetup();
  bool ok = true;

  /* We use the same routines as for the ApplyForceStiff case, but we
   * do not update the global force vector, which is used for the
   * solution process in the NOX library.
   * This is meaningful, since the computational overhead, which is
   * generated by evaluating the right hand side is negligible */

  // -------------------------------------------------------------------------
  // (1) EXTRERNAL FORCES and STIFFNESS ENTRIES
  // -------------------------------------------------------------------------
  // so far the Neumann loads implemented especially for statmech don't
  // have a contribution to the jacobian
//   ApplyForceStiffExternal();

  // -------------------------------------------------------------------------
  // (2) BROWNIAN FORCES and STIFFNESS ENTRIES
  // -------------------------------------------------------------------------
  ApplyForceStiffBrownian();

  if (not stiff_brownian_ptr_->Filled())
    stiff_brownian_ptr_->Complete();

  // that's it
  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::BrownianDyn::EvaluateForceStiff()
{
  CheckInitSetup();
  bool ok = true;

  // -------------------------------------------------------------------------
  // (1) EXTRERNAL FORCES and STIFFNESS ENTRIES
  // -------------------------------------------------------------------------
  ApplyForceStiffExternal();
  // -------------------------------------------------------------------------
  // (2) BROWNIAN FORCES and STIFFNESS ENTRIES
  // -------------------------------------------------------------------------
  ApplyForceStiffBrownian();

  if (not stiff_brownian_ptr_->Filled())
    stiff_brownian_ptr_->Complete();

  // that's it
  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::BrownianDyn::AssembleForce(Epetra_Vector& f,
    const double & timefac_np) const
{
  CheckInitSetup();

  // -------------------------------------------------------------------------
  // *********** finally put everything together ***********
  // build residual  Res = F_{brw;n+1}
  //                     - F_{ext;n+1}
  // -------------------------------------------------------------------------
  STR::AssembleVector(1.0,f,-timefac_np,*f_ext_np_ptr_);
  STR::AssembleVector(1.0,f,timefac_np,*f_brown_np_ptr_);

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::BrownianDyn::AssembleJacobian(
    LINALG::SparseOperator& jac,
    const double & timefac_np) const
{
  CheckInitSetup();

  Teuchos::RCP<LINALG::SparseMatrix> jac_dd_ptr =
      GState().ExtractDisplBlock(jac);
  jac_dd_ptr->Add(*stiff_brownian_ptr_,false,timefac_np,1.0);
  // no need to keep it
  stiff_brownian_ptr_->Zero();
  // nothing to do
  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::BrownianDyn::ApplyForceExternal()
{
  CheckInitSetup();
  bool ok = true;
  // -------------------------------------------------------------------------
  // Set to default value  as it is unnecessary for the
  // EvaluateNeumann routine.
  // -------------------------------------------------------------------------
  EvalData().SetActionType(DRT::ELEMENTS::none);
  // -------------------------------------------------------------------------
  // set vector values needed by elements
  // -------------------------------------------------------------------------
  Discret().ClearState();
  Discret().SetState(0, "displacement", GState().GetDisN());
  // -------------------------------------------------------------------------
  // Evaluate statmech specific neumann conditions
  // -------------------------------------------------------------------------
  EvaluateNeumannBrownianDyn(f_ext_np_ptr_,Teuchos::null);

  // that's it
  return ok;
} // ApplyForceExternal()

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::BrownianDyn::ApplyForceBrownian()
{
  CheckInitSetup();
  bool ok = true;
  // -------------------------------------------------------------------------
  // currently a fixed number of matrix and vector pointers are supported
  // set default matrices and vectors
  // -------------------------------------------------------------------------
  Teuchos::RCP<Epetra_Vector> eval_vec [3] =
      {Teuchos::null,Teuchos::null,Teuchos::null};
  Teuchos::RCP<LINALG::SparseOperator> eval_mat[2] =
      {Teuchos::null,Teuchos::null};
  // -------------------------------------------------------------------------
  // set brwonian force vector (gets filled on element level)
  // -------------------------------------------------------------------------
  eval_vec[0] = f_brown_np_ptr_;
  // -------------------------------------------------------------------------
  // set action for elements
  // -------------------------------------------------------------------------
  EvalData().SetActionType(DRT::ELEMENTS::struct_calc_brownianforce);
  // -------------------------------------------------------------------------
  // set vector values needed by elements
  // -------------------------------------------------------------------------
  Discret().ClearState();
  Discret().SetState(0,"displacement", GState().GetMutableDisNp());
  Discret().SetState(0,"velocity", GState().GetVelNp());
  // -------------------------------------------------------------------------
  // Evaluate Browian (stochastic and damping forces)
  // -------------------------------------------------------------------------
  EvaluateBrownian(&eval_mat[0],&eval_vec[0]);

  // that's it
  return ok;

} // ApplyForceBrownian()

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::BrownianDyn::ApplyForceStiffExternal()
{
  /* so far statmech specific neumann loads need no linearization,
   therefore ApplyForceStiffExternal is equal to ApplyForceExternal*/

  CheckInitSetup();
  bool ok = true;
  // -------------------------------------------------------------------------
  // Set to default value, as it is unnecessary for the
  // EvaluateNeumann routine.
  // -------------------------------------------------------------------------
  EvalData().SetActionType(DRT::ELEMENTS::none);
  // -------------------------------------------------------------------------
  // set vector values needed by elements
  // -------------------------------------------------------------------------
  Discret().ClearState();
  Discret().SetState(0, "displacement", GState().GetDisN());
  // -------------------------------------------------------------------------
  // Evaluate statmech specific neumann conditions
  // -------------------------------------------------------------------------
  EvaluateNeumannBrownianDyn(f_ext_np_ptr_,Teuchos::null);

  return ok;
} // ApplyForceStiffExternal()

/*----------------------------------------------------------------------*
 |  Apply brownian force and stiff            (private)  eichinger 06/16|
 *----------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::BrownianDyn::ApplyForceStiffBrownian()
{
  CheckInitSetup();
  bool ok = true;
  // -------------------------------------------------------------------------
  // currently a fixed number of matrix and vector pointers are supported
  // set default matrices and vectors
  // -------------------------------------------------------------------------
  Teuchos::RCP<Epetra_Vector> eval_vec [3] =
      {Teuchos::null,Teuchos::null,Teuchos::null};
  Teuchos::RCP<LINALG::SparseOperator> eval_mat[2] =
      {Teuchos::null,Teuchos::null};
  // -------------------------------------------------------------------------
  // set jac matrix and brownian force vector (filled on element level)
  // -------------------------------------------------------------------------
  eval_mat[0] = stiff_brownian_ptr_;
  eval_vec[0] = f_brown_np_ptr_;
  // -------------------------------------------------------------------------
  // set action for elements
  // -------------------------------------------------------------------------
  EvalData().SetActionType(DRT::ELEMENTS::struct_calc_brownianstiff);
  // -------------------------------------------------------------------------
  // set vector values needed by elements
  // -------------------------------------------------------------------------
  Discret().ClearState();
  Discret().SetState(0,"displacement", GState().GetMutableDisNp());
  Discret().SetState(0,"velocity", GState().GetVelNp());
  // -------------------------------------------------------------------------
  // Evaluate brownian (stochastic and damping) forces
  EvaluateBrownian(&eval_mat[0],&eval_vec[0]);

  // that's it
  return ok;
} // ApplyForceStiffBrownian()

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::EvaluateBrownian(
    Teuchos::RCP<LINALG::SparseOperator>* eval_mat,
    Teuchos::RCP<Epetra_Vector>* eval_vec)
{
  CheckInitSetup();

  // todo: just give ParamsInterface to elements (not a parameter list)
  Teuchos::ParameterList p;
  p.set<Teuchos::RCP<DRT::ELEMENTS::ParamsInterface> >("interface",
      EvalDataPtr());
  // -------------------------------------------------------------------------
  // Evaluate brownian (stochastic and damping) forces on element level
  // -------------------------------------------------------------------------
  EvaluateBrownian(p,eval_mat,eval_vec);

  // that's it
  return;
} //EvaluateBrownian()

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::EvaluateBrownian(
    Teuchos::ParameterList& p,
    Teuchos::RCP<LINALG::SparseOperator>* eval_mat,
    Teuchos::RCP<Epetra_Vector>* eval_vec)
{
  CheckInitSetup();

  // todo: this needs to go, just pass ParamsInterface to elements
  if (p.numParams()>1)
    dserror("Please use the STR::ELEMENTS::Interface and its derived "
        "classes to set and get parameters.");
  // -------------------------------------------------------------------------
  // Evaluate brownian on element level
  // -------------------------------------------------------------------------
  Discret().Evaluate(p, eval_mat[0], eval_mat[1],
     eval_vec[0], eval_vec[1], eval_vec[2]);
  Discret().ClearState();

  // that's it
  return;
} // EvaluateBrownian()

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::EvaluateNeumannBrownianDyn(
    Teuchos::RCP<Epetra_Vector> eval_vec,
    Teuchos::RCP<LINALG::SparseOperator> eval_mat)
{
  CheckInitSetup();
  // -------------------------------------------------------------------------
  // get interface pointer
  // -------------------------------------------------------------------------
  Teuchos::RCP<DRT::ELEMENTS::ParamsInterface> interface_ptr = EvalDataPtr();
  // -------------------------------------------------------------------------
  // evaluate statmech specific Neumann boundary conditions
  // -------------------------------------------------------------------------
//  sm_manager_ptr_->EvaluateNeumannStatMech(interface_ptr,eval_vec,eval_mat);
  DiscretPtr()->ClearState();

  return;
} //EvaluateNeumannBrownianDyn()

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::WriteRestart(
        IO::DiscretizationWriter& iowriter,
        const bool& forced_writerestart) const
{
  CheckInitSetup();


  return;
} // WriteRestart()

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::ReadRestart(
    IO::DiscretizationReader& ioreader)
{
  CheckInitSetup();

  return;
} // ReadRestart()

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::RecoverState(
    const Epetra_Vector& xold,
    const Epetra_Vector& dir,
    const Epetra_Vector& xnew)
{
 // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::UpdateStepState(
    const double& timefac_n)
{
  CheckInitSetup();
  // -------------------------------------------------------------------------
  // add statmech force contributions to the old structural
  // residual state vector
  // -------------------------------------------------------------------------
  Teuchos::RCP<Epetra_Vector>& fstructold_ptr =
      GState().GetMutableFstructureOld();
  fstructold_ptr->Update(timefac_n,*f_brown_np_ptr_,1.0);
  fstructold_ptr->Update(-timefac_n,*f_ext_np_ptr_,1.0);

  return;
} // UpdateStepState()

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::UpdateStepElement()
{

  // -------------------------------------------------------------------------
  // check if timestep changes according to action dt in input file
  // -------------------------------------------------------------------------
  // todo: this needs to go somewhere else, to a more global/general place
  // (console output at this point is also very unflattering)
//  sm_manager_ptr_->UpdateTimeAndStepSize((*GStatePtr()->GetMutableDeltaTime())[0],
//                                           GStatePtr()->GetTimeN());

  return;

} // UpdateStepElement()

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::DetermineStressStrain()
{
  // nothing to do
  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::DetermineEnergy()
{
  // nothing to do
  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::OutputStepState(
    IO::DiscretizationWriter& iowriter) const
{
  // nothing to do
  return;
} //OutputStepState()

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> STR::MODELEVALUATOR::BrownianDyn::
    GetBlockDofRowMapPtr() const
{
  CheckInitSetup();
  return GState().DofRowMap();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> STR::MODELEVALUATOR::BrownianDyn::
    GetCurrentSolutionPtr() const
{
  // there are no model specific solution entries
  return Teuchos::null;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> STR::MODELEVALUATOR::BrownianDyn::
    GetLastTimeStepSolutionPtr() const
{
  // there are no model specific solution entries
  return Teuchos::null;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::PostOutput()
{
  CheckInitSetup();
  // -------------------------------------------------------------------------
  // seed random generator to generate the same random
  // numbers even if the simulation was interrupted by a restart
  // -------------------------------------------------------------------------
  SeedRandomGenerator();
  // -------------------------------------------------------------------------
  // Generate new random forces
  // -------------------------------------------------------------------------
  eval_statmech_ptr_->ResizeRandomForceMVector(DiscretPtr(), maxrandnumelement_);
  GenerateGaussianRandomNumbers();

  return;
} // PostOutput()

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::ResetStepState()
{
  CheckInitSetup();
  // -------------------------------------------------------------------------
  // Generate new random forces
  // -------------------------------------------------------------------------
  eval_statmech_ptr_->ResizeRandomForceMVector(DiscretPtr(), maxrandnumelement_);
  GenerateGaussianRandomNumbers();
  // -------------------------------------------------------------------------
  // Update number of unconverged steps
  // -------------------------------------------------------------------------
//  sm_manager_ptr_->UpdateNumberOfUnconvergedSteps();

  /* special part in statmech for predictor: initialize disn_ and veln_ with zero;
   * this is necessary only for the following case: Assume that an iteration
   * step did not converge and is repeated with new random numbers; if the
   * failure of convergence lead to disn_ = NaN and veln_ = NaN this would affect
   * also the next trial as e.g. disn_->Update(1.0,*((*dis_)(0)),0.0); would set
   * disn_ to NaN as even 0*NaN = NaN!; this would defeat the purpose of the
   * repeated iterations with new random numbers and has thus to be avoided;
   * therefore we initialized disn_ and veln_ with zero which has no effect
   * in any other case*/
  // todo: is this the right place for this (originally done in statmech predictor,
  // should work as prediction is the next thing that is done)
  GStatePtr()->GetMutableDisNp()->PutScalar(0.0);
  GStatePtr()->GetMutableVelNp()->PutScalar(0.0);
  // we only need this in case we use Lie Group gen alpha and calculate a consistent
  // mass matrix and acc vector (i.e. we are not neclecting inertia forces)
  GStatePtr()->GetMutableAccNp()->PutScalar(0.0);

  return;
} //ResetStepState()

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::RandomNumbersPerElement()
{
  CheckInit();
  // -------------------------------------------------------------------------
  // maximal number of random numbers to be generated per time step for
  // any column map element of this processor
  // -------------------------------------------------------------------------
  int randomnumbersperlocalelement = 0;
  // -------------------------------------------------------------------------
  // check maximal number of nodes of an element with stochastic forces
  // on this processor
  // -------------------------------------------------------------------------
  for (int i=0; i<DiscretPtr()->NumMyColElements(); ++i)
    // see whether current element needs more random numbers per time step
    // than any other before
    randomnumbersperlocalelement = std::max(randomnumbersperlocalelement,
        dynamic_cast<DRT::ELEMENTS::Beam3Base*>(
            DiscretPtr()->lColElement(i))->HowManyRandomNumbersINeed());
  // -------------------------------------------------------------------------
  // so far the maximal number of random numbers required per element
  // has been checked only locally on this processor; now we compare the
  // results of each processor and store the maximal one in
  // maxrandnumelement_
  // -------------------------------------------------------------------------
  DiscretPtr()->Comm().MaxAll(&randomnumbersperlocalelement,&maxrandnumelement_ ,1);

  return;
} // RandomNumbersPerElement()

/*----------------------------------------------------------------------------*
 | seed all random generators of this object with fixed seed if given and     |
 | with system time otherwise; seedparameter is used only in the first        |
 | case to calculate the actual seed variable based on some given fixed       |
 | seed value; note that seedparameter may be any integer, but has to be      |
 | been set in a deterministic way so that it for a certain call of this      |
 | method at a certain point in the program always the same number            |
 | whenever the program is used                           eichinger 06/16     |
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::SeedRandomGenerator()
{
  CheckInit();

  const int    stepn  = GStatePtr()->GetStepN() + 1;
  const double timenp = GStatePtr()->GetTimeNp();
  const double dt     = (*GStatePtr()->GetDeltaTime())[0];
  // -------------------------------------------------------------------------
  // if input flag RADNSEED >= -1: Use same random numbers in each program start;
  // to this end compute int seedvariable from given parameter RANDSEED and some
  // other deterministic parameter at runtime
  // -------------------------------------------------------------------------
  int seedvariable = 0;
  // -------------------------------------------------------------------------
  // seed random generator wit fixed seed (overwrite seed written in ReadParameter
  // drt_globalproblem.cpp). We have same random numbers in each program start
  // (not for different numbers of procs though)
  // -------------------------------------------------------------------------
  if (rs_ >= 0)
  {
    // -----------------------------------------------------------------------
    // Decide if random numbers should change in every time step...
    // read time interval within the random numbers remain constant (-1.0 means no
    // prescribed time interval). This means new random numbers every
    // randnumtimeinc seconds.
    // -----------------------------------------------------------------------
    double time_interv_with_const_rn =
        eval_statmech_ptr_->GetDataSMDynPtr()->TimeIntConstRandNumb_();
    if(time_interv_with_const_rn != -1.0)
    {
      // new random numbers every time step (same in each program start though)
      seedvariable = (rs_ + stepn)*(DiscretPtr()->Comm().MyPID() + 1);
    }
    //...or only every time_interv_with_const_rn seconds
    else
    {
      // this variable changes every time_interv_with_const_rn seconds
      int seed_differs_every_time_int =
          static_cast<int>( (timenp-dt) / time_interv_with_const_rn + 1.0e-8);
      // calculate seed variable
      seedvariable =
          (rs_ + seed_differs_every_time_int)*(DiscretPtr()->Comm().MyPID() + 1);
    }
    // -----------------------------------------------------------------------
    // seed random number generartor
    // -----------------------------------------------------------------------
    DRT::Problem::Instance()->Random()->SetRandSeed((unsigned int)seedvariable);
  }
  // -------------------------------------------------------------------------
  // else set seed according to system time and different for each processor
  // once in the beginning (done in ReadParameter drt_globalproblem.cpp)
  // in this case we have different random numbers in each program start
  // -------------------------------------------------------------------------

  // -------------------------------------------------------------------------
  // set range for uniform random number generator
  // -------------------------------------------------------------------------
  DRT::Problem::Instance()->Random()->SetRandRange(0.0,1.0);

  return;
} // SeedRandomGenerators()

/*----------------------------------------------------------------------------*
 | (public) generate gaussian randomnumbers with mean "meanvalue" and         |
 | standarddeviation "standarddeviation" for parallel use           cyron10/09|
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::GenerateGaussianRandomNumbers()
{
  CheckInit();

  // initialize mean value and standard deviation
  double meanvalue = 0.0;
  double standarddeviation = 0.0;
  double randnumtimeinc =
      eval_statmech_ptr_->GetDataSMDynPtr()->TimeIntConstRandNumb_();

  // generate gaussian random numbers for parallel use with mean value 0 and
  // standard deviation (2KT / dt)^0.5
  if(randnumtimeinc==-1.0)
    standarddeviation = pow(2.0 * eval_statmech_ptr_->GetDataSMDynPtr()->KT() /
        (*GStatePtr()->GetDeltaTime())[0],0.5);
  else
    standarddeviation = pow(2.0 * eval_statmech_ptr_->GetDataSMDynPtr()->KT() /
        randnumtimeinc,0.5);

  // Set mean value and standard deviation of normal distribution
  DRT::Problem::Instance()->Random()->SetMeanVariance(meanvalue,standarddeviation);

  //multivector for stochastic forces evaluated by each element based on row map
  Teuchos::RCP<Epetra_MultiVector> randomnumbersrow =
      eval_statmech_ptr_->GetMutableRadomForces();

  int numele = randomnumbersrow->MyLength();
  int numperele = randomnumbersrow->NumVectors();
  int count = numele*numperele;
  std::vector<double> randvec(count);
  DRT::Problem::Instance()->Random()->Normal(randvec,count);

  //MAXRANDFORCE is a multiple of the standard deviation
  double maxrandforcefac = eval_statmech_ptr_->GetDataSMDynPtr()->MaxRandRorce();
  if(maxrandforcefac==-1.0)
  {
    for (int i=0; i<numele; ++i)
      for (int j=0; j<numperele; ++j)
        (*randomnumbersrow)[j][i] = randvec[i*numperele+j];
  }
  else
  {
    for (int i=0; i<numele; i++)
      for (int j=0; j<numperele; j++)
      {
        (*randomnumbersrow)[j][i] = randvec[i*numperele+j];

        if((*randomnumbersrow)[j][i]>maxrandforcefac*standarddeviation + meanvalue)
        {
          (*randomnumbersrow)[j][i]=maxrandforcefac*standarddeviation + meanvalue;
        }
        else if((*randomnumbersrow)[j][i]<-maxrandforcefac*standarddeviation + meanvalue)
        {
          (*randomnumbersrow)[j][i]=-maxrandforcefac*standarddeviation + meanvalue;
        }
      }
  }

  return;
} // BrownianDynManager::GenerateGaussianRandomNumbers()