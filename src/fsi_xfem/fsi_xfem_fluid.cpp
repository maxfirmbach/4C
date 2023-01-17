/*----------------------------------------------------------------------*/
/*! \file

\brief ...

\level 2

*/

#include "fsi_xfem_fluid.H"

#include "lib_colors.H"
#include "lib_globalproblem.H"
#include "inpar_validparameters.H"
#include "io_pstream.H"

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FSI::FluidXFEMAlgorithm::FluidXFEMAlgorithm(const Epetra_Comm& comm)
    : FluidMovingBoundaryBaseAlgorithm(
          DRT::Problem::Instance()->FluidDynamicParams(), "FSICoupling"),
      comm_(comm)
{
  const Teuchos::ParameterList& fluiddyn = DRT::Problem::Instance()->FluidDynamicParams();

  if (comm_.MyPID() == 0) DRT::INPUT::PrintDefaultParameters(IO::cout, fluiddyn);

  step_ = 0;
  time_ = 0.;
  dt_ = fluiddyn.get<double>("TIMESTEP");
  nstep_ = fluiddyn.get<int>("NUMSTEP");
  maxtime_ = fluiddyn.get<double>("MAXTIME");
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FSI::FluidXFEMAlgorithm::~FluidXFEMAlgorithm() {}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidXFEMAlgorithm::Timeloop()
{
  if (DRT::Problem::Instance()->GetProblemType() == ProblemType::fluid_xfem)
  {
    if (Comm().MyPID() == 0)
      std::cout << YELLOW_LIGHT << "Integrate routine for MOVING INTERFACES" << END_COLOR << "\n"
                << std::endl;


    while (NotFinished())
    {
      PrepareTimeStep();
      Solve();
      Update();
      Output();
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidXFEMAlgorithm::ReadRestart(int step)
{
  time_ = MBFluidField()->ReadRestart(step);
  step_ = step;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidXFEMAlgorithm::PrepareTimeStep()
{
  step_ += 1;
  time_ += dt_;


  MBFluidField()->PrepareTimeStep();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidXFEMAlgorithm::Solve() { MBFluidField()->NonlinearSolve(); }


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidXFEMAlgorithm::Update() { MBFluidField()->Update(); }


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::FluidXFEMAlgorithm::Output() { MBFluidField()->Output(); }