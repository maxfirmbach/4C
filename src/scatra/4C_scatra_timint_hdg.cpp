/*----------------------------------------------------------------------*/
/*! \file
\brief HDG time-integration scheme

\level 3

*/
/*----------------------------------------------------------------------*/

#include "4C_scatra_timint_hdg.hpp"

#include "4C_binstrategy.hpp"
#include "4C_fem_discretization_hdg.hpp"
#include "4C_fem_dofset.hpp"
#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_hdg.hpp"
#include "4C_scatra_resulttest_hdg.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Constructor (public)                                 hoermann 09/15 |
 *----------------------------------------------------------------------*/
ScaTra::TimIntHDG::TimIntHDG(const Teuchos::RCP<Core::FE::Discretization> &actdis,
    const Teuchos::RCP<Core::LinAlg::Solver> &solver,
    const Teuchos::RCP<Teuchos::ParameterList> &params,
    const Teuchos::RCP<Teuchos::ParameterList> &extraparams,
    Teuchos::RCP<Core::IO::DiscretizationWriter> output)
    : ScaTraTimIntImpl(actdis, solver, params, extraparams, output),
      TimIntGenAlpha(actdis, solver, params, extraparams, output),
      nds_intvar_(2),
      intphinp_(Teuchos::null),
      intphin_(Teuchos::null),
      interpolatedPhinp_(Teuchos::null),
      timealgoset_(Inpar::ScaTra::timeint_gen_alpha),
      startalgo_(true),
      theta_(-1),
      hdgdis_(nullptr),
      padaptivity_(Core::UTILS::IntegralValue<bool>(*params, "PADAPTIVITY")),
      padapterrortol_(params->get<double>("PADAPTERRORTOL")),
      padapterrorbase_(params->get<double>("PADAPTERRORBASE")),
      padaptdegreemax_(params->get<int>("PADAPTDEGREEMAX")),
      elementdegree_(Teuchos::null)

{
}


/*----------------------------------------------------------------------*
 |  initialize algorithm                                 hoermann 09/15 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::setup()
{
  hdgdis_ = dynamic_cast<Core::FE::DiscretizationHDG *>(discret_.get());
  if (hdgdis_ == nullptr) FOUR_C_THROW("Did not receive an HDG discretization");

  // vector to store the dofs per element
  const Teuchos::RCP<Epetra_IntVector> eledofs =
      Teuchos::rcp(new Epetra_IntVector(*discret_->ElementColMap()));

  // loop over elements
  for (int iele = 0; iele < discret_->NumMyColElements(); ++iele)
  {
    Discret::ELEMENTS::ScaTraHDG *hdgele =
        dynamic_cast<Discret::ELEMENTS::ScaTraHDG *>(discret_->lColElement(iele));
    (*eledofs)[iele] = hdgele->num_dof_per_element_auxiliary();
  }

  // add proxy for interior degrees of freedom to scatra discretization
  Teuchos::RCP<Core::DOFSets::DofSetInterface> dofsetaux =
      Teuchos::rcp(new Core::DOFSets::DofSetPredefinedDoFNumber(0, eledofs, 0, false));
  if (discret_->AddDofSet(dofsetaux) != 2)
    FOUR_C_THROW("Scatra discretization has illegal number of dofsets!");
  discret_->fill_complete();

  // HDG vectors passed to the element
  const Epetra_Map *intdofrowmap = discret_->dof_row_map(nds_intvar_);
  intphinp_ = Core::LinAlg::CreateVector(*intdofrowmap, true);
  intphin_ = Core::LinAlg::CreateVector(*intdofrowmap, true);

  // write number of degrees of freedom for hdg and interior variables to screen output
  if (discret_->Comm().MyPID() == 0)
  {
    std::cout << "Number of degrees of freedom in HDG system: "
              << discret_->dof_row_map(0)->NumGlobalElements() << std::endl;
    std::cout << "Number of degrees of freedom of interior variables: "
              << discret_->dof_row_map(nds_intvar_)->NumGlobalElements() << std::endl;
  }

  // implement ost and bdf2 through gen-alpha facilities
  // TO DO: implement other time integration schemes, at the moment only one-step-theta and
  // stationary are implemented
  switch (timealgo_)
  {
    case Inpar::ScaTra::timeint_bdf2:
    {
      FOUR_C_THROW("At the moment only one step theta implemented");
      alphaM_ = 1.5;
      alphaF_ = 1.0;
      gamma_ = 1.0;
      break;
    }
    case Inpar::ScaTra::timeint_one_step_theta:
    {
      alphaM_ = 1.0;
      alphaF_ = 1.0;
      gamma_ = params_->get<double>("THETA");
      break;
    }
    case Inpar::ScaTra::timeint_stationary:
    {
      break;
    }
    default:
      FOUR_C_THROW("At the moment only one step theta implemented");
  }

  timealgoset_ = timealgo_;
  if (timealgo_ != Inpar::ScaTra::timeint_stationary) timealgo_ = Inpar::ScaTra::timeint_gen_alpha;

  // call init()-functions of base classes
  // note: this order is important
  ScaTra::TimIntGenAlpha::setup();

  // create vector for concentration at nodes for output
  interpolatedPhinp_ = Core::LinAlg::CreateVector(*discret_->NodeRowMap(), true);

  // vector to store the elementdegree at each time step
  elementdegree_ = Core::LinAlg::CreateVector(*(discret_->ElementRowMap()), true);
}


/*------------------------------------------------------------------------*
| calculate pseudo-theta for startalgo_, modified for HDG  hoermann 09/15 |
*-------------------------------------------------------------------------*/
void ScaTra::TimIntHDG::SetTheta()
{
  // -------------------------------------------------------------------
  //  For af-generalized-alpha time-integration scheme:
  //  set "pseudo-theta", calculate initial accelerations according to
  //  prescribed Dirichlet values for generalized-alpha time
  //  integration and values at intermediate time steps
  // -------------------------------------------------------------------
  // starting algorithm
  if (startalgo_ || (step_ <= 2 && timealgoset_ == Inpar::ScaTra::timeint_bdf2))
  {
    // use backward-Euler-type parameter combination
    if (step_ <= 1 && timealgoset_ == Inpar::ScaTra::timeint_bdf2)
    {
      if (myrank_ == 0)
      {
        std::cout << "Starting algorithm for Af_GenAlpha active. "
                  //<<"Performing step "<<step_ <<" of "<<numstasteps_
                  << " Backward Euler starting steps" << std::endl;
      }
      alphaM_ = 1.0;
      alphaF_ = 1.0;
      gamma_ = 1.0;
    }
    else
    {
      // recall original user wish
      switch (timealgoset_)
      {
        case Inpar::ScaTra::timeint_one_step_theta:
        {
          alphaM_ = alphaF_ = 1.0;
          gamma_ = params_->get<double>("THETA");
          break;
        }
        case Inpar::ScaTra::timeint_bdf2:
        {
          alphaF_ = gamma_ = 1.0;
          alphaM_ = 3. / 2.;
          break;
        }
        case Inpar::ScaTra::timeint_stationary:
        {
          // Setting the parameters as for Inpar::ScaTra::timeint_one_step_theta with theta = 1
          // (basically BDF1 and therefore we only compute the RHS and Dirich at t+1)
          alphaM_ = alphaF_ = gamma_ = 1.0;
          // The time step can not be set to zero because there is plenty of divisions by dt.
          // Dt is therefore to 1.0
          dta_ = 1.0;
          // Set time equal -dta, this way the steady state is given as the solution at t=0.0.
          // This is necessary otherwhise the solver would recognise that we are at the end of a
          // simulation and skip the solution altogheter.
          time_ = -dta_;
          maxtime_ = dta_;
          // stepmax is 1 to avoid waste computation (it's stationary after all)
          stepmax_ = 1;
          break;
        }
        default:
        {
          alphaM_ = params_->get<double>("alpha_M");
          alphaF_ = params_->get<double>("alpha_F");
          gamma_ = params_->get<double>("gamma");
          break;
        }
      }

      // do not enter starting algorithm section in the future
      startalgo_ = false;
    }
  }

  // compute "pseudo-theta" for af-generalized-alpha scheme
  theta_ = alphaF_ * gamma_ / alphaM_;
}


/*----------------------------------------------------------------------*
| set HDG state vectors                                  hoermann 09/15 |
*-----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::add_time_integration_specific_vectors(bool forcedincrementalsolver)
{
  // set hdg vector and interior variables vector
  discret_->set_state(0, "phin", phin_);
  discret_->set_state(0, "phiaf", phinp_);
  discret_->set_state(nds_intvar_, "intphinp", intphinp_);
  discret_->set_state(nds_intvar_, "intphin", intphin_);
}  // add_time_integration_specific_vectors

/*----------------------------------------------------------------------*
 | compute values at intermediate time steps for gen.-alpha  hoer 09/15 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::gen_alpha_intermediate_values()
{
  // set intermediate values for concentration derivatives
  //
  //       n+alphaM                n+1                      n
  //  dtphi       = alpha_M * dtphi    + (1-alpha_M) * dtphi
  //       (i)                     (i)
  phidtam_->Update((alphaM_), *phidtnp_, (1.0 - alphaM_), *phidtn_, 0.0);

  // set intermediate values for concentration, concentration gradient
  //
  //       n+alphaF              n+1                   n
  //      u         = alpha_F * u     + (1-alpha_F) * u
  //       (i)                   (i)
  //
  // note that its af-genalpha with mid-point treatment of the pressure,
  // not implicit treatment as for the genalpha according to Whiting
  phiaf_->Update((alphaF_), *phinp_, (1.0 - alphaF_), *phin_, 0.0);

  phiam_->Update(alphaM_, *phinp_, (1.0 - alphaM_), *phin_, 0.0);

}  // gen_alpha_intermediate_values

/*----------------------------------------------------------------------*
| set old part of right hand side                        hoermann 09/15 |
*-----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::set_old_part_of_righthandside()
{
  SetTheta();
  hist_->PutScalar(0.0);

  // This code is entered at the beginning of the nonlinear iteration, so
  // store that the assembly to be done next is going to be the first one
  // (without combined vector update) for HDG.
  //  ScaTra::TimIntGenAlpha::set_old_part_of_righthandside();
}

/*----------------------------------------------------------------------*
 * Update
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::update()
{
  ScaTra::TimIntGenAlpha::update();

  // concentrations of this step become most recent
  // concentrations of the last step
  intphin_->Update(1.0, *intphinp_, 0.0);

  if (padaptivity_) adapt_degree();

}  // Update

namespace
{
  // internal helper function for output
  void getNodeVectorsHDG(Core::FE::Discretization &dis,
      const Teuchos::RCP<Epetra_Vector> &interiorValues,
      const Teuchos::RCP<Epetra_Vector> &traceValues, const int ndim,
      Teuchos::RCP<Epetra_Vector> &phi, Teuchos::RCP<Epetra_MultiVector> &gradphi,
      Teuchos::RCP<Epetra_Vector> &tracephi, int nds_intvar_, int ndofs)
  {
    dis.ClearState(true);

    // create dofsets for concentration at nodes
    tracephi.reset(new Epetra_Vector(phi->Map()));
    gradphi.reset(new Epetra_MultiVector(*dis.NodeRowMap(), ndim));

    // call element routine to interpolate HDG to elements
    Teuchos::ParameterList eleparams;
    Core::UTILS::AddEnumClassToParameterList<ScaTra::Action>(
        "action", ScaTra::Action::interpolate_hdg_to_node, eleparams);
    dis.set_state(0, "phiaf", traceValues);
    dis.set_state(nds_intvar_, "intphinp", interiorValues);
    Core::Elements::Element::LocationArray la(ndofs);
    Core::LinAlg::SerialDenseMatrix dummyMat;
    Core::LinAlg::SerialDenseVector dummyVec;
    Core::LinAlg::SerialDenseVector interpolVec;
    std::vector<unsigned char> touchCount(dis.NumMyRowNodes());

    phi->PutScalar(0.);

    for (int el = 0; el < dis.NumMyColElements(); ++el)
    {
      Core::Elements::Element *ele = dis.lColElement(el);
      ele->LocationVector(dis, la, false);
      interpolVec.size(ele->num_node() * (2 + ndim));

      ele->evaluate(eleparams, dis, la, dummyMat, dummyMat, interpolVec, dummyVec, dummyVec);

      // sum values on nodes into vectors and record the touch count (build average of values)
      for (int i = 0; i < ele->num_node(); ++i)
      {
        Core::Nodes::Node *node = ele->Nodes()[i];
        const int localIndex = dis.NodeRowMap()->LID(node->Id());
        if (localIndex < 0) continue;
        touchCount[localIndex]++;
        (*phi)[localIndex] += interpolVec(i);
        (*tracephi)[localIndex] += interpolVec(i + ele->num_node());
        for (int d = 0; d < ndim; ++d)
          (*gradphi)[d][localIndex] += interpolVec(i + (d + 2) * ele->num_node());
      }
    }

    // build average of values
    for (int i = 0; i < phi->MyLength(); ++i)
    {
      (*phi)[i] /= touchCount[i];
      (*tracephi)[i] /= touchCount[i];
      for (int d = 0; d < ndim; ++d) (*gradphi)[d][i] /= touchCount[i];
    }
    dis.ClearState(true);
  }
}  // namespace


/*----------------------------------------------------------------------*
 | output of solution vector to binio                     hoermann 09/15|
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::output_state()
{
  // output of solution

  Teuchos::RCP<Epetra_MultiVector> interpolatedGradPhi;
  Teuchos::RCP<Epetra_Vector> interpolatedtracePhi;
  // get (averaged) values at element nodes
  getNodeVectorsHDG(*discret_, intphinp_, phinp_, Global::Problem::Instance()->NDim(),
      interpolatedPhinp_, interpolatedGradPhi, interpolatedtracePhi, nds_intvar_,
      discret_->NumDofSets());

  // write vector to output file
  output_->write_vector("phi_hdg", interpolatedPhinp_, Core::IO::nodevector);
  output_->write_vector("gradphi_hdg", interpolatedGradPhi, Core::IO::nodevector);
  output_->write_vector("tracephi_hdg", interpolatedtracePhi, Core::IO::nodevector);

  write_problem_specific_output(interpolatedPhinp_);

  output_->write_vector("elementdegree", elementdegree_, Core::IO::elementvector);

}  // output_state

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::write_restart() const
{
  ScaTra::TimIntGenAlpha::write_restart();
  output_->write_vector("intphinp", intphinp_);
  output_->write_vector("phinp_trace", phinp_);
  output_->write_vector("intphin", intphin_);

  output_->write_mesh(
      step_, time_);  // add info to control file for reading all variables in restart
}

/*----------------------------------------------------------------------*
 | read restart                                          hoermann 09/15 |
 -----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::read_restart(const int step, Teuchos::RCP<Core::IO::InputControl> input)
{
  Core::IO::DiscretizationReader reader(
      discret_, Global::Problem::Instance()->InputControlFile(), step);

  time_ = reader.read_double("time");
  step_ = reader.read_int("step");

  reader.read_history_data(step);  // Read all saved data in nodes and elements and call nodal and
                                   // element Unpacking each global variable has to be read

  if (padaptivity_)
  {
    // redistribute discr. with help of binning strategy
    if (discret_->Comm().NumProc() > 1)
    {
      // create vector of discr.
      std::vector<Teuchos::RCP<Core::FE::Discretization>> dis;
      dis.push_back(discret_);

      // binning strategy for parallel redistribution
      Teuchos::RCP<Core::Binstrategy::BinningStrategy> binningstrategy;

      std::vector<Teuchos::RCP<Epetra_Map>> stdelecolmap;
      std::vector<Teuchos::RCP<Epetra_Map>> stdnodecolmap;

      // binning strategy is created and parallel redistribution is performed
      Teuchos::ParameterList binning_params =
          Global::Problem::Instance()->binning_strategy_params();
      Core::UTILS::AddEnumClassToParameterList<Core::FE::ShapeFunctionType>(
          "spatial_approximation_type", Global::Problem::Instance()->spatial_approximation_type(),
          binning_params);
      auto element_filter = [](const Core::Elements::Element *element)
      { return Core::Binstrategy::Utils::SpecialElement::none; };
      auto rigid_sphere_radius = [](const Core::Elements::Element *element) { return 0.0; };
      auto correct_beam_center_node = [](const Core::Nodes::Node *node) { return node; };
      binningstrategy = Teuchos::rcp(new Core::Binstrategy::BinningStrategy(binning_params,
          Global::Problem::Instance()->OutputControlFile(), discret_->Comm(),
          discret_->Comm().MyPID(), element_filter, rigid_sphere_radius, correct_beam_center_node,
          dis));
      binningstrategy
          ->do_weighted_partitioning_of_bins_and_extend_ghosting_of_discret_to_one_bin_layer(
              dis, stdelecolmap, stdnodecolmap);
    }
  }

  // vector to store the dofs per element
  const Teuchos::RCP<Epetra_IntVector> eledofs =
      Teuchos::rcp(new Epetra_IntVector(*discret_->ElementColMap()));

  // build new maps for face dofs with adapted element order
  hdgdis_->BuildFaces();
  hdgdis_->BuildFaceRowMap();
  hdgdis_->BuildFaceColMap();

  // assign the degrees of freedom to the adapted dofsets
  hdgdis_->assign_degrees_of_freedom(0);

  // replace all ghosted element with the original thus the correct polynomial degree is used
  discret_->export_column_elements(*discret_->ElementColMap(), false, false);

  hdgdis_->fill_complete();

  // store the number of dofs per element on vector
  for (int iele = 0; iele < discret_->NumMyColElements(); ++iele)
  {
    Discret::ELEMENTS::ScaTraHDG *hdgele =
        dynamic_cast<Discret::ELEMENTS::ScaTraHDG *>(discret_->lColElement(iele));
    // store the number of dofs for the element
    (*eledofs)[iele] = hdgele->num_dof_per_element_auxiliary();
  }

  // create new local dofset for the new interior element dofs with adapted element order
  Teuchos::RCP<Core::DOFSets::DofSetPredefinedDoFNumber> eledofs_new =
      Teuchos::rcp(new Core::DOFSets::DofSetPredefinedDoFNumber(0, eledofs, 0, false));
  // replace old interior element dofs with the new created dofset
  discret_->ReplaceDofSet(nds_intvar_, eledofs_new, false);

  hdgdis_->assign_degrees_of_freedom(0);

  // clear map cache since after every fill_complete() / assign_degrees_of_freedom() old maps are
  // stored in the mapstack
  output_->clear_map_cache();

  // reset the residual, increment and sysmat to the size
  residual_.reset(new Epetra_Vector(*(discret_->dof_row_map())));
  increment_.reset(new Epetra_Vector(*(discret_->dof_row_map())));
  neumann_loads_.reset(new Epetra_Vector(*(discret_->dof_row_map())));
  sysmat_ = Teuchos::null;
  sysmat_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(*(discret_->dof_row_map()), 27));

  // reset the state vectors
  intphinp_.reset(new Epetra_Vector(*(discret_->dof_row_map(nds_intvar_)), true));
  intphin_.reset(new Epetra_Vector(*(discret_->dof_row_map(nds_intvar_)), true));
  phinp_.reset(new Epetra_Vector(*(discret_->dof_row_map())));
  phin_.reset(new Epetra_Vector(*(discret_->dof_row_map())));

  // read state vectors that are needed for hdg
  reader.read_vector(phinp_, "phinp_trace");
  reader.read_vector(intphinp_, "intphinp");

  intphin_->Update(1.0, *intphinp_, 0.0);
  phin_->Update(1.0, *phinp_, 0.0);

  // reset vector
  interpolatedPhinp_.reset(new Epetra_Vector(*(discret_->NodeRowMap())));
  elementdegree_.reset(new Epetra_Vector(*(discret_->ElementRowMap())));

  return;
}

/*----------------------------------------------------------------------*
 |  set initial field for phi                            hoermann 09/15 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::SetInitialField(
    const Inpar::ScaTra::InitialField init, const int startfuncno)
{
  switch (init)
  {
    case Inpar::ScaTra::initfield_zero_field:
    {
      // set initial field to zero
      phin_->PutScalar(0.0);
      phinp_->PutScalar(0.0);
      intphin_->PutScalar(0.0);
      intphinp_->PutScalar(0.0);
      break;
    }
    case Inpar::ScaTra::initfield_field_by_function:
    {
      // set initial field defined by function
      Teuchos::ParameterList eleparams;
      Core::UTILS::AddEnumClassToParameterList<ScaTra::Action>(
          "action", ScaTra::Action::set_initial_field, eleparams);
      eleparams.set<int>("funct", startfuncno);

      discret_->set_state("phiaf", phinp_);
      discret_->set_state("phin", phin_);
      discret_->set_state(nds_intvar_, "intphin", intphin_);
      discret_->set_state(nds_intvar_, "intphinp", intphinp_);

      Core::LinAlg::SerialDenseMatrix dummyMat;
      Core::LinAlg::SerialDenseVector updateVec1, updateVec2, dummyVec;
      Core::Elements::Element::LocationArray la(discret_->NumDofSets());

      const Epetra_Map *dofrowmap = discret_->dof_row_map();
      const Epetra_Map *intdofrowmap = discret_->dof_row_map(nds_intvar_);
      double error = 0;

      for (int iele = 0; iele < discret_->NumMyColElements(); ++iele)
      {
        Core::Elements::Element *ele = discret_->lColElement(iele);
        ele->LocationVector(*discret_, la, false);
        if (static_cast<std::size_t>(updateVec1.numRows()) != la[0].lm_.size())
          updateVec1.size(la[0].lm_.size());
        else
          updateVec1.putScalar(0.0);
        if (updateVec2.numRows() != discret_->NumDof(nds_intvar_, ele))
          updateVec2.size(discret_->NumDof(nds_intvar_, ele));
        else
          updateVec2.putScalar(0.0);
        ele->evaluate(
            eleparams, *discret_, la, dummyMat, dummyMat, updateVec1, updateVec2, dummyVec);

        if (ele->Owner() == discret_->Comm().MyPID())
        {
          std::vector<int> localDofs = discret_->Dof(nds_intvar_, ele);
          FOUR_C_ASSERT(
              localDofs.size() == static_cast<std::size_t>(updateVec2.numRows()), "Internal error");
          for (unsigned int i = 0; i < localDofs.size(); ++i)
            localDofs[i] = intdofrowmap->LID(localDofs[i]);
          intphinp_->ReplaceMyValues(localDofs.size(), updateVec2.values(), localDofs.data());
        }

        // now fill the element vector into the discretization
        for (unsigned int i = 0; i < la[0].lm_.size(); ++i)
        {
          const int lid = dofrowmap->LID(la[0].lm_[i]);
          if (lid >= 0)
          {
            // safety check if initial value for trace dof is set for all elements the same
            // (interior face)
            if ((*phinp_)[lid] != 0) error += std::abs((*phinp_)[lid] - updateVec1(i));
            (*phinp_)[lid] = updateVec1(i);
            (*phin_)[lid] = updateVec1(i);
          }
        }
      }

      double globerror = 0;
      discret_->Comm().SumAll(&error, &globerror, 1);
      if (discret_->Comm().MyPID() == 0)
        std::cout << "Error project when setting face twice: " << globerror << std::endl;

      // initialize also the solution vector. These values are a pretty good guess for the
      // solution after the first time step (much better than starting with a zero vector)
      intphin_->Update(1.0, *intphinp_, 0.0);

      break;
    }

    default:
      FOUR_C_THROW("Option for initial field not implemented: %d", init);
      break;
  }  // switch(init)

}  // SetInitialField


/*----------------------------------------------------------------------*
 | calculate intermediate solution                        hoermann 09/15|
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::compute_intermediate_values()
{
  // time derivatives are not independent but rather have to be computed
  // from phinp_, phin_ and phidtn_
  gen_alpha_compute_time_derivative();
  // compute values at intermediate time steps
  gen_alpha_intermediate_values();

  return;
}

/*----------------------------------------------------------------------*
 | compute values at the interior of the elements         hoermann 09/15|
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::compute_interior_values()
{
  // Update the interior variables
  update_interior_variables(intphinp_);
  return;
}

/*----------------------------------------------------------------------*
 | update time derivative for gen-alpha time integration hoermann 09/15 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::gen_alpha_compute_time_derivative()
{
  //                                  n+1     n
  //                               phi   - phi
  //       n+1      n  gamma-1.0      (i)
  // phidt    = phidt * --------- + ------------
  //       (i)           gamma      gamma * dt
  //

  // compute factors
  const double fact1 = 1.0 / (gamma_ * dta_);
  const double fact2 = 1.0 - (1.0 / gamma_);

  phidtnp_->Update(fact2, *phidtn_, 0.0);
  phidtnp_->Update(fact1, *phinp_, -fact1, *phin_, 1.0);

}  // gen_alpha_compute_time_derivative


/*----------------------------------------------------------------------*
 | update interior variables                             hoermann 09/15 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::update_interior_variables(Teuchos::RCP<Epetra_Vector> updatevector)
{
  discret_->ClearState(true);
  Teuchos::ParameterList eleparams;
  Core::UTILS::AddEnumClassToParameterList<ScaTra::Action>(
      "action", ScaTra::Action::update_interior_variables, eleparams);
  discret_->set_state("phiaf", phinp_);
  discret_->set_state("phin", phin_);
  discret_->set_state(nds_intvar_, "intphin", intphin_);
  discret_->set_state(nds_intvar_, "intphinp", intphinp_);

  Core::LinAlg::SerialDenseMatrix dummyMat;
  Core::LinAlg::SerialDenseVector dummyVec;
  Core::LinAlg::SerialDenseVector updateVec;
  Core::Elements::Element::LocationArray la(discret_->NumDofSets());
  const Epetra_Map *intdofrowmap = discret_->dof_row_map(nds_intvar_);

  for (int iele = 0; iele < discret_->NumMyColElements(); ++iele)
  {
    Core::Elements::Element *ele = discret_->lColElement(iele);
    if (ele->Owner() != discret_->Comm().MyPID()) continue;

    ele->LocationVector(*discret_, la, false);
    updateVec.size(discret_->NumDof(nds_intvar_, ele));

    ele->evaluate(eleparams, *discret_, la, dummyMat, dummyMat, updateVec, dummyVec, dummyVec);

    std::vector<int> localDofs = discret_->Dof(nds_intvar_, ele);
    FOUR_C_ASSERT(
        localDofs.size() == static_cast<std::size_t>(updateVec.numRows()), "Internal error");
    for (unsigned int i = 0; i < localDofs.size(); ++i)
    {
      localDofs[i] = intdofrowmap->LID(localDofs[i]);
    }
    updatevector->ReplaceMyValues(localDofs.size(), updateVec.values(), localDofs.data());
  }

  discret_->ClearState(true);
}


/*-------------------------------------------------------------------------------------*
 | finite difference check for system matrix (for debugging only)       hoermann 09/15 |
 *-------------------------------------------------------------------------------------*/
void ScaTra::TimIntHDG::fd_check()
{
  // make a copy of state variables to undo perturbations later
  Teuchos::RCP<Epetra_Vector> phinp_original = Teuchos::rcp(new Epetra_Vector(*phinp_));

  discret_->ClearState(true);

  const Epetra_Map *dofrowmap = discret_->dof_row_map(0);
  const Epetra_Map *intdofrowmap = discret_->dof_row_map(nds_intvar_);

  Teuchos::RCP<Core::LinAlg::SparseOperator> systemmatrix1, systemmatrix2;
  Teuchos::RCP<Epetra_Vector> systemvector1, systemvector2, systemvector3;

  // create matrix and vector for calculation of sysmat and assemble
  systemmatrix1 = Teuchos::rcp(new Core::LinAlg::SparseMatrix(*(discret_->dof_row_map()), 27));
  systemvector1 = Core::LinAlg::CreateVector(*dofrowmap, true);
  Core::FE::AssembleStrategy strategy(
      0, 0, systemmatrix1, systemmatrix2, systemvector1, systemvector2, systemvector3);

  // fill state vector with original state variables
  phinp_->Update(1., *phinp_original, 0.);

  // make temporary vector for interior variables for the update of the last step and use this
  // vector for the calculation of the original residual the temporary vector is necessary because
  // afterwards we need to calculate also the interior vectors with the state vector with
  // perturbation without influence for this update
  Teuchos::RCP<Epetra_Vector> intphitemp;
  intphitemp = Core::LinAlg::CreateVector(*intdofrowmap, true);

  strategy.Zero();

  // calculate of residual vector
  update_interior_variables(intphitemp);

  discret_->ClearState(true);
  discret_->set_state("phiaf", phinp_);
  discret_->set_state(nds_intvar_, "intphin", intphin_);
  discret_->set_state(0, "phin", phin_);
  discret_->set_state(nds_intvar_, "intphinp", intphitemp);
  Teuchos::ParameterList eleparams;
  Core::UTILS::AddEnumClassToParameterList<ScaTra::Action>(
      "action", ScaTra::Action::calc_mat_and_rhs, eleparams);
  Core::Elements::Element::LocationArray la(discret_->NumDofSets());

  // loop over elements
  for (int iele = 0; iele < discret_->NumMyColElements(); ++iele)
  {
    Core::Elements::Element *ele = discret_->lColElement(iele);
    ele->LocationVector(*discret_, la, false);

    strategy.ClearElementStorage(la[0].Size(), la[0].Size());

    // evaluate
    ele->evaluate(eleparams, *discret_, la, strategy.Elematrix1(), strategy.Elematrix2(),
        strategy.Elevector1(), strategy.Elevector2(), strategy.Elevector3());
    int eid = ele->Id();
    strategy.AssembleMatrix1(eid, la[0].lm_, la[0].lm_, la[0].lmowner_, la[0].stride_);
    strategy.AssembleVector1(la[0].lm_, la[0].lmowner_);
  }
  strategy.Complete();

  // make a copy of system matrix as Epetra_CrsMatrix
  Teuchos::RCP<Epetra_CrsMatrix> sysmatcopy = Teuchos::null;
  sysmatcopy = (new Core::LinAlg::SparseMatrix(
                    *(Teuchos::rcp_static_cast<Core::LinAlg::SparseMatrix>(systemmatrix1))))
                   ->EpetraMatrix();
  sysmatcopy->FillComplete();

  // make a copy of system right-hand side vector
  Teuchos::RCP<Epetra_Vector> residualVec = Teuchos::rcp(new Epetra_Vector(*systemvector1));
  Teuchos::RCP<Epetra_Vector> fdvec = Core::LinAlg::CreateVector(*dofrowmap, true);

  for (int k = 0; k < 16; ++k)
  {
    double eps = 1000;
    for (int j = 0; j < k; ++j) eps *= 0.1;

    // initialize tracking variable for maximum absolute and relative errors
    double maxabserr(0.);
    double maxrelerr(0.);

    // calculate fd matrix
    for (int colgid = 0; colgid <= sysmatcopy->ColMap().MaxAllGID(); ++colgid)
    {
      // check whether current column index is a valid global column index and continue loop if not
      int collid(sysmatcopy->ColMap().LID(colgid));
      int maxcollid(-1);
      discret_->Comm().MaxAll(&collid, &maxcollid, 1);
      if (maxcollid < 0) continue;

      strategy.Zero();

      // fill state vector with original state variables
      phinp_->Update(1., *phinp_original, 0.);

      // impose perturbation and update interior variables
      if (phinp_->Map().MyGID(colgid))
        if (phinp_->SumIntoGlobalValue(colgid, 0, eps))
          FOUR_C_THROW(
              "Perturbation could not be imposed on state vector for finite difference check!");
      update_interior_variables(intphitemp);

      discret_->ClearState(true);

      discret_->set_state("phiaf", phinp_);
      discret_->set_state(nds_intvar_, "intphin", intphin_);
      discret_->set_state(0, "phin", phin_);
      discret_->set_state(nds_intvar_, "intphinp", intphitemp);

      Teuchos::ParameterList eleparams;
      Core::UTILS::AddEnumClassToParameterList<ScaTra::Action>(
          "action", ScaTra::Action::calc_mat_and_rhs, eleparams);

      Core::Elements::Element::LocationArray la(discret_->NumDofSets());

      for (int iele = 0; iele < discret_->NumMyColElements(); ++iele)
      {
        Core::Elements::Element *ele = discret_->lColElement(iele);
        ele->LocationVector(*discret_, la, false);

        strategy.ClearElementStorage(la[0].Size(), la[0].Size());
        ele->evaluate(eleparams, *discret_, la, strategy.Elematrix1(), strategy.Elematrix2(),
            strategy.Elevector1(), strategy.Elevector2(), strategy.Elevector3());
        int eid = ele->Id();
        strategy.AssembleMatrix1(eid, la[0].lm_, la[0].lm_, la[0].lmowner_, la[0].stride_);
        strategy.AssembleVector1(la[0].lm_, la[0].lmowner_);
      }
      strategy.Complete();
      fdvec->PutScalar(0.0);

      // finite difference suggestion (first divide by epsilon and then subtract for better
      // conditioning)
      for (int j = 0; j < phinp_->MyLength(); ++j)
        (*fdvec)[j] = -(*systemvector1)[j] / eps + (*residualVec)[j] / eps;

      for (int rowlid = 0; rowlid < discret_->dof_row_map()->NumMyElements(); ++rowlid)
      {
        // get global index of current matrix row
        const int rowgid = sysmatcopy->RowMap().GID(rowlid);
        if (rowgid < 0) FOUR_C_THROW("Invalid global ID of matrix row!");

        // get current entry in original system matrix
        double entry(0.);
        int length = sysmatcopy->NumMyEntries(rowlid);
        int numentries;
        std::vector<double> values(length);
        std::vector<int> indices(length);
        sysmatcopy->ExtractMyRowCopy(rowlid, length, numentries, values.data(), indices.data());

        for (int ientry = 0; ientry < length; ++ientry)
        {
          if (sysmatcopy->ColMap().GID(indices[ientry]) == colgid)
          {
            entry = values[ientry];
            break;
          }
        }

        // absolute and relative errors in first comparison
        const double abserr1 = entry - (*fdvec)[rowlid];
        double relerr1 = 0;
        if (abs(entry) > 1.e-17)
          relerr1 = abserr1 / abs(entry);
        else if (abs((*fdvec)[rowlid]) > 1.e-17)
          relerr1 = abserr1 / abs((*fdvec)[rowlid]);
        // store max abs and rel error
        if (abs(abserr1) > maxabserr) maxabserr = abs(abserr1);
        if (abs(relerr1) > maxrelerr) maxrelerr = abs(relerr1);
      }
    }
    // end calculate fd matrix

    // screen output
    if (myrank_ == 0)
    {
      std::cout << std::endl << "FINITE DIFFERENCE CHECK FOR SCATRA HDG SYSTEM MATRIX" << std::endl;
      std::cout << "EPS:        " << eps << std::endl;
      std::cout << "ABSOLUT: " << maxabserr << std::endl;
      std::cout << "RELATIVE: " << maxrelerr << std::endl;
    }
  }
  FOUR_C_THROW("FD check END");
}

/*----------------------------------------------------------------------------------*
 | compute relative error with reference to analytical solution    berardocco 05/20 |
 *----------------------------------------------------------------------------------*/
void ScaTra::TimIntHDG::evaluate_error_compared_to_analytical_sol()
{
  switch (calcerror_)
  {
    case Inpar::ScaTra::calcerror_byfunction:
    case Inpar::ScaTra::calcerror_spherediffusion:
    {
      Teuchos::RCP<Core::LinAlg::SerialDenseVector> errors = compute_error();
      if (errors == Teuchos::null)
        FOUR_C_THROW("It was not possible to compute error. Check the error function number.");

      if (std::abs((*errors)[1]) > 1e-14)
        (*relerrors_)[0] = std::sqrt((*errors)[0]) / std::sqrt((*errors)[1]);
      else
        FOUR_C_THROW(
            "Can't compute scalar's relative L2 error due to numerical roundoff sensitivity!");
      if (std::abs((*errors)[3]) > 1e-14)
        (*relerrors_)[1] = std::sqrt((*errors)[2]) / std::sqrt((*errors)[3]);
      else
        FOUR_C_THROW(
            "Can't compute grandient's relative L2 error due to numerical roundoff sensitivity!");

      if (myrank_ == 0)
      {
        // print last error in a separate file
        const std::string simulation = problem_->OutputControlFile()->file_name();
        const std::string fname = simulation + "_time.relerror";
        std::ofstream f;

        // create new error file and write initial error
        if (step_ == 0)
        {
          f.open(fname.c_str());
          f << "| Step | Time | abs. L2-error (phi) | rel. L2-error (phi)| abs. L2-error "
               "(gradPhi)| rel. L2-error (gradPhi)|"
            << std::endl;
        }

        // append error of the last time step to the error file
        else
        {
          f.open(fname.c_str(), std::fstream::ate | std::fstream::app);

          f << step_ << " " << time_ << " " << std::setprecision(6) << std::sqrt((*errors)[0])
            << " " << (*relerrors_)[0] << " " << std::sqrt((*errors)[2]) << " " << (*relerrors_)[1]
            << std::endl;
        }

        f.flush();
        f.close();
      }

      break;
    }
    case Inpar::ScaTra::calcerror_no:
    {
      // do nothing
      break;
    }

    default:
    {
      FOUR_C_THROW("Cannot calculate error. Unknown type of analytical test problem!");
      break;
    }
  }

  return;
}  // ScaTra::TimIntHDG::evaluate_error_compared_to_analytical_sol

/*----------------------------------------------------------------------------------*
 | compute relative error with reference to analytical solution    berardocco 08/20 |
 *----------------------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::SerialDenseVector> ScaTra::TimIntHDG::compute_error() const
{
  // If we are here it means that we either arrived at the end and we are checking the results or
  // that we specified that we want to compute the error. In any case, if the error function was not
  // specified, we just return a null pointer.
  const int errorfunctnumber = params_->get<int>("CALCERRORNO", -1);
  if (errorfunctnumber < 1) return Teuchos::null;

  // create the parameters for the error calculation
  Teuchos::ParameterList eleparams;
  Core::UTILS::AddEnumClassToParameterList<ScaTra::Action>(
      "action", ScaTra::Action::calc_error, eleparams);

  eleparams.set<int>("error function number", errorfunctnumber);
  eleparams.set<double>("time", time_);

  // set vector values needed by elements
  discret_->ClearState();
  discret_->set_state("phiaf", phinp_);
  discret_->set_state(nds_intvar_, "intphinp", intphinp_);
  // get (squared) error values
  // The error is computed for the transported scalar and its gradient. Notice that so far only
  // the L2 error is computed, feel free to extend the calculations to any error measure needed
  unsigned int NumErrorEntries = 4;
  Teuchos::RCP<Core::LinAlg::SerialDenseVector> errors =
      Teuchos::rcp(new Core::LinAlg::SerialDenseVector(NumErrorEntries));

  discret_->EvaluateScalars(eleparams, errors);
  discret_->ClearState();

  return errors;
}

/*----------------------------------------------------------------------*
 | prepare time loop                                     hoermann 09/15 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::prepare_time_loop()
{
  // calculate matrices on element
  calc_mat_initial();

  // call base class routine
  ScaTraTimIntImpl::prepare_time_loop();

}  // ScaTra::TimIntHDG::prepare_time_loop


/*----------------------------------------------------------------------*
 | calculate matrices on element                        hoermann 07/16 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::calc_mat_initial()
{
  TEUCHOS_FUNC_TIME_MONITOR("ScaTra::TimIntHDG::CalcMat");

  discret_->ClearState(true);

  // check validity of material and element formulation
  Teuchos::ParameterList eleparams;
  Core::UTILS::AddEnumClassToParameterList<ScaTra::Action>(
      "action", ScaTra::Action::calc_mat_initial, eleparams);

  discret_->set_state("phiaf", phinp_);
  discret_->set_state("phin", phin_);
  discret_->set_state(nds_intvar_, "intphin", intphin_);
  discret_->set_state(nds_intvar_, "intphinp", intphinp_);

  Teuchos::RCP<Core::LinAlg::SparseOperator> systemmatrix1, systemmatrix2;
  Teuchos::RCP<Epetra_Vector> systemvector1, systemvector2, systemvector3;

  // create matrix and vector for calculation of sysmat and assemble
  systemmatrix1 = Teuchos::rcp(new Core::LinAlg::SparseMatrix(*(discret_->dof_row_map()), 27));
  Core::FE::AssembleStrategy strategy(
      0, 0, sysmat_, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null);

  strategy.Zero();
  Core::Elements::Element::LocationArray la(discret_->NumDofSets());

  //    // get cpu time
  //    const double tcmatinit = Teuchos::Time::wallTime();

  // loop over elements
  for (int iele = 0; iele < discret_->NumMyColElements(); ++iele)
  {
    Core::Elements::Element *ele = discret_->lColElement(iele);

    // if the element has only ghosted nodes it will not assemble -> skip evaluation
    if (ele->HasOnlyGhostNodes(discret_->Comm().MyPID())) continue;
    ele->LocationVector(*discret_, la, false);

    strategy.ClearElementStorage(la[0].Size(), la[0].Size());

    // evaluate
    int err = ele->evaluate(eleparams, *discret_, la, strategy.Elematrix1(), strategy.Elematrix2(),
        strategy.Elevector1(), strategy.Elevector2(), strategy.Elevector3());
    if (err)
      FOUR_C_THROW("Proc %d: Element %d returned err=%d", discret_->Comm().MyPID(), ele->Id(), err);

    int eid = ele->Id();
    strategy.AssembleMatrix1(eid, la[0].lm_, la[0].lm_, la[0].lmowner_, la[0].stride_);
  }
  sysmat_->Complete();

  //    // end time measurement for element
  //    double dtmatinit=Teuchos::Time::wallTime()-tcmatinit;
  //    std::cout << "Time measurement evaluate: " << dtmatinit << std::endl;

  // Output of non-zeros in system matrix
  if (step_ == 0 and discret_->Comm().MyPID() == 0)
  {
    int numglobalnonzeros = SystemMatrix()->EpetraMatrix()->NumGlobalNonzeros();
    std::cout << "Number of non-zeros in system matrix: " << numglobalnonzeros << std::endl;
  }

  return;
}  // ScaTra::TimIntHDG::CalcMatIntitial


/*----------------------------------------------------------------------*
 | adapt degree of test function on element               hoermann 07/16|
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::adapt_degree()
{
  // time measurement: element calls
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + adapt degree");

  // get cpu time
  const double tcadapt = Teuchos::Time::wallTime();

  // cast and check if hdg discretization is provided
  Core::FE::DiscretizationHDG *hdgdis = dynamic_cast<Core::FE::DiscretizationHDG *>(discret_.get());
  if (hdgdis == nullptr) FOUR_C_THROW("Did not receive an HDG discretization");

  // vector to store the dofs per single element
  const Teuchos::RCP<Epetra_IntVector> eledofs =
      Teuchos::rcp(new Epetra_IntVector(*discret_->ElementColMap()));

  // vector to store the location array of the dofsets before the adaption with the new order
  std::vector<Core::Elements::Element::LocationArray> la_old;

  // copy the old face dof map and the old interior element dof map
  Teuchos::RCP<Epetra_Map> facedofs_old = Teuchos::rcp(new Epetra_Map(*discret_->DofColMap(0)));
  Teuchos::RCP<Epetra_Map> eledofs_old =
      Teuchos::rcp(new Epetra_Map(*discret_->DofColMap(nds_intvar_)));

  // set action
  Teuchos::ParameterList eleparams;
  Core::UTILS::AddEnumClassToParameterList<ScaTra::Action>(
      "action", ScaTra::Action::calc_padaptivity, eleparams);

  Core::LinAlg::SerialDenseMatrix dummyMat;
  Core::LinAlg::SerialDenseVector dummyVec;

  discret_->set_state("phiaf", phinp_);
  discret_->set_state(nds_intvar_, "intphinp", intphinp_);

  // get cpu time
  //  const double tccalcerr = Teuchos::Time::wallTime();


  // store if degree changes
  int degchange(0);

  // loop over elements
  for (int iele = 0; iele < discret_->NumMyColElements(); ++iele)
  {
    // add new location array in vector for each element
    la_old.push_back(Core::Elements::Element::LocationArray(discret_->NumDofSets()));

    Core::Elements::Element *ele = discret_->lColElement(iele);

    // fill location array and store it for later use
    ele->LocationVector(*discret_, la_old[iele], false);

    if (ele->Owner() == discret_->Comm().MyPID())
    {
      Discret::ELEMENTS::ScaTraHDG *hdgele =
          dynamic_cast<Discret::ELEMENTS::ScaTraHDG *>(discret_->lColElement(iele));

      // call routine on elements to calculate error on element
      ele->evaluate(
          eleparams, *discret_, la_old[iele], dummyMat, dummyMat, dummyVec, dummyVec, dummyVec);

      double error = eleparams.get<double>("error");
      double errorlog = 0;

      if (error < 0) FOUR_C_THROW("Error is negative!");

      if (error > 0)
        errorlog = log(error / padapterrortol_);
      else
        errorlog = 0.;

      int deg = hdgele->Degree() + ceil(errorlog / padapterrorbase_);

      if (deg < 0)
        deg = 0;
      else if (deg > padaptdegreemax_)
        deg = padaptdegreemax_;

      if (hdgele->Degree() != deg) degchange = 1;

      // set degree on element
      hdgele->SetDegree(deg);

      // store element degree (only for output)
      const int eleIndex = discret_->ElementRowMap()->LID(ele->Id());
      if (eleIndex >= 0) (*elementdegree_)[eleIndex] = deg;
    }
  }

  int degchangeall;
  discret_->Comm().SumAll(&degchange, &degchangeall, 1);

  if (!degchangeall) return;

  pack_material();

  //  // end time measurement for element
  //  double dtcalcerr=Teuchos::Time::wallTime()-tccalcerr;
  //  std::cout << "Time measurement for error calculation: " << dtcalcerr << std::endl;

  //  // get cpu time
  //  const double tcfillcomplete = Teuchos::Time::wallTime();

  // number of dofset in location array
  int nds_intvar_old(discret_->NumDofSets() + 2);
  int nds_var_old(discret_->NumDofSets());

  // build new maps for face dofs with adapted element order
  hdgdis_->BuildFaces();
  hdgdis_->BuildFaceRowMap();
  hdgdis_->BuildFaceColMap();

  // assign the degrees of freedom to the adapted dofsets
  hdgdis_->assign_degrees_of_freedom(0);

  // replace all ghosted element with the original thus the correct polynomial degree is used
  discret_->export_column_elements(*discret_->ElementColMap(), false, false);

  hdgdis_->fill_complete();

  // store the number of dofs per element on vector
  for (int iele = 0; iele < discret_->NumMyColElements(); ++iele)
  {
    Discret::ELEMENTS::ScaTraHDG *hdgele =
        dynamic_cast<Discret::ELEMENTS::ScaTraHDG *>(discret_->lColElement(iele));
    // store the number of dofs for the element
    (*eledofs)[iele] = hdgele->num_dof_per_element_auxiliary();
  }

  // create new local dofset for the new interior element dofs with adapted element order
  Teuchos::RCP<Core::DOFSets::DofSetPredefinedDoFNumber> eledofs_new =
      Teuchos::rcp(new Core::DOFSets::DofSetPredefinedDoFNumber(0, eledofs, 0, false));
  // replace old interior element dofs with the new created dofset
  discret_->ReplaceDofSet(nds_intvar_, eledofs_new, false);

  hdgdis_->assign_degrees_of_freedom(0);

  // clear map cache since after every fill_complete() / assign_degrees_of_freedom() old maps are
  // stored in the mapstack
  output_->clear_map_cache();

  // copy old values of the state vectors phi and intphi into vectors, which are then used for the
  // projection
  Teuchos::RCP<Epetra_Vector> phinp_old = Core::LinAlg::CreateVector(*facedofs_old, true);
  Core::LinAlg::Export(*phinp_, *phinp_old);

  Teuchos::RCP<Epetra_Vector> intphinp_old = Core::LinAlg::CreateVector(*eledofs_old, true);
  Core::LinAlg::Export(*intphinp_, *intphinp_old);

  // reset the residual, increment and sysmat to the size of the adapted new dofset
  residual_.reset(new Epetra_Vector(*(discret_->dof_row_map())));
  increment_.reset(new Epetra_Vector(*(discret_->dof_row_map())));
  neumann_loads_.reset(new Epetra_Vector(*(discret_->dof_row_map())));
  sysmat_ = Teuchos::null;
  sysmat_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(*(discret_->dof_row_map()), 27));

  // reset the state vectors
  intphinp_.reset(new Epetra_Vector(*(discret_->dof_row_map(nds_intvar_)), true));
  intphin_.reset(new Epetra_Vector(*(discret_->dof_row_map(nds_intvar_)), true));
  phinp_.reset(new Epetra_Vector(*(discret_->dof_row_map())));
  phin_.reset(new Epetra_Vector(*(discret_->dof_row_map())));

  //  // end time measurement for element
  //  double dtfillcomplete=Teuchos::Time::wallTime()-tcfillcomplete;
  //  std::cout << "Time measurement fill complete: " << dtfillcomplete << std::endl;

  //  // get cpu time
  //  const double tcproject = Teuchos::Time::wallTime();

  // unpack material data
  unpack_material();

  adapt_variable_vector(
      phinp_, phinp_old, intphinp_, intphinp_old, nds_var_old, nds_intvar_old, la_old);

  intphin_->Update(1.0, *intphinp_, 0.0);
  phin_->Update(1.0, *phinp_, 0.0);

  project_material();

  //  // end time measurement for element
  //  double dtproject=Teuchos::Time::wallTime()-tcproject;
  //  std::cout << "Time measurement for projection: " << dtproject << std::endl;
  //
  //  // get cpu time
  //  const double tcmatinit = Teuchos::Time::wallTime();

  calc_mat_initial();

  //  // end time measurement for element
  //  double dtmatinit=Teuchos::Time::wallTime()-tcmatinit;
  //  std::cout << "Time measurement calc mat initial: " << dtmatinit << std::endl;

  // end time measurement for element
  double dtadapt = Teuchos::Time::wallTime() - tcadapt;

  if (myrank_ == 0)
    std::cout << "Time measurement for adaption of element degree: " << dtadapt << std::endl;


  return;
}

/*----------------------------------------------------------------------*
 | adapt trace vector and interior variables when adapting element      |
 | degrees                                                hoermann 07/16|
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::adapt_variable_vector(Teuchos::RCP<Epetra_Vector> phi_new,
    Teuchos::RCP<Epetra_Vector> phi_old, Teuchos::RCP<Epetra_Vector> intphi_new,
    Teuchos::RCP<Epetra_Vector> intphi_old, int nds_var_old, int nds_intvar_old,
    std::vector<Core::Elements::Element::LocationArray> la_old)
{
  // set action
  Teuchos::ParameterList eleparams;
  Core::UTILS::AddEnumClassToParameterList<ScaTra::Action>(
      "action", ScaTra::Action::project_field, eleparams);

  // set number of dofset for the old dofsets on the parameter list to extract the correct location
  // array
  eleparams.set<int>("nds_var_old", nds_var_old);
  eleparams.set<int>("nds_intvar_old", nds_intvar_old);

  // dof row map for adapted dofset
  const Epetra_Map *intdofrowmap = discret_->dof_row_map(nds_intvar_);
  const Epetra_Map *dofrowmap = discret_->dof_row_map(0);


  // set old state vector on parameter list
  eleparams.set<Teuchos::RCP<Epetra_Vector>>("phi", phi_old);
  eleparams.set<Teuchos::RCP<Epetra_Vector>>("intphi", intphi_old);

  Core::LinAlg::SerialDenseMatrix dummyMat;
  Core::LinAlg::SerialDenseVector intphi_ele, phi_ele, dummyVec;

  // create location array for new and old dofsets (old ones are already filled and only copied to
  // the location array)
  Core::Elements::Element::LocationArray la(2 * discret_->NumDofSets());
  // create location array for new dofsets
  Core::Elements::Element::LocationArray la_temp(discret_->NumDofSets());

  for (int iele = 0; iele < discret_->NumMyColElements(); ++iele)
  {
    Core::Elements::Element *ele = discret_->lColElement(iele);

    // if the element has only ghosted nodes it will not assemble -> skip evaluation
    if (ele->HasOnlyGhostNodes(discret_->Comm().MyPID())) continue;

    // fill location array for adapted dofsets
    ele->LocationVector(*discret_, la_temp, false);

    for (int i = 0; i < discret_->NumDofSets(); i++)
    {
      // copy old and new location arrays to global location array la
      la[i] = la_temp[i];
      la[discret_->NumDofSets() + i] = la_old[iele][i];
    }

    const unsigned size = la_temp[0].lm_.size();

    if (static_cast<std::size_t>(phi_ele.numRows()) != size)
      phi_ele.size(la[0].lm_.size());
    else
      phi_ele.putScalar(0.0);
    if (intphi_ele.numRows() != discret_->NumDof(nds_intvar_, ele))
      intphi_ele.size(discret_->NumDof(nds_intvar_, ele));
    else
      intphi_ele.putScalar(0.0);

    // call routine on elements to project values from old to new element vector
    ele->evaluate(eleparams, *discret_, la, dummyMat, dummyMat, phi_ele, intphi_ele, dummyVec);

    // store projected values of the element on the new state vector for the interior variables
    if (ele->Owner() == discret_->Comm().MyPID())
    {
      std::vector<int> localDofs = discret_->Dof(nds_intvar_, ele);
      FOUR_C_ASSERT(
          localDofs.size() == static_cast<std::size_t>(intphi_ele.numRows()), "Internal error");
      for (unsigned int i = 0; i < localDofs.size(); ++i)
        localDofs[i] = intdofrowmap->LID(localDofs[i]);
      (intphi_new)->ReplaceMyValues(localDofs.size(), intphi_ele.values(), localDofs.data());
    }

    // now fill the element vector into the new state vector for the trace values
    for (unsigned int i = 0; i < la[0].lm_.size(); ++i)
    {
      const int lid = dofrowmap->LID(la[0].lm_[i]);

      if (lid >= 0) (*phi_new)[lid] = phi_ele(i);
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 | chooses the assembly process for matrix and rhs       hoermann 06/16 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::assemble_mat_and_rhs()
{
  if (!Core::UTILS::IntegralValue<int>(*params_, "SEMIIMPLICIT"))
    ScaTra::ScaTraTimIntImpl::assemble_mat_and_rhs();
  else  // in semi-implicit evaluation matrix does not change, thus only rhs is assembled in every
        // step
    assemble_rhs();

  return;
}  // TimIntHDG::assemble_mat_and_rhs

/*----------------------------------------------------------------------*
 | contains the assembly process only for rhs            hoermann 06/16 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntHDG::assemble_rhs()
{
  // time measurement: element calls
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + element calls");

  // get cpu time
  const double tcpuele = Teuchos::Time::wallTime();

  // reset the residual vector
  residual_->PutScalar(0.0);

  // create parameter list for elements
  Teuchos::ParameterList eleparams;

  // action for elements
  Core::UTILS::AddEnumClassToParameterList<ScaTra::Action>(
      "action", ScaTra::Action::calc_mat_and_rhs, eleparams);

  // set vector values needed by elements
  discret_->ClearState();

  // add state vectors according to time-integration scheme
  add_time_integration_specific_vectors();

  // add problem specific time-integration parameters
  add_problem_specific_parameters_and_vectors(eleparams);

  Core::FE::AssembleStrategy strategy(
      0, 0, Teuchos::null, Teuchos::null, residual_, Teuchos::null, Teuchos::null);

  strategy.Zero();

  Core::Elements::Element::LocationArray la(discret_->NumDofSets());

  // loop over elements
  for (int iele = 0; iele < discret_->NumMyColElements(); ++iele)
  {
    Core::Elements::Element *ele = discret_->lColElement(iele);

    // if the element has only ghosted nodes it will not assemble -> skip evaluation
    if (ele->HasOnlyGhostNodes(discret_->Comm().MyPID())) continue;

    ele->LocationVector(*discret_, la, false);

    strategy.ClearElementStorage(la[0].Size(), la[0].Size());

    // evaluate
    ele->evaluate(eleparams, *discret_, la, strategy.Elematrix1(), strategy.Elematrix2(),
        strategy.Elevector1(), strategy.Elevector2(), strategy.Elevector3());
    strategy.AssembleVector1(la[0].lm_, la[0].lmowner_);
  }

  discret_->ClearState();

  // potential residual scaling and potential addition of Neumann terms
  scaling_and_neumann();

  // end time measurement for element
  dtele_ = Teuchos::Time::wallTime() - tcpuele;

  return;
}  // TimIntHDG::AssembleRHS

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::UTILS::ResultTest> ScaTra::TimIntHDG::create_sca_tra_field_test()
{
  return Teuchos::rcp(new ScaTra::HDGResultTest(Teuchos::rcp(this, false)));
}

FOUR_C_NAMESPACE_CLOSE