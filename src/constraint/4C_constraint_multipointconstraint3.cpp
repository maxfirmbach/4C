/*----------------------------------------------------------------------*/
/*! \file
\brief Basic constraint class, dealing with multi point constraints
\level 2


*----------------------------------------------------------------------*/



#include "4C_constraint_multipointconstraint3.hpp"

#include "4C_constraint_element3.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_dofset_transparent.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_rebalance_binning_based.hpp"
#include "4C_utils_function_of_time.hpp"

#include <iostream>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor (public)                                               tk 07/08|
 *----------------------------------------------------------------------*/
CONSTRAINTS::MPConstraint3::MPConstraint3(Teuchos::RCP<Core::FE::Discretization> discr,
    const std::string& conditionname, int& offsetID, int& maxID)
    : MPConstraint(discr, conditionname)
{
  if (constrcond_.size())
  {
    maxID++;
    // control the constraint by absolute or relative values
    for (auto* cond : constrcond_)
    {
      const int condID = cond->parameters().get<int>("ConditionID");
      if (offsetID > maxID) offsetID = maxID;
      const std::string type = cond->parameters().get<std::string>("control");
      if (type == "abs")
        absconstraint_[condID] = true;
      else
        absconstraint_[condID] = false;
    }

    constraintdis_ = create_discretization_from_condition(
        actdisc_, constrcond_, "ConstrDisc", "CONSTRELE3", maxID);

    std::map<int, Teuchos::RCP<Core::FE::Discretization>>::iterator discriter;
    for (discriter = constraintdis_.begin(); discriter != constraintdis_.end(); discriter++)
    {
      // ReplaceNumDof(actdisc_,discriter->second);
      Teuchos::RCP<Epetra_Map> newcolnodemap =
          Core::Rebalance::ComputeNodeColMap(actdisc_, discriter->second);
      actdisc_->Redistribute(*(actdisc_->NodeRowMap()), *newcolnodemap);
      Teuchos::RCP<Core::DOFSets::DofSet> newdofset =
          Teuchos::rcp(new Core::DOFSets::TransparentDofSet(actdisc_));
      (discriter->second)->ReplaceDofSet(newdofset);
      newdofset = Teuchos::null;
      (discriter->second)->fill_complete();
    }
  }
}

/*------------------------------------------------------------------------*
|(public)                                                       tk 08/08  |
|Initialization routine activates conditions (restart)                    |
*------------------------------------------------------------------------*/
void CONSTRAINTS::MPConstraint3::initialize(const double& time)
{
  for (auto* cond : constrcond_)
  {
    // Get ConditionID of current condition if defined and write value in parameterlist
    int condID = cond->parameters().get<int>("ConditionID");

    // if current time (at) is larger than activation time of the condition, activate it
    if ((inittimes_.find(condID)->second < time) && (!activecons_.find(condID)->second))
    {
      activecons_.find(condID)->second = true;
      if (actdisc_->Comm().MyPID() == 0)
      {
        std::cout << "Encountered another active condition (Id = " << condID
                  << ")  for restart time t = " << time << std::endl;
      }
    }
  }
}

/*-----------------------------------------------------------------------*
|(public)                                                        tk 07/08|
|Evaluate Constraints, choose the right action based on type             |
*-----------------------------------------------------------------------*/
void CONSTRAINTS::MPConstraint3::initialize(
    Teuchos::ParameterList& params, Teuchos::RCP<Epetra_Vector> systemvector)
{
  const double time = params.get("total time", -1.0);
  // in case init is set to true we want to set systemvector1 to the amplitudes defined
  // in the input file
  // allocate vectors for amplitudes and IDs

  std::vector<double> amplit(constrcond_.size());
  std::vector<int> IDs(constrcond_.size());
  // read data of the input files

  for (unsigned int i = 0; i < constrcond_.size(); i++)
  {
    Core::Conditions::Condition& cond = *(constrcond_[i]);

    int condID = cond.parameters().get<int>("ConditionID");
    if (inittimes_.find(condID)->second <= time && (!(activecons_.find(condID)->second)))
    {
      // control absolute values
      if (absconstraint_.find(condID)->second)
      {
        // in case of a mpcnormalcomp3d-condition amplitude is always 0
        //        if (Type()==mpcnormalcomp3d)
        //          amplit[i]=0.0;
        //        else
        //        {
        double MPCampl = constrcond_[i]->parameters().get<double>("amplitude");
        amplit[i] = MPCampl;
        //        }
        const int mid = params.get("OffsetID", 0);
        IDs[i] = condID - mid;
      }
      // control relative values
      else
      {
        switch (Type())
        {
          case mpcnormalcomp3d:
          case mpcnodeonplane3d:
            params.set("action", "calc_MPC_state");
            break;
          case none:
            return;
          default:
            FOUR_C_THROW("Constraint is not an multi point constraint!");
        }
        initialize_constraint(constraintdis_.find(condID)->second, params, systemvector);
      }
      activecons_.find(condID)->second = true;
      if (actdisc_->Comm().MyPID() == 0)
      {
        std::cout << "Encountered a new active condition (Id = " << condID
                  << ")  at time t = " << time << std::endl;
      }
    }
  }

  if (actdisc_->Comm().MyPID() == 0)
    systemvector->SumIntoGlobalValues(amplit.size(), amplit.data(), IDs.data());

  return;
}

/*-----------------------------------------------------------------------*
|(public)                                                        tk 07/08|
|Evaluate Constraints, choose the right action based on type             |
*-----------------------------------------------------------------------*/
void CONSTRAINTS::MPConstraint3::evaluate(Teuchos::ParameterList& params,
    Teuchos::RCP<Core::LinAlg::SparseOperator> systemmatrix1,
    Teuchos::RCP<Core::LinAlg::SparseOperator> systemmatrix2,
    Teuchos::RCP<Epetra_Vector> systemvector1, Teuchos::RCP<Epetra_Vector> systemvector2,
    Teuchos::RCP<Epetra_Vector> systemvector3)
{
  switch (Type())
  {
    case mpcnodeonplane3d:
    case mpcnormalcomp3d:
      params.set("action", "calc_MPC_stiff");
      break;
    case none:
      return;
    default:
      FOUR_C_THROW("Constraint/monitor is not an multi point constraint!");
  }
  std::map<int, Teuchos::RCP<Core::FE::Discretization>>::iterator discriter;
  for (discriter = constraintdis_.begin(); discriter != constraintdis_.end(); discriter++)
    evaluate_constraint(discriter->second, params, systemmatrix1, systemmatrix2, systemvector1,
        systemvector2, systemvector3);

  return;
}

/*------------------------------------------------------------------------*
 |(private)                                                   tk 04/08    |
 |subroutine creating a new discretization containing constraint elements |
 *------------------------------------------------------------------------*/
std::map<int, Teuchos::RCP<Core::FE::Discretization>>
CONSTRAINTS::MPConstraint3::create_discretization_from_condition(
    Teuchos::RCP<Core::FE::Discretization> actdisc,
    std::vector<Core::Conditions::Condition*> constrcondvec, const std::string& discret_name,
    const std::string& element_name, int& startID)
{
  // start with empty map
  std::map<int, Teuchos::RCP<Core::FE::Discretization>> newdiscmap;

  if (!actdisc->Filled())
  {
    actdisc->fill_complete();
  }

  if (constrcondvec.size() == 0)
    FOUR_C_THROW(
        "number of multi point constraint conditions = 0 --> cannot create constraint "
        "discretization");


  // Loop all conditions in constrcondvec and build discretization for any condition ID

  int index = 0;  // counter for the index of condition in vector
  std::vector<Core::Conditions::Condition*>::iterator conditer;
  for (conditer = constrcondvec.begin(); conditer != constrcondvec.end(); conditer++)
  {
    // initialize a new discretization
    Teuchos::RCP<Epetra_Comm> com = Teuchos::rcp(actdisc->Comm().Clone());
    Teuchos::RCP<Core::FE::Discretization> newdis =
        Teuchos::rcp(new Core::FE::Discretization(discret_name, com, actdisc->n_dim()));
    const int myrank = newdis->Comm().MyPID();
    std::set<int> rownodeset;
    std::set<int> colnodeset;
    const Epetra_Map* actnoderowmap = actdisc->NodeRowMap();
    // get node IDs, this vector will only contain FREE nodes in the end
    std::vector<int> ngid = *((*conditer)->GetNodes());
    std::vector<int> defnv;
    switch (Type())
    {
      case mpcnodeonplane3d:
      {
        // take three nodes defining plane as specified by user and put them into a set
        const auto& defnvp = (*conditer)->parameters().get<std::vector<int>>("planeNodes");
        defnv = defnvp;
      }
      break;
      case mpcnormalcomp3d:
      {
        // take master node
        const int defn = (*conditer)->parameters().get<int>("masterNode");
        defnv.push_back(defn);
      }
      break;
      default:
        FOUR_C_THROW("not good!");
    }
    std::set<int> defns(defnv.begin(), defnv.end());
    std::set<int>::iterator nsit;
    // safe gids of definition nodes in a vector
    std::vector<int> defnodeIDs;

    int counter = 1;  // counter is used to keep track of deleted node ids from the vector, input
                      // starts with 1

    for (nsit = defns.begin(); nsit != defns.end(); ++nsit)
    {
      defnodeIDs.push_back(ngid.at((*nsit) - counter));
      ngid.erase(ngid.begin() + ((*nsit) - counter));
      counter++;
    }

    unsigned int nodeiter;
    // loop over all free nodes of condition
    for (nodeiter = 0; nodeiter < ngid.size(); nodeiter++)
    {
      std::vector<int> ngid_ele = defnodeIDs;
      ngid_ele.push_back(ngid[nodeiter]);
      const int numnodes = ngid_ele.size();

      remove_copy_if(ngid_ele.data(), ngid_ele.data() + numnodes,
          inserter(rownodeset, rownodeset.begin()),
          std::not_fn(Core::Conditions::MyGID(actnoderowmap)));
      // copy node ids specified in condition to colnodeset
      copy(ngid_ele.data(), ngid_ele.data() + numnodes, inserter(colnodeset, colnodeset.begin()));

      // construct constraint nodes, which use the same global id as the standard nodes
      for (int i = 0; i < actnoderowmap->NumMyElements(); ++i)
      {
        const int gid = actnoderowmap->GID(i);
        if (rownodeset.find(gid) != rownodeset.end())
        {
          const Core::Nodes::Node* standardnode = actdisc->lRowNode(i);
          newdis->AddNode(Teuchos::rcp(new Core::Nodes::Node(gid, standardnode->X(), myrank)));
        }
      }

      if (myrank == 0)
      {
        Teuchos::RCP<Core::Elements::Element> constraintele =
            Core::Communication::Factory(element_name, "Polynomial", nodeiter + startID, myrank);
        // set the same global node ids to the ale element
        constraintele->SetNodeIds(ngid_ele.size(), ngid_ele.data());
        // add constraint element
        newdis->add_element(constraintele);
      }
      // save the connection between element and condition
      eletocond_id_[nodeiter + startID] = (*conditer)->parameters().get<int>("ConditionID");
      eletocondvecindex_[nodeiter + startID] = index;
    }
    // adjust starting ID for next condition, in this case nodeiter=ngid.size(), hence the counter
    // is larger than the ID
    // of the last element
    startID += nodeiter;

    // now care about the parallel distribution and ghosting.
    // So far every processor only knows about his nodes

    // build unique node row map
    std::vector<int> boundarynoderowvec(rownodeset.begin(), rownodeset.end());
    rownodeset.clear();
    Teuchos::RCP<Epetra_Map> constraintnoderowmap = Teuchos::rcp(new Epetra_Map(
        -1, boundarynoderowvec.size(), boundarynoderowvec.data(), 0, newdis->Comm()));
    boundarynoderowvec.clear();

    // build overlapping node column map
    std::vector<int> constraintnodecolvec(colnodeset.begin(), colnodeset.end());
    colnodeset.clear();
    Teuchos::RCP<Epetra_Map> constraintnodecolmap = Teuchos::rcp(new Epetra_Map(
        -1, constraintnodecolvec.size(), constraintnodecolvec.data(), 0, newdis->Comm()));

    constraintnodecolvec.clear();
    newdis->Redistribute(*constraintnoderowmap, *constraintnodecolmap);
    // put new discretization into the map
    newdiscmap[(*conditer)->parameters().get<int>("ConditionID")] = newdis;
    // increase counter
    index++;
  }

  startID--;  // set counter back to ID of the last element
  return newdiscmap;
}



/*-----------------------------------------------------------------------*
 |(private)                                                     tk 07/08 |
 |Evaluate method, calling element evaluates of a condition and          |
 |assembing results based on this conditions                             |
 *----------------------------------------------------------------------*/
void CONSTRAINTS::MPConstraint3::evaluate_constraint(Teuchos::RCP<Core::FE::Discretization> disc,
    Teuchos::ParameterList& params, Teuchos::RCP<Core::LinAlg::SparseOperator> systemmatrix1,
    Teuchos::RCP<Core::LinAlg::SparseOperator> systemmatrix2,
    Teuchos::RCP<Epetra_Vector> systemvector1, Teuchos::RCP<Epetra_Vector> systemvector2,
    Teuchos::RCP<Epetra_Vector> systemvector3)
{
  if (!(disc->Filled())) FOUR_C_THROW("fill_complete() was not called");
  if (!(disc->HaveDofs())) FOUR_C_THROW("assign_degrees_of_freedom() was not called");

  // see what we have for input
  bool assemblemat1 = systemmatrix1 != Teuchos::null;
  bool assemblemat2 = systemmatrix2 != Teuchos::null;
  bool assemblevec1 = systemvector1 != Teuchos::null;
  bool assemblevec3 = systemvector3 != Teuchos::null;

  // define element matrices and vectors
  Core::LinAlg::SerialDenseMatrix elematrix1;
  Core::LinAlg::SerialDenseMatrix elematrix2;
  Core::LinAlg::SerialDenseVector elevector1;
  Core::LinAlg::SerialDenseVector elevector2;
  Core::LinAlg::SerialDenseVector elevector3;

  const double time = params.get("total time", -1.0);
  const int numcolele = disc->NumMyColElements();

  // get values from time integrator to scale matrices with
  double scStiff = params.get("scaleStiffEntries", 1.0);
  double scConMat = params.get("scaleConstrMat", 1.0);

  // loop over column elements
  for (int i = 0; i < numcolele; ++i)
  {
    // some useful data for computation
    Core::Elements::Element* actele = disc->lColElement(i);
    int eid = actele->Id();
    int condID = eletocond_id_.find(eid)->second;
    Core::Conditions::Condition* cond = constrcond_[eletocondvecindex_.find(eid)->second];
    params.set<Teuchos::RCP<Core::Conditions::Condition>>("condition", Teuchos::rcp(cond, false));

    // computation only if time is larger or equal than initialization time for constraint
    if (inittimes_.find(condID)->second <= time)
    {
      // initialize if it is the first time condition is evaluated
      if (activecons_.find(condID)->second == false)
      {
        const std::string action = params.get<std::string>("action");
        Teuchos::RCP<Epetra_Vector> displast = params.get<Teuchos::RCP<Epetra_Vector>>("old disp");
        SetConstrState("displacement", displast);
        // last converged step is used reference
        initialize(params, systemvector2);
        Teuchos::RCP<Epetra_Vector> disp = params.get<Teuchos::RCP<Epetra_Vector>>("new disp");
        SetConstrState("displacement", disp);
        params.set("action", action);
      }

      // define global and local index of this bc in redundant vectors
      const int offsetID = params.get<int>("OffsetID");
      int gindex = eid - offsetID;
      const int lindex = (systemvector3->Map()).LID(gindex);

      // Get the current lagrange multiplier value for this condition
      const Teuchos::RCP<Epetra_Vector> lagramul =
          params.get<Teuchos::RCP<Epetra_Vector>>("LagrMultVector");
      const double lagraval = (*lagramul)[lindex];

      // get element location vector, dirichlet flags and ownerships
      std::vector<int> lm;
      std::vector<int> lmowner;
      std::vector<int> lmstride;
      actele->LocationVector(*disc, lm, lmowner, lmstride);
      // get dimension of element matrices and vectors
      // Reshape element matrices and vectors and init to zero
      const int eledim = (int)lm.size();
      elematrix1.shape(eledim, eledim);
      elematrix2.shape(eledim, eledim);
      elevector1.size(eledim);
      elevector2.size(eledim);
      elevector3.size(1);
      params.set("ConditionID", eid);

      // call the element evaluate method
      int err = actele->evaluate(
          params, *disc, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);
      if (err) FOUR_C_THROW("Proc %d: Element %d returned err=%d", disc->Comm().MyPID(), eid, err);

      if (assemblemat1)
      {
        // scale with time integrator dependent value
        elematrix1.scale(scStiff * lagraval);
        systemmatrix1->Assemble(eid, lmstride, elematrix1, lm, lmowner);
      }
      if (assemblemat2)
      {
        std::vector<int> colvec(1);
        colvec[0] = gindex;
        elevector2.scale(scConMat);
        systemmatrix2->Assemble(eid, lmstride, elevector2, lm, lmowner, colvec);
      }
      if (assemblevec1)
      {
        elevector1.scale(lagraval);
        Core::LinAlg::Assemble(*systemvector1, elevector1, lm, lmowner);
      }
      if (assemblevec3)
      {
        std::vector<int> constrlm;
        std::vector<int> constrowner;
        constrlm.push_back(gindex);
        constrowner.push_back(actele->Owner());
        Core::LinAlg::Assemble(*systemvector3, elevector3, constrlm, constrowner);
      }

      // loadcurve business
      const auto* curve = cond->parameters().GetIf<int>("curve");
      int curvenum = -1;
      if (curve) curvenum = (*curve);
      double curvefac = 1.0;
      bool usetime = true;
      if (time < 0.0) usetime = false;
      if (curvenum >= 0 && usetime)
        curvefac = Global::Problem::Instance()
                       ->FunctionById<Core::UTILS::FunctionOfTime>(curvenum)
                       .evaluate(time);
      Teuchos::RCP<Epetra_Vector> timefact =
          params.get<Teuchos::RCP<Epetra_Vector>>("vector curve factors");
      timefact->ReplaceGlobalValues(1, &curvefac, &gindex);
    }
  }
}  // end of evaluate_condition

/*-----------------------------------------------------------------------*
 |(private)                                                     tk 07/08 |
 |Evaluate method, calling element evaluates of a condition and          |
 |assembing results based on this conditions                             |
 *----------------------------------------------------------------------*/
void CONSTRAINTS::MPConstraint3::initialize_constraint(Teuchos::RCP<Core::FE::Discretization> disc,
    Teuchos::ParameterList& params, Teuchos::RCP<Epetra_Vector> systemvector)
{
  if (!(disc->Filled())) FOUR_C_THROW("fill_complete() was not called");
  if (!(disc->HaveDofs())) FOUR_C_THROW("assign_degrees_of_freedom() was not called");

  // define element matrices and vectors
  Core::LinAlg::SerialDenseMatrix elematrix1;
  Core::LinAlg::SerialDenseMatrix elematrix2;
  Core::LinAlg::SerialDenseVector elevector1;
  Core::LinAlg::SerialDenseVector elevector2;
  Core::LinAlg::SerialDenseVector elevector3;

  // loop over column elements
  const double time = params.get("total time", -1.0);
  const int numcolele = disc->NumMyColElements();
  for (int i = 0; i < numcolele; ++i)
  {
    // some useful data for computation
    Core::Elements::Element* actele = disc->lColElement(i);
    int eid = actele->Id();
    int condID = eletocond_id_.find(eid)->second;
    Core::Conditions::Condition* cond = constrcond_[eletocondvecindex_.find(eid)->second];
    params.set<Teuchos::RCP<Core::Conditions::Condition>>("condition", Teuchos::rcp(cond, false));

    // get element location vector, dirichlet flags and ownerships
    std::vector<int> lm;
    std::vector<int> lmowner;
    std::vector<int> lmstride;
    actele->LocationVector(*disc, lm, lmowner, lmstride);
    // get dimension of element matrices and vectors
    // Reshape element matrices and vectors and init to zero
    const int eledim = (int)lm.size();
    elematrix1.shape(eledim, eledim);
    elematrix2.shape(eledim, eledim);
    elevector1.size(eledim);
    elevector2.size(eledim);
    elevector3.size(1);
    // call the element evaluate method
    int err = actele->evaluate(
        params, *disc, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);
    if (err)
      FOUR_C_THROW("Proc %d: Element %d returned err=%d", disc->Comm().MyPID(), actele->Id(), err);

    // assembly
    std::vector<int> constrlm;
    std::vector<int> constrowner;
    int offsetID = params.get<int>("OffsetID");
    constrlm.push_back(eid - offsetID);
    constrowner.push_back(actele->Owner());
    Core::LinAlg::Assemble(*systemvector, elevector3, constrlm, constrowner);

    // loadcurve business
    const auto* curve = cond->parameters().GetIf<int>("curve");
    int curvenum = -1;
    if (curve) curvenum = (*curve);
    double curvefac = 1.0;
    bool usetime = true;
    if (time < 0.0) usetime = false;
    if (curvenum >= 0 && usetime)
      curvefac =
          Global::Problem::Instance()->FunctionById<Core::UTILS::FunctionOfTime>(curvenum).evaluate(
              time);

    // Get ConditionID of current condition if defined and write value in parameterlist
    char factorname[30];
    sprintf(factorname, "LoadCurveFactor %d", condID);
    params.set(factorname, curvefac);
  }
}  // end of initialize_constraint

FOUR_C_NAMESPACE_CLOSE