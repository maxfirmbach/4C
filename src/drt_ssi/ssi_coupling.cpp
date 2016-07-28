/*----------------------------------------------------------------------*/
/*!
 \file ssi_coupling.cpp

 \brief helper classes for  scalar-structure coupling

   \level 3

   \maintainer  Anh-Tu Vuong
                vuong@lnm.mw.tum.de
                http://www.lnm.mw.tum.de
                089 - 289-15251
 *----------------------------------------------------------------------*/

#include "ssi_coupling.H"

//for coupling of nonmatching meshes
#include "../drt_adapter/adapter_coupling_volmortar.H"
#include "../drt_adapter/adapter_coupling_mortar.H"

#include "../drt_adapter/adapter_scatra_base_algorithm.H"
#include "../drt_adapter/ad_str_wrapper.H"

#include "../drt_scatra/scatra_timint_implicit.H"

#include"../drt_inpar/inpar_volmortar.H"
#include "../drt_volmortar/volmortar_utils.H"


#include "../drt_lib/drt_nodematchingoctree.H"
#include "../drt_lib/drt_condition_utils.H"
#include "../drt_lib/drt_dofset_mapped_proxy.H"

#include "../linalg/linalg_mapextractor.H"
#include "../linalg/linalg_utils.H"

#include "../drt_particle/binning_strategy.H"

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolume::Setup(
    const int                         ndim,          /// dimension of the problem
    Teuchos::RCP<DRT::Discretization> structdis,     /// underlying structure discretization
    Teuchos::RCP<DRT::Discretization> scatradis      /// underlying scatra discretization
    )
{
  structdis->FillComplete();
  scatradis->FillComplete();

  // build a proxy of the structure discretization for the scatra field
  Teuchos::RCP<DRT::DofSet> structdofset = structdis->GetDofSetProxy();
  // build a proxy of the scatra discretization for the structure field
  Teuchos::RCP<DRT::DofSet> scatradofset = scatradis->GetDofSetProxy();

  // check if scatra field has 2 discretizations, so that coupling is possible
  if (scatradis->AddDofSet(structdofset)!=1)
    dserror("unexpected dof sets in scatra field");
  if (structdis->AddDofSet(scatradofset)!=1)
    dserror("unexpected dof sets in structure field");


  AssignMaterialPointers(structdis,scatradis);

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolume::AssignMaterialPointers(
        Teuchos::RCP<DRT::Discretization> structdis,     /// underlying structure discretization
        Teuchos::RCP<DRT::Discretization> scatradis      /// underlying scatra discretization
        )
{
  const int numelements = scatradis->NumMyColElements();

  for (int i=0; i<numelements; ++i)
  {
    DRT::Element* scatratele = scatradis->lColElement(i);
    const int gid = scatratele->Id();

    DRT::Element* structele = structdis->gElement(gid);

    //for coupling we add the source material to the target element and vice versa
    scatratele->AddMaterial(structele->Material());
    structele->AddMaterial(scatratele->Material());
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolume::SetMeshDisp(
    Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> scatra, /// underlying scatra problem of the SSI problem
    Teuchos::RCP<const Epetra_Vector> disp             /// displacement field to set
    )
{
  scatra->ScaTraField()->ApplyMeshMovement(
      disp,
      1);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolume::SetVelocityFields(
    Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> scatra, /// underlying scatra problem of the SSI problem
    Teuchos::RCP<const Epetra_Vector> convvel,         /// convective velocity field to set
    Teuchos::RCP<const Epetra_Vector> vel              /// velocity field to set
    )
{
  scatra->ScaTraField()->SetVelocityField(
        convvel, //convective vel.
        Teuchos::null, //acceleration
        vel, //velocity
        Teuchos::null, //fsvel
        1);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolume::SetScalarField(
    Teuchos::RCP< ::ADAPTER::Structure> structure,     /// underlying structure of the SSI problem,
    Teuchos::RCP<const Epetra_Vector> phi              /// scalar field to set
    )
{
  structure->Discretization()->SetState(1,"temperature",phi);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingBoundary::Setup(
    const int                             ndim,      /// dimension of the problem
    Teuchos::RCP<DRT::Discretization> structdis,     /// underlying structure discretization
    Teuchos::RCP<DRT::Discretization> scatradis      /// underlying scatra discretization
    )
{
  //first call FillComplete for single discretizations.
  //This way the physical dofs are numbered successively
  structdis->FillComplete();
  scatradis->FillComplete();

  //build auxiliary dofsets, i.e. pseudo dofs on each discretization
  const int ndofpernode_scatra = scatradis->NumDof(0,scatradis->lRowNode(0));
  const int ndofperelement_scatra  = 0;
  const int ndofpernode_struct = structdis->NumDof(0,structdis->lRowNode(0));
  const int ndofperelement_struct = 0;
  if (structdis->BuildDofSetAuxProxy(ndofpernode_scatra, ndofperelement_scatra, 0, true ) != 1)
    dserror("unexpected dof sets in structure field");
  if (scatradis->BuildDofSetAuxProxy(ndofpernode_struct, ndofperelement_struct, 0, true) != 1)
    dserror("unexpected dof sets in scatra field");

  //call AssignDegreesOfFreedom also for auxiliary dofsets
  //note: the order of FillComplete() calls determines the gid numbering!
  // 1. structure dofs
  // 2. scatra dofs
  // 3. structure auxiliary dofs
  // 4. scatra auxiliary dofs
  structdis->FillComplete(true, false,false);
  scatradis->FillComplete(true, false,false);

  // setup mortar adapter for surface volume coupling
  adaptermeshtying_ = Teuchos::rcp(new ADAPTER::CouplingMortar());

  std::vector<int> coupleddof(ndim, 1);
  // Setup of meshtying adapter
  adaptermeshtying_->Setup(
      structdis,
      scatradis,
      Teuchos::null,
      coupleddof,
      "SSICoupling",
      structdis->Comm(),
      false,
      false,
      0,
      1
      );

  //extractor for coupled surface of structure discretization with surface scatra
  extractor_= Teuchos::rcp(new LINALG::MapExtractor(
      *structdis->DofRowMap(0),
      adaptermeshtying_->MasterDofMap(),
      true));

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingBoundary::AssignMaterialPointers(
        Teuchos::RCP<DRT::Discretization> structdis,     /// underlying structure discretization
        Teuchos::RCP<DRT::Discretization> scatradis      /// underlying scatra discretization
        )
{
  //nothing to do in this case, since
  //transferring scalar state to structure discretization not implemented for
  //transport on structural boundary. Only SolidToScatra coupling available.
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingBoundary::SetMeshDisp(
    Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> scatra, /// underlying scatra problem of the SSI problem
    Teuchos::RCP<const Epetra_Vector> disp             /// displacement field to set
    )
{
  scatra->ScaTraField()->ApplyMeshMovement(
      adaptermeshtying_->MasterToSlave(extractor_->ExtractCondVector(disp)),
      1);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingBoundary::SetVelocityFields(
    Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> scatra, /// underlying scatra problem of the SSI problem
    Teuchos::RCP<const Epetra_Vector> convvel,         /// convective velocity field to set
    Teuchos::RCP<const Epetra_Vector> vel              /// velocity field to set
    )
{
  scatra->ScaTraField()->SetVelocityField(
      adaptermeshtying_->MasterToSlave(extractor_->ExtractCondVector(convvel)), //convective vel.
      Teuchos::null, //acceleration
      adaptermeshtying_->MasterToSlave(extractor_->ExtractCondVector(vel)), //velocity
      Teuchos::null, //fsvel
      1);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingBoundary::SetScalarField(
    Teuchos::RCP< ::ADAPTER::Structure> structure,     /// underlying structure of the SSI problem,
    Teuchos::RCP<const Epetra_Vector> phi              /// scalar field to set
    )
{
  dserror("transferring scalar state to structure discretization not implemented for "
      "transport on structural boundary. Only SolidToScatra coupling available.");
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingVolume::Setup(
    const int                         ndim,          /// dimension of the problem
    Teuchos::RCP<DRT::Discretization> structdis,     /// underlying structure discretization
    Teuchos::RCP<DRT::Discretization> scatradis      /// underlying scatra discretization
    )
{
  //first call FillComplete for single discretizations.
  //This way the physical dofs are numbered successively
  structdis->FillComplete();
  scatradis->FillComplete();

  //build auxiliary dofsets, i.e. pseudo dofs on each discretization
  const int ndofpernode_scatra = scatradis->NumDof(0,scatradis->lRowNode(0));
  const int ndofperelement_scatra  = 0;
  const int ndofpernode_struct = structdis->NumDof(0,structdis->lRowNode(0));
  const int ndofperelement_struct = 0;
  if (structdis->BuildDofSetAuxProxy(ndofpernode_scatra, ndofperelement_scatra, 0, true ) != 1)
    dserror("unexpected dof sets in structure field");
  if (scatradis->BuildDofSetAuxProxy(ndofpernode_struct, ndofperelement_struct, 0, true) != 1)
    dserror("unexpected dof sets in scatra field");

  //call AssignDegreesOfFreedom also for auxiliary dofsets
  //note: the order of FillComplete() calls determines the gid numbering!
  // 1. structure dofs
  // 2. scatra dofs
  // 3. structure auxiliary dofs
  // 4. scatra auxiliary dofs
  structdis->FillComplete(true, false,false);
  scatradis->FillComplete(true, false,false);

  // Scheme: non matching meshes --> volumetric mortar coupling...
  volcoupl_structurescatra_=Teuchos::rcp(new ADAPTER::MortarVolCoupl() );

  //setup projection matrices (use default material strategy)
  volcoupl_structurescatra_->Setup(
      structdis,
      scatradis);

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingVolume::AssignMaterialPointers(
        Teuchos::RCP<DRT::Discretization> structdis,     /// underlying structure discretization
        Teuchos::RCP<DRT::Discretization> scatradis      /// underlying scatra discretization
        )
{
  volcoupl_structurescatra_->AssignMaterials(
      structdis,
      scatradis);
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingVolume::SetMeshDisp(
    Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> scatra, /// underlying scatra problem of the SSI problem
    Teuchos::RCP<const Epetra_Vector> disp             /// displacement field to set
    )
{
  scatra->ScaTraField()->ApplyMeshMovement(
      volcoupl_structurescatra_->ApplyVectorMapping21(disp),
      1);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingVolume::SetVelocityFields(
    Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> scatra, /// underlying scatra problem of the SSI problem
    Teuchos::RCP<const Epetra_Vector> convvel,         /// convective velocity field to set
    Teuchos::RCP<const Epetra_Vector> vel              /// velocity field to set
    )
{
  scatra->ScaTraField()->SetVelocityField(
      volcoupl_structurescatra_->ApplyVectorMapping21(convvel), //convective vel.
      Teuchos::null, //acceleration
      volcoupl_structurescatra_->ApplyVectorMapping21(vel), //velocity
      Teuchos::null, //fsvel
      1);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingVolume::SetScalarField(
    Teuchos::RCP< ::ADAPTER::Structure> structure,     /// underlying structure of the SSI problem,
    Teuchos::RCP<const Epetra_Vector> phi              /// scalar field to set
    )
{
  structure->Discretization()->SetState(1,"temperature",volcoupl_structurescatra_->ApplyVectorMapping12(phi));
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolumeAndBoundary::Setup(
    const int                         ndim,          /// dimension of the problem
    Teuchos::RCP<DRT::Discretization> structdis,     /// underlying structure discretization
    Teuchos::RCP<DRT::Discretization> scatradis      /// underlying scatra discretization
    )
{
  // redistribute discr. with help of binning strategy
  if(scatradis->Comm().NumProc()>1)
  {
    scatradis->FillComplete();
    structdis->FillComplete();
    // create vector of discr.
    std::vector<Teuchos::RCP<DRT::Discretization> > dis;
    dis.push_back(structdis);
    dis.push_back(scatradis);

    std::vector<Teuchos::RCP<Epetra_Map> > stdelecolmap;
    std::vector<Teuchos::RCP<Epetra_Map> > stdnodecolmap;

    /// binning strategy is created and parallel redistribution is performed
    Teuchos::RCP<BINSTRATEGY::BinningStrategy> binningstrategy =
      Teuchos::rcp(new BINSTRATEGY::BinningStrategy(dis,stdelecolmap,stdnodecolmap));
  }

  {
    Teuchos::RCP<DRT::DofSet> newdofset =
        Teuchos::rcp(new DRT::DofSetMappedProxy(structdis->GetDofSetProxy(),structdis,"SSICoupling"));

    // add dofset and check if scatra field has 2 discretizations, so that coupling is possible
    if (scatradis->AddDofSet(newdofset)!=1)
      dserror("unexpected dof sets in scatra field");

    Teuchos::RCP<DRT::DofSet> newdofset_struct =
        Teuchos::rcp(new DRT::DofSetMappedProxy(scatradis->GetDofSetProxy(),scatradis,"SSICoupling"));

    // add dofset  check if scatra field has 2 discretizations, so that coupling is possible
    if (structdis->AddDofSet(newdofset_struct)!=1)
      dserror("unexpected dof sets in structure field");
  }

  // exchange material pointers for coupled material formulations
  AssignMaterialPointers(structdis,scatradis);

  structdis->FillComplete();
  scatradis->FillComplete();

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolumeAndBoundary::AssignMaterialPointers(
        Teuchos::RCP<DRT::Discretization> structdis,     /// underlying structure discretization
        Teuchos::RCP<DRT::Discretization> scatradis      /// underlying scatra discretization
        )
{
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolumeAndBoundary::SetMeshDisp(
    Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> scatra, /// underlying scatra problem of the SSI problem
    Teuchos::RCP<const Epetra_Vector> disp             /// displacement field to set
    )
{
  scatra->ScaTraField()->ApplyMeshMovement(
      disp,
      1);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolumeAndBoundary::SetVelocityFields(
    Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> scatra, /// underlying scatra problem of the SSI problem
    Teuchos::RCP<const Epetra_Vector> convvel,         /// convective velocity field to set
    Teuchos::RCP<const Epetra_Vector> vel              /// velocity field to set
    )
{
  scatra->ScaTraField()->SetVelocityField(
        convvel, //convective vel.
        Teuchos::null, //acceleration
        vel, //velocity
        Teuchos::null, //fsvel
        1);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolumeAndBoundary::SetScalarField(
    Teuchos::RCP< ::ADAPTER::Structure> structure,     /// underlying structure of the SSI problem,
    Teuchos::RCP<const Epetra_Vector> phi              /// scalar field to set
    )
{
  structure->Discretization()->SetState(1,"temperature",phi);
}