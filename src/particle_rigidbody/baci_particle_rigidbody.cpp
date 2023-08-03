/*---------------------------------------------------------------------------*/
/*! \file
\brief rigid body handler for particle problem
\level 2
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "baci_particle_rigidbody.H"

#include "baci_inpar_particle.H"
#include "baci_io.H"
#include "baci_io_pstream.H"
#include "baci_lib_pack_buffer.H"
#include "baci_lib_parobject.H"
#include "baci_particle_engine_communication_utils.H"
#include "baci_particle_engine_interface.H"
#include "baci_particle_engine_unique_global_id.H"
#include "baci_particle_interaction_utils.H"
#include "baci_particle_rigidbody_affiliation_pairs.H"
#include "baci_particle_rigidbody_datastate.H"
#include "baci_particle_rigidbody_runtime_vtp_writer.H"
#include "baci_particle_rigidbody_utils.H"

#include <Teuchos_TimeMonitor.hpp>

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
PARTICLERIGIDBODY::RigidBodyHandler::RigidBodyHandler(
    const Epetra_Comm& comm, const Teuchos::ParameterList& params)
    : comm_(comm), myrank_(comm.MyPID()), params_(params)
{
  // empty constructor
}

PARTICLERIGIDBODY::RigidBodyHandler::~RigidBodyHandler() = default;

void PARTICLERIGIDBODY::RigidBodyHandler::Init()
{
  // init rigid body unique global identifier handler
  InitRigidBodyUniqueGlobalIdHandler();

  // init rigid body data state container
  InitRigidBodyDataState();

  // init rigid body runtime vtp writer
  InitRigidBodyVtpWriter();

  // init affiliation pair handler
  InitAffiliationPairHandler();
}

void PARTICLERIGIDBODY::RigidBodyHandler::Setup(
    const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // setup unique global identifier handler
  rigidbodyuniqueglobalidhandler_->Setup();

  // setup rigid body data state container
  rigidbodydatastate_->Setup();

  // setup rigid body runtime vtp writer
  SetupRigidBodyVtpWriter();

  // setup affiliation pair handler
  affiliationpairs_->Setup(particleengineinterface);

  // safety check
  {
    // get particle container bundle
    PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle =
        particleengineinterface_->GetParticleContainerBundle();

    if (not particlecontainerbundle->GetParticleTypes().count(PARTICLEENGINE::RigidPhase))
      dserror("no particle container for particle type '%s' found!",
          PARTICLEENGINE::EnumToTypeName(PARTICLEENGINE::RigidPhase).c_str());
  }

  // short screen output
  if (particleengineinterface_->HavePeriodicBoundaryConditions() and myrank_ == 0)
    IO::cout << "Warning: rigid bodies not transferred over periodic boundary!" << IO::endl;
}

void PARTICLERIGIDBODY::RigidBodyHandler::WriteRestart() const
{
  // get bin discretization writer
  std::shared_ptr<IO::DiscretizationWriter> binwriter =
      particleengineinterface_->GetBinDiscretizationWriter();

  // write restart of unique global identifier handler
  rigidbodyuniqueglobalidhandler_->WriteRestart(binwriter);

  // write restart of affiliation pair handler
  affiliationpairs_->WriteRestart();

  // get packed rigid body state data
  Teuchos::RCP<std::vector<char>> buffer = Teuchos::rcp(new std::vector<char>);
  GetPackedRigidBodyStates(*buffer);

  // write rigid body state data
  binwriter->WriteCharVector("RigidBodyStateData", buffer);
}

void PARTICLERIGIDBODY::RigidBodyHandler::ReadRestart(
    const std::shared_ptr<IO::DiscretizationReader> reader)
{
  // read restart of unique global identifier handler
  rigidbodyuniqueglobalidhandler_->ReadRestart(reader);

  // read restart of runtime vtp writer
  rigidbodyvtpwriter_->ReadRestart(reader);

  // read restart of affiliation pair handler
  affiliationpairs_->ReadRestart(reader);

  // allocate rigid body states
  AllocateRigidBodyStates();

  // read rigid body state data
  Teuchos::RCP<std::vector<char>> buffer = Teuchos::rcp(new std::vector<char>);
  reader->ReadCharVector(buffer, "RigidBodyStateData");

  // extract packed rigid body state data
  ExtractPackedRigidBodyStates(*buffer);
}

void PARTICLERIGIDBODY::RigidBodyHandler::InsertParticleStatesOfParticleTypes(
    std::map<PARTICLEENGINE::TypeEnum, std::set<PARTICLEENGINE::StateEnum>>& particlestatestotypes)
    const
{
  // iterate over particle types
  for (auto& typeIt : particlestatestotypes)
  {
    // get type of particles
    PARTICLEENGINE::TypeEnum type = typeIt.first;

    // set of particle states for current particle type
    std::set<PARTICLEENGINE::StateEnum>& particlestates = typeIt.second;

    if (type == PARTICLEENGINE::RigidPhase)
    {
      // insert states of rigid particles
      particlestates.insert(
          {PARTICLEENGINE::RigidBodyColor, PARTICLEENGINE::RelativePositionBodyFrame,
              PARTICLEENGINE::RelativePosition, PARTICLEENGINE::Inertia, PARTICLEENGINE::Force});
    }
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::WriteRigidBodyRuntimeOutput(
    const int step, const double time) const
{
  rigidbodyvtpwriter_->ResetTimeAndTimeStep(time, step);
  rigidbodyvtpwriter_->SetRigidBodyPositionsAndStates(ownedrigidbodies_);
  rigidbodyvtpwriter_->WriteFiles();
  rigidbodyvtpwriter_->WriteCollectionFileOfAllWrittenFiles();
}

void PARTICLERIGIDBODY::RigidBodyHandler::SetInitialAffiliationPairData()
{
  // get reference to affiliation pair data
  std::unordered_map<int, int>& affiliationpairdata =
      affiliationpairs_->GetRefToAffiliationPairData();

  // get particle container bundle
  PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->GetParticleContainerBundle();

  // get container of owned particles of rigid phase
  PARTICLEENGINE::ParticleContainer* container_i = particlecontainerbundle->GetSpecificContainer(
      PARTICLEENGINE::RigidPhase, PARTICLEENGINE::Owned);

  // loop over particles in container
  for (int particle_i = 0; particle_i < container_i->ParticlesStored(); ++particle_i)
  {
    // get global id of particle i
    const int* globalid_i = container_i->GetPtrToGlobalID(particle_i);

    // get pointer to particle states
    const double* rigidbodycolor_i =
        container_i->GetPtrToState(PARTICLEENGINE::RigidBodyColor, particle_i);

    // get global id of affiliated rigid body k
    const int rigidbody_k = std::round(rigidbodycolor_i[0]);

    // insert affiliation pair
    affiliationpairdata.insert(std::make_pair(globalid_i[0], rigidbody_k));
  }

#ifdef DEBUG
  if (static_cast<int>(affiliationpairdata.size()) != container_i->ParticlesStored())
    dserror("number of affiliation pairs and rigid particles not equal!");
#endif
}

void PARTICLERIGIDBODY::RigidBodyHandler::SetUniqueGlobalIdsForAllRigidBodies()
{
  // get reference to affiliation pair data
  std::unordered_map<int, int>& affiliationpairdata =
      affiliationpairs_->GetRefToAffiliationPairData();

  // maximum global id of rigid bodies on this processor
  int maxglobalid = -1;

  // get maximum global id of rigid bodies on this processor
  for (const auto& it : affiliationpairdata) maxglobalid = std::max(maxglobalid, it.second);

  // get maximum global id of rigid bodies on all processors
  int allprocmaxglobalid = -1;
  comm_.MaxAll(&maxglobalid, &allprocmaxglobalid, 1);

  // number of global ids on all processors
  const int numglobalids = allprocmaxglobalid + 1;

#ifdef DEBUG
  if (not(rigidbodyuniqueglobalidhandler_->GetMaxGlobalId() < 0))
    dserror("maximum global id of rigid body unique global identifier handler already touched!");
#endif

  // request number of global ids of all rigid bodies on processor 0
  std::vector<int> requesteduniqueglobalids;
  if (myrank_ == 0) requesteduniqueglobalids.reserve(numglobalids);

  // draw requested number of global ids
  rigidbodyuniqueglobalidhandler_->DrawRequestedNumberOfGlobalIds(requesteduniqueglobalids);

#ifdef DEBUG
  if (myrank_ == 0)
    for (int i = 0; i < numglobalids; ++i)
      if (requesteduniqueglobalids[i] != i) dserror("drawn requested global ids not consecutive!");
#endif

  // used global ids on all processors
  std::vector<int> usedglobalids(numglobalids, 0);

  // get used global ids on this processor
  for (const auto& it : affiliationpairdata) usedglobalids[it.second] = 1;

  // mpi communicator
  const auto* mpicomm = dynamic_cast<const Epetra_MpiComm*>(&comm_);
  if (!mpicomm) dserror("dynamic cast to Epetra_MpiComm failed!");

  // get used global ids on all processors
  MPI_Allreduce(
      MPI_IN_PLACE, usedglobalids.data(), numglobalids, MPI_INT, MPI_MAX, mpicomm->Comm());

  // free unused global ids on processor 0
  if (myrank_ == 0)
    for (int i = 0; i < numglobalids; ++i)
      if (usedglobalids[i] == 0)
        rigidbodyuniqueglobalidhandler_->InsertFreedGlobalId(requesteduniqueglobalids[i]);
}

void PARTICLERIGIDBODY::RigidBodyHandler::AllocateRigidBodyStates()
{
  // number of global ids
  const int numglobalids = rigidbodyuniqueglobalidhandler_->GetMaxGlobalId() + 1;

  // allocate stored states
  rigidbodydatastate_->AllocateStoredStates(numglobalids);
}

void PARTICLERIGIDBODY::RigidBodyHandler::InitializeRigidBodyMassQuantitiesAndOrientation()
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "PARTICLERIGIDBODY::RigidBodyHandler::InitializeRigidBodyMassQuantitiesAndOrientation");

  // compute mass quantities of rigid bodies
  ComputeRigidBodyMassQuantities();

  // clear orientation of rigid bodies
  ClearRigidBodyOrientation();

  // broadcast positions of rigid bodies
  BroadcastRigidBodyPositions();

  // set relative position of rigid particles in body frame
  SetRigidParticleRelativePositionInBodyFrame();

  // update relative position of rigid particles
  UpdateRigidParticleRelativePosition();
}

void PARTICLERIGIDBODY::RigidBodyHandler::DistributeRigidBody()
{
  TEUCHOS_FUNC_TIME_MONITOR("PARTICLERIGIDBODY::RigidBodyHandler::DistributeRigidBody");

  // distribute affiliation pairs
  affiliationpairs_->DistributeAffiliationPairs();

  // update rigid body ownership
  UpdateRigidBodyOwnership();
}

void PARTICLERIGIDBODY::RigidBodyHandler::CommunicateRigidBody()
{
  TEUCHOS_FUNC_TIME_MONITOR("PARTICLERIGIDBODY::RigidBodyHandler::CommunicateRigidBody");

  // communicate affiliation pairs
  affiliationpairs_->CommunicateAffiliationPairs();

  // update rigid body ownership
  UpdateRigidBodyOwnership();
}

void PARTICLERIGIDBODY::RigidBodyHandler::ClearForcesAndTorques()
{
  TEUCHOS_FUNC_TIME_MONITOR("PARTICLERIGIDBODY::RigidBodyHandler::ClearForcesAndTorques");

  // clear force and torque acting on rigid bodies
  ClearRigidBodyForceAndTorque();

  // clear force acting on rigid particles
  ClearRigidParticleForce();
}

void PARTICLERIGIDBODY::RigidBodyHandler::AddGravityAcceleration(std::vector<double>& gravity)
{
  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    // get pointer to rigid body states
    double* acc_k = rigidbodydatastate_->GetRefMutableAcceleration()[rigidbody_k].data();

    // set gravity acceleration
    PARTICLEINTERACTION::UTILS::VecAdd(acc_k, gravity.data());
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::ComputeAccelerations()
{
  TEUCHOS_FUNC_TIME_MONITOR("PARTICLERIGIDBODY::RigidBodyHandler::ComputeAccelerations");

  // compute partial force and torque acting on rigid bodies
  ComputePartialForceAndTorque();

  // gather partial and compute full force and torque acting on rigid bodies
  GatherPartialAndComputeFullForceAndTorque();

  // compute accelerations of rigid bodies from force and torque
  ComputeAccelerationsFromForceAndTorque();

  // broadcast accelerations of rigid bodies
  BroadcastRigidBodyAccelerations();

  // set accelerations of rigid particles
  SetRigidParticleAccelerations();
}

void PARTICLERIGIDBODY::RigidBodyHandler::UpdatePositions(const double timeincrement)
{
  TEUCHOS_FUNC_TIME_MONITOR("PARTICLERIGIDBODY::RigidBodyHandler::UpdatePositions");

  // update positions of rigid bodies with given time increment
  UpdateRigidBodyPositions(timeincrement);

  // broadcast positions of rigid bodies
  BroadcastRigidBodyPositions();

  // update relative position of rigid particles
  UpdateRigidParticleRelativePosition();

  // set position of rigid particles
  SetRigidParticlePosition();
}

void PARTICLERIGIDBODY::RigidBodyHandler::UpdateVelocities(const double timeincrement)
{
  TEUCHOS_FUNC_TIME_MONITOR("PARTICLERIGIDBODY::RigidBodyHandler::UpdateVelocities");

  // update velocities of rigid bodies with given time increment
  UpdateRigidBodyVelocities(timeincrement);

  // broadcast velocities of rigid bodies
  BroadcastRigidBodyVelocities();

  // set velocities of rigid particles
  SetRigidParticleVelocities();
}

void PARTICLERIGIDBODY::RigidBodyHandler::ClearAccelerations()
{
  TEUCHOS_FUNC_TIME_MONITOR("PARTICLERIGIDBODY::RigidBodyHandler::ClearAccelerations");

  // clear accelerations of rigid bodies
  ClearRigidBodyAccelerations();
}

bool PARTICLERIGIDBODY::RigidBodyHandler::HaveRigidBodyPhaseChange(
    const std::vector<PARTICLEENGINE::ParticleTypeToType>& particlesfromphasetophase)
{
  TEUCHOS_FUNC_TIME_MONITOR("PARTICLERIGIDBODY::RigidBodyHandler::HaveRigidBodyPhaseChange");

  int localhavephasechange = 0;

  // iterate over particle phase change tuples
  for (const auto& particletypetotype : particlesfromphasetophase)
  {
    PARTICLEENGINE::TypeEnum type_source;
    PARTICLEENGINE::TypeEnum type_target;
    std::tie(type_source, type_target, std::ignore) = particletypetotype;

    if (type_source == PARTICLEENGINE::RigidPhase or type_target == PARTICLEENGINE::RigidPhase)
    {
      localhavephasechange = 1;
      break;
    }
  }

  // check among all processors
  int globalhavephasechange = 0;
  comm_.MaxAll(&localhavephasechange, &globalhavephasechange, 1);

  return globalhavephasechange;
}

void PARTICLERIGIDBODY::RigidBodyHandler::EvaluateRigidBodyPhaseChange(
    const std::vector<PARTICLEENGINE::ParticleTypeToType>& particlesfromphasetophase)
{
  TEUCHOS_FUNC_TIME_MONITOR("PARTICLERIGIDBODY::RigidBodyHandler::EvaluateRigidBodyPhaseChange");

  // evaluate melting of rigid bodies
  EvaluateRigidBodyMelting(particlesfromphasetophase);

  // evaluate solidification of rigid bodies
  EvaluateRigidBodySolidification(particlesfromphasetophase);

  // update rigid body ownership
  UpdateRigidBodyOwnership();

  // store previous position of rigid bodies
  const std::vector<std::vector<double>> previousposition = rigidbodydatastate_->GetRefPosition();

  // initialize rigid body mass quantities and orientation
  InitializeRigidBodyMassQuantitiesAndOrientation();

  // set velocities of rigid bodies after phase change
  SetRigidBodyVelocitiesAfterPhaseChange(previousposition);

  // broadcast velocities of rigid bodies
  BroadcastRigidBodyVelocities();

  // set velocities of rigid particles
  SetRigidParticleVelocities();
}

void PARTICLERIGIDBODY::RigidBodyHandler::InitRigidBodyUniqueGlobalIdHandler()
{
  // create and init unique global identifier handler
  rigidbodyuniqueglobalidhandler_ = std::unique_ptr<PARTICLEENGINE::UniqueGlobalIdHandler>(
      new PARTICLEENGINE::UniqueGlobalIdHandler(comm_, "rigidbody"));
  rigidbodyuniqueglobalidhandler_->Init();
}

void PARTICLERIGIDBODY::RigidBodyHandler::InitRigidBodyDataState()
{
  // create rigid body data state container
  rigidbodydatastate_ = std::make_shared<PARTICLERIGIDBODY::RigidBodyDataState>();

  // init rigid body data state container
  rigidbodydatastate_->Init();
}

void PARTICLERIGIDBODY::RigidBodyHandler::InitRigidBodyVtpWriter()
{
  // construct and init rigid body runtime vtp writer
  rigidbodyvtpwriter_ = std::unique_ptr<PARTICLERIGIDBODY::RigidBodyRuntimeVtpWriter>(
      new PARTICLERIGIDBODY::RigidBodyRuntimeVtpWriter(comm_));
  rigidbodyvtpwriter_->Init(rigidbodydatastate_);
}

void PARTICLERIGIDBODY::RigidBodyHandler::InitAffiliationPairHandler()
{
  // create affiliation pair handler
  affiliationpairs_ = std::unique_ptr<PARTICLERIGIDBODY::RigidBodyAffiliationPairs>(
      new PARTICLERIGIDBODY::RigidBodyAffiliationPairs(comm_));

  // init affiliation pair handler
  affiliationpairs_->Init();
}

void PARTICLERIGIDBODY::RigidBodyHandler::SetupRigidBodyVtpWriter()
{
  // get data format for written numeric data via vtp
  bool write_binary_output = (DRT::INPUT::IntegralValue<INPAR::PARTICLE::OutputDataFormat>(
                                  params_, "OUTPUT_DATA_FORMAT") == INPAR::PARTICLE::binary);

  // setup rigid body runtime vtp writer
  rigidbodyvtpwriter_->Setup(write_binary_output);
}

void PARTICLERIGIDBODY::RigidBodyHandler::GetPackedRigidBodyStates(std::vector<char>& buffer) const
{
  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    // get reference to rigid body states
    const double& mass_k = rigidbodydatastate_->GetRefMass()[rigidbody_k];
    const std::vector<double>& inertia_k = rigidbodydatastate_->GetRefInertia()[rigidbody_k];
    const std::vector<double>& pos_k = rigidbodydatastate_->GetRefPosition()[rigidbody_k];
    const std::vector<double>& rot_k = rigidbodydatastate_->GetRefRotation()[rigidbody_k];
    const std::vector<double>& vel_k = rigidbodydatastate_->GetRefVelocity()[rigidbody_k];
    const std::vector<double>& angvel_k = rigidbodydatastate_->GetRefAngularVelocity()[rigidbody_k];
    const std::vector<double>& acc_k = rigidbodydatastate_->GetRefAcceleration()[rigidbody_k];
    const std::vector<double>& angacc_k =
        rigidbodydatastate_->GetRefAngularAcceleration()[rigidbody_k];

    // pack data for sending
    DRT::PackBuffer data;
    data.StartPacking();

    data.AddtoPack(rigidbody_k);
    data.AddtoPack(mass_k);
    for (int i = 0; i < 6; ++i) data.AddtoPack(inertia_k[i]);
    for (int i = 0; i < 3; ++i) data.AddtoPack(pos_k[i]);
    for (int i = 0; i < 4; ++i) data.AddtoPack(rot_k[i]);
    for (int i = 0; i < 3; ++i) data.AddtoPack(vel_k[i]);
    for (int i = 0; i < 3; ++i) data.AddtoPack(angvel_k[i]);
    for (int i = 0; i < 3; ++i) data.AddtoPack(acc_k[i]);
    for (int i = 0; i < 3; ++i) data.AddtoPack(angacc_k[i]);

    buffer.insert(buffer.end(), data().begin(), data().end());
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::ExtractPackedRigidBodyStates(std::vector<char>& buffer)
{
  std::vector<char>::size_type position = 0;

  while (position < buffer.size())
  {
    const int rigidbody_k = DRT::ParObject::ExtractInt(position, buffer);

    // get global ids of rigid bodies owned by this processor
    ownedrigidbodies_.push_back(rigidbody_k);

    // get reference to rigid body states
    double& mass_k = rigidbodydatastate_->GetRefMutableMass()[rigidbody_k];
    std::vector<double>& inertia_k = rigidbodydatastate_->GetRefMutableInertia()[rigidbody_k];
    std::vector<double>& pos_k = rigidbodydatastate_->GetRefMutablePosition()[rigidbody_k];
    std::vector<double>& rot_k = rigidbodydatastate_->GetRefMutableRotation()[rigidbody_k];
    std::vector<double>& vel_k = rigidbodydatastate_->GetRefMutableVelocity()[rigidbody_k];
    std::vector<double>& angvel_k =
        rigidbodydatastate_->GetRefMutableAngularVelocity()[rigidbody_k];
    std::vector<double>& acc_k = rigidbodydatastate_->GetRefMutableAcceleration()[rigidbody_k];
    std::vector<double>& angacc_k =
        rigidbodydatastate_->GetRefMutableAngularAcceleration()[rigidbody_k];

    DRT::ParObject::ExtractfromPack(position, buffer, mass_k);
    for (int i = 0; i < 6; ++i) DRT::ParObject::ExtractfromPack(position, buffer, inertia_k[i]);
    for (int i = 0; i < 3; ++i) DRT::ParObject::ExtractfromPack(position, buffer, pos_k[i]);
    for (int i = 0; i < 4; ++i) DRT::ParObject::ExtractfromPack(position, buffer, rot_k[i]);
    for (int i = 0; i < 3; ++i) DRT::ParObject::ExtractfromPack(position, buffer, vel_k[i]);
    for (int i = 0; i < 3; ++i) DRT::ParObject::ExtractfromPack(position, buffer, angvel_k[i]);
    for (int i = 0; i < 3; ++i) DRT::ParObject::ExtractfromPack(position, buffer, acc_k[i]);
    for (int i = 0; i < 3; ++i) DRT::ParObject::ExtractfromPack(position, buffer, angacc_k[i]);
  }

  if (position != buffer.size())
    dserror("mismatch in size of data %d <-> %d", static_cast<int>(buffer.size()), position);
}

void PARTICLERIGIDBODY::RigidBodyHandler::UpdateRigidBodyOwnership()
{
  // store rigid bodies previously owned by this processor
  std::vector<int> previouslyownedrigidbodies = ownedrigidbodies_;

  // determine owned and hosted rigid bodies
  DetermineOwnedAndHostedRigidBodies();

  // relate owned rigid bodies to all hosting processors
  RelateOwnedRigidBodiesToHostingProcs();

  // communicate rigid body states
  CommunicateRigidBodyStates(previouslyownedrigidbodies);
}

void PARTICLERIGIDBODY::RigidBodyHandler::DetermineOwnedAndHostedRigidBodies()
{
  ownedrigidbodies_.clear();
  hostedrigidbodies_.clear();
  ownerofrigidbodies_.clear();

  // number of global ids
  const int numglobalids = rigidbodyuniqueglobalidhandler_->GetMaxGlobalId() + 1;

  // maximum number of particles per rigid body over all processors
  std::vector<std::pair<int, int>> maxnumberofparticlesperrigidbodyonproc(
      numglobalids, std::make_pair(0, myrank_));

  // get number of particle per rigid body on this processor
  for (const auto& it : affiliationpairs_->GetRefToAffiliationPairData())
    maxnumberofparticlesperrigidbodyonproc[it.second].first++;

  // get global ids of rigid bodies hosted (owned and non-owned) by this processor
  for (int rigidbody_k = 0; rigidbody_k < numglobalids; ++rigidbody_k)
    if (maxnumberofparticlesperrigidbodyonproc[rigidbody_k].first > 0)
      hostedrigidbodies_.push_back(rigidbody_k);

  // mpi communicator
  const auto* mpicomm = dynamic_cast<const Epetra_MpiComm*>(&comm_);
  if (!mpicomm) dserror("dynamic cast to Epetra_MpiComm failed!");

  // get maximum number of particles per rigid body over all processors
  MPI_Allreduce(MPI_IN_PLACE, maxnumberofparticlesperrigidbodyonproc.data(), numglobalids, MPI_2INT,
      MPI_MAXLOC, mpicomm->Comm());

  // get owner of all rigid bodies
  ownerofrigidbodies_.reserve(numglobalids);
  for (const auto& it : maxnumberofparticlesperrigidbodyonproc)
    ownerofrigidbodies_.push_back(it.second);

  // get global ids of rigid bodies owned by this processor
  for (const int rigidbody_k : hostedrigidbodies_)
    if (ownerofrigidbodies_[rigidbody_k] == myrank_) ownedrigidbodies_.push_back(rigidbody_k);
}

void PARTICLERIGIDBODY::RigidBodyHandler::RelateOwnedRigidBodiesToHostingProcs()
{
  // number of global ids
  const int numglobalids = rigidbodyuniqueglobalidhandler_->GetMaxGlobalId() + 1;

  // allocate memory
  ownedrigidbodiestohostingprocs_.assign(numglobalids, std::vector<int>(0));

  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  // iterate over hosted rigid bodies
  for (const int rigidbody_k : hostedrigidbodies_)
  {
    // owner of rigid body k
    const int owner_k = ownerofrigidbodies_[rigidbody_k];

    // communicate global id of rigid body to owning processor
    if (owner_k != myrank_)
    {
      // pack data for sending
      DRT::PackBuffer data;
      data.StartPacking();

      data.AddtoPack(rigidbody_k);

      sdata[owner_k].insert(sdata[owner_k].end(), data().begin(), data().end());
    }
  }

  // communicate data via non-buffered send from proc to proc
  PARTICLEENGINE::COMMUNICATION::ImmediateRecvBlockingSend(comm_, sdata, rdata);

  // unpack and store received data
  for (auto& p : rdata)
  {
    int msgsource = p.first;
    std::vector<char>& rmsg = p.second;

    std::vector<char>::size_type position = 0;

    while (position < rmsg.size())
    {
      const int rigidbody_k = DRT::ParObject::ExtractInt(position, rmsg);

      // insert processor id the gathered global id of rigid body is received from
      ownedrigidbodiestohostingprocs_[rigidbody_k].push_back(msgsource);
    }

    if (position != rmsg.size())
      dserror("mismatch in size of data %d <-> %d", static_cast<int>(rmsg.size()), position);
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::CommunicateRigidBodyStates(
    std::vector<int>& previouslyownedrigidbodies)
{
  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  // iterate over previously owned rigid bodies
  for (const int rigidbody_k : previouslyownedrigidbodies)
  {
    // owner of rigid body k
    const int owner_k = ownerofrigidbodies_[rigidbody_k];

    // get reference to rigid body states
    const double& mass_k = rigidbodydatastate_->GetRefMass()[rigidbody_k];
    const std::vector<double>& inertia_k = rigidbodydatastate_->GetRefInertia()[rigidbody_k];
    const std::vector<double>& pos_k = rigidbodydatastate_->GetRefPosition()[rigidbody_k];
    const std::vector<double>& rot_k = rigidbodydatastate_->GetRefRotation()[rigidbody_k];
    const std::vector<double>& vel_k = rigidbodydatastate_->GetRefVelocity()[rigidbody_k];
    const std::vector<double>& angvel_k = rigidbodydatastate_->GetRefAngularVelocity()[rigidbody_k];
    const std::vector<double>& acc_k = rigidbodydatastate_->GetRefAcceleration()[rigidbody_k];
    const std::vector<double>& angacc_k =
        rigidbodydatastate_->GetRefAngularAcceleration()[rigidbody_k];

    // communicate states to owning processor
    if (owner_k != myrank_)
    {
      // pack data for sending
      DRT::PackBuffer data;
      data.StartPacking();

      data.AddtoPack(rigidbody_k);
      data.AddtoPack(mass_k);
      for (int i = 0; i < 6; ++i) data.AddtoPack(inertia_k[i]);
      for (int i = 0; i < 3; ++i) data.AddtoPack(pos_k[i]);
      for (int i = 0; i < 4; ++i) data.AddtoPack(rot_k[i]);
      for (int i = 0; i < 3; ++i) data.AddtoPack(vel_k[i]);
      for (int i = 0; i < 3; ++i) data.AddtoPack(angvel_k[i]);
      for (int i = 0; i < 3; ++i) data.AddtoPack(acc_k[i]);
      for (int i = 0; i < 3; ++i) data.AddtoPack(angacc_k[i]);

      sdata[owner_k].insert(sdata[owner_k].end(), data().begin(), data().end());
    }
  }

  // communicate data via non-buffered send from proc to proc
  PARTICLEENGINE::COMMUNICATION::ImmediateRecvBlockingSend(comm_, sdata, rdata);

  // unpack and store received data
  for (auto& p : rdata)
  {
    std::vector<char>& rmsg = p.second;

    std::vector<char>::size_type position = 0;

    while (position < rmsg.size())
    {
      const int rigidbody_k = DRT::ParObject::ExtractInt(position, rmsg);

      // get reference to rigid body states
      double& mass_k = rigidbodydatastate_->GetRefMutableMass()[rigidbody_k];
      std::vector<double>& inertia_k = rigidbodydatastate_->GetRefMutableInertia()[rigidbody_k];
      std::vector<double>& pos_k = rigidbodydatastate_->GetRefMutablePosition()[rigidbody_k];
      std::vector<double>& rot_k = rigidbodydatastate_->GetRefMutableRotation()[rigidbody_k];
      std::vector<double>& vel_k = rigidbodydatastate_->GetRefMutableVelocity()[rigidbody_k];
      std::vector<double>& angvel_k =
          rigidbodydatastate_->GetRefMutableAngularVelocity()[rigidbody_k];
      std::vector<double>& acc_k = rigidbodydatastate_->GetRefMutableAcceleration()[rigidbody_k];
      std::vector<double>& angacc_k =
          rigidbodydatastate_->GetRefMutableAngularAcceleration()[rigidbody_k];

      DRT::ParObject::ExtractfromPack(position, rmsg, mass_k);
      for (int i = 0; i < 6; ++i) DRT::ParObject::ExtractfromPack(position, rmsg, inertia_k[i]);
      for (int i = 0; i < 3; ++i) DRT::ParObject::ExtractfromPack(position, rmsg, pos_k[i]);
      for (int i = 0; i < 4; ++i) DRT::ParObject::ExtractfromPack(position, rmsg, rot_k[i]);
      for (int i = 0; i < 3; ++i) DRT::ParObject::ExtractfromPack(position, rmsg, vel_k[i]);
      for (int i = 0; i < 3; ++i) DRT::ParObject::ExtractfromPack(position, rmsg, angvel_k[i]);
      for (int i = 0; i < 3; ++i) DRT::ParObject::ExtractfromPack(position, rmsg, acc_k[i]);
      for (int i = 0; i < 3; ++i) DRT::ParObject::ExtractfromPack(position, rmsg, angacc_k[i]);
    }

    if (position != rmsg.size())
      dserror("mismatch in size of data %d <-> %d", static_cast<int>(rmsg.size()), position);
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::ComputeRigidBodyMassQuantities()
{
  // clear partial mass quantities of rigid bodies
  ClearPartialMassQuantities();

  // compute partial mass quantities of rigid bodies
  ComputePartialMassQuantities();

  // gathered partial mass quantities of rigid bodies from all corresponding processors
  std::unordered_map<int, std::vector<double>> gatheredpartialmass;
  std::unordered_map<int, std::vector<std::vector<double>>> gatheredpartialinertia;
  std::unordered_map<int, std::vector<std::vector<double>>> gatheredpartialposition;

  // gather partial mass quantities of rigid bodies
  GatherPartialMassQuantities(gatheredpartialmass, gatheredpartialinertia, gatheredpartialposition);

  // compute full mass quantities of rigid bodies
  ComputeFullMassQuantities(gatheredpartialmass, gatheredpartialinertia, gatheredpartialposition);
}

void PARTICLERIGIDBODY::RigidBodyHandler::ClearPartialMassQuantities()
{
  // iterate over hosted rigid bodies
  for (const int rigidbody_k : hostedrigidbodies_)
  {
    // get pointer to rigid body states
    double* mass_k = &rigidbodydatastate_->GetRefMutableMass()[rigidbody_k];
    double* inertia_k = rigidbodydatastate_->GetRefMutableInertia()[rigidbody_k].data();
    double* pos_k = rigidbodydatastate_->GetRefMutablePosition()[rigidbody_k].data();

    // clear mass quantities
    mass_k[0] = 0.0;
    for (int i = 0; i < 6; ++i) inertia_k[i] = 0.0;
    PARTICLEINTERACTION::UTILS::VecClear(pos_k);
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::ComputePartialMassQuantities()
{
  // get reference to affiliation pair data
  const std::unordered_map<int, int>& affiliationpairdata =
      affiliationpairs_->GetRefToAffiliationPairData();

  // get particle container bundle
  PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->GetParticleContainerBundle();

  // get container of owned particles of rigid phase
  PARTICLEENGINE::ParticleContainer* container_i = particlecontainerbundle->GetSpecificContainer(
      PARTICLEENGINE::RigidPhase, PARTICLEENGINE::Owned);

#ifdef DEBUG
  if (static_cast<int>(affiliationpairdata.size()) != container_i->ParticlesStored())
    dserror("number of affiliation pairs and rigid particles not equal!");
#endif

  // loop over particles in container
  for (int particle_i = 0; particle_i < container_i->ParticlesStored(); ++particle_i)
  {
    // get global id of particle i
    const int* globalid_i = container_i->GetPtrToGlobalID(particle_i);

    auto it = affiliationpairdata.find(globalid_i[0]);

#ifdef DEBUG
    // no affiliation pair for current global id
    if (it == affiliationpairdata.end())
      dserror("no affiliated rigid body found for particle with global id %d", globalid_i[0]);
#endif

    // get global id of affiliated rigid body k
    const int rigidbody_k = it->second;

    // get pointer to rigid body states
    double* mass_k = &rigidbodydatastate_->GetRefMutableMass()[rigidbody_k];
    double* pos_k = rigidbodydatastate_->GetRefMutablePosition()[rigidbody_k].data();

    // get pointer to particle states
    const double* mass_i = container_i->GetPtrToState(PARTICLEENGINE::Mass, particle_i);
    const double* pos_i = container_i->GetPtrToState(PARTICLEENGINE::Position, particle_i);

    // sum contribution of particle i
    mass_k[0] += mass_i[0];
    PARTICLEINTERACTION::UTILS::VecAddScale(pos_k, mass_i[0], pos_i);
  }

  // iterate over hosted rigid bodies
  for (const int rigidbody_k : hostedrigidbodies_)
  {
    // get pointer to rigid body states
    const double* mass_k = &rigidbodydatastate_->GetRefMass()[rigidbody_k];
    double* pos_k = rigidbodydatastate_->GetRefMutablePosition()[rigidbody_k].data();

#ifdef DEBUG
    if (not(mass_k[0] > 0.0)) dserror("partial mass of rigid body %d is zero!", rigidbody_k);
#endif

    // determine center of gravity of (partial) rigid body k
    PARTICLEINTERACTION::UTILS::VecScale(pos_k, 1.0 / mass_k[0]);
  }

  // loop over particles in container
  for (int particle_i = 0; particle_i < container_i->ParticlesStored(); ++particle_i)
  {
    // get global id of particle i
    const int* globalid_i = container_i->GetPtrToGlobalID(particle_i);

    auto it = affiliationpairdata.find(globalid_i[0]);

#ifdef DEBUG
    // no affiliation pair for current global id
    if (it == affiliationpairdata.end())
      dserror("no affiliated rigid body found for particle with global id %d", globalid_i[0]);
#endif

    // get global id of affiliated rigid body k
    const int rigidbody_k = it->second;

    // get pointer to rigid body states
    const double* pos_k = rigidbodydatastate_->GetRefPosition()[rigidbody_k].data();
    double* inertia_k = rigidbodydatastate_->GetRefMutableInertia()[rigidbody_k].data();

    // get pointer to particle states
    const double* mass_i = container_i->GetPtrToState(PARTICLEENGINE::Mass, particle_i);
    const double* pos_i = container_i->GetPtrToState(PARTICLEENGINE::Position, particle_i);
    const double* inertia_i = container_i->GetPtrToState(PARTICLEENGINE::Inertia, particle_i);

    double r_ki[3];
    PARTICLEINTERACTION::UTILS::VecSet(r_ki, pos_k);
    PARTICLEINTERACTION::UTILS::VecSub(r_ki, pos_i);

    // sum contribution of particle i
    inertia_k[0] += inertia_i[0] + (r_ki[1] * r_ki[1] + r_ki[2] * r_ki[2]) * mass_i[0];
    inertia_k[1] += inertia_i[0] + (r_ki[0] * r_ki[0] + r_ki[2] * r_ki[2]) * mass_i[0];
    inertia_k[2] += inertia_i[0] + (r_ki[0] * r_ki[0] + r_ki[1] * r_ki[1]) * mass_i[0];
    inertia_k[3] -= r_ki[0] * r_ki[1] * mass_i[0];
    inertia_k[4] -= r_ki[0] * r_ki[2] * mass_i[0];
    inertia_k[5] -= r_ki[1] * r_ki[2] * mass_i[0];
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::GatherPartialMassQuantities(
    std::unordered_map<int, std::vector<double>>& gatheredpartialmass,
    std::unordered_map<int, std::vector<std::vector<double>>>& gatheredpartialinertia,
    std::unordered_map<int, std::vector<std::vector<double>>>& gatheredpartialposition)
{
  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  // iterate over hosted rigid bodies
  for (const int rigidbody_k : hostedrigidbodies_)
  {
    // owner of rigid body k
    const int owner_k = ownerofrigidbodies_[rigidbody_k];

    // get reference to rigid body states
    const double& mass_k = rigidbodydatastate_->GetRefMass()[rigidbody_k];
    const std::vector<double>& inertia_k = rigidbodydatastate_->GetRefInertia()[rigidbody_k];
    const std::vector<double>& pos_k = rigidbodydatastate_->GetRefPosition()[rigidbody_k];

    // rigid body k owned by this processor
    if (owner_k == myrank_)
    {
      // append to gathered partial mass quantities
      gatheredpartialmass[rigidbody_k].push_back(mass_k);
      gatheredpartialinertia[rigidbody_k].push_back(inertia_k);
      gatheredpartialposition[rigidbody_k].push_back(pos_k);
    }
    // communicate partial mass quantities to owning processor
    else
    {
      // pack data for sending
      DRT::PackBuffer data;
      data.StartPacking();

      data.AddtoPack(rigidbody_k);
      data.AddtoPack(mass_k);
      for (int i = 0; i < 6; ++i) data.AddtoPack(inertia_k[i]);
      for (int i = 0; i < 3; ++i) data.AddtoPack(pos_k[i]);

      sdata[owner_k].insert(sdata[owner_k].end(), data().begin(), data().end());
    }
  }

  // communicate data via non-buffered send from proc to proc
  PARTICLEENGINE::COMMUNICATION::ImmediateRecvBlockingSend(comm_, sdata, rdata);

  // unpack and store received data
  for (auto& p : rdata)
  {
    std::vector<char>& rmsg = p.second;

    std::vector<char>::size_type position = 0;

    while (position < rmsg.size())
    {
      const int rigidbody_k = DRT::ParObject::ExtractInt(position, rmsg);
      double mass_k = DRT::ParObject::ExtractDouble(position, rmsg);

      std::vector<double> inertia_k(6);
      for (int i = 0; i < 6; ++i) DRT::ParObject::ExtractfromPack(position, rmsg, inertia_k[i]);

      std::vector<double> pos_k(3);
      for (int i = 0; i < 3; ++i) DRT::ParObject::ExtractfromPack(position, rmsg, pos_k[i]);

      // append to gathered partial mass quantities
      gatheredpartialmass[rigidbody_k].push_back(mass_k);
      gatheredpartialinertia[rigidbody_k].push_back(inertia_k);
      gatheredpartialposition[rigidbody_k].push_back(pos_k);
    }

    if (position != rmsg.size())
      dserror("mismatch in size of data %d <-> %d", static_cast<int>(rmsg.size()), position);
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::ComputeFullMassQuantities(
    std::unordered_map<int, std::vector<double>>& gatheredpartialmass,
    std::unordered_map<int, std::vector<std::vector<double>>>& gatheredpartialinertia,
    std::unordered_map<int, std::vector<std::vector<double>>>& gatheredpartialposition)
{
  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    std::vector<double>& partialmass_k = gatheredpartialmass[rigidbody_k];
    std::vector<std::vector<double>>& partialpos_k = gatheredpartialposition[rigidbody_k];

    // number of partial mass quantities of rigid body k including this processor
    const int numpartial_k = ownedrigidbodiestohostingprocs_[rigidbody_k].size() + 1;

#ifdef DEBUG
    if (static_cast<int>(partialmass_k.size()) != numpartial_k or
        static_cast<int>(partialpos_k.size()) != numpartial_k)
      dserror("the number of partial mass quantities of rigid body %d do not match!", rigidbody_k);
#endif

    // get pointer to rigid body states
    double* mass_k = &rigidbodydatastate_->GetRefMutableMass()[rigidbody_k];
    double* pos_k = rigidbodydatastate_->GetRefMutablePosition()[rigidbody_k].data();

    // clear mass and position
    mass_k[0] = 0.0;
    PARTICLEINTERACTION::UTILS::VecClear(pos_k);

    // iterate over partial quantities
    for (int p = 0; p < numpartial_k; ++p)
    {
      // sum contribution of partial quantity
      mass_k[0] += partialmass_k[p];
      PARTICLEINTERACTION::UTILS::VecAddScale(pos_k, partialmass_k[p], partialpos_k[p].data());
    }

    // determine center of gravity of rigid body k
    PARTICLEINTERACTION::UTILS::VecScale(pos_k, 1.0 / mass_k[0]);
  }

  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    std::vector<double>& partialmass_k = gatheredpartialmass[rigidbody_k];
    std::vector<std::vector<double>>& partialpos_k = gatheredpartialposition[rigidbody_k];
    std::vector<std::vector<double>>& partialinertia_k = gatheredpartialinertia[rigidbody_k];

    // number of partial mass quantities of rigid body k including this processor
    const int numpartial_k = ownedrigidbodiestohostingprocs_[rigidbody_k].size() + 1;

#ifdef DEBUG
    if (static_cast<int>(partialmass_k.size()) != numpartial_k or
        static_cast<int>(partialinertia_k.size()) != numpartial_k)
      dserror("the number of partial mass quantities of rigid body %d do not match!", rigidbody_k);
#endif

    // get pointer to rigid body states
    const double* pos_k = rigidbodydatastate_->GetRefPosition()[rigidbody_k].data();
    double* inertia_k = rigidbodydatastate_->GetRefMutableInertia()[rigidbody_k].data();

    // clear inertia
    for (int i = 0; i < 6; ++i) inertia_k[i] = 0.0;

    // iterate over partial quantities
    for (int p = 0; p < numpartial_k; ++p)
    {
      double r_kp[3];
      PARTICLEINTERACTION::UTILS::VecSet(r_kp, pos_k);
      PARTICLEINTERACTION::UTILS::VecSub(r_kp, partialpos_k[p].data());

      // sum contribution of partial quantity
      for (int i = 0; i < 6; ++i) inertia_k[i] += partialinertia_k[p][i];

      inertia_k[0] += (r_kp[1] * r_kp[1] + r_kp[2] * r_kp[2]) * partialmass_k[p];
      inertia_k[1] += (r_kp[0] * r_kp[0] + r_kp[2] * r_kp[2]) * partialmass_k[p];
      inertia_k[2] += (r_kp[0] * r_kp[0] + r_kp[1] * r_kp[1]) * partialmass_k[p];
      inertia_k[3] -= r_kp[0] * r_kp[1] * partialmass_k[p];
      inertia_k[4] -= r_kp[0] * r_kp[2] * partialmass_k[p];
      inertia_k[5] -= r_kp[1] * r_kp[2] * partialmass_k[p];
    }
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::ClearRigidBodyForceAndTorque()
{
  // iterate over hosted rigid bodies
  for (const int rigidbody_k : hostedrigidbodies_)
  {
    // get pointer to rigid body states
    double* force_k = rigidbodydatastate_->GetRefMutableForce()[rigidbody_k].data();
    double* torque_k = rigidbodydatastate_->GetRefMutableTorque()[rigidbody_k].data();

    // clear force and torque of rigid body k
    PARTICLEINTERACTION::UTILS::VecClear(force_k);
    PARTICLEINTERACTION::UTILS::VecClear(torque_k);
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::ClearRigidParticleForce()
{
  // get particle container bundle
  PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->GetParticleContainerBundle();

  // get container of owned particles of rigid phase
  PARTICLEENGINE::ParticleContainer* container_i = particlecontainerbundle->GetSpecificContainer(
      PARTICLEENGINE::RigidPhase, PARTICLEENGINE::Owned);

  // clear force of all particles
  container_i->ClearState(PARTICLEENGINE::Force);
}

void PARTICLERIGIDBODY::RigidBodyHandler::ComputePartialForceAndTorque()
{
  // get reference to affiliation pair data
  const std::unordered_map<int, int>& affiliationpairdata =
      affiliationpairs_->GetRefToAffiliationPairData();

  // get particle container bundle
  PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->GetParticleContainerBundle();

  // get container of owned particles of rigid phase
  PARTICLEENGINE::ParticleContainer* container_i = particlecontainerbundle->GetSpecificContainer(
      PARTICLEENGINE::RigidPhase, PARTICLEENGINE::Owned);

#ifdef DEBUG
  if (static_cast<int>(affiliationpairdata.size()) != container_i->ParticlesStored())
    dserror("number of affiliation pairs and rigid particles not equal!");
#endif

  // loop over particles in container
  for (int particle_i = 0; particle_i < container_i->ParticlesStored(); ++particle_i)
  {
    // get global id of particle i
    const int* globalid_i = container_i->GetPtrToGlobalID(particle_i);

    auto it = affiliationpairdata.find(globalid_i[0]);

#ifdef DEBUG
    // no affiliation pair for current global id
    if (it == affiliationpairdata.end())
      dserror("no affiliated rigid body found for particle with global id %d", globalid_i[0]);
#endif

    // get global id of affiliated rigid body k
    const int rigidbody_k = it->second;

    // get pointer to rigid body states
    double* force_k = rigidbodydatastate_->GetRefMutableForce()[rigidbody_k].data();
    double* torque_k = rigidbodydatastate_->GetRefMutableTorque()[rigidbody_k].data();

    // get pointer to particle states
    const double* relpos_i =
        container_i->GetPtrToState(PARTICLEENGINE::RelativePosition, particle_i);
    const double* force_i = container_i->GetPtrToState(PARTICLEENGINE::Force, particle_i);

    // sum contribution of particle i
    PARTICLEINTERACTION::UTILS::VecAdd(force_k, force_i);
    PARTICLEINTERACTION::UTILS::VecAddCross(torque_k, relpos_i, force_i);
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::GatherPartialAndComputeFullForceAndTorque()
{
  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  // iterate over hosted rigid bodies
  for (const int rigidbody_k : hostedrigidbodies_)
  {
    // owner of rigid body k
    const int owner_k = ownerofrigidbodies_[rigidbody_k];

    // get reference to rigid body states
    const std::vector<double>& force_k = rigidbodydatastate_->GetRefForce()[rigidbody_k];
    const std::vector<double>& torque_k = rigidbodydatastate_->GetRefTorque()[rigidbody_k];

    // communicate partial force and torque to owning processor
    if (owner_k != myrank_)
    {
      // pack data for sending
      DRT::PackBuffer data;
      data.StartPacking();

      data.AddtoPack(rigidbody_k);
      for (int i = 0; i < 3; ++i) data.AddtoPack(force_k[i]);
      for (int i = 0; i < 3; ++i) data.AddtoPack(torque_k[i]);

      sdata[owner_k].insert(sdata[owner_k].end(), data().begin(), data().end());
    }
  }

  // communicate data via non-buffered send from proc to proc
  PARTICLEENGINE::COMMUNICATION::ImmediateRecvBlockingSend(comm_, sdata, rdata);

  // unpack and store received data
  for (auto& p : rdata)
  {
    std::vector<char>& rmsg = p.second;

    std::vector<char>::size_type position = 0;

    while (position < rmsg.size())
    {
      const int rigidbody_k = DRT::ParObject::ExtractInt(position, rmsg);

      std::vector<double> tmp_force_k(3);
      for (int i = 0; i < 3; ++i) DRT::ParObject::ExtractfromPack(position, rmsg, tmp_force_k[i]);

      std::vector<double> tmp_torque_k(3);
      for (int i = 0; i < 3; ++i) DRT::ParObject::ExtractfromPack(position, rmsg, tmp_torque_k[i]);

      // get pointer to rigid body states
      double* force_k = rigidbodydatastate_->GetRefMutableForce()[rigidbody_k].data();
      double* torque_k = rigidbodydatastate_->GetRefMutableTorque()[rigidbody_k].data();

      // sum gathered contribution to full force and torque
      PARTICLEINTERACTION::UTILS::VecAdd(force_k, tmp_force_k.data());
      PARTICLEINTERACTION::UTILS::VecAdd(torque_k, tmp_torque_k.data());
    }

    if (position != rmsg.size())
      dserror("mismatch in size of data %d <-> %d", static_cast<int>(rmsg.size()), position);
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::ComputeAccelerationsFromForceAndTorque()
{
  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    // get pointer to rigid body states
    const double* mass_k = &rigidbodydatastate_->GetRefMass()[rigidbody_k];
    const double* inertia_k = rigidbodydatastate_->GetRefInertia()[rigidbody_k].data();
    const double* rot_k = rigidbodydatastate_->GetRefRotation()[rigidbody_k].data();
    const double* force_k = rigidbodydatastate_->GetRefForce()[rigidbody_k].data();
    const double* torque_k = rigidbodydatastate_->GetRefTorque()[rigidbody_k].data();
    double* acc_k = rigidbodydatastate_->GetRefMutableAcceleration()[rigidbody_k].data();
    double* angacc_k = rigidbodydatastate_->GetRefMutableAngularAcceleration()[rigidbody_k].data();

    // compute acceleration of rigid body k
    PARTICLEINTERACTION::UTILS::VecAddScale(acc_k, 1 / mass_k[0], force_k);

    // compute inverse of rotation
    double invrot_k[4];
    UTILS::QuaternionInvert(invrot_k, rot_k);

    // get torque in the reference frame
    double reftorque_k[3];
    UTILS::QuaternionRotateVector(reftorque_k, invrot_k, torque_k);

    // determinant of mass moment of inertia
    const double det_inertia_k =
        inertia_k[0] * inertia_k[1] * inertia_k[2] + inertia_k[3] * inertia_k[4] * inertia_k[5] +
        inertia_k[3] * inertia_k[4] * inertia_k[5] - inertia_k[1] * inertia_k[4] * inertia_k[4] -
        inertia_k[2] * inertia_k[3] * inertia_k[3] - inertia_k[0] * inertia_k[5] * inertia_k[5];

    // no mass moment of inertia
    if (std::abs(det_inertia_k) < 1E-14) continue;

    // evaluate angular acceleration of rigid body k in the reference frame
    double refangacc_k[3];
    refangacc_k[0] = reftorque_k[0] * (inertia_k[1] * inertia_k[2] - inertia_k[5] * inertia_k[5]) +
                     reftorque_k[1] * (inertia_k[4] * inertia_k[5] - inertia_k[2] * inertia_k[3]) +
                     reftorque_k[2] * (inertia_k[3] * inertia_k[5] - inertia_k[1] * inertia_k[4]);

    refangacc_k[1] = reftorque_k[0] * (inertia_k[4] * inertia_k[5] - inertia_k[2] * inertia_k[3]) +
                     reftorque_k[1] * (inertia_k[0] * inertia_k[2] - inertia_k[4] * inertia_k[4]) +
                     reftorque_k[2] * (inertia_k[3] * inertia_k[4] - inertia_k[0] * inertia_k[5]);

    refangacc_k[2] = reftorque_k[0] * (inertia_k[3] * inertia_k[5] - inertia_k[1] * inertia_k[4]) +
                     reftorque_k[1] * (inertia_k[3] * inertia_k[4] - inertia_k[0] * inertia_k[5]) +
                     reftorque_k[2] * (inertia_k[0] * inertia_k[1] - inertia_k[3] * inertia_k[3]);

    PARTICLEINTERACTION::UTILS::VecScale(refangacc_k, 1.0 / det_inertia_k);

    // compute angular acceleration of rigid body k in the rotating frame
    double temp[3];
    UTILS::QuaternionRotateVector(temp, rot_k, refangacc_k);
    PARTICLEINTERACTION::UTILS::VecAdd(angacc_k, temp);
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::ClearRigidBodyOrientation()
{
  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    // get pointer to rigid body states
    double* rot_k = rigidbodydatastate_->GetRefMutableRotation()[rigidbody_k].data();

    // initialize rotation of rigid body k
    UTILS::QuaternionClear(rot_k);
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::UpdateRigidBodyPositions(const double timeincrement)
{
  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    // get pointer to rigid body states
    double* pos_k = rigidbodydatastate_->GetRefMutablePosition()[rigidbody_k].data();
    double* rot_k = rigidbodydatastate_->GetRefMutableRotation()[rigidbody_k].data();
    const double* vel_k = rigidbodydatastate_->GetRefVelocity()[rigidbody_k].data();
    const double* angvel_k = rigidbodydatastate_->GetRefAngularVelocity()[rigidbody_k].data();

    // update position
    PARTICLEINTERACTION::UTILS::VecAddScale(pos_k, timeincrement, vel_k);

    // save current rotation
    double curr_rot_k[4];
    UTILS::QuaternionSet(curr_rot_k, rot_k);

    // get rotation increment
    double phi_k[3];
    PARTICLEINTERACTION::UTILS::VecSetScale(phi_k, timeincrement, angvel_k);

    double incr_rot_k[4];
    UTILS::QuaternionFromAngle(incr_rot_k, phi_k);

    // update rotation
    UTILS::QuaternionProduct(rot_k, incr_rot_k, curr_rot_k);
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::UpdateRigidBodyVelocities(const double timeincrement)
{
  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    // get pointer to rigid body states
    double* vel_k = rigidbodydatastate_->GetRefMutableVelocity()[rigidbody_k].data();
    double* angvel_k = rigidbodydatastate_->GetRefMutableAngularVelocity()[rigidbody_k].data();
    const double* acc_k = rigidbodydatastate_->GetRefAcceleration()[rigidbody_k].data();
    const double* angacc_k = rigidbodydatastate_->GetRefAngularAcceleration()[rigidbody_k].data();

    // update velocities
    PARTICLEINTERACTION::UTILS::VecAddScale(vel_k, timeincrement, acc_k);
    PARTICLEINTERACTION::UTILS::VecAddScale(angvel_k, timeincrement, angacc_k);
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::ClearRigidBodyAccelerations()
{
  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    // get pointer to rigid body states
    double* acc_k = rigidbodydatastate_->GetRefMutableAcceleration()[rigidbody_k].data();
    double* angacc_k = rigidbodydatastate_->GetRefMutableAngularAcceleration()[rigidbody_k].data();

    // clear accelerations of rigid body k
    PARTICLEINTERACTION::UTILS::VecClear(acc_k);
    PARTICLEINTERACTION::UTILS::VecClear(angacc_k);
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::BroadcastRigidBodyPositions()
{
  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    // get reference to hosting processors of rigid body k
    std::vector<int>& hostingprocs_k = ownedrigidbodiestohostingprocs_[rigidbody_k];

    // get reference to rigid body states
    const std::vector<double>& pos_k = rigidbodydatastate_->GetRefPosition()[rigidbody_k];
    const std::vector<double>& rot_k = rigidbodydatastate_->GetRefRotation()[rigidbody_k];

    // pack data for sending
    DRT::PackBuffer data;
    data.StartPacking();

    data.AddtoPack(rigidbody_k);

    for (int i = 0; i < 3; ++i) data.AddtoPack(pos_k[i]);
    for (int i = 0; i < 4; ++i) data.AddtoPack(rot_k[i]);

    for (int torank : hostingprocs_k)
      sdata[torank].insert(sdata[torank].end(), data().begin(), data().end());
  }

  // communicate data via non-buffered send from proc to proc
  PARTICLEENGINE::COMMUNICATION::ImmediateRecvBlockingSend(comm_, sdata, rdata);

  // unpack and store received data
  for (auto& p : rdata)
  {
    std::vector<char>& rmsg = p.second;

    std::vector<char>::size_type position = 0;

    while (position < rmsg.size())
    {
      const int rigidbody_k = DRT::ParObject::ExtractInt(position, rmsg);

      // get reference to rigid body states
      std::vector<double>& pos_k = rigidbodydatastate_->GetRefMutablePosition()[rigidbody_k];
      std::vector<double>& rot_k = rigidbodydatastate_->GetRefMutableRotation()[rigidbody_k];

      for (int i = 0; i < 3; ++i) DRT::ParObject::ExtractfromPack(position, rmsg, pos_k[i]);
      for (int i = 0; i < 4; ++i) DRT::ParObject::ExtractfromPack(position, rmsg, rot_k[i]);
    }

    if (position != rmsg.size())
      dserror("mismatch in size of data %d <-> %d", static_cast<int>(rmsg.size()), position);
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::BroadcastRigidBodyVelocities()
{
  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    // get reference to hosting processors of rigid body k
    std::vector<int>& hostingprocs_k = ownedrigidbodiestohostingprocs_[rigidbody_k];

    // get reference to rigid body states
    const std::vector<double>& vel_k = rigidbodydatastate_->GetRefVelocity()[rigidbody_k];
    const std::vector<double>& angvel_k = rigidbodydatastate_->GetRefAngularVelocity()[rigidbody_k];

    // pack data for sending
    DRT::PackBuffer data;
    data.StartPacking();

    data.AddtoPack(rigidbody_k);

    for (int i = 0; i < 3; ++i) data.AddtoPack(vel_k[i]);
    for (int i = 0; i < 3; ++i) data.AddtoPack(angvel_k[i]);

    for (int torank : hostingprocs_k)
      sdata[torank].insert(sdata[torank].end(), data().begin(), data().end());
  }

  // communicate data via non-buffered send from proc to proc
  PARTICLEENGINE::COMMUNICATION::ImmediateRecvBlockingSend(comm_, sdata, rdata);

  // unpack and store received data
  for (auto& p : rdata)
  {
    std::vector<char>& rmsg = p.second;

    std::vector<char>::size_type position = 0;

    while (position < rmsg.size())
    {
      const int rigidbody_k = DRT::ParObject::ExtractInt(position, rmsg);

      // get reference to rigid body states
      std::vector<double>& vel_k = rigidbodydatastate_->GetRefMutableVelocity()[rigidbody_k];
      std::vector<double>& angvel_k =
          rigidbodydatastate_->GetRefMutableAngularVelocity()[rigidbody_k];

      for (int i = 0; i < 3; ++i) DRT::ParObject::ExtractfromPack(position, rmsg, vel_k[i]);
      for (int i = 0; i < 3; ++i) DRT::ParObject::ExtractfromPack(position, rmsg, angvel_k[i]);
    }

    if (position != rmsg.size())
      dserror("mismatch in size of data %d <-> %d", static_cast<int>(rmsg.size()), position);
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::BroadcastRigidBodyAccelerations()
{
  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    // get reference to hosting processors of rigid body k
    std::vector<int>& hostingprocs_k = ownedrigidbodiestohostingprocs_[rigidbody_k];

    // get reference to rigid body states
    const std::vector<double>& acc_k = rigidbodydatastate_->GetRefAcceleration()[rigidbody_k];
    const std::vector<double>& angacc_k =
        rigidbodydatastate_->GetRefAngularAcceleration()[rigidbody_k];

    // pack data for sending
    DRT::PackBuffer data;
    data.StartPacking();

    data.AddtoPack(rigidbody_k);

    for (int i = 0; i < 3; ++i) data.AddtoPack(acc_k[i]);
    for (int i = 0; i < 3; ++i) data.AddtoPack(angacc_k[i]);

    for (int torank : hostingprocs_k)
      sdata[torank].insert(sdata[torank].end(), data().begin(), data().end());
  }

  // communicate data via non-buffered send from proc to proc
  PARTICLEENGINE::COMMUNICATION::ImmediateRecvBlockingSend(comm_, sdata, rdata);

  // unpack and store received data
  for (auto& p : rdata)
  {
    std::vector<char>& rmsg = p.second;

    std::vector<char>::size_type position = 0;

    while (position < rmsg.size())
    {
      const int rigidbody_k = DRT::ParObject::ExtractInt(position, rmsg);

      // get reference to rigid body states
      std::vector<double>& acc_k = rigidbodydatastate_->GetRefMutableAcceleration()[rigidbody_k];
      std::vector<double>& angacc_k =
          rigidbodydatastate_->GetRefMutableAngularAcceleration()[rigidbody_k];

      for (int i = 0; i < 3; ++i) DRT::ParObject::ExtractfromPack(position, rmsg, acc_k[i]);
      for (int i = 0; i < 3; ++i) DRT::ParObject::ExtractfromPack(position, rmsg, angacc_k[i]);
    }

    if (position != rmsg.size())
      dserror("mismatch in size of data %d <-> %d", static_cast<int>(rmsg.size()), position);
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::SetRigidParticleRelativePositionInBodyFrame()
{
  // get reference to affiliation pair data
  const std::unordered_map<int, int>& affiliationpairdata =
      affiliationpairs_->GetRefToAffiliationPairData();

  // get particle container bundle
  PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->GetParticleContainerBundle();

  // get container of owned particles of rigid phase
  PARTICLEENGINE::ParticleContainer* container_i = particlecontainerbundle->GetSpecificContainer(
      PARTICLEENGINE::RigidPhase, PARTICLEENGINE::Owned);

#ifdef DEBUG
  if (static_cast<int>(affiliationpairdata.size()) != container_i->ParticlesStored())
    dserror("number of affiliation pairs and rigid particles not equal!");
#endif

  // loop over particles in container
  for (int particle_i = 0; particle_i < container_i->ParticlesStored(); ++particle_i)
  {
    // get global id of particle i
    const int* globalid_i = container_i->GetPtrToGlobalID(particle_i);

    auto it = affiliationpairdata.find(globalid_i[0]);

#ifdef DEBUG
    // no affiliation pair for current global id
    if (it == affiliationpairdata.end())
      dserror("no affiliated rigid body found for particle with global id %d", globalid_i[0]);
#endif

    // get global id of affiliated rigid body k
    const int rigidbody_k = it->second;

    // get pointer to rigid body states
    const double* pos_k = rigidbodydatastate_->GetRefPosition()[rigidbody_k].data();

    // get pointer to particle states
    const double* pos_i = container_i->GetPtrToState(PARTICLEENGINE::Position, particle_i);
    double* relposbody_i =
        container_i->GetPtrToState(PARTICLEENGINE::RelativePositionBodyFrame, particle_i);

    PARTICLEINTERACTION::UTILS::VecSet(relposbody_i, pos_i);
    PARTICLEINTERACTION::UTILS::VecSub(relposbody_i, pos_k);
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::UpdateRigidParticleRelativePosition()
{
  // get reference to affiliation pair data
  const std::unordered_map<int, int>& affiliationpairdata =
      affiliationpairs_->GetRefToAffiliationPairData();

  // get particle container bundle
  PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->GetParticleContainerBundle();

  // get container of owned particles of rigid phase
  PARTICLEENGINE::ParticleContainer* container_i = particlecontainerbundle->GetSpecificContainer(
      PARTICLEENGINE::RigidPhase, PARTICLEENGINE::Owned);

#ifdef DEBUG
  if (static_cast<int>(affiliationpairdata.size()) != container_i->ParticlesStored())
    dserror("number of affiliation pairs and rigid particles not equal!");
#endif

  // loop over particles in container
  for (int particle_i = 0; particle_i < container_i->ParticlesStored(); ++particle_i)
  {
    // get global id of particle i
    const int* globalid_i = container_i->GetPtrToGlobalID(particle_i);

    auto it = affiliationpairdata.find(globalid_i[0]);

#ifdef DEBUG
    // no affiliation pair for current global id
    if (it == affiliationpairdata.end())
      dserror("no affiliated rigid body found for particle with global id %d", globalid_i[0]);
#endif

    // get global id of affiliated rigid body k
    const int rigidbody_k = it->second;

    // get pointer to rigid body states
    const double* rot_k = rigidbodydatastate_->GetRefRotation()[rigidbody_k].data();

    // get pointer to particle states
    const double* relposbody_i =
        container_i->GetPtrToState(PARTICLEENGINE::RelativePositionBodyFrame, particle_i);
    double* relpos_i = container_i->GetPtrToState(PARTICLEENGINE::RelativePosition, particle_i);

    // update relative position of particle i
    UTILS::QuaternionRotateVector(relpos_i, rot_k, relposbody_i);
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::SetRigidParticlePosition()
{
  // get reference to affiliation pair data
  const std::unordered_map<int, int>& affiliationpairdata =
      affiliationpairs_->GetRefToAffiliationPairData();

  // get particle container bundle
  PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->GetParticleContainerBundle();

  // get container of owned particles of rigid phase
  PARTICLEENGINE::ParticleContainer* container_i = particlecontainerbundle->GetSpecificContainer(
      PARTICLEENGINE::RigidPhase, PARTICLEENGINE::Owned);

#ifdef DEBUG
  if (static_cast<int>(affiliationpairdata.size()) != container_i->ParticlesStored())
    dserror("number of affiliation pairs and rigid particles not equal!");
#endif

  // loop over particles in container
  for (int particle_i = 0; particle_i < container_i->ParticlesStored(); ++particle_i)
  {
    // get global id of particle i
    const int* globalid_i = container_i->GetPtrToGlobalID(particle_i);

    auto it = affiliationpairdata.find(globalid_i[0]);

#ifdef DEBUG
    // no affiliation pair for current global id
    if (it == affiliationpairdata.end())
      dserror("no affiliated rigid body found for particle with global id %d", globalid_i[0]);
#endif

    // get global id of affiliated rigid body k
    const int rigidbody_k = it->second;

    // get pointer to rigid body states
    const double* pos_k = rigidbodydatastate_->GetRefPosition()[rigidbody_k].data();

    // get pointer to particle states
    const double* relpos_i =
        container_i->GetPtrToState(PARTICLEENGINE::RelativePosition, particle_i);
    double* pos_i = container_i->GetPtrToState(PARTICLEENGINE::Position, particle_i);

    // set position of particle i
    PARTICLEINTERACTION::UTILS::VecSet(pos_i, pos_k);
    PARTICLEINTERACTION::UTILS::VecAdd(pos_i, relpos_i);
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::SetRigidParticleVelocities()
{
  // get reference to affiliation pair data
  const std::unordered_map<int, int>& affiliationpairdata =
      affiliationpairs_->GetRefToAffiliationPairData();

  // get particle container bundle
  PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->GetParticleContainerBundle();

  // get container of owned particles of rigid phase
  PARTICLEENGINE::ParticleContainer* container_i = particlecontainerbundle->GetSpecificContainer(
      PARTICLEENGINE::RigidPhase, PARTICLEENGINE::Owned);

#ifdef DEBUG
  if (static_cast<int>(affiliationpairdata.size()) != container_i->ParticlesStored())
    dserror("number of affiliation pairs and rigid particles not equal!");
#endif

  // loop over particles in container
  for (int particle_i = 0; particle_i < container_i->ParticlesStored(); ++particle_i)
  {
    // get global id of particle i
    const int* globalid_i = container_i->GetPtrToGlobalID(particle_i);

    auto it = affiliationpairdata.find(globalid_i[0]);

#ifdef DEBUG
    // no affiliation pair for current global id
    if (it == affiliationpairdata.end())
      dserror("no affiliated rigid body found for particle with global id %d", globalid_i[0]);
#endif

    // get global id of affiliated rigid body k
    const int rigidbody_k = it->second;

    // get pointer to rigid body states
    const double* vel_k = rigidbodydatastate_->GetRefVelocity()[rigidbody_k].data();
    const double* angvel_k = rigidbodydatastate_->GetRefAngularVelocity()[rigidbody_k].data();

    // get pointer to particle states
    const double* relpos_i =
        container_i->GetPtrToState(PARTICLEENGINE::RelativePosition, particle_i);
    double* vel_i = container_i->GetPtrToState(PARTICLEENGINE::Velocity, particle_i);
    double* angvel_i = container_i->CondGetPtrToState(PARTICLEENGINE::AngularVelocity, particle_i);

    // set velocities of particle i
    PARTICLEINTERACTION::UTILS::VecSet(vel_i, vel_k);
    PARTICLEINTERACTION::UTILS::VecAddCross(vel_i, angvel_k, relpos_i);
    if (angvel_i) PARTICLEINTERACTION::UTILS::VecSet(angvel_i, angvel_k);
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::SetRigidParticleAccelerations()
{
  // get reference to affiliation pair data
  const std::unordered_map<int, int>& affiliationpairdata =
      affiliationpairs_->GetRefToAffiliationPairData();

  // get particle container bundle
  PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->GetParticleContainerBundle();

  // get container of owned particles of rigid phase
  PARTICLEENGINE::ParticleContainer* container_i = particlecontainerbundle->GetSpecificContainer(
      PARTICLEENGINE::RigidPhase, PARTICLEENGINE::Owned);

#ifdef DEBUG
  if (static_cast<int>(affiliationpairdata.size()) != container_i->ParticlesStored())
    dserror("number of affiliation pairs and rigid particles not equal!");
#endif

  // loop over particles in container
  for (int particle_i = 0; particle_i < container_i->ParticlesStored(); ++particle_i)
  {
    // get global id of particle i
    const int* globalid_i = container_i->GetPtrToGlobalID(particle_i);

    auto it = affiliationpairdata.find(globalid_i[0]);

#ifdef DEBUG
    // no affiliation pair for current global id
    if (it == affiliationpairdata.end())
      dserror("no affiliated rigid body found for particle with global id %d", globalid_i[0]);
#endif

    // get global id of affiliated rigid body k
    const int rigidbody_k = it->second;

    // get pointer to rigid body states
    const double* angvel_k = rigidbodydatastate_->GetRefAngularVelocity()[rigidbody_k].data();
    const double* acc_k = rigidbodydatastate_->GetRefAcceleration()[rigidbody_k].data();
    const double* angacc_k = rigidbodydatastate_->GetRefAngularAcceleration()[rigidbody_k].data();

    // get pointer to particle states
    const double* relpos_i =
        container_i->GetPtrToState(PARTICLEENGINE::RelativePosition, particle_i);
    double* acc_i = container_i->GetPtrToState(PARTICLEENGINE::Acceleration, particle_i);
    double* angacc_i =
        container_i->CondGetPtrToState(PARTICLEENGINE::AngularAcceleration, particle_i);

    // evaluate relative velocity of particle i
    double relvel_i[3];
    PARTICLEINTERACTION::UTILS::VecSetCross(relvel_i, angvel_k, relpos_i);

    // set accelerations of particle i
    PARTICLEINTERACTION::UTILS::VecSet(acc_i, acc_k);
    PARTICLEINTERACTION::UTILS::VecAddCross(acc_i, angacc_k, relpos_i);
    PARTICLEINTERACTION::UTILS::VecAddCross(acc_i, angvel_k, relvel_i);
    if (angacc_i) PARTICLEINTERACTION::UTILS::VecSet(angacc_i, angacc_k);
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::EvaluateRigidBodyMelting(
    const std::vector<PARTICLEENGINE::ParticleTypeToType>& particlesfromphasetophase)
{
  // get reference to affiliation pair data
  std::unordered_map<int, int>& affiliationpairdata =
      affiliationpairs_->GetRefToAffiliationPairData();

  // iterate over particle phase change tuples
  for (const auto& particletypetotype : particlesfromphasetophase)
  {
    PARTICLEENGINE::TypeEnum type_source;
    int globalid_i;
    std::tie(type_source, std::ignore, globalid_i) = particletypetotype;

    if (type_source == PARTICLEENGINE::RigidPhase)
    {
      auto it = affiliationpairdata.find(globalid_i);

#ifdef DEBUG
      // no affiliation pair for current global id
      if (it == affiliationpairdata.end())
        dserror("no affiliated rigid body found for particle with global id %d", globalid_i);
#endif

      // erase affiliation pair
      affiliationpairdata.erase(it);
    }
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::EvaluateRigidBodySolidification(
    const std::vector<PARTICLEENGINE::ParticleTypeToType>& particlesfromphasetophase)
{
  // get search radius
  const double searchradius = params_.get<double>("RIGID_BODY_PHASECHANGE_RADIUS");
  if (not(searchradius > 0.0)) dserror("search radius not positive!");

  // get reference to affiliation pair data
  std::unordered_map<int, int>& affiliationpairdata =
      affiliationpairs_->GetRefToAffiliationPairData();

  // get particle container bundle
  PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->GetParticleContainerBundle();

  // get container of owned particles of rigid phase
  PARTICLEENGINE::ParticleContainer* container_i = particlecontainerbundle->GetSpecificContainer(
      PARTICLEENGINE::RigidPhase, PARTICLEENGINE::Owned);

  // iterate over particle phase change tuples
  for (const auto& particletypetotype : particlesfromphasetophase)
  {
    PARTICLEENGINE::TypeEnum type_target;
    int globalid_i;
    std::tie(std::ignore, type_target, globalid_i) = particletypetotype;

    if (type_target == PARTICLEENGINE::RigidPhase)
    {
      // get local index in specific particle container
      PARTICLEENGINE::LocalIndexTupleShrdPtr localindextuple =
          particleengineinterface_->GetLocalIndexInSpecificContainer(globalid_i);

#ifdef DEBUG
      if (not localindextuple)
        dserror("particle with global id %d not found on this processor!", globalid_i);

      // access values of local index tuples of particle i
      PARTICLEENGINE::TypeEnum type_i;
      PARTICLEENGINE::StatusEnum status_i;
      std::tie(type_i, status_i, std::ignore) = *localindextuple;

      if (type_i != PARTICLEENGINE::RigidPhase)
        dserror("particle with global id %d not of particle type '%s'!", globalid_i,
            PARTICLEENGINE::EnumToTypeName(PARTICLEENGINE::RigidPhase).c_str());

      if (status_i == PARTICLEENGINE::Ghosted)
        dserror("particle with global id %d not owned on this processor!", globalid_i);
#endif

      // access values of local index tuples of particle i
      int particle_i;
      std::tie(std::ignore, std::ignore, particle_i) = *localindextuple;

      // get pointer to particle states
      const double* pos_i = container_i->GetPtrToState(PARTICLEENGINE::Position, particle_i);
      double* rigidbodycolor_i =
          container_i->GetPtrToState(PARTICLEENGINE::RigidBodyColor, particle_i);

      // get particles within radius
      std::vector<PARTICLEENGINE::LocalIndexTuple> neighboringparticles;
      particleengineinterface_->GetParticlesWithinRadius(pos_i, searchradius, neighboringparticles);

      // minimum distance between particles
      double mindist = searchradius;

      // iterate over neighboring particles
      for (const auto& neighboringparticle : neighboringparticles)
      {
        // access values of local index tuple of particle j
        PARTICLEENGINE::TypeEnum type_j;
        PARTICLEENGINE::StatusEnum status_j;
        int particle_j;
        std::tie(type_j, status_j, particle_j) = neighboringparticle;

        // evaluation only for rigid particles
        if (type_j != PARTICLEENGINE::RigidPhase) continue;

        // get container of particles of current particle type
        PARTICLEENGINE::ParticleContainer* container_j =
            particlecontainerbundle->GetSpecificContainer(type_j, status_j);

        // get pointer to particle states
        const double* pos_j = container_j->GetPtrToState(PARTICLEENGINE::Position, particle_j);
        const double* rigidbodycolor_j =
            container_j->GetPtrToState(PARTICLEENGINE::RigidBodyColor, particle_j);

        // vector from particle i to j
        double r_ji[3];
        PARTICLEINTERACTION::UTILS::VecSet(r_ji, pos_j);
        PARTICLEINTERACTION::UTILS::VecSub(r_ji, pos_i);

        // absolute distance between particles
        const double absdist = PARTICLEINTERACTION::UTILS::VecNormTwo(r_ji);

        // set rigid body color to the one of the closest rigid particle j
        if (absdist < mindist)
        {
          mindist = absdist;
          rigidbodycolor_i[0] = rigidbodycolor_j[0];
        }
      }

      // get global id of affiliated rigid body k
      const int rigidbody_k = std::round(rigidbodycolor_i[0]);

      // insert affiliation pair
      affiliationpairdata.insert(std::make_pair(globalid_i, rigidbody_k));
    }
  }
}

void PARTICLERIGIDBODY::RigidBodyHandler::SetRigidBodyVelocitiesAfterPhaseChange(
    const std::vector<std::vector<double>>& previousposition)
{
  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    // get pointer to rigid body states
    const double* pos_k = rigidbodydatastate_->GetRefPosition()[rigidbody_k].data();
    const double* angvel_k = rigidbodydatastate_->GetRefAngularVelocity()[rigidbody_k].data();
    double* vel_k = rigidbodydatastate_->GetRefMutableVelocity()[rigidbody_k].data();

    const double* prevpos_k = previousposition[rigidbody_k].data();

    // vector from previous to current position of rigid body k
    double prev_r_kk[3];
    PARTICLEINTERACTION::UTILS::VecSet(prev_r_kk, pos_k);
    PARTICLEINTERACTION::UTILS::VecSub(prev_r_kk, prevpos_k);

    // update velocity
    PARTICLEINTERACTION::UTILS::VecAddCross(vel_k, angvel_k, prev_r_kk);
  }
}