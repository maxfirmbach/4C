/*---------------------------------------------------------------------------*/
/*! \file
\brief rigid body result test for particle simulations
\level 1
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_particle_rigidbody_result_test.hpp"

#include "4C_io_linedefinition.hpp"
#include "4C_particle_rigidbody_datastate.hpp"
#include "4C_particle_rigidbody_interface.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
ParticleRigidBody::RigidBodyResultTest::RigidBodyResultTest() : Core::UTILS::ResultTest("RIGIDBODY")
{
  // empty constructor
}

void ParticleRigidBody::RigidBodyResultTest::init()
{
  // nothing to do
}

void ParticleRigidBody::RigidBodyResultTest::setup(
    const std::shared_ptr<ParticleRigidBody::RigidBodyHandlerInterface> particlerigidbodyinterface)
{
  // set interface to rigid body handler
  particlerigidbodyinterface_ = particlerigidbodyinterface;
}

void ParticleRigidBody::RigidBodyResultTest::TestSpecial(
    Input::LineDefinition& res, int& nerr, int& test_count)
{
  // get owned rigid bodies by this processor
  const std::vector<int>& ownedrigidbodies = particlerigidbodyinterface_->GetOwnedRigidBodies();

  // extract global id of rigid body
  int globalid;
  res.extract_int("ID", globalid);

  // rigid body owned by this processor
  if (std::find(ownedrigidbodies.begin(), ownedrigidbodies.end(), globalid) !=
      ownedrigidbodies.end())
  {
    // get rigid body data state container
    std::shared_ptr<ParticleRigidBody::RigidBodyDataState> rigidbodydatastate =
        particlerigidbodyinterface_->get_rigid_body_data_state();

    // get result
    std::string quantity;
    res.extract_string("QUANTITY", quantity);

    // init actual result
    double actresult = 0.0;

    // position
    if (quantity == "posx" or quantity == "posy" or quantity == "posz")
    {
      // get reference to rigid body position
      const std::vector<std::vector<double>>& pos = rigidbodydatastate->GetRefPosition();

      // get actual result
      if (quantity == "posx")
        actresult = pos[globalid][0];
      else if (quantity == "posy")
        actresult = pos[globalid][1];
      else if (quantity == "posz")
        actresult = pos[globalid][2];
    }
    // velocity
    else if (quantity == "velx" or quantity == "vely" or quantity == "velz")
    {
      // get reference to rigid body velocity
      const std::vector<std::vector<double>>& vel = rigidbodydatastate->GetRefVelocity();

      // get actual result
      if (quantity == "velx")
        actresult = vel[globalid][0];
      else if (quantity == "vely")
        actresult = vel[globalid][1];
      else if (quantity == "velz")
        actresult = vel[globalid][2];
    }
    // velocity
    else if (quantity == "angvelx" or quantity == "angvely" or quantity == "angvelz")
    {
      // get reference to rigid body angular velocity
      const std::vector<std::vector<double>>& angvel =
          rigidbodydatastate->get_ref_angular_velocity();

      // get actual result
      if (quantity == "angvelx")
        actresult = angvel[globalid][0];
      else if (quantity == "angvely")
        actresult = angvel[globalid][1];
      else if (quantity == "angvelz")
        actresult = angvel[globalid][2];
    }
    // mass
    else if (quantity == "mass")
    {
      // get reference to rigid body mass
      const std::vector<double>& mass = rigidbodydatastate->GetRefMass();

      // get actual result
      actresult = mass[globalid];
    }
    else
      FOUR_C_THROW("result check failed with unknown quantity '%s'!", quantity.c_str());

    // compare values
    const int err = compare_values(actresult, "SPECIAL", res);
    nerr += err;
    test_count++;
  }
}

FOUR_C_NAMESPACE_CLOSE