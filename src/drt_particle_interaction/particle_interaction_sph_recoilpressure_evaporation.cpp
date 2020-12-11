/*---------------------------------------------------------------------------*/
/*! \file
\brief evaporation induced recoil pressure handler for smoothed particle hydrodynamics (SPH)
       interactions
\level 3
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "particle_interaction_sph_recoilpressure_evaporation.H"

#include "particle_interaction_material_handler.H"

#include "particle_interaction_utils.H"

#include "../drt_particle_engine/particle_engine_interface.H"
#include "../drt_particle_engine/particle_container.H"

#include "../drt_lib/drt_dserror.H"

#include <Teuchos_TimeMonitor.hpp>

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
PARTICLEINTERACTION::SPHRecoilPressureEvaporation::SPHRecoilPressureEvaporation(
    const Teuchos::ParameterList& params)
    : params_sph_(params),
      evaporatingphase_(PARTICLEENGINE::Phase1),
      recoilboilingtemp_(params_sph_.get<double>("VAPOR_RECOIL_BOILINGTEMPERATURE")),
      recoil_pfac_(params_sph_.get<double>("VAPOR_RECOIL_PFAC")),
      recoil_tfac_(params_sph_.get<double>("VAPOR_RECOIL_TFAC"))
{
  // empty constructor
}

void PARTICLEINTERACTION::SPHRecoilPressureEvaporation::Init()
{
  // safety check
  if (DRT::INPUT::IntegralValue<INPAR::PARTICLE::SurfaceTensionFormulation>(
          params_sph_, "SURFACETENSIONFORMULATION") == INPAR::PARTICLE::NoSurfaceTension)
    dserror("surface tension evaluation needed for evaporation induced recoil pressure!");
}

void PARTICLEINTERACTION::SPHRecoilPressureEvaporation::Setup(
    const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->GetParticleContainerBundle();
}

void PARTICLEINTERACTION::SPHRecoilPressureEvaporation::AddAccelerationContribution() const
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "PARTICLEINTERACTION::SPHRecoilPressureEvaporation::AddAccelerationContribution");

  // get container of owned particles of evaporating phase
  PARTICLEENGINE::ParticleContainer* container_i =
      particlecontainerbundle_->GetSpecificContainer(evaporatingphase_, PARTICLEENGINE::Owned);

  // iterate over particles in container
  for (int particle_i = 0; particle_i < container_i->ParticlesStored(); ++particle_i)
  {
    const double* dens_i = container_i->GetPtrToParticleState(PARTICLEENGINE::Density, particle_i);
    const double* temp_i =
        container_i->GetPtrToParticleState(PARTICLEENGINE::Temperature, particle_i);
    const double* colorfieldgrad_i =
        container_i->GetPtrToParticleState(PARTICLEENGINE::ColorfieldGradient, particle_i);
    const double* interfacenormal_i =
        container_i->GetPtrToParticleState(PARTICLEENGINE::InterfaceNormal, particle_i);
    double* acc_i = container_i->GetPtrToParticleState(PARTICLEENGINE::Acceleration, particle_i);

    // evaluation only for non-zero interface normal
    if (not(UTILS::vec_norm2(interfacenormal_i) > 0.0)) continue;

    // recoil pressure contribution only for temperature above boiling temperature
    if (not(temp_i[0] > recoilboilingtemp_)) continue;

    // compute evaporation induced recoil pressure
    const double recoil_press_i =
        recoil_pfac_ * std::exp(-recoil_tfac_ * (1.0 / temp_i[0] - 1.0 / recoilboilingtemp_));

    // add contribution to acceleration
    UTILS::vec_addscale(acc_i, -recoil_press_i / dens_i[0], colorfieldgrad_i);
  }
}