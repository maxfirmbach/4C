/*---------------------------------------------------------------------------*/
/*! \file
\brief history pair struct for discrete element method (DEM) interactions
\level 3
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
#ifndef FOUR_C_PARTICLE_INTERACTION_DEM_HISTORY_PAIR_STRUCT_HPP
#define FOUR_C_PARTICLE_INTERACTION_DEM_HISTORY_PAIR_STRUCT_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_comm_parobject.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace ParticleInteraction
{
  //! struct to store tangential contact history of interacting particles
  struct DEMHistoryPairTangential final
  {
    //! tangential stick flag
    bool stick_ = true;

    //! tangential gap
    double gap_t_[3] = {0.0, 0.0, 0.0};

    //! pack history pair data
    void pack(Core::Communication::PackBuffer& data) const
    {
      data.add_to_pack(stick_);

      for (int i = 0; i < 3; ++i) data.add_to_pack(gap_t_[i]);
    }

    //! unpack history pair data
    void unpack(std::vector<char>::size_type& position, const std::vector<char>& data)
    {
      Core::Communication::ParObject::extract_from_pack(position, data, stick_);

      for (int i = 0; i < 3; ++i)
        Core::Communication::ParObject::extract_from_pack(position, data, gap_t_[i]);
    }
  };

  //! struct to store rolling contact history of interacting particles
  struct DEMHistoryPairRolling final
  {
    //! rolling stick flag
    bool stick_ = true;

    //! rolling gap
    double gap_r_[3] = {0.0, 0.0, 0.0};

    //! pack history pair data
    void pack(Core::Communication::PackBuffer& data) const
    {
      data.add_to_pack(stick_);

      for (int i = 0; i < 3; ++i) data.add_to_pack(gap_r_[i]);
    }

    //! unpack history pair data
    void unpack(std::vector<char>::size_type& position, const std::vector<char>& data)
    {
      Core::Communication::ParObject::extract_from_pack(position, data, stick_);

      for (int i = 0; i < 3; ++i)
        Core::Communication::ParObject::extract_from_pack(position, data, gap_r_[i]);
    }
  };

  //! struct to store adhesion history of interacting particles
  struct DEMHistoryPairAdhesion final
  {
    //! surface energy
    double surface_energy_ = 0.0;

    //! adhesion force
    double adhesion_force_ = 0.0;

    //! pack history pair data
    void pack(Core::Communication::PackBuffer& data) const
    {
      data.add_to_pack(surface_energy_);

      data.add_to_pack(adhesion_force_);
    }

    //! unpack history pair data
    void unpack(std::vector<char>::size_type& position, const std::vector<char>& data)
    {
      Core::Communication::ParObject::extract_from_pack(position, data, surface_energy_);

      Core::Communication::ParObject::extract_from_pack(position, data, adhesion_force_);
    }
  };
}  // namespace ParticleInteraction

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif