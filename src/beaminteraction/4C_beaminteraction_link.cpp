/*----------------------------------------------------------------------*/
/*! \file

\brief One beam-to-beam pair (two beam elements) connected by a mechanical link

\level 3

*/
/*----------------------------------------------------------------------*/

#include "4C_beaminteraction_link.hpp"

#include "4C_fem_general_largerotations.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

BEAMINTERACTION::BeamLinkType BEAMINTERACTION::BeamLinkType::instance_;


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BEAMINTERACTION::BeamLink::BeamLink()
    : ParObject(),
      isinit_(false),
      issetup_(false),
      id_(-1),
      bspotpos1_(true),
      bspotpos2_(true),
      linkertype_(Inpar::BEAMINTERACTION::linkertype_arbitrary),
      timelinkwasset_(-1.0),
      reflength_(-1.0)
{
  bspot_ids_.clear();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
BEAMINTERACTION::BeamLink::BeamLink(const BEAMINTERACTION::BeamLink& old)
    : ParObject(old),
      isinit_(old.isinit_),
      issetup_(old.issetup_),
      id_(old.id_),
      bspot_ids_(old.bspot_ids_),
      bspotpos1_(old.bspotpos1_),
      bspotpos2_(old.bspotpos2_),
      linkertype_(old.linkertype_),
      timelinkwasset_(old.timelinkwasset_),
      reflength_(old.reflength_)
{
  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::BeamLink::init(const int id, const std::vector<std::pair<int, int>>& eleids,
    const std::vector<Core::LinAlg::Matrix<3, 1>>& initpos,
    const std::vector<Core::LinAlg::Matrix<3, 3>>& inittriad,
    Inpar::BEAMINTERACTION::CrosslinkerType linkertype, double timelinkwasset)
{
  issetup_ = false;

  id_ = id;
  bspot_ids_ = eleids;

  bspotpos1_ = initpos[0];
  bspotpos2_ = initpos[1];

  linkertype_ = linkertype;

  timelinkwasset_ = timelinkwasset;

  reflength_ = 0.0;
  for (unsigned int i = 0; i < 3; ++i)
    reflength_ += (initpos[1](i) - initpos[0](i)) * (initpos[1](i) - initpos[0](i));
  reflength_ = sqrt(reflength_);

  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::BeamLink::setup(const int matnum)
{
  check_init();

  // the flag issetup_ will be set in the derived method!
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BEAMINTERACTION::BeamLink::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::PackBuffer::SizeMarker sm(data);

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  add_to_pack(data, type);
  // isinit_
  add_to_pack(data, isinit_);
  // issetup
  add_to_pack(data, issetup_);
  // add id
  add_to_pack(data, id_);

  // add eleids_
  add_to_pack(data, bspot_ids_);
  // bspotpos1_
  add_to_pack(data, bspotpos1_);
  // bspotpos2_
  add_to_pack(data, bspotpos2_);
  // linkertype
  add_to_pack(data, linkertype_);
  // timelinkwasset
  add_to_pack(data, timelinkwasset_);
  // reflength
  add_to_pack(data, reflength_);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BEAMINTERACTION::BeamLink::unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;

  Core::Communication::ExtractAndAssertId(position, data, UniqueParObjectId());

  // isinit_
  isinit_ = Core::Communication::ParObject::extract_int(position, data);
  // issetup
  issetup_ = Core::Communication::ParObject::extract_int(position, data);
  // id_
  extract_from_pack(position, data, id_);

  // eleids_
  extract_from_pack(position, data, bspot_ids_);
  // bspotpos1
  extract_from_pack(position, data, bspotpos1_);
  // bspotpos2
  extract_from_pack(position, data, bspotpos2_);
  // linkertype
  linkertype_ = static_cast<Inpar::BEAMINTERACTION::CrosslinkerType>(extract_int(position, data));
  // timelinkwasset
  extract_from_pack(position, data, timelinkwasset_);
  // reflength
  extract_from_pack(position, data, reflength_);

  if (position != data.size())
    FOUR_C_THROW("Mismatch in size of data %d <-> %d", (int)data.size(), position);

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::BeamLink::ResetState(std::vector<Core::LinAlg::Matrix<3, 1>>& bspotpos,
    std::vector<Core::LinAlg::Matrix<3, 3>>& bspottriad)
{
  check_init_setup();

  /* the two positions of the linkage element coincide with the positions of the
   * binding spots on the parent elements */
  bspotpos1_ = bspotpos[0];
  bspotpos2_ = bspotpos[1];
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::BeamLink::print(std::ostream& out) const
{
  check_init();

  out << "\nBeamLinkRigidJointed (ID " << Id() << "):";
  out << "\nbspotIds_[0] = ";
  out << "EleGID " << GetEleGid(0) << " locbspotnum " << GetLocBSpotNum(0);
  out << "\nbspotIds_[1] = ";
  out << "EleGID " << GetEleGid(1) << " locbspotnum " << GetLocBSpotNum(1);
  out << "\n";
  out << "\nbspotpos1_ = ";
  GetBindSpotPos1().print(out);
  out << "\nbspotpos2_ = ";
  GetBindSpotPos2().print(out);

  out << "\n";
}

FOUR_C_NAMESPACE_CLOSE