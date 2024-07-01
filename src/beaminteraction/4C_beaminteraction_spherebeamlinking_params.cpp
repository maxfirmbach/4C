/*----------------------------------------------------------------------------*/
/*! \file

\brief data container holding all contractile cells input parameters

\level 3

*/
/*----------------------------------------------------------------------------*/

#include "4C_beaminteraction_spherebeamlinking_params.hpp"

#include "4C_global_data.hpp"
#include "4C_inpar_beaminteraction.hpp"
#include "4C_mat_crosslinkermat.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BEAMINTERACTION::SphereBeamLinkingParams::SphereBeamLinkingParams()
    : isinit_(false), issetup_(false), deltatime_(-1.0), own_deltatime_(true)
{
  mat_.clear();
  contractionrate_.clear();
  maxnumlinkerpertype_.clear();
  matlinkerpertype_.clear();
  linkertypes_.clear();
  filamentbspotintervalglobal_.clear();
  filamentbspotintervallocal_.clear();
  filamentbspotrangeglobal_.clear();
  filamentbspotrangelocal_.clear();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SphereBeamLinkingParams::init(STR::TimeInt::BaseDataGlobalState const& gstate)
{
  issetup_ = false;

  const Teuchos::ParameterList& spherebeamlink_params_list =
      Global::Problem::Instance()->beam_interaction_params().sublist("SPHERE BEAM LINK");

  // time step for stochastic events concering crosslinking
  deltatime_ = spherebeamlink_params_list.get<double>("TIMESTEP");

  // safety check
  // todo: maybe make input of time step obligatory
  if (deltatime_ < 0.0)
  {
    own_deltatime_ = false;
    deltatime_ = (*gstate.get_delta_time())[0];
    if (gstate.get_my_rank() == 0)
      std::cout << " Time step " << (*gstate.get_delta_time())[0]
                << " from Structural Dynamic section "
                   "used for sphere beam link.\n"
                   "Force dependent unbinding of beam-sphere linker is activated for dt > 0"
                << std::endl;
  }

  // number of linker in simulation volume
  {
    maxnumlinkerpertype_.clear();
    std::istringstream PL(
        Teuchos::getNumericStringParameter(spherebeamlink_params_list, "MAXNUMLINKERPERTYPE"));
    std::string word;
    char* input;
    while (PL >> word) maxnumlinkerpertype_.push_back(std::strtod(word.c_str(), &input));

    // safety check
    for (unsigned int i = 0; i < maxnumlinkerpertype_.size(); ++i)
      if (maxnumlinkerpertype_[i] < 0)
        FOUR_C_THROW(" negative number of linker does not make sense.");
  }

  // material numbers for crosslinker types
  {
    matlinkerpertype_.clear();
    std::istringstream PL(
        Teuchos::getNumericStringParameter(spherebeamlink_params_list, "MATLINKERPERTYPE"));
    std::string word;
    char* input;
    while (PL >> word) matlinkerpertype_.push_back(std::strtod(word.c_str(), &input));

    for (unsigned int i = 0; i < matlinkerpertype_.size(); ++i)
    {
      // safety check
      if (matlinkerpertype_[i] < 0) FOUR_C_THROW(" negative material number does not make sense.");

      // store materials
      mat_.push_back(
          Teuchos::rcp_dynamic_cast<Mat::CrosslinkerMat>(Mat::Factory(matlinkerpertype_[i])));
      if (mat_.back() == Teuchos::null)
        FOUR_C_THROW("Invalid material given for beam sphere link. \n");
    }
  }

  // safety check
  if (maxnumlinkerpertype_.size() != matlinkerpertype_.size())
    FOUR_C_THROW("number of crosslinker types does not fit number of assigned materials");

  // store number of different linker types
  linkertypes_.clear();
  for (unsigned int type_i = 0; type_i < matlinkerpertype_.size(); ++type_i)
  {
    if (not(std::find(linkertypes_.begin(), linkertypes_.end(), matlinkerpertype_[type_i]) !=
            linkertypes_.end()))
      linkertypes_.push_back(
          Teuchos::rcp_dynamic_cast<Mat::CrosslinkerMat>(Mat::Factory(matlinkerpertype_[type_i]))
              ->LinkerType());
  }

  // store contraction rate, each linker type (not material) can have its own
  {
    contractionrate_.clear();
    std::istringstream PL(
        Teuchos::getNumericStringParameter(spherebeamlink_params_list, "CONTRACTIONRATE"));
    std::string word;
    char* input;
    int count = 0;
    while (PL >> word)
    {
      contractionrate_[linkertypes_[count]] = std::strtod(word.c_str(), &input);
      ++count;
    }
  }

  if (contractionrate_.size() != linkertypes_.size())
    FOUR_C_THROW("You need to specify a contractionrate for each linker type. Exciting ... ");

  // distance between the two binding spots on each filament the same
  {
    filamentbspotintervalglobal_.clear();
    std::istringstream PL(Teuchos::getNumericStringParameter(
        spherebeamlink_params_list, "FILAMENTBSPOTINTERVALGLOBAL"));
    std::string word;
    char* input;
    int count = 0;
    while (PL >> word)
    {
      filamentbspotintervalglobal_[linkertypes_[count]] = std::strtod(word.c_str(), &input);
      ++count;
    }
  }

  // distance between the two binding spots on a filament as percentage of current filament
  // reference length
  {
    filamentbspotintervallocal_.clear();
    std::istringstream PL(Teuchos::getNumericStringParameter(
        spherebeamlink_params_list, "FILAMENTBSPOTINTERVALLOCAL"));
    std::string word;
    char* input;
    int count = 0;
    while (PL >> word)
    {
      filamentbspotintervallocal_[linkertypes_[count]] = std::strtod(word.c_str(), &input);
      ++count;
    }
  }

  if (linkertypes_.size() != filamentbspotintervalglobal_.size() and
      linkertypes_.size() != filamentbspotintervallocal_.size())
    FOUR_C_THROW("You need to specify filament binding spots for all your linker types");

  // safety checks for feasibility of input
  if (filamentbspotintervalglobal_.size() == filamentbspotintervallocal_.size())
  {
    for (auto const& iter : filamentbspotintervalglobal_)
    {
      // safety feasibility checks
      if (iter.second <= 0.0 and not(filamentbspotintervallocal_.at(iter.first) > 0.0 and
                                     filamentbspotintervallocal_.at(iter.first) <= 1.0))
        FOUR_C_THROW(
            " Choose realistic value for FILAMENTBSPOTINTERVAL (i.e. distance between "
            "two binding spots on a filament) in input file. ");
      if (iter.second > 0.0 and filamentbspotintervallocal_.at(iter.first) > 0.0)
        FOUR_C_THROW(" You can only set either a global or a local filament binding spot interval");
    }
  }

  // start and end arc parameter for binding spots on a filament
  {
    filamentbspotrangeglobal_.clear();
    std::istringstream PL(
        Teuchos::getNumericStringParameter(spherebeamlink_params_list, "FILAMENTBSPOTRANGEGLOBAL"));
    std::string word;
    char* input;
    int count = 0;
    while (PL >> word)
    {
      std::pair<double, double> pair;
      pair.first = std::strtod(word.c_str(), &input);
      if (PL >> word)
        pair.second = std::strtod(word.c_str(), &input);
      else
        FOUR_C_THROW("Filament binding spot range needs to be specified via two values");

      // store global range
      filamentbspotrangeglobal_[linkertypes_[count]] = pair;

      if (pair.first > 0.0 and pair.second > 0.0 and (pair.first > pair.second))
        FOUR_C_THROW(" lower bound > upper bound, fix FILAMENTBSPOTRANGEGLOBAL in input file ");

      ++count;
    }
  }

  // start and end arc parameter for binding spots on a filament
  {
    filamentbspotrangelocal_.clear();
    std::istringstream PL(
        Teuchos::getNumericStringParameter(spherebeamlink_params_list, "FILAMENTBSPOTRANGELOCAL"));
    std::string word;
    char* input;
    int count = 0;
    while (PL >> word)
    {
      std::pair<double, double> pair;
      pair.first = std::strtod(word.c_str(), &input);
      if (PL >> word)
        pair.second = std::strtod(word.c_str(), &input);
      else
        FOUR_C_THROW("Filament binding spot range needs to be specified via two values");

      // store local range
      filamentbspotrangelocal_[linkertypes_[count]] = pair;

      if (pair.first > 0.0 and pair.second > 0.0 and (pair.first > pair.second))
        FOUR_C_THROW(" lower bound > upper bound, fix FILAMENTBSPOTRANGEGLOCAL in input file ");
      if (pair.first > 1.0 or pair.second > 1.0)
        FOUR_C_THROW("values > 1.0 do not make sense for local filament binding spot range");

      ++count;
    }
  }

  if (linkertypes_.size() != filamentbspotrangeglobal_.size() and
      linkertypes_.size() != filamentbspotrangelocal_.size())
    FOUR_C_THROW("You need to specify filament binding spots for all your linker types");

  // safety checks for feasibility of input
  if (filamentbspotrangeglobal_.size() == filamentbspotrangelocal_.size())
  {
    for (auto const& iter : filamentbspotrangeglobal_)
    {
      if (filamentbspotrangelocal_.at(iter.first).first > 0.0 and
          filamentbspotrangelocal_.at(iter.first).second > 0.0 and
          (iter.second.first > 0.0 or iter.second.second > 0.0))
        FOUR_C_THROW("either local or global binding spot range can be specified");
    }
  }


  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SphereBeamLinkingParams::setup()
{
  check_init();

  // empty for now

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SphereBeamLinkingParams::ResetTimeStep(double structure_delta_time)
{
  check_init_setup();

  if (not own_deltatime_) deltatime_ = structure_delta_time;
}

FOUR_C_NAMESPACE_CLOSE