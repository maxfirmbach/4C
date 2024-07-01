/*----------------------------------------------------------------------*/
/*! \file
\brief Implementation of the mixture material holding a general mixturerule and mixture constituents

\level 3

*/
/*----------------------------------------------------------------------*/
#include "4C_mat_mixture.hpp"

#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

// constructor of the parameters
Mat::PAR::Mixture::Mixture(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata), constituents_(0)
{
  const int num_constituents = matdata.parameters.get<int>("NUMCONST");
  const auto& constituent_matids = matdata.parameters.get<std::vector<int>>("MATIDSCONST");

  // check, if size of constituents fits to the number of constituents
  if (num_constituents != (int)constituent_matids.size())
  {
    FOUR_C_THROW(
        "number of constituents %d does not fit to the size of the constituents material vector"
        " %d",
        num_constituents, constituent_matids.size());
  }

  // Create constituents
  for (int i = 0; i < num_constituents; ++i)
  {
    // Create constituent material
    constituents_.emplace_back(MIXTURE::PAR::MixtureConstituent::factory(constituent_matids[i]));
  }

  // Create mixture rule
  mixture_rule_ =
      MIXTURE::PAR::MixtureRule::factory(matdata.parameters.get<int>("MATIDMIXTURERULE"));
}

// Create a material instance from parameters
Teuchos::RCP<Core::Mat::Material> Mat::PAR::Mixture::create_material()
{
  return Teuchos::rcp(new Mat::Mixture(this));
}

Mat::MixtureType Mat::MixtureType::instance_;

// Create a material instance from packed data
Core::Communication::ParObject* Mat::MixtureType::Create(const std::vector<char>& data)
{
  auto* mix_elhy = new Mat::Mixture();
  mix_elhy->unpack(data);

  return mix_elhy;
}

// constructor
Mat::Mixture::Mixture()
    : params_(nullptr),
      constituents_(std::make_shared<std::vector<std::unique_ptr<MIXTURE::MixtureConstituent>>>(0)),
      setup_(false),
      anisotropy_()
{
}

// constructor
Mat::Mixture::Mixture(Mat::PAR::Mixture* params)
    : params_(params),
      constituents_(std::make_shared<std::vector<std::unique_ptr<MIXTURE::MixtureConstituent>>>(0)),
      setup_(false),
      anisotropy_()
{
  // create instances of constituents
  int id = 0;
  for (auto const& constituent : params_->constituents_)
  {
    constituents_->emplace_back(constituent->create_constituent(id));
    constituents_->back()->register_anisotropy_extensions(anisotropy_);

    ++id;
  }

  // create instance of mixture rule
  mixture_rule_ = params->mixture_rule_->create_rule();
  mixture_rule_->set_constituents(constituents_);
  mixture_rule_->register_anisotropy_extensions(anisotropy_);
}

// Pack data
void Mat::Mixture::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::PackBuffer::SizeMarker sm(data);

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  add_to_pack(data, type);

  // Pack material id
  int matid = -1;
  if (params_ != nullptr) matid = params_->Id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  // pack setup flag
  add_to_pack(data, static_cast<int>(setup_));

  // Pack isPreEvaluated flag
  std::vector<int> isPreEvaluatedInt;
  isPreEvaluatedInt.resize(is_pre_evaluated_.size());
  for (unsigned i = 0; i < is_pre_evaluated_.size(); ++i)
  {
    isPreEvaluatedInt[i] = static_cast<int>(is_pre_evaluated_[i]);
  }
  add_to_pack(data, isPreEvaluatedInt);

  anisotropy_.pack_anisotropy(data);

  // pack all constituents
  // constituents are not accessible during post processing
  if (params_ != nullptr)
  {
    for (const auto& constituent : *constituents_)
    {
      constituent->pack_constituent(data);
    }

    // pack mixturerule
    mixture_rule_->pack_mixture_rule(data);
  }
}

// Unpack data
void Mat::Mixture::unpack(const std::vector<char>& data)
{
  params_ = nullptr;
  constituents_->clear();
  setup_ = false;

  std::vector<char>::size_type position = 0;

  Core::Communication::ExtractAndAssertId(position, data, UniqueParObjectId());

  // matid and recover params_
  int matid;
  extract_from_pack(position, data, matid);
  if (Global::Problem::Instance()->Materials() != Teuchos::null)
  {
    if (Global::Problem::Instance()->Materials()->Num() != 0)
    {
      const int probinst = Global::Problem::Instance()->Materials()->GetReadFromProblem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::Instance(probinst)->Materials()->ParameterById(matid);
      if (mat->Type() == MaterialType())
      {
        params_ = dynamic_cast<Mat::PAR::Mixture*>(mat);
      }
      else
      {
        FOUR_C_THROW("Type of parameter material %d does not fit to calling type %d", mat->Type(),
            MaterialType());
      }
    }

    // Extract setup flag
    setup_ = (bool)extract_int(position, data);


    // Extract is isPreEvaluated
    std::vector<int> isPreEvaluatedInt(0);
    Core::Communication::ParObject::extract_from_pack(position, data, isPreEvaluatedInt);
    is_pre_evaluated_.resize(isPreEvaluatedInt.size());
    for (unsigned i = 0; i < isPreEvaluatedInt.size(); ++i)
    {
      is_pre_evaluated_[i] = static_cast<bool>(isPreEvaluatedInt[i]);
    }

    anisotropy_.unpack_anisotropy(data, position);

    // extract constituents
    // constituents are not accessible during post processing
    if (params_ != nullptr)
    {
      // create instances of constituents
      int id = 0;
      for (auto const& constituent : params_->constituents_)
      {
        constituents_->emplace_back(constituent->create_constituent(id));

        ++id;
      }

      // create instance of mixture rule
      mixture_rule_ = params_->mixture_rule_->create_rule();

      // make sure the referenced materials in material list have quick access parameters
      for (const auto& constituent : *constituents_)
      {
        constituent->unpack_constituent(position, data);
        constituent->register_anisotropy_extensions(anisotropy_);
      }

      // unpack mixturerule
      mixture_rule_->unpack_mixture_rule(position, data);
      mixture_rule_->set_constituents(constituents_);
      mixture_rule_->register_anisotropy_extensions(anisotropy_);

      // position checking is not available in post processing mode
      if (position != data.size())
      {
        FOUR_C_THROW("Mismatch in size of data to unpack (%d <-> %d)", data.size(), position);
      }
    }
  }
}

// Read element and create arrays for the quantities at the Gauss points
void Mat::Mixture::setup(const int numgp, Input::LineDefinition* linedef)
{
  So3Material::setup(numgp, linedef);

  // resize preevaluation flag
  is_pre_evaluated_.resize(numgp, false);

  // Setup anisotropy
  anisotropy_.set_number_of_gauss_points(numgp);
  anisotropy_.read_anisotropy_from_element(linedef);

  // Let all constituents read the line definition
  for (const auto& constituent : *constituents_)
  {
    constituent->read_element(numgp, linedef);
  }

  mixture_rule_->read_element(numgp, linedef);
}

// Post setup routine -> Call Setup of constituents and mixture rule
void Mat::Mixture::post_setup(Teuchos::ParameterList& params, const int eleGID)
{
  So3Material::post_setup(params, eleGID);
  anisotropy_.read_anisotropy_from_parameter_list(params);
  if (constituents_ != nullptr)
  {
    for (const auto& constituent : *constituents_)
    {
      constituent->setup(params, eleGID);
    }
  }

  if (mixture_rule_ != nullptr)
  {
    mixture_rule_->setup(params, eleGID);
  }

  setup_ = true;
}

// This method is called between two timesteps
void Mat::Mixture::update(Core::LinAlg::Matrix<3, 3> const& defgrd, const int gp,
    Teuchos::ParameterList& params, const int eleGID)
{
  // Update all constituents
  for (const auto& constituent : *constituents_)
  {
    constituent->update(defgrd, params, gp, eleGID);
  }

  mixture_rule_->update(defgrd, params, gp, eleGID);
}

// Evaluates the material
void Mat::Mixture::evaluate(const Core::LinAlg::Matrix<3, 3>* defgrd,
    const Core::LinAlg::Matrix<6, 1>* glstrain, Teuchos::ParameterList& params,
    Core::LinAlg::Matrix<6, 1>* stress, Core::LinAlg::Matrix<6, 6>* cmat, const int gp,
    const int eleGID)
{
  // check, whether the post_setup method was already called
  if (!setup_) FOUR_C_THROW("The material's post_setup() method has not been called yet.");

  if (!is_pre_evaluated_[gp])
  {
    for (const auto& constituent : *constituents_)
    {
      is_pre_evaluated_[gp] = true;
      constituent->pre_evaluate(*mixture_rule_, params, gp, eleGID);
    }

    mixture_rule_->pre_evaluate(params, gp, eleGID);
  }

  // Evaluate mixturerule
  mixture_rule_->evaluate(*defgrd, *glstrain, params, *stress, *cmat, gp, eleGID);
}

void Mat::Mixture::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  mixture_rule_->register_output_data_names(names_and_size);
  for (const auto& constituent : *constituents_)
  {
    constituent->register_output_data_names(names_and_size);
  }
}

bool Mat::Mixture::EvaluateOutputData(
    const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
{
  bool out = mixture_rule_->evaluate_output_data(name, data);
  for (const auto& constituent : *constituents_)
  {
    out = out || constituent->evaluate_output_data(name, data);
  }

  return out;
}
FOUR_C_NAMESPACE_CLOSE