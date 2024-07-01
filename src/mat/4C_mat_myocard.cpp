/*----------------------------------------------------------------------*/
/*! \file
\brief myocard material

\level 3

*/


/*----------------------------------------------------------------------*
 |  headers                                                  cbert 09/12 |
 *----------------------------------------------------------------------*/

#include "4C_mat_myocard.hpp"

#include "4C_global_data.hpp"
#include "4C_io_linedefinition.hpp"
#include "4C_mat_myocard_fitzhugh_nagumo.hpp"
#include "4C_mat_myocard_inada.hpp"
#include "4C_mat_myocard_minimal.hpp"
#include "4C_mat_myocard_san_garny.hpp"
#include "4C_mat_myocard_tentusscher.hpp"
#include "4C_mat_par_bundle.hpp"

#include <fstream>  // For plotting ion concentrations
#include <vector>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |                                                          cbert 09/12 |
 *----------------------------------------------------------------------*/
Mat::PAR::Myocard::Myocard(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      diff1(matdata.parameters.get<double>("DIFF1")),
      diff2(matdata.parameters.get<double>("DIFF2")),
      diff3(0.0),
      dt_deriv(matdata.parameters.get<double>("PERTUBATION_DERIV")),
      model(matdata.parameters.get<std::string>("MODEL")),
      tissue(matdata.parameters.get<std::string>("TISSUE")),
      time_scale(matdata.parameters.get<double>("TIME_SCALE")),
      num_gp(0)
{
}

Teuchos::RCP<Core::Mat::Material> Mat::PAR::Myocard::create_material()
{
  return Teuchos::rcp(new Mat::Myocard(this));
}


Mat::MyocardType Mat::MyocardType::instance_;


Core::Communication::ParObject* Mat::MyocardType::Create(const std::vector<char>& data)
{
  Mat::Myocard* myocard = new Mat::Myocard();
  myocard->unpack(data);
  return myocard;
}


/*----------------------------------------------------------------------*
 |  Constructor                                    (public)  cbert 08/13 |
 *----------------------------------------------------------------------*/
Mat::Myocard::Myocard()
    : params_(nullptr),
      difftensor_(0),
      nb_state_variables_(0),
      myocard_mat_(Teuchos::null),
      diff_at_ele_center_(false)
{
}


/*----------------------------------------------------------------------*
 |  Constructor                                   (public)   cbert 08/13 |
 *----------------------------------------------------------------------*/
Mat::Myocard::Myocard(Mat::PAR::Myocard* params)
    : params_(params),
      difftensor_(0),
      nb_state_variables_(0),
      myocard_mat_(Teuchos::null),
      diff_at_ele_center_(false)
{
}


/*----------------------------------------------------------------------*
 |  Pack                                           (public)  cbert 09/12 |
 *----------------------------------------------------------------------*/
void Mat::Myocard::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::PackBuffer::SizeMarker sm(data);

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->Id();  // in case we are in post-process mode
  add_to_pack(data, matid);
  add_to_pack(data, nb_state_variables_);
  add_to_pack(data, difftensor_);
  int num;
  if (diff_at_ele_center_)
    num = 1;
  else
    num = 0;
  add_to_pack(data, num);
  if (myocard_mat_ != Teuchos::null)
    add_to_pack(data, myocard_mat_->GetNumberOfGP());
  else
    add_to_pack(data, 0);

  // pack history data
  if (myocard_mat_ != Teuchos::null)
    for (int k = -1; k < nb_state_variables_; ++k)  // Starting from -1 for mechanical activation
      for (int i = 0; i < myocard_mat_->GetNumberOfGP(); ++i)  // loop over Gauss points
        add_to_pack(data, myocard_mat_->GetInternalState(k, i));

  return;
}

/*----------------------------------------------------------------------*
 |  Unpack                                         (public)  cbert 09/12 |
 *----------------------------------------------------------------------*/
void Mat::Myocard::unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;

  Core::Communication::ExtractAndAssertId(position, data, UniqueParObjectId());

  // matid
  int matid;
  int unpack_nb_state_variables;
  int num_gp;
  int num;
  extract_from_pack(position, data, matid);
  extract_from_pack(position, data, unpack_nb_state_variables);
  extract_from_pack(position, data, difftensor_);
  extract_from_pack(position, data, num);
  if (num)
    diff_at_ele_center_ = true;
  else
    diff_at_ele_center_ = false;
  extract_from_pack(position, data, num_gp);

  params_ = nullptr;
  if (Global::Problem::Instance()->Materials() !=
      Teuchos::null)  // it does not enter here in postprocessing
  {
    if (Global::Problem::Instance()->Materials()->Num() != 0)
    {
      const int probinst = Global::Problem::Instance()->Materials()->GetReadFromProblem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::Instance(probinst)->Materials()->ParameterById(matid);
      if (mat->Type() == MaterialType())
      {
        params_ = static_cast<Mat::PAR::Myocard*>(mat);
      }
      else
        FOUR_C_THROW("Type of parameter material %d does not fit to calling type %d", mat->Type(),
            MaterialType());

      // Set number of Gauss points
      SetGP(num_gp);

      if (num_gp > 0)
      {
        // Initialize material
        initialize();

        // unpack history data
        double val;
        for (int k = -1; k < unpack_nb_state_variables;
             ++k)                                    // Starting from -1 for mechanical activation
          for (int i = 0; i < params_->num_gp; ++i)  // loop over Gauss points
          {
            extract_from_pack(position, data, val);
            myocard_mat_->SetInternalState(k, val, i);
          }

        if (position != data.size())
          FOUR_C_THROW("Mismatch in size of data %d <-> %d", data.size(), position);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 |  unpack_material                                       hoermann 12/16 |
 *----------------------------------------------------------------------*/
void Mat::Myocard::unpack_material(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;

  Core::Communication::ExtractAndAssertId(position, data, UniqueParObjectId());

  // matid
  int matid;
  int num_gp;
  int num;
  extract_from_pack(position, data, matid);
  extract_from_pack(position, data, nb_state_variables_);
  extract_from_pack(position, data, difftensor_);
  extract_from_pack(position, data, num);
  if (num)
    diff_at_ele_center_ = true;
  else
    diff_at_ele_center_ = false;
  extract_from_pack(position, data, num_gp);

  params_ = nullptr;
  if (Global::Problem::Instance()->Materials() !=
      Teuchos::null)  // it does not enter here in postprocessing
  {
    if (Global::Problem::Instance()->Materials()->Num() != 0)
    {
      const int probinst = Global::Problem::Instance()->Materials()->GetReadFromProblem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::Instance(probinst)->Materials()->ParameterById(matid);
      if (mat->Type() == MaterialType())
      {
        params_ = static_cast<Mat::PAR::Myocard*>(mat);
      }
      else
        FOUR_C_THROW("Type of parameter material %d does not fit to calling type %d", mat->Type(),
            MaterialType());

      // Set number of Gauss points
      SetGP(myocard_mat_->GetNumberOfGP());

      // unpack history data
      double val;
      for (int k = -1; k < nb_state_variables_; ++k)  // Starting from -1 for mechanical activation
        for (int i = 0; i < params_->num_gp; ++i)     // loop over Gauss points
        {
          extract_from_pack(position, data, val);
          myocard_mat_->SetInternalState(k, val, i);
        }

      if (position != data.size())
        FOUR_C_THROW("Mismatch in size of data %d <-> %d", data.size(), position);
    }
  }
}


/*----------------------------------------------------------------------*
 |  Setup conductivity tensor                                cbert 02/13 |
 *----------------------------------------------------------------------*/
void Mat::Myocard::setup(const Core::LinAlg::Matrix<3, 1>& fiber1)
{
  setup_diffusion_tensor(fiber1);
}

void Mat::Myocard::setup(const Core::LinAlg::Matrix<2, 1>& fiber1)
{
  setup_diffusion_tensor(fiber1);
}

void Mat::Myocard::setup(Input::LineDefinition* linedef)
{
  std::vector<double> fiber1(3);
  if (linedef->has_named("FIBER1"))
  {
    diff_at_ele_center_ = true;
    linedef->extract_double_vector("FIBER1", fiber1);
    setup_diffusion_tensor(fiber1);
  }
}


void Mat::Myocard::setup_diffusion_tensor(const std::vector<double>& fiber1)
{
  // Normalize fiber1
  double fiber1normS = fiber1[0] * fiber1[0] + fiber1[1] * fiber1[1] + fiber1[2] * fiber1[2];

  // get conductivity values of main fiber direction and perpendicular to fiber direction (rot
  // symmetry)
  const double diff1 = params_->diff1;
  const double diff2 = params_->diff2;

  Core::LinAlg::Matrix<3, 3> difftensor;

  // ******** SETUP ORTHOTROPIC DIFFUSION TENSOR: diff2*Id + (diff1-diff2)*fiber1*fiber1'
  // first row
  difftensor(0, 0) = diff2 + (diff1 - diff2) * fiber1[0] * fiber1[0] / fiber1normS;
  difftensor(0, 1) = (diff1 - diff2) * fiber1[0] * fiber1[1] / fiber1normS;
  difftensor(0, 2) = (diff1 - diff2) * fiber1[0] * fiber1[2] / fiber1normS;
  // second row
  difftensor(1, 0) = (diff1 - diff2) * fiber1[1] * fiber1[0] / fiber1normS;
  difftensor(1, 1) = diff2 + (diff1 - diff2) * fiber1[1] * fiber1[1] / fiber1normS;
  difftensor(1, 2) = (diff1 - diff2) * fiber1[1] * fiber1[2] / fiber1normS;
  // third row
  difftensor(2, 0) = (diff1 - diff2) * fiber1[2] * fiber1[0] / fiber1normS;
  difftensor(2, 1) = (diff1 - diff2) * fiber1[2] * fiber1[1] / fiber1normS;
  difftensor(2, 2) = diff2 + (diff1 - diff2) * fiber1[2] * fiber1[2] / fiber1normS;
  // done

  difftensor_.push_back(difftensor);
  return;
}

void Mat::Myocard::setup_diffusion_tensor(const Core::LinAlg::Matrix<3, 1>& fiber1)
{
  // Normalize fiber1
  double fiber1normS = fiber1(0) * fiber1(0) + fiber1(1) * fiber1(1) + fiber1(2) * fiber1(2);

  // get conductivity values of main fiber direction and perpendicular to fiber direction (rot
  // symmetry)
  const double diff1 = params_->diff1;
  const double diff2 = params_->diff2;

  Core::LinAlg::Matrix<3, 3> difftensor;

  // ******** SETUP ORTHOTROPIC DIFFUSION TENSOR: diff2*Id + (diff1-diff2)*fiber1*fiber1'
  // first row
  difftensor(0, 0) = diff2 + (diff1 - diff2) * fiber1(0) * fiber1(0) / fiber1normS;
  difftensor(0, 1) = (diff1 - diff2) * fiber1(0) * fiber1(1) / fiber1normS;
  difftensor(0, 2) = (diff1 - diff2) * fiber1(0) * fiber1(2) / fiber1normS;
  // second row
  difftensor(1, 0) = (diff1 - diff2) * fiber1(1) * fiber1(0) / fiber1normS;
  difftensor(1, 1) = diff2 + (diff1 - diff2) * fiber1(1) * fiber1(1) / fiber1normS;
  difftensor(1, 2) = (diff1 - diff2) * fiber1(1) * fiber1(2) / fiber1normS;
  // third row
  difftensor(2, 0) = (diff1 - diff2) * fiber1(2) * fiber1(0) / fiber1normS;
  difftensor(2, 1) = (diff1 - diff2) * fiber1(2) * fiber1(1) / fiber1normS;
  difftensor(2, 2) = diff2 + (diff1 - diff2) * fiber1(2) * fiber1(2) / fiber1normS;
  // done

  difftensor_.push_back(difftensor);

  return;
}

void Mat::Myocard::setup_diffusion_tensor(const Core::LinAlg::Matrix<2, 1>& fiber1)
{
  // Normalize fiber1
  double fiber1normS = fiber1(0) * fiber1(0) + fiber1(1) * fiber1(1);

  // get conductivity values of main fiber direction and perpendicular to fiber direction (rot
  // symmetry)
  const double diff1 = params_->diff1;
  const double diff2 = params_->diff2;

  Core::LinAlg::Matrix<3, 3> difftensor;

  // ******** SETUP ORTHOTROPIC DIFFUSION TENSOR: diff2*Id + (diff1-diff2)*fiber1*fiber1'
  // first row
  difftensor(0, 0) = diff2 + (diff1 - diff2) * fiber1(0) * fiber1(0) / fiber1normS;
  difftensor(0, 1) = (diff1 - diff2) * fiber1(0) * fiber1(1) / fiber1normS;
  // second row
  difftensor(1, 0) = (diff1 - diff2) * fiber1(1) * fiber1(0) / fiber1normS;
  difftensor(1, 1) = diff2 + (diff1 - diff2) * fiber1(1) * fiber1(1) / fiber1normS;
  // done

  difftensor_.push_back(difftensor);

  return;
}

void Mat::Myocard::Diffusivity(Core::LinAlg::Matrix<1, 1>& diffus3, int gp) const
{
  diffus3(0, 0) = difftensor_[gp](0, 0);
  return;
}

void Mat::Myocard::Diffusivity(Core::LinAlg::Matrix<2, 2>& diffus3, int gp) const
{
  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      diffus3(i, j) = difftensor_[gp](i, j);
    }
  }

  return;
}

void Mat::Myocard::Diffusivity(Core::LinAlg::Matrix<3, 3>& diffus3, int gp) const
{
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      diffus3(i, j) = difftensor_[gp](i, j);
    }
  }
  return;
}

/*----------------------------------------------------------------------*
 |                                                           cbert 09/13 |
 *----------------------------------------------------------------------*/
double Mat::Myocard::ReaCoeff(const double phi, const double dt) const
{
  double reacoeff = params_->time_scale;
  reacoeff *= myocard_mat_->ReaCoeff(phi, dt * params_->time_scale);

  return reacoeff;
}

/*----------------------------------------------------------------------*
 |                                                       hoermann 09/15 |
 *----------------------------------------------------------------------*/
double Mat::Myocard::ReaCoeff(const double phi, const double dt, int gp) const
{
  double reacoeff = params_->time_scale;
  if (gp == -1)
    reacoeff *= myocard_mat_->ReaCoeff(phi, dt * params_->time_scale);
  else
    reacoeff *= myocard_mat_->ReaCoeff(phi, dt * params_->time_scale, gp);

  return reacoeff;
}

/*----------------------------------------------------------------------*
 | reaction coefficient at time n                        hoermann 11/16 |
 *----------------------------------------------------------------------*/
double Mat::Myocard::ReaCoeffN(const double phi, const double dt, int gp) const
{
  double reacoeff = params_->time_scale;
  if (gp == -1)
    reacoeff *= myocard_mat_->ReaCoeffN(phi, dt * params_->time_scale);
  else
    reacoeff *= myocard_mat_->ReaCoeffN(phi, dt * params_->time_scale, gp);

  return reacoeff;
}


/*----------------------------------------------------------------------*
 |                                                           cbert 09/13 |
 *----------------------------------------------------------------------*/
double Mat::Myocard::ReaCoeffDeriv(const double phi, const double dt) const
{
  double ReaCoeffDeriv = 0.0;
  if (params_->dt_deriv != 0.0)
  {
    double ReaCoeff_t2 = ReaCoeff((phi + params_->dt_deriv), dt);
    double ReaCoeff_t1 = ReaCoeff(phi, dt);
    ReaCoeffDeriv = (ReaCoeff_t2 - ReaCoeff_t1) / (params_->dt_deriv);
  }
  return ReaCoeffDeriv;
}


/*----------------------------------------------------------------------*
 |                                                       hoermann 09/15 |
 *----------------------------------------------------------------------*/
double Mat::Myocard::ReaCoeffDeriv(const double phi, const double dt, int gp) const
{
  double ReaCoeffDeriv = 0.0;
  if (params_->dt_deriv != 0.0)
  {
    double ReaCoeff_t2 = ReaCoeff((phi + params_->dt_deriv), dt, gp);
    double ReaCoeff_t1 = ReaCoeff(phi, dt, gp);
    ReaCoeffDeriv = (ReaCoeff_t2 - ReaCoeff_t1) / (params_->dt_deriv);
  }
  return ReaCoeffDeriv;
}


/*----------------------------------------------------------------------*
 |  returns number of internal state variables              cbert 08/13 |
 *----------------------------------------------------------------------*/
int Mat::Myocard::get_number_of_internal_state_variables() const
{
  int val = 0;
  val = myocard_mat_->get_number_of_internal_state_variables();
  return val;
}

/*----------------------------------------------------------------------*
 |  returns current internal state of the material          cbert 08/13 |
 *----------------------------------------------------------------------*/
double Mat::Myocard::GetInternalState(const int k) const
{
  double val = 0.0;
  val = myocard_mat_->GetInternalState(k);
  return val;
}

/*----------------------------------------------------------------------*
 |  returns current internal state of the material       hoermann 09/15 |
 |  for multiple points per element                                     |
 *----------------------------------------------------------------------*/
double Mat::Myocard::GetInternalState(const int k, int gp) const
{
  double val = 0.0;
  val = myocard_mat_->GetInternalState(k, gp);
  return val;
}

/*----------------------------------------------------------------------*
 |  returns current internal state of the material          cbert 08/13 |
 *----------------------------------------------------------------------*/
void Mat::Myocard::SetInternalState(const int k, const double val)
{
  myocard_mat_->SetInternalState(k, val);
}

/*----------------------------------------------------------------------*
 |  returns current internal state of the material       hoermann 09/15 |
 |  for multiple points per element                                     |
 *----------------------------------------------------------------------*/
void Mat::Myocard::SetInternalState(const int k, const double val, int gp)
{
  myocard_mat_->SetInternalState(k, val, gp);
}

/*----------------------------------------------------------------------*
 |  returns number of internal state variables of the material  cbert 08/13 |
 *----------------------------------------------------------------------*/
int Mat::Myocard::get_number_of_ionic_currents() const
{
  int val = 0;
  val = myocard_mat_->get_number_of_ionic_currents();
  return val;
}

/*----------------------------------------------------------------------*
 |  returns current internal currents                    hoermann 09/15 |
 |  for multiple points per element                                     |
 *----------------------------------------------------------------------*/
double Mat::Myocard::GetIonicCurrents(const int k, int gp) const
{
  double val = 0.0;
  val = myocard_mat_->GetIonicCurrents(k, gp);
  return val;
}

/*----------------------------------------------------------------------*
 |  returns current internal currents                       cbert 08/13 |
 *----------------------------------------------------------------------*/
double Mat::Myocard::GetIonicCurrents(const int k) const
{
  double val = 0.0;
  val = myocard_mat_->GetIonicCurrents(k);
  return val;
}


/*----------------------------------------------------------------------*
 |  initialize internal variables (called by constructors)   cbert 09/12 |
 *----------------------------------------------------------------------*/
void Mat::Myocard::initialize()
{
  if ((params_->model) == "MV")
    myocard_mat_ =
        Teuchos::rcp(new MyocardMinimal(params_->dt_deriv, (params_->tissue), params_->num_gp));
  else if ((params_->model) == "FHN")
    myocard_mat_ = Teuchos::rcp(
        new MyocardFitzhughNagumo(params_->dt_deriv, (params_->tissue), params_->num_gp));
  else if ((params_->model) == "INADA")
    myocard_mat_ = Teuchos::rcp(new MyocardInada(params_->dt_deriv, (params_->tissue)));
  else if ((params_->model) == "TNNP")
    myocard_mat_ = Teuchos::rcp(new MyocardTenTusscher(params_->dt_deriv, (params_->tissue)));
  else if ((params_->model) == "SAN")
    myocard_mat_ = Teuchos::rcp(new MyocardSanGarny(params_->dt_deriv, (params_->tissue)));
  else
    FOUR_C_THROW(
        "Myocard Material type is not supported! (for the moment only MV,FHN,INADA,TNNP and SAN)");

  nb_state_variables_ = myocard_mat_->get_number_of_internal_state_variables();

  return;
}

/*----------------------------------------------------------------------*
 |  resize internal state variables                      hoermann 12/16 |
 *----------------------------------------------------------------------*/
void Mat::Myocard::resize_internal_state_variables()
{
  myocard_mat_->resize_internal_state_variables(params_->num_gp);
  return;
}

/*----------------------------------------------------------------------*
 |  update of material at the end of a time step             ljag 07/12 |
 *----------------------------------------------------------------------*/
void Mat::Myocard::update(const double phi, const double dt)
{
  myocard_mat_->update(phi, dt * (params_->time_scale));

  return;
}

/*----------------------------------------------------------------------*
 |  get number of Gauss points                           hoermann 12/16 |
 *----------------------------------------------------------------------*/
int Mat::Myocard::GetNumberOfGP() const { return myocard_mat_->GetNumberOfGP(); };

FOUR_C_NAMESPACE_CLOSE