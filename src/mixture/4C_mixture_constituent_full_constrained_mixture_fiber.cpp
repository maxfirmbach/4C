/*----------------------------------------------------------------------*/
/*! \file
\brief Definition of a quasi 1D full constrained mixture constituent
equations
\level 3
*/
/*----------------------------------------------------------------------*/
#include "4C_mixture_constituent_full_constrained_mixture_fiber.hpp"

#include "4C_comm_parobject.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_material.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_matelast_aniso_structuraltensor_strategy.hpp"
#include "4C_mixture_constituent_remodelfiber_lib.hpp"
#include "4C_mixture_growth_evolution_linear_cauchy_poisson_turnover.hpp"
#include "4C_utils_function_of_time.hpp"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <numeric>

FOUR_C_NAMESPACE_OPEN

// anonymous namespace for helper classes and functions
namespace
{
  [[nodiscard]] static inline Core::LinAlg::Matrix<3, 3> EvaluateC(
      const Core::LinAlg::Matrix<3, 3>& F)
  {
    Core::LinAlg::Matrix<3, 3> C(false);
    C.multiply_tn(F, F);
    return C;
  }

  [[nodiscard]] static inline double GetTotalTime(const Teuchos::ParameterList& params)
  {
    double time = params.get<double>("total time");
    if (time < 0) return 0.0;  // Time has not been set by the time integrator during setup
    return time;
  }

  [[nodiscard]] static inline double GetDeltaTime(const Teuchos::ParameterList& params)
  {
    return params.get<double>("delta time");
  }

  MIXTURE::HistoryAdaptionStrategy GetHistoryAdaptionStrategyFromInput(const std::string& input)
  {
    if (input == "none")
    {
      return MIXTURE::HistoryAdaptionStrategy::none;
    }
    else if (input == "window")
    {
      return MIXTURE::HistoryAdaptionStrategy::window;
    }
    else if (input == "model_equation")
    {
      return MIXTURE::HistoryAdaptionStrategy::model_equation;
    }
    else if (input == "higher_order")
    {
      return MIXTURE::HistoryAdaptionStrategy::higher_order_integration;
    }
    else
    {
      FOUR_C_THROW("Unknown history adaption strategy %s!", input.c_str());
    }
  }
}  // namespace

MIXTURE::PAR::MixtureConstituentFullConstrainedMixtureFiber::
    MixtureConstituentFullConstrainedMixtureFiber(const Core::Mat::PAR::Parameter::Data& matdata)
    : MixtureConstituent(matdata),
      fiber_id_(matdata.parameters.get<int>("FIBER_ID") - 1),
      init_(matdata.parameters.get<int>("INIT")),
      fiber_material_id_(matdata.parameters.get<int>("FIBER_MATERIAL_ID")),
      fiber_material_(FiberMaterialFactory(fiber_material_id_)),
      growth_enabled_(matdata.parameters.get<bool>("GROWTH_ENABLED")),
      poisson_decay_time_(matdata.parameters.get<double>("DECAY_TIME")),
      growth_constant_(matdata.parameters.get<double>("GROWTH_CONSTANT")),
      deposition_stretch_(matdata.parameters.get<double>("DEPOSITION_STRETCH")),
      initial_deposition_stretch_timefunc_num_(
          matdata.parameters.get<int>("INITIAL_DEPOSITION_STRETCH_TIMEFUNCT")),
      adaptive_history_strategy_(GetHistoryAdaptionStrategyFromInput(
          matdata.parameters.get<std::string>("ADAPTIVE_HISTORY_STRATEGY"))),
      adaptive_history_tolerance_(matdata.parameters.get<double>("ADAPTIVE_HISTORY_TOLERANCE"))
{
}

std::unique_ptr<MIXTURE::MixtureConstituent>
MIXTURE::PAR::MixtureConstituentFullConstrainedMixtureFiber::create_constituent(int id)
{
  return std::make_unique<MIXTURE::MixtureConstituentFullConstrainedMixtureFiber>(this, id);
}

MIXTURE::MixtureConstituentFullConstrainedMixtureFiber::
    MixtureConstituentFullConstrainedMixtureFiber(
        MIXTURE::PAR::MixtureConstituentFullConstrainedMixtureFiber* params, int id)
    : MixtureConstituent(params, id),
      params_(params),
      full_constrained_mixture_fiber_(),
      anisotropy_extension_(params_->init_, 0.0, false,
          Teuchos::rcp(new Mat::Elastic::StructuralTensorStrategyStandard(nullptr)),
          {params_->fiber_id_})
{
  anisotropy_extension_.register_needed_tensors(
      Mat::FiberAnisotropyExtension<1>::STRUCTURAL_TENSOR_STRESS |
      Mat::FiberAnisotropyExtension<1>::STRUCTURAL_TENSOR);
}

Core::Materials::MaterialType
MIXTURE::MixtureConstituentFullConstrainedMixtureFiber::material_type() const
{
  return Core::Materials::mix_full_constrained_mixture_fiber;
}

void MIXTURE::MixtureConstituentFullConstrainedMixtureFiber::pack_constituent(
    Core::Communication::PackBuffer& data) const
{
  MIXTURE::MixtureConstituent::pack_constituent(data);
  anisotropy_extension_.pack_anisotropy(data);

  for (const FullConstrainedMixtureFiber<double>& fiber : full_constrained_mixture_fiber_)
    fiber.pack(data);

  Core::Communication::ParObject::add_to_pack(data, last_lambda_f_);
}

void MIXTURE::MixtureConstituentFullConstrainedMixtureFiber::unpack_constituent(
    std::vector<char>::size_type& position, const std::vector<char>& data)
{
  MIXTURE::MixtureConstituent::unpack_constituent(position, data);
  initialize();

  anisotropy_extension_.unpack_anisotropy(data, position);

  for (FullConstrainedMixtureFiber<double>& fiber : full_constrained_mixture_fiber_)
    fiber.unpack(position, data);

  Core::Communication::ParObject::extract_from_pack(position, data, last_lambda_f_);

  if (params_->growth_enabled_)
  {
    for (auto gp = 0; gp < num_gp(); ++gp)
    {
      full_constrained_mixture_fiber_[gp].reinitialize_history(
          last_lambda_f_[gp], full_constrained_mixture_fiber_[gp].get_last_time_in_history());
    }
  }
}

void MIXTURE::MixtureConstituentFullConstrainedMixtureFiber::register_anisotropy_extensions(
    Mat::Anisotropy& anisotropy)
{
  anisotropy.register_anisotropy_extension(anisotropy_extension_);
}

void MIXTURE::MixtureConstituentFullConstrainedMixtureFiber::initialize()
{
  full_constrained_mixture_fiber_.clear();
  std::shared_ptr<const RemodelFiberMaterial<double>> material =
      params_->fiber_material_->create_remodel_fiber_material();

  last_lambda_f_.resize(num_gp(), 1.0);

  for (int gp = 0; gp < num_gp(); ++gp)
  {
    LinearCauchyGrowthWithPoissonTurnoverGrowthEvolution<double> growth_evolution(
        params_->growth_constant_, params_->poisson_decay_time_);
    full_constrained_mixture_fiber_.emplace_back(material, growth_evolution,
        evaluate_initial_deposition_stretch(0.0), params_->adaptive_history_strategy_,
        params_->growth_enabled_);
    full_constrained_mixture_fiber_[gp].adaptive_tolerance_ = params_->adaptive_history_tolerance_;
  }
}

void MIXTURE::MixtureConstituentFullConstrainedMixtureFiber::read_element(
    int numgp, Input::LineDefinition* linedef)
{
  MIXTURE::MixtureConstituent::read_element(numgp, linedef);
  initialize();
}

void MIXTURE::MixtureConstituentFullConstrainedMixtureFiber::setup(
    Teuchos::ParameterList& params, int eleGID)
{
  MIXTURE::MixtureConstituent::setup(params, eleGID);

  if (params_->growth_enabled_)
  {
    for (auto& fiber : full_constrained_mixture_fiber_)
    {
      // No deformation at t=0
      fiber.reinitialize_history(1.0, 0.0);
    }
  }
}

void MIXTURE::MixtureConstituentFullConstrainedMixtureFiber::update(
    const Core::LinAlg::Matrix<3, 3>& F, Teuchos::ParameterList& params, const int gp,
    const int eleGID)
{
  MixtureConstituent::update(F, params, gp, eleGID);

  const double time = GetTotalTime(params);
  full_constrained_mixture_fiber_[gp].set_deposition_stretch(
      evaluate_initial_deposition_stretch(time));
  last_lambda_f_[gp] = evaluate_lambdaf(EvaluateC(F), gp, eleGID);

  // Update state
  full_constrained_mixture_fiber_[gp].update();
}

void MIXTURE::MixtureConstituentFullConstrainedMixtureFiber::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  MixtureConstituent::register_output_data_names(names_and_size);
  names_and_size["mixture_constituent_" + std::to_string(id()) + "_sig_h"] = 1;
  names_and_size["mixture_constituent_" + std::to_string(id()) + "_sig"] = 1;
  names_and_size["mixture_constituent_" + std::to_string(id()) + "_growth_scalar"] = 1;
  names_and_size["mixture_constituent_" + std::to_string(id()) + "_history_size"] = 1;
}

bool MIXTURE::MixtureConstituentFullConstrainedMixtureFiber::evaluate_output_data(
    const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
{
  if (name == "mixture_constituent_" + std::to_string(id()) + "_sig_h")
  {
    for (int gp = 0; gp < num_gp(); ++gp)
    {
      data(gp, 0) = full_constrained_mixture_fiber_[gp].sig_h_;
    }
    return true;
  }
  else if (name == "mixture_constituent_" + std::to_string(id()) + "_sig")
  {
    for (int gp = 0; gp < num_gp(); ++gp)
    {
      data(gp, 0) = full_constrained_mixture_fiber_[gp].computed_sigma_;
    }
    return true;
  }
  else if (name == "mixture_constituent_" + std::to_string(id()) + "_growth_scalar")
  {
    for (int gp = 0; gp < num_gp(); ++gp)
    {
      data(gp, 0) = full_constrained_mixture_fiber_[gp].computed_growth_scalar_;
    }
    return true;
  }
  else if (name == "mixture_constituent_" + std::to_string(id()) + "_history_size")
  {
    for (int gp = 0; gp < num_gp(); ++gp)
    {
      data(gp, 0) = std::accumulate(full_constrained_mixture_fiber_[gp].history_.begin(),
          full_constrained_mixture_fiber_[gp].history_.end(), 0,
          [](std::size_t number, const DepositionHistoryInterval<double>& item)
          { return number + item.timesteps.size(); });
    }
    return true;
  }
  return MixtureConstituent::evaluate_output_data(name, data);
}

Core::LinAlg::Matrix<1, 6>
MIXTURE::MixtureConstituentFullConstrainedMixtureFiber::evaluate_d_lambdafsq_dc(
    int gp, int eleGID) const
{
  Core::LinAlg::Matrix<1, 6> dLambdafDC(false);
  dLambdafDC.update_t(anisotropy_extension_.get_structural_tensor_stress(gp, 0));
  return dLambdafDC;
}

Core::LinAlg::Matrix<6, 1>
MIXTURE::MixtureConstituentFullConstrainedMixtureFiber::evaluate_current_p_k2(
    int gp, int eleGID) const
{
  Core::LinAlg::Matrix<6, 1> S_stress(false);
  const double fiber_pk2 = full_constrained_mixture_fiber_[gp].evaluate_current_second_pk_stress();

  S_stress.update(fiber_pk2, anisotropy_extension_.get_structural_tensor_stress(gp, 0));

  return S_stress;
}

Core::LinAlg::Matrix<6, 6>
MIXTURE::MixtureConstituentFullConstrainedMixtureFiber::evaluate_current_cmat(
    const int gp, const int eleGID) const
{
  const double dPK2dlambdafsq =
      full_constrained_mixture_fiber_[gp].evaluate_d_current_fiber_p_k2_stress_d_lambdafsq();

  Core::LinAlg::Matrix<6, 6> cmat(false);
  cmat.multiply_nn(2.0 * dPK2dlambdafsq, anisotropy_extension_.get_structural_tensor_stress(gp, 0),
      evaluate_d_lambdafsq_dc(gp, eleGID));

  return cmat;
}

void MIXTURE::MixtureConstituentFullConstrainedMixtureFiber::evaluate(
    const Core::LinAlg::Matrix<3, 3>& F, const Core::LinAlg::Matrix<6, 1>& E_strain,
    Teuchos::ParameterList& params, Core::LinAlg::Matrix<6, 1>& S_stress,
    Core::LinAlg::Matrix<6, 6>& cmat, int gp, int eleGID)
{
  const double time = GetTotalTime(params);
  const double delta_time = GetDeltaTime(params);

  Core::LinAlg::Matrix<3, 3> C = EvaluateC(F);

  const double lambda_f = evaluate_lambdaf(C, gp, eleGID);
  full_constrained_mixture_fiber_[gp].recompute_state(lambda_f, time, delta_time);

  S_stress.update(evaluate_current_p_k2(gp, eleGID));
  cmat.update(evaluate_current_cmat(gp, eleGID));
}

void MIXTURE::MixtureConstituentFullConstrainedMixtureFiber::evaluate_elastic_part(
    const Core::LinAlg::Matrix<3, 3>& FM, const Core::LinAlg::Matrix<3, 3>& iFextin,
    Teuchos::ParameterList& params, Core::LinAlg::Matrix<6, 1>& S_stress,
    Core::LinAlg::Matrix<6, 6>& cmat, int gp, int eleGID)
{
  FOUR_C_THROW(
      "The full constrained mixture fiber cannot be evaluated with an additional inelastic "
      "deformation.");
}

double MIXTURE::MixtureConstituentFullConstrainedMixtureFiber::get_growth_scalar(int gp) const
{
  return full_constrained_mixture_fiber_[gp].computed_growth_scalar_;
}

Core::LinAlg::Matrix<1, 6>
MIXTURE::MixtureConstituentFullConstrainedMixtureFiber::get_d_growth_scalar_d_cg(
    int gp, int eleGID) const
{
  if (!params_->growth_enabled_) return Core::LinAlg::Matrix<1, 6>(true);
  Core::LinAlg::Matrix<1, 6> dGrowthScalarDE = evaluate_d_lambdafsq_dc(gp, eleGID);
  dGrowthScalarDE.scale(
      2.0 * full_constrained_mixture_fiber_[gp].computed_dgrowth_scalar_dlambda_f_sq_);
  return dGrowthScalarDE;
}

double MIXTURE::MixtureConstituentFullConstrainedMixtureFiber::evaluate_initial_deposition_stretch(
    const double time) const
{
  if (params_->initial_deposition_stretch_timefunc_num_ == 0)
  {
    return params_->deposition_stretch_;
  }

  return Global::Problem::Instance()
      ->FunctionById<Core::UTILS::FunctionOfTime>(
          params_->initial_deposition_stretch_timefunc_num_ - 1)
      .evaluate(time);
}

double MIXTURE::MixtureConstituentFullConstrainedMixtureFiber::evaluate_lambdaf(
    const Core::LinAlg::Matrix<3, 3>& C, const int gp, const int eleGID) const
{
  return std::sqrt(C.dot(anisotropy_extension_.get_structural_tensor(gp, 0)));
}
FOUR_C_NAMESPACE_CLOSE