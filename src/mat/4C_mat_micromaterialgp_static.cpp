/*----------------------------------------------------------------------*/
/*! \file
\brief
class for handling of micro-macro transitions

\level 3


*----------------------------------------------------------------------*/


#include "4C_mat_micromaterialgp_static.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_so3_hex8.hpp"
#include "4C_so3_shw6.hpp"
#include "4C_stru_multi_microstatic.hpp"

#include <filesystem>

FOUR_C_NAMESPACE_OPEN


std::map<int, Teuchos::RCP<MultiScale::MicroStatic>> Mat::MicroMaterialGP::microstaticmap_;
std::map<int, int> Mat::MicroMaterialGP::microstaticcounter_;


/// construct an instance of MicroMaterial for a given Gauss point and
/// microscale discretization

Mat::MicroMaterialGP::MicroMaterialGP(
    const int gp, const int ele_ID, const bool eleowner, const int microdisnum, const double V0)
    : gp_(gp), ele_id_(ele_ID), microdisnum_(microdisnum)
{
  Global::Problem* microproblem = Global::Problem::Instance(microdisnum_);
  Teuchos::RCP<Core::FE::Discretization> microdis = microproblem->GetDis("structure");
  dis_ = Core::LinAlg::CreateVector(*microdis->dof_row_map(), true);
  disn_ = Core::LinAlg::CreateVector(*microdis->dof_row_map(), true);
  lastalpha_ = Teuchos::rcp(new std::map<int, Teuchos::RCP<Core::LinAlg::SerialDenseMatrix>>);
  oldalpha_ = Teuchos::rcp(new std::map<int, Teuchos::RCP<Core::LinAlg::SerialDenseMatrix>>);
  oldfeas_ = Teuchos::rcp(new std::map<int, Teuchos::RCP<Core::LinAlg::SerialDenseMatrix>>);
  old_kaainv_ = Teuchos::rcp(new std::map<int, Teuchos::RCP<Core::LinAlg::SerialDenseMatrix>>);
  old_kda_ = Teuchos::rcp(new std::map<int, Teuchos::RCP<Core::LinAlg::SerialDenseMatrix>>);

  // data must be consistent between micro and macro input file
  const Teuchos::ParameterList& sdyn_macro =
      Global::Problem::Instance()->structural_dynamic_params();
  const Teuchos::ParameterList& sdyn_micro = microproblem->structural_dynamic_params();

  dt_ = sdyn_macro.get<double>("TIMESTEP");
  microdis->Comm().Broadcast(&dt_, 1, 0);
  step_ = 0;
  stepn_ = step_ + 1;
  time_ = 0.;
  timen_ = time_ + dt_;

  // if class handling microscale simulations is not yet initialized
  // -> set up

  if (microstaticmap_.find(microdisnum_) == microstaticmap_.end() or
      microstaticmap_[microdisnum_] == Teuchos::null)
  {
    // create "time integration" class for this microstructure
    microstaticmap_[microdisnum_] = Teuchos::rcp(new MultiScale::MicroStatic(microdisnum_, V0));
    // create a counter of macroscale GP associated with this "time integration" class
    // note that the counter is immediately updated afterwards!
    microstaticcounter_[microdisnum_] = 0;
  }

  microstaticcounter_[microdisnum] += 1;
  density_ = (microstaticmap_[microdisnum_])->Density();

  // create and initialize "empty" EAS history map (if necessary)
  eas_init();

  std::string newfilename;
  new_result_file(eleowner, newfilename);

  // check whether we are using modified Newton as a nonlinear solver
  // on the macroscale or not
  if (Core::UTILS::IntegralValue<Inpar::STR::NonlinSolTech>(sdyn_micro, "NLNSOL") ==
      Inpar::STR::soltech_newtonmod)
    mod_newton_ = true;
  else
    mod_newton_ = false;

  build_stiff_ = true;
}

/// destructor

Mat::MicroMaterialGP::~MicroMaterialGP()
{
  microstaticcounter_[microdisnum_] -= 1;
  if (microstaticcounter_[microdisnum_] == 0) microstaticmap_[microdisnum_] = Teuchos::null;
}


/// Read restart

void Mat::MicroMaterialGP::read_restart()
{
  step_ = Global::Problem::Instance()->restart();
  microstaticmap_[microdisnum_]->read_restart(step_, dis_, lastalpha_, restartname_);

  *oldalpha_ = *lastalpha_;

  disn_->Update(1.0, *dis_, 0.0);
}


/// New resultfile

void Mat::MicroMaterialGP::new_result_file(bool eleowner, std::string& newfilename)
{
  // set up micro output
  //
  // Get the macro output prefix and insert element and gauss point
  // identifier. We use the original name here and rely on our (micro)
  // OutputControl object below to act just like the macro (default)
  // OutputControl. In particular we assume that there are always micro and
  // macro control files on restart.
  Teuchos::RCP<Core::IO::OutputControl> macrocontrol =
      Global::Problem::Instance(0)->OutputControlFile();
  std::string microprefix = macrocontrol->restart_name();
  std::string micronewprefix = macrocontrol->new_output_file_name();

  Global::Problem* microproblem = Global::Problem::Instance(microdisnum_);
  Teuchos::RCP<Core::FE::Discretization> microdis = microproblem->GetDis("structure");

  if (microdis->Comm().MyPID() == 0)
  {
    // figure out prefix of micro-scale restart files
    restartname_ = new_result_file_path(microprefix);

    // figure out new prefix for micro-scale output files
    newfilename = new_result_file_path(micronewprefix);
  }

  // restart file name and new output file name are sent to supporting procs
  if (microdis->Comm().NumProc() > 1)
  {
    {
      // broadcast restartname_ for micro scale
      int length = restartname_.length();
      std::vector<int> name(restartname_.begin(), restartname_.end());
      int err = microdis->Comm().Broadcast(&length, 1, 0);
      if (err) FOUR_C_THROW("communication error");
      name.resize(length);
      err = microdis->Comm().Broadcast(name.data(), length, 0);
      if (err) FOUR_C_THROW("communication error");
      restartname_.assign(name.begin(), name.end());
    }

    {
      // broadcast newfilename for micro scale
      int length = newfilename.length();
      std::vector<int> name(newfilename.begin(), newfilename.end());
      int err = microdis->Comm().Broadcast(&length, 1, 0);
      if (err) FOUR_C_THROW("communication error");
      name.resize(length);
      err = microdis->Comm().Broadcast(name.data(), length, 0);
      if (err) FOUR_C_THROW("communication error");
      newfilename.assign(name.begin(), name.end());
    }
  }

  if (eleowner)
  {
    const int ndim = Global::Problem::Instance()->NDim();
    const int restart = Global::Problem::Instance()->restart();
    bool adaptname = true;
    // in case of restart, the new output file name is already adapted
    if (restart) adaptname = false;

    Teuchos::RCP<Core::IO::OutputControl> microcontrol =
        Teuchos::rcp(new Core::IO::OutputControl(microdis->Comm(), "Structure",
            microproblem->spatial_approximation_type(), "micro-input-file-not-known", restartname_,
            newfilename, ndim, restart, macrocontrol->file_steps(),
            Core::UTILS::IntegralValue<bool>(microproblem->IOParams(), "OUTPUT_BIN"), adaptname));

    micro_output_ = Teuchos::rcp(new Core::IO::DiscretizationWriter(
        microdis, microcontrol, microproblem->spatial_approximation_type()));
    micro_output_->set_output(microcontrol);

    micro_output_->write_mesh(step_, time_);
  }
}

std::string Mat::MicroMaterialGP::new_result_file_path(const std::string& newprefix)
{
  std::string newfilename;

  // create path from string to extract only filename prefix
  const std::filesystem::path path(newprefix);
  const std::string newfileprefix = path.filename().string();

  const size_t posn = newfileprefix.rfind('-');
  if (posn != std::string::npos)
  {
    std::string number = newfileprefix.substr(posn + 1);
    std::string prefix = newfileprefix.substr(0, posn);

    // recombine path and file
    const std::filesystem::path parent_path(path.parent_path());
    const std::filesystem::path filen_name(prefix);
    const std::filesystem::path recombined_path = parent_path / filen_name;

    std::ostringstream s;
    s << recombined_path.string() << "_el" << ele_id_ << "_gp" << gp_ << "-" << number;
    newfilename = s.str();
  }
  else
  {
    std::ostringstream s;
    s << newprefix << "_el" << ele_id_ << "_gp" << gp_;
    newfilename = s.str();
  }
  return newfilename;
}


void Mat::MicroMaterialGP::eas_init()
{
  Teuchos::RCP<Core::FE::Discretization> discret =
      (Global::Problem::Instance(microdisnum_))->GetDis("structure");

  for (int lid = 0; lid < discret->ElementRowMap()->NumMyElements(); ++lid)
  {
    Core::Elements::Element* actele = discret->lRowElement(lid);

    if (actele->ElementType() == Discret::ELEMENTS::SoHex8Type::Instance() or
        actele->ElementType() == Discret::ELEMENTS::SoShw6Type::Instance())
    {
      // create the parameters for the discretization
      Teuchos::ParameterList p;
      // action for elements
      p.set("action", "multi_eas_init");
      p.set("lastalpha", lastalpha_);
      p.set("oldalpha", oldalpha_);
      p.set("oldfeas", oldfeas_);
      p.set("oldKaainv", old_kaainv_);
      p.set("oldKda", old_kda_);

      Core::LinAlg::SerialDenseMatrix elematrix1;
      Core::LinAlg::SerialDenseMatrix elematrix2;
      Core::LinAlg::SerialDenseVector elevector1;
      Core::LinAlg::SerialDenseVector elevector2;
      Core::LinAlg::SerialDenseVector elevector3;
      std::vector<int> lm;

      actele->evaluate(p, *discret, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);
    }
  }

  return;
}



void Mat::MicroMaterialGP::ResetTimeAndStep()
{
  time_ = 0.0;
  timen_ = time_ + dt_;
  step_ = 0;
  stepn_ = step_ + 1;
}


/// perform microscale simulation

void Mat::MicroMaterialGP::perform_micro_simulation(Core::LinAlg::Matrix<3, 3>* defgrd,
    Core::LinAlg::Matrix<6, 1>* stress, Core::LinAlg::Matrix<6, 6>* cmat)
{
  // select corresponding "time integration class" for this microstructure
  Teuchos::RCP<MultiScale::MicroStatic> microstatic = microstaticmap_[microdisnum_];

  // set displacements and EAS data of last step
  microstatic->set_state(dis_, disn_, stress_, strain_, plstrain_, lastalpha_, oldalpha_, oldfeas_,
      old_kaainv_, old_kda_);

  // set current time, time step size and step number
  microstatic->set_time(time_, timen_, dt_, step_, stepn_);

  microstatic->Predictor(defgrd);
  microstatic->FullNewton();
  microstatic->static_homogenization(stress, cmat, defgrd, mod_newton_, build_stiff_);

  // note that it is not necessary to save displacements and EAS data
  // explicitly since we dealt with Teuchos::RCP's -> any update in class
  // microstatic and the elements, respectively, inherently updates the
  // micromaterialgp_static data!

  // clear displacements in MicroStruGenAlpha for next usage
  microstatic->ClearState();
}


void Mat::MicroMaterialGP::update()
{
  // select corresponding "time integration class" for this microstructure
  Teuchos::RCP<MultiScale::MicroStatic> microstatic = microstaticmap_[microdisnum_];

  time_ = timen_;
  timen_ += dt_;
  step_ = stepn_;
  stepn_++;

  dis_->Update(1.0, *disn_, 0.0);

  Global::Problem* microproblem = Global::Problem::Instance(microdisnum_);
  Teuchos::RCP<Core::FE::Discretization> microdis = microproblem->GetDis("structure");
  const Epetra_Map* elemap = microdis->ElementRowMap();

  for (int i = 0; i < elemap->NumMyElements(); ++i) (*lastalpha_)[i] = (*oldalpha_)[i];

  // in case of modified Newton, the stiffness matrix needs to be rebuilt at
  // the beginning of the new time step
  build_stiff_ = true;
}


void Mat::MicroMaterialGP::prepare_output()
{
  // select corresponding "time integration class" for this microstructure
  Teuchos::RCP<MultiScale::MicroStatic> microstatic = microstaticmap_[microdisnum_];

  stress_ = Teuchos::rcp(new std::vector<char>());
  strain_ = Teuchos::rcp(new std::vector<char>());
  plstrain_ = Teuchos::rcp(new std::vector<char>());

  microstatic->set_state(dis_, disn_, stress_, strain_, plstrain_, lastalpha_, oldalpha_, oldfeas_,
      old_kaainv_, old_kda_);
  microstatic->set_time(time_, timen_, dt_, step_, stepn_);
  microstatic->prepare_output();
}


void Mat::MicroMaterialGP::output()
{
  // select corresponding "time integration class" for this microstructure
  Teuchos::RCP<MultiScale::MicroStatic> microstatic = microstaticmap_[microdisnum_];

  // set displacements and EAS data of last step
  microstatic->set_state(dis_, disn_, stress_, strain_, plstrain_, lastalpha_, oldalpha_, oldfeas_,
      old_kaainv_, old_kda_);
  microstatic->output(micro_output_, time_, step_, dt_);

  // we don't need these containers anymore
  stress_ = Teuchos::null;
  strain_ = Teuchos::null;
  plstrain_ = Teuchos::null;
}

FOUR_C_NAMESPACE_CLOSE