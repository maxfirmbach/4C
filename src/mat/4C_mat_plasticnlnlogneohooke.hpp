
/*! \file
\brief Contains the functions to establish local material law /
       stress-strain law for isotropic material following large strain
       von-Mises plasticity with linear isotropic hardening
       and logarithmic hyperelastic material (i.e. linear relation
       between Kirchhoff-stress and logarithmic strain; also known as Hencky
       material model).
       The principal stress-based implementation follows
       Bonet and Wood: "Nonlinear continuum mechanics for finite element analysis.",
       Cambridge University Press, Cambridge, 2008

       geometrically nonlinear, finite strains, visco-plastic

       example input line:
       MAT 1 MAT_Struct_PlasticNlnLogNeoHooke YOUNG 206.9 NUE 0.29 DENS 0.0
         YIELD 0.45 ISOHARD 0.12924 SATHARDENING 0.715 HARDEXPO 16.93 VISC 1.0 RATE_DEPENDENCY 0.1

\level 2


*/
/*----------------------------------------------------------------------*
 | definitions                                                          |
 *----------------------------------------------------------------------*/
#ifndef FOUR_C_MAT_PLASTICNLNLOGNEOHOOKE_HPP
#define FOUR_C_MAT_PLASTICNLNLOGNEOHOOKE_HPP

/*----------------------------------------------------------------------*
 | headers                                                              |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core
{
  namespace UTILS
  {
    class FunctionOfAnything;
  }
}  // namespace Core

namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    //! material parameters for neo-Hooke
    class PlasticNlnLogNeoHooke : public Core::Mat::PAR::Parameter
    {
     public:
      //! standard constructor
      PlasticNlnLogNeoHooke(const Core::Mat::PAR::Parameter::Data& matdata);

      //! @name material parameters
      //@{

      //! Young's modulus
      const double youngs_;
      //! Possion's ratio
      const double poissonratio_;
      //! mass density
      const double density_;
      //! initial yield stress (constant)
      const double yield_;
      //! linear isotropic hardening modulus
      const double isohard_;
      //! saturation yield stress
      const double infyield_;
      //! nonlinear hardening exponent
      const double hardexp_;
      //! viscosity
      const double visc_;
      //! rate dependency
      const double rate_dependency_;
      //! function ID for evaluation of saturation
      const int functionID_hardening_;
      //! maximum number of Newton Raphson iterations
      const int max_iterations_;
      //! newton Raphson tolerance
      const double tolerance_nr_;


      //@}

      //! create material instance of matching type with my parameters
      Teuchos::RCP<Core::Mat::Material> create_material() override;

    };  // class PlasticNlnLogNeoHooke

  }  // namespace PAR


  class PlasticNlnLogNeoHookeType : public Core::Communication::ParObjectType
  {
   public:
    std::string Name() const override { return "PlasticNlnLogNeoHookeType"; }

    static PlasticNlnLogNeoHookeType& Instance() { return instance_; };

    Core::Communication::ParObject* Create(const std::vector<char>& data) override;

   private:
    static PlasticNlnLogNeoHookeType instance_;

  };  // class PlasticNlnLogNeoHookeType

  /*----------------------------------------------------------------------*/
  //! wrapper for finite strain elasto-plastic material

  class PlasticNlnLogNeoHooke : public So3Material
  {
   public:
    //! construct empty material object
    PlasticNlnLogNeoHooke();

    //! construct the material object given material parameters
    explicit PlasticNlnLogNeoHooke(Mat::PAR::PlasticNlnLogNeoHooke* params);

    //! @name Packing and Unpacking

    /*!
    \brief Return unique ParObject id

    every class implementing ParObject needs a unique id defined at the
    top of parobject.H (this file) and should return it in this method.
    */
    int UniqueParObjectId() const override
    {
      return PlasticNlnLogNeoHookeType::Instance().UniqueParObjectId();
    }

    /*!
    \brief Pack this class so it can be communicated

    Resizes the vector data and stores all information of a class in it.
    The first information to be stored in data has to be the
    unique parobject id delivered by UniqueParObjectId() which will then
    identify the exact class on the receiving processor.

    \param data (in/out): char vector to store class information
    */
    void pack(Core::Communication::PackBuffer& data) const override;

    /*!
    \brief Unpack data from a char vector into this class

    The vector data contains all information to rebuild the
    exact copy of an instance of a class on a different processor.
    The first entry in data has to be an integer which is the unique
    parobject id defined at the top of this file and delivered by
    UniqueParObjectId().

    \param data (in) : vector storing all data to be unpacked into this
    instance.
    */
    void unpack(const std::vector<char>& data) override;

    //@}

    //! @name Access methods

    //! material type
    Core::Materials::MaterialType MaterialType() const override
    {
      return Core::Materials::m_plnlnlogneohooke;
    }

    /// check if element kinematics and material kinematics are compatible
    void ValidKinematics(Inpar::STR::KinemType kinem) override
    {
      if (!(kinem == Inpar::STR::KinemType::nonlinearTotLag))
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    //! return copy of this material object
    Teuchos::RCP<Core::Mat::Material> Clone() const override
    {
      return Teuchos::rcp(new PlasticNlnLogNeoHooke(*this));
    }

    //! density
    double Density() const override { return params_->density_; }

    //! return quick accessible material parameter data
    Core::Mat::PAR::Parameter* Parameter() const override { return params_; }

    //! return accumulated strain at Gauss points
    //! use the old vector (last_) for postprocessing
    //! Output is called after(!!) Update, so that newest values are included in
    //! the old history vectors last_, while the current history vectors curr_
    //! are reset
    double AccumulatedStrain(int gp) const { return (accplstrainlast_.at(gp)); }

    //! return 1 if the material point is actively yielding (gamma > 0); else 0
    double ActiveYielding(int gp) const { return (activeyield_.at(gp)); }

    //! //! check if history variables are already initialized
    bool Initialized() const { return (isinit_ and !accplstraincurr_.empty()); }

    //! return names of visualization data
    void VisNames(std::map<std::string, int>& names) override;

    //! return visualization data
    bool VisData(const std::string& name, std::vector<double>& data, int numgp, int eleID) override;

    //! return names of visualization data available for direct VTK output
    void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const override;

    //! return visualization data for direct VTK output
    bool EvaluateOutputData(
        const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const override;

    /// Return whether the material requires the deformation gradient for its evaluation
    bool needs_defgrd() override { return true; };

    //@}

    //! @name Evaluation methods

    //! initialise internal stress variables
    void setup(int numgp, Input::LineDefinition* linedef) override;

    //! update internal stress variables
    void update() override;

    //! evaluate material law
    void evaluate(const Core::LinAlg::Matrix<3, 3>*
                      defgrd,  //!< input deformation gradient for multiplicative sp
        const Core::LinAlg::Matrix<6, 1>*
            glstrain,                        //!< input Green-Lagrange strain (redundant with defo
                                             //   but used for neo-hooke evaluation; maybe remove
        Teuchos::ParameterList& params,      //!< input parameter list (e.g. Young's, ...)
        Core::LinAlg::Matrix<6, 1>* stress,  //!< output (mandatory) second Piola-Kirchhoff stress
        Core::LinAlg::Matrix<6, 6>* cmat,    //!< output (mandatory) material stiffness matrix
        int gp,                              //!< Gauss point
        int eleGID) override;

   private:
    //! my material parameters
    Mat::PAR::PlasticNlnLogNeoHooke* params_;

    //! inverse right cauchy green of plastic strain
    std::vector<Core::LinAlg::Matrix<3, 3>> invplrcglast_;
    //! inverse right cauchy green of plastic strain
    std::vector<Core::LinAlg::Matrix<3, 3>> invplrcgcurr_;

    //! old (i.e. at t_n) accumulated plastic strain
    std::vector<double> accplstrainlast_;
    //! current (i.e. at t_n+1) accumulated plastic strain
    std::vector<double> accplstraincurr_;
    //! active yielding between t_n and t_n+1
    std::vector<double> activeyield_;

    const Core::UTILS::FunctionOfAnything* hardening_function_{nullptr};

    //! indicator if #Initialize routine has been called
    bool isinit_;
  };  // class PlasticNlnLogNeoHooke

}  // namespace Mat


/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif