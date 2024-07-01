/*----------------------------------------------------------------------*/
/*! \file

\brief Base of four-element Maxwell material model for reduced dimensional
acinus elements

Four-element Maxwell model consists of a parallel configuration of a spring (Stiffness1),
spring-dashpot (Stiffness2 and Viscosity1) and dashpot (Viscosity2) element
(derivation: see Ismail Mahmoud's dissertation, chapter 3.4)


\level 3
*/
/*----------------------------------------------------------------------*/
#ifndef FOUR_C_MAT_MAXWELL_0D_ACINUS_HPP
#define FOUR_C_MAT_MAXWELL_0D_ACINUS_HPP


#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_red_airways_elem_params.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// material parameters for Maxwell 0D acinar material
    ///
    class Maxwell0dAcinus : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      Maxwell0dAcinus(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{
      /// first stiffness of the Maxwell model
      const double stiffness1_;
      /// first stiffness of the Maxwell model
      const double stiffness2_;
      /// first viscosity of the Maxwell model
      const double viscosity1_;
      /// first viscosity of the Maxwell model
      const double viscosity2_;
      //@}

      /// create material instance of matching type with my parameters
      Teuchos::RCP<Core::Mat::Material> create_material() override;

    };  // class Maxwell_0d_acinus

  }  // namespace PAR

  class Maxwell0dAcinusType : public Core::Communication::ParObjectType
  {
   public:
    std::string Name() const override { return "maxwell_0d_acinusType"; }

    static Maxwell0dAcinusType& Instance() { return instance_; };

    Core::Communication::ParObject* Create(const std::vector<char>& data) override;

   private:
    static Maxwell0dAcinusType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for Maxwell 0D acinar material
  ///
  /// This object exists (several times) at every element
  class Maxwell0dAcinus : public Core::Mat::Material
  {
   public:
    /// construct empty material object
    Maxwell0dAcinus();

    /// construct the material object given material parameters
    explicit Maxwell0dAcinus(Mat::PAR::Maxwell0dAcinus* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int UniqueParObjectId() const override
    {
      return Maxwell0dAcinusType::Instance().UniqueParObjectId();
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

    /*!
      \brief
    */
    virtual void setup(Input::LineDefinition* linedef)
    {
      FOUR_C_THROW(
          "Setup not implemented yet! Check your material type, "
          "maybe you are still using the base class MAT_0D_MAXWELL_ACINUS.");
    }

    /*!
       \brief
     */
    virtual void evaluate(Core::LinAlg::SerialDenseVector& epnp,
        Core::LinAlg::SerialDenseVector& epn, Core::LinAlg::SerialDenseVector& epnm,
        Core::LinAlg::SerialDenseMatrix& sysmat, Core::LinAlg::SerialDenseVector& rhs,
        const Discret::ReducedLung::ElemParams& params, const double NumOfAcini, const double Vo,
        double time, double dt)
    {
      FOUR_C_THROW("Evaluate not implemented yet !");
    }

    //@}

    /// material type
    Core::Materials::MaterialType MaterialType() const override
    {
      return Core::Materials::m_0d_maxwell_acinus;
    }

    /// return copy of this material object
    Teuchos::RCP<Core::Mat::Material> Clone() const override
    {
      return Teuchos::rcp(new Maxwell0dAcinus(*this));
    }

    /// return density
    double Density() const override { return -1; }

    /// return first stiffness of the Maxwell model
    double Stiffness1() const { return params_->stiffness1_; }

    /// return first stiffness of the Maxwell model
    double Stiffness2() const { return params_->stiffness2_; }

    /// return first viscosity of the Maxwell model
    double Viscosity1() const { return params_->viscosity1_; }

    /// return first viscosity of the Maxwell model
    double Viscosity2() const { return params_->viscosity2_; }

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* Parameter() const override { return params_; }

    /// Return value of class parameter
    virtual double GetParams(std::string parametername);

    /// Set value of class parameter
    virtual void SetParams(std::string parametername, double new_value);

    /// Return names of visualization data
    virtual void VisNames(std::map<std::string, int>& names){
        /* do nothing for simple material models */
    };

    /// Return visualization data
    virtual bool VisData(const std::string& name, std::vector<double>& data, int eleID)
    { /* do nothing for simple material models */
      return false;
    };

   protected:
    /// my material parameters
    Mat::PAR::Maxwell0dAcinus* params_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif