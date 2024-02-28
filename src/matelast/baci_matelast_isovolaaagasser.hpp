/*----------------------------------------------------------------------*/
/*! \file
\brief Definition of classes for the isochoric contribution of the AAA Gasser material and the
corresponding volumetric contribution.

\level 1
*/
/*----------------------------------------------------------------------*/

#ifndef BACI_MATELAST_ISOVOLAAAGASSER_HPP
#define BACI_MATELAST_ISOVOLAAAGASSER_HPP

#include "baci_config.hpp"

#include "baci_mat_par_parameter.hpp"
#include "baci_matelast_summand.hpp"

BACI_NAMESPACE_OPEN

namespace MAT
{
  namespace ELASTIC
  {
    namespace PAR
    {
      /*!
       * \brief This material provides the isochoric and(!) the volumetric
       * contribution for a AAA thromubs.
       *
       * The input line should read like this:
       *     MAT 21 ELAST_IsoVolAAAGasser CLUM 2.62E3 CMED 1.98E3
       *     CABLUM 1.73E3 NUE 0.49 BETA -2.0
       *
       * An Ogden type material
       * \f$\Psi=c\underset{i=1}{\overset{3}{\sum}}(\lambda_{i}^{4}-1)\f$ is chosen for the
       * isochoric contribution with spatially varying stiffness parameters governed by luminal
       * thrombus layer stiffness CLUM, the medial layer stiffness CMED and the abluminal thrombus
       * layer stiffness CABLUM (see: Gasser et al., Failure properties of intraluminal thrombus in
       * abdominal aortic aneurysm under static and pulsating
       * mechanical loads, J Vasc Surg, 2008). The volumetric
       * contribution is modeled by an Ogden-Simo_Miehe type SEF:
       * \f$\Psi=\frac {\kappa}{\beta^2}(\beta lnJ + J^{-\beta}-1)\f$ (see  Doll and Schweizerhof,
       * On the Development of Volumetric Strain Energy Functions, Journal of Applied Mechanics,
       * 2000).
       *
       * Strain energy function is given by
       * \f[
       *   \Psi = c (\overline{I}_1^2 - 2 \overline{I}_2 -3)
       *          + \frac{8c}{(1-2 \nu) \beta^2} \big(\beta log(J) + J^{-\beta} -1 \big)
       * \f]
       */
      class IsoVolAAAGasser : public MAT::PAR::Parameter
      {
       public:
        /// standard constructor
        IsoVolAAAGasser(const Teuchos::RCP<MAT::PAR::Material>& matdata);

        /// @name material parameters
        //@{

        // !brief enum for mapping between material parameter and entry in the matparams_ vector
        enum matparamnames_
        {
          clum,      /// stiffness parameter (luminal)
          cmed,      /// stiffness parameter (medial)
          cablum,    /// stiffness parameter (abluminal)
          nue,       /// poisson ration
          beta,      /// numerical parameter (should be -2.0 (Doll and Schwizerhof)
                     /// or 9.0(Holzapfel and Ogden))
          normdist,  /// normalized ILT thickness
          cele,
          mu_lum,
          mu_med,
          mu_ablum,
          sigma_lum,
          sigma_med,
          sigma_ablum,
          xi,
          first = clum,
          last = xi
        };

        //@}

        /// Override this method and throw error, as the material should be created in within the
        /// Factory method of the elastic summand
        Teuchos::RCP<MAT::Material> CreateMaterial() override
        {
          dserror(
              "Cannot create a material from this method, as it should be created in "
              "MAT::ELASTIC::Summand::Factory.");
          return Teuchos::null;
        };

        void SetInitToTrue() { isinit_ = true; };

        bool IsInit() { return isinit_; };

       private:
        // has this material been properly initialized?
        double isinit_;

      };  // class IsoVolAAAGasser

    }  // namespace PAR

    class IsoVolAAAGasser : public Summand
    {
     public:
      /// constructor with given material parameters
      IsoVolAAAGasser(MAT::ELASTIC::PAR::IsoVolAAAGasser* params);

      /// @name Access material constants
      //@{

      /// material type
      INPAR::MAT::MaterialType MaterialType() const override
      {
        return INPAR::MAT::mes_isovolaaagasser;
      }

      //@}

      void CalcCele(const int eleGID);

      /// Setup of patient-specific stuff
      void SetupAAA(Teuchos::ParameterList& params, const int eleGID) override;

      // add strain energy
      void AddStrainEnergy(double& psi,  ///< strain energy function
          const CORE::LINALG::Matrix<3, 1>&
              prinv,  ///< principal invariants of right Cauchy-Green tensor
          const CORE::LINALG::Matrix<3, 1>&
              modinv,  ///< modified invariants of right Cauchy-Green tensor
          const CORE::LINALG::Matrix<6, 1>& glstrain,  ///< Green-Lagrange strain
          int gp,                                      ///< Gauss point
          const int eleGID                             ///< element GID
          ) override;

      // Add derivatives with respect to modified invariants.
      void AddDerivativesModified(
          CORE::LINALG::Matrix<3, 1>&
              dPmodI,  ///< first derivative with respect to modified invariants
          CORE::LINALG::Matrix<6, 1>&
              ddPmodII,  ///< second derivative with respect to modified invariants
          const CORE::LINALG::Matrix<3, 1>&
              modinv,       ///< modified invariants of right Cauchy-Green tensor
          int gp,           ///< Gauss point
          const int eleGID  ///< element GID
          ) override;

      /// Indicator for formulation
      void SpecifyFormulation(
          bool& isoprinc,     ///< global indicator for isotropic principal formulation
          bool& isomod,       ///< global indicator for isotropic splitted formulation
          bool& anisoprinc,   ///< global indicator for anisotropic principal formulation
          bool& anisomod,     ///< global indicator for anisotropic splitted formulation
          bool& viscogeneral  ///< global indicator, if one viscoelastic formulation is used
          ) override
      {
        isomod = true;
        return;
      };

      //! @name Visualization methods

      /// Return names of visualization data
      void VisNames(std::map<std::string, int>& names) override;

      /// Return visualization data
      bool VisData(
          const std::string& name, std::vector<double>& data, int numgp, int eleGID) override;

      //@}

     private:
      /// my material parameters
      MAT::ELASTIC::PAR::IsoVolAAAGasser* params_;

    };  // class IsoVolAAAGasser

  }  // namespace ELASTIC
}  // namespace MAT

BACI_NAMESPACE_CLOSE

#endif