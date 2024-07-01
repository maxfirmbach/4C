/*----------------------------------------------------------------------------*/
/*! \file

\brief spherical particle element for brownian dynamics

\level 3

*/
/*----------------------------------------------------------------------------*/

#ifndef FOUR_C_RIGIDSPHERE_HPP
#define FOUR_C_RIGIDSPHERE_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_largerotations.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_integration.hpp"

#include <Epetra_Vector.h>
#include <Teuchos_RCP.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <unordered_map>

FOUR_C_NAMESPACE_OPEN

// forward declaration ...
namespace BEAMINTERACTION
{
  class BeamLinkPinJointed;
}
namespace STR
{
  namespace ELEMENTS
  {
    class ParamsInterface;
  }
}  // namespace STR

namespace Discret
{
  namespace ELEMENTS
  {
    class RigidsphereType : public Core::Elements::ElementType
    {
     public:
      std::string Name() const override { return ("RigidsphereType"); }

      static RigidsphereType& Instance();

      Core::Communication::ParObject* Create(const std::vector<char>& data) override;

      Teuchos::RCP<Core::Elements::Element> Create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      Teuchos::RCP<Core::Elements::Element> Create(const int id, const int owner) override;

      int initialize(Core::FE::Discretization& dis) override;

      void nodal_block_information(
          Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override;

      Core::LinAlg::SerialDenseMatrix ComputeNullSpace(
          Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
          override;

     private:
      static RigidsphereType instance_;
    };

    /*!
    \brief Spherical particle element for brownian dynamics

    */
    class Rigidsphere : public Core::Elements::Element
    {
     public:
      //! @name Friends
      friend class RigidsphereType;


      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id    (in): A globally unique element id
      \param etype (in): Type of element
      \param owner (in): owner processor of the element
      */
      Rigidsphere(int id, int owner);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element
      */
      Rigidsphere(const Rigidsphere& old);



      /*!
      \brief Deep copy this instance of Beam3eb and return pointer to the copy

      The Clone() method is used by the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed
    .
      */
      Core::Elements::Element* Clone() const override;

      /*!
     \brief Get shape type of element
     */
      Core::FE::CellType Shape() const override;


      /*!
      \brief Return unique ParObject id

      Every class implementing ParObject needs a unique id defined at the
      top of parobject.H
      */
      int UniqueParObjectId() const override
      {
        return (RigidsphereType::Instance().UniqueParObjectId());
      }

      /*!
      \brief Pack this class so it can be communicated

      \ref pack and \ref unpack are used to communicate this element

      */
      void pack(Core::Communication::PackBuffer& data) const override;

      /*!
      \brief Unpack data from a char vector into this class

      \ref pack and \ref unpack are used to communicate this element

      */
      void unpack(const std::vector<char>& data) override;

      Core::Elements::ElementType& ElementType() const override
      {
        return (RigidsphereType::Instance());
      }

      //@}

      /*!
      \brief Return number of lines to this element
      */
      int NumLine() const override { return (1); }

      /*!
      \brief Get vector of Teuchos::RCPs to the lines of this element
      */
      std::vector<Teuchos::RCP<Core::Elements::Element>> Lines() override;


      /*!
      \brief Get number of degrees of freedom of a single node
      */
      int NumDofPerNode(const Core::Nodes::Node& node) const override
      {
        /*note: this is not necessarily the number of DOF assigned to this node by the
         *discretization finally, but only the number of DOF requested for this node by this
         *element; the discretization will finally assign the maximal number of DOF to this node
         *requested by any element connected to this node*/
        return (3);
      }

      /*!
      \brief Get number of degrees of freedom per element not including nodal degrees of freedom
      */
      int num_dof_per_element() const override { return (0); }

      /*!
      \brief Print this element
      */
      void print(std::ostream& os) const override;

      /*!
      \brief Return radius of sphere
      */
      const double& Radius() const { return (radius_); };

      //! @name Construction

      /*!
      \brief Read input for this element

      This class implements a dummy of this method that prints a warning and
      returns false. A derived class would read one line from the input file and
      store all necessary information.

      */
      // virtual bool ReadElement();

      /*!
      \brief Read input for this element
      */
      bool ReadElement(const std::string& eletype, const std::string& distype,
          Input::LineDefinition* linedef) override;

      //@}

      //! @name Evaluation methods


      /*!
      \brief Evaluate an element

      An element derived from this class uses the Evaluate method to receive commands
      and parameters from some control routine in params and evaluates element matrices and
      vectors accoring to the command in params.

      \note This class implements a dummy of this method that prints a warning and
            returns false.

      \param params (in/out)    : ParameterList for communication between control routine
                                  and elements
      \param discretization (in): A reference to the underlying discretization
      \param lm (in)            : location vector of this element
      \param elemat1 (out)      : matrix to be filled by element depending on commands
                                  given in params
      \param elemat2 (out)      : matrix to be filled by element depending on commands
                                  given in params
      \param elevec1 (out)      : vector to be filled by element depending on commands
                                  given in params
      \param elevec2 (out)      : vector to be filled by element depending on commands
                                  given in params
      \param elevec3 (out)      : vector to be filled by element depending on commands
                                  given in params
      \return 0 if successful, negative otherwise
      */
      int evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          std::vector<int>& lm, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3) override;


      /*!
      \brief Evaluate a Neumann boundary condition

      An element derived from this class uses the evaluate_neumann method to receive commands
      and parameters from some control routine in params and evaluates a Neumann boundary condition
      given in condition

      \note This class implements a dummy of this method that prints a warning and
            returns false.

      \param params (in/out)    : ParameterList for communication between control routine
                                  and elements
      \param discretization (in): A reference to the underlying discretization
      \param condition (in)     : The condition to be evaluated
      \param lm (in)            : location vector of this element
      \param elevec1 (out)      : Force vector to be filled by element

      \return 0 if successful, negative otherwise
      */
      int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          Core::Conditions::Condition& condition, std::vector<int>& lm,
          Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override
      {
        return (0);
      };

      /*!
      \brief Evaluate mass matrix
      */
      void nlnstiffmass(Teuchos::ParameterList& params, std::vector<double>& acc,
          std::vector<double>& vel, std::vector<double>& disp,
          Core::LinAlg::SerialDenseMatrix* stiffmatrix, Core::LinAlg::SerialDenseMatrix* massmatrix,
          Core::LinAlg::SerialDenseVector* force, Core::LinAlg::SerialDenseVector* inertia_force);

      /*! \brief set the parameter interface ptr for the solid elements
       *
       *  \param p (in): Parameter list coming from the time integrator.
       *
       *  \author hiermeier
       *  \date 04/16 */
      void set_params_interface_ptr(const Teuchos::ParameterList& p) override;

      /*! \brief returns true if the parameter interface is defined and initialized, otherwise false
       *
       *  \author hiermeier
       *  \date 04/16 */
      inline bool IsParamsInterface() const override { return (not interface_ptr_.is_null()); }

      /*! \brief get access to the parameter interface pointer
       *
       *  \author hiermeier
       *  \date 04/16 */
      Teuchos::RCP<Core::Elements::ParamsInterface> ParamsInterfacePtr() override;
      //@}

      //! @name methods for biopolymer network simulations

      //! computes the number of different random numbers required in each time step for generation
      //! of stochastic forces
      int how_many_random_numbers_i_need();

      /// \brief get generalized interpolation matrix which yields the variation of the position
      virtual void get_generalized_interpolation_matrix_variations_at_xi(
          Core::LinAlg::SerialDenseMatrix& Ivar, const double& dummy1,
          const std::vector<double>& dummy2) const;

      /// \brief get generalized interpolation matrix which yields the increments of the position
      virtual void get_generalized_interpolation_matrix_increments_at_xi(
          Core::LinAlg::SerialDenseMatrix& Iinc, const double& dummy1,
          const std::vector<double>& dummy2) const;

      /** \brief get linearization of the product of (generalized interpolation matrix for
       * variations (see above) and applied force vector) with respect to the primary DoFs of this
       * element
       *
       *  \author grill
       *  \date 01/17 */
      virtual void get_stiffmat_resulting_from_generalized_interpolation_matrix_at_xi(
          Core::LinAlg::SerialDenseMatrix& stiffmat, const double& xi,
          const std::vector<double>& disp, const Core::LinAlg::SerialDenseVector& force) const
      {
        // nothing to do here
        stiffmat.putScalar(0.0);
      }

      /*!
      \brief set radius of sphere
      */
      void SetRadius(double radius) { radius_ = radius; };

      /*!
      \brief set radius of sphere
      */
      void ScaleRadius(double scalefac) { radius_ *= scalefac; };

      /// return binding spot xi function dummy
      double GetBindingSpotXi(Inpar::BEAMINTERACTION::CrosslinkerType dummy1, int dummy2) const
      {
        return 0.0;
      }

      /** \brief get number of bonds
       *
       *  \author eichinger
       *  \date 06/17 */
      int GetNumberOfBonds() const { return static_cast<int>(mybondstobeams_.size()); }

      /** \brief check if bond exists
       *
       *  \author eichinger
       *  \date 06/17 */
      bool DoesBondExist(int bond) const
      {
        return (mybondstobeams_.find(bond) != mybondstobeams_.end()) ? true : false;
      }

      /** \brief check bond with certain bond gid
       *
       *  \author eichinger
       *  \date 06/17 */
      Teuchos::RCP<BEAMINTERACTION::BeamLinkPinJointed> GetBond(int bond_id)
      {
        return mybondstobeams_[bond_id];
      }

      /** \brief check bond with certain bond gid
       *
       *  \author eichinger
       *  \date 06/17 */
      std::map<int, Teuchos::RCP<BEAMINTERACTION::BeamLinkPinJointed>> const& GetBondMap() const
      {
        return mybondstobeams_;
      }

      /** \brief add bond
       *
       *  \author eichinger
       *  \date 06/17 */
      void AddBond(int id, Teuchos::RCP<BEAMINTERACTION::BeamLinkPinJointed> newbondpartner)
      {
        mybondstobeams_[id] = newbondpartner;
      };

      /** \brief disolve bond
       *
       *  \author eichinger
       *  \date 06/17 */
      void dissolve_bond(int id)
      {
#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (mybondstobeams_.find(id) == mybondstobeams_.end())
          FOUR_C_THROW(" You want to dissolve not existing bond. Something went wrong.");
#endif

        mybondstobeams_.erase(id);
      };

      /**
       * \brief Get the bounding volume of the element for geometric search
       *
       * @param discret discretization of the respective field
       * @param result_data_dofbased Result data vector used for extracting positions
       * @return bounding volume of the respective element
       */
      Core::GeometricSearch::BoundingVolume GetBoundingVolume(
          const Core::FE::Discretization& discret, const Epetra_Vector& result_data_dofbased,
          const Core::GeometricSearch::GeometricSearchParams& params) const override;

      //@}

     protected:
      /** \brief get access to the interface
       *
       *  \author hiermeier
       *  \date 04/16 */
      inline STR::ELEMENTS::ParamsInterface& params_interface()
      {
        if (not IsParamsInterface()) FOUR_C_THROW("The interface ptr is not set!");
        return *interface_ptr_;
      }

     private:
      /*! \brief interface ptr
       *
       *  data exchange between the element and the time integrator. */
      Teuchos::RCP<STR::ELEMENTS::ParamsInterface> interface_ptr_;

      //! radius of the sphere
      double radius_;

      //! density of the sphere
      double rho_;

      //! @name variables for biopolymer network simulations

      /// holds unique id of beam element binding spot to which sphere is bonded to (size equals
      /// number of bonds)
      std::map<int, Teuchos::RCP<BEAMINTERACTION::BeamLinkPinJointed>> mybondstobeams_;

      //@}

      //! @name methods for initialization of the element

      //@}

      //! @name Internal calculation methods

      //! calculation of thermal (i.e. stochastic) and damping forces according to Brownian dynamics
      void calc_brownian_forces_and_stiff(Teuchos::ParameterList& params, std::vector<double>& vel,
          std::vector<double>& disp, Core::LinAlg::SerialDenseMatrix* stiffmatrix,
          Core::LinAlg::SerialDenseVector* force);

      //! calculation of drag force and corresponding stiffness contribution
      void calc_drag_force(Teuchos::ParameterList& params, const std::vector<double>& vel,
          const std::vector<double>& disp, Core::LinAlg::SerialDenseMatrix* stiffmatrix,
          Core::LinAlg::SerialDenseVector* force);

      //! calculation of stochastic force and corresponding stiffness contribution
      void calc_stochastic_force(Teuchos::ParameterList& params, const std::vector<double>& vel,
          const std::vector<double>& disp, Core::LinAlg::SerialDenseMatrix* stiffmatrix,
          Core::LinAlg::SerialDenseVector* force);

      //  //!calculation of inertia force and corresponding stiffness contribution
      //  void CalcInertiaForce( Teuchos::ParameterList&   params,
      //                         std::vector<double>&      vel,
      //                         std::vector<double>&      disp,
      //                         Core::LinAlg::SerialDenseMatrix* stiffmatrix,
      //                         Core::LinAlg::SerialDenseVector* force);

      //! calculation of background fluid velocity and gradient of velocity
      void get_background_velocity(Teuchos::ParameterList& params,
          Core::LinAlg::Matrix<3, 1>& velbackground, Core::LinAlg::Matrix<3, 3>& velbackgroundgrad);

      //! computes damping coefficient
      double my_damping_constant();

      // don't want = operator
      Rigidsphere& operator=(const Rigidsphere& old);

    };  // class Rigidsphere

    // << operator
    std::ostream& operator<<(std::ostream& os, const Core::Elements::Element& ele);

  }  // namespace ELEMENTS
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif