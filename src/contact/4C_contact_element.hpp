/*-----------------------------------------------------------------------*/
/*! \file
\brief Contact element

\level 2

*/
/*-----------------------------------------------------------------------*/
#ifndef FOUR_C_CONTACT_ELEMENT_HPP
#define FOUR_C_CONTACT_ELEMENT_HPP

#include "4C_config.hpp"

#include "4C_mortar_element.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations

namespace CONTACT
{
  // forward declarations

  class ElementType : public Core::Elements::ElementType
  {
   public:
    std::string Name() const override { return "CONTACT::ElementType"; }

    static ElementType& Instance();

    Core::Communication::ParObject* Create(const std::vector<char>& data) override;

    Teuchos::RCP<Core::Elements::Element> Create(const int id, const int owner) override;

    void nodal_block_information(
        Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override;

    Core::LinAlg::SerialDenseMatrix ComputeNullSpace(
        Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override;

   private:
    static ElementType instance_;
  };

  /*!
   \brief A contact element


   */
  class Element : public Mortar::Element
  {
   public:
    //! @name Constructors and destructors and related methods

    /*!
     \brief Standard Constructor

     \param id    (in): A globally unique element id
     \param owner (in): owner processor of the element
     \param shape (in): shape of this element
     \param numnode (in): Number of nodes to this element
     \param nodeids (in): ids of nodes adjacent to this element
     \param isslave (in): flag indicating whether element is slave or master side
     \param isnurbs (in): flag indicating whether element is nurbs element or not
     */
    Element(int id, int owner, const Core::FE::CellType& shape, const int numnode,
        const int* nodeids, const bool isslave, bool isnurbs = false);

    /*!
     \brief Copy Constructor

     Makes a deep copy of this class

     */
    Element(const CONTACT::Element& old);



    /*!
     \brief Deep copy the derived class and return pointer to it

     */
    Core::Elements::Element* Clone() const override;

    /*!
     \brief Return unique ParObject id

     Every class implementing ParObject needs a unique id defined at the
     top of parobject.H

     */
    int UniqueParObjectId() const override { return ElementType::Instance().UniqueParObjectId(); }

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

    CONTACT::ElementType& ElementType() const override { return ElementType::Instance(); }

    //@}

    //! @name Query methods

    /*!
     \brief Get number of degrees of freedom of a certain node

     This Element is picky: It cooperates only with CNodes, not with
     standard Node objects!

     */
    int NumDofPerNode(const Core::Nodes::Node& node) const override;

    /*!
     \brief Print this element

     */
    void print(std::ostream& os) const override;

    //! @name Evaluation methods

    /*!
     \brief Evaluate an element

     An element derived from this class uses the Evaluate method to receive commands
     and parameters from some control routine in params and evaluates element matrices and
     vectors accoring to the command in params.

     \note This class implements a dummy of this method that prints a dserror and
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
     \brief Evaluate a Neumann boundary condition dummy

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
      return 0;
    }

    /*!
     \brief Build element normal derivative at node passed in
     */
    virtual void DerivNormalAtNode(int nid, int& i, Core::LinAlg::SerialDenseMatrix& elens,
        std::vector<Core::Gen::Pairedvector<int, double>>& derivn);

    virtual void OldUnitNormalAtXi(const double* xi, Core::LinAlg::Matrix<3, 1>& n_old,
        Core::LinAlg::Matrix<3, 2>& d_n_old_dxi);

    /*!
     \brief Evaluate derivative J,xi of Jacobian determinant
     */
    virtual void DJacDXi(
        double* djacdxi, double* xi, const Core::LinAlg::SerialDenseMatrix& secderiv);

    /*! \brief Evaluate derivative J,xi of Jacobian determinant
     *
     *  \author hiermeier \date 03/17 */
    template <unsigned elenumnode>
    inline void DJacDXi(double* djacdxi, double* xi, Core::LinAlg::Matrix<elenumnode, 3>& secderiv)
    {
      Core::LinAlg::SerialDenseMatrix sdm_secderiv(
          Teuchos::View, secderiv.data(), elenumnode, elenumnode, 3);
      DJacDXi(djacdxi, xi, sdm_secderiv);
    }

    /*!
     \brief Prepare D-Matrix deriv integration of contribution of one slave element
     */
    virtual void PrepareDderiv(const std::vector<Mortar::Element*>& meles);

    /*!
     \brief Prepare D-Matrix deriv integration of contribution of one slave element
     */
    virtual void PrepareMderiv(const std::vector<Mortar::Element*>& meles, const int m);

    /*!
     \brief Access to D-Matrix deriv to add Gauss point contribution
     */
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& GetDderiv()
    {
      if (d_matrix_deriv_ == Teuchos::null)
        FOUR_C_THROW("trying to get Dderiv, but not initialized");
      return *d_matrix_deriv_;
    }

    /*!
     \brief Access to M-Matrix deriv to add Gauss point contribution
     */
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& GetMderiv()
    {
      if (m_matrix_deriv_ == Teuchos::null)
        FOUR_C_THROW("trying to get Mderiv, but not initialized");
      return *m_matrix_deriv_;
    }

    /*!
     \brief Assemble D-Matrix deriv contribution of one slave element into the adjacent nodes
     */
    virtual void assemble_dderiv_to_nodes(bool dual);

    /*!
     \brief Assemble M-Matrix deriv contribution of one slave/master pair into the adjacent nodes
     */
    virtual void assemble_mderiv_to_nodes(Mortar::Element& mele);

    //@}
   private:
    /*!
     \brief Compute element normal derivative at local coordinate xi
     Caution: This function cannot be called stand-alone! It is
     integrated into the whole nodal normal calculation process.
     */
    virtual void deriv_normal_at_xi(double* xi, int& i, Core::LinAlg::SerialDenseMatrix& elens,
        std::vector<Core::Gen::Pairedvector<int, double>>& derivn);

   private:
    Teuchos::RCP<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>>
        d_matrix_deriv_;  //< temporary matrix for D linearization during integration
    Teuchos::RCP<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>>
        m_matrix_deriv_;  //< temporary matrix for M linearization during integration
  };
  // class Element
}  // namespace CONTACT

// << operator
std::ostream& operator<<(std::ostream& os, const CONTACT::Element& ele);

FOUR_C_NAMESPACE_CLOSE

#endif