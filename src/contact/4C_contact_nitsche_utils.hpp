/*---------------------------------------------------------------------*/
/*! \file
\brief Some helpers for nitsche contact

\level 3


*/
/*---------------------------------------------------------------------*/
#ifndef FOUR_C_CONTACT_NITSCHE_UTILS_HPP
#define FOUR_C_CONTACT_NITSCHE_UTILS_HPP

#include "4C_config.hpp"

#include "4C_contact_utils.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_mortar_element.hpp"

#include <Epetra_CrsMatrix.h>
#include <Epetra_FEVector.h>
#include <Teuchos_RCP.hpp>

#include <unordered_map>

FOUR_C_NAMESPACE_OPEN

namespace Mortar
{
  class ElementNitscheContainer
  {
   public:
    /**
     * Virtual destructor.
     */
    virtual ~ElementNitscheContainer() = default;

    ElementNitscheContainer() = default;

    virtual void clear() = 0;

    virtual void AssembleRHS(Mortar::Element* mele, CONTACT::VecBlockType row,
        Teuchos::RCP<Epetra_FEVector> fc) const = 0;

    virtual void AssembleMatrix(Mortar::Element* mele, CONTACT::MatBlockType block,
        Teuchos::RCP<Core::LinAlg::SparseMatrix> kc) const = 0;

    virtual double* Rhs(int dof) = 0;
    virtual double* Rhs() = 0;
    virtual double* K(int col) = 0;
    virtual double* K(int col, int dof) = 0;

    virtual double* RhsT(int dof) = 0;
    virtual double* RhsT() = 0;
    virtual double* Ktt(int col) = 0;
    virtual double* Ktd(int col) = 0;
    virtual double* Kdt(int col) = 0;

    virtual double* RhsP(int dof) = 0;
    virtual double* Kpp(int col) = 0;
    virtual double* Kpd(int col) = 0;
    virtual double* Kdp(int col) = 0;

    virtual double* RhsS(int dof) = 0;
    virtual double* Kss(int col) = 0;
    virtual double* Ksd(int col) = 0;
    virtual double* Kds(int col) = 0;

    virtual double* RhsE(int dof) = 0;
    virtual double* Kee(int col) = 0;
    virtual double* Ked(int col) = 0;
    virtual double* Ked(int col, int dof) = 0;
    virtual double* Kde(int col) = 0;
  };

  template <Core::FE::CellType parent_distype>
  class ElementNitscheDataTsi
  {
   public:
    void clear()
    {
      rhs_t_.clear();
      k_tt_.clear();
      k_td_.clear();
      k_dt_.clear();
    }

    static constexpr int num_parent_disp_dof =
        Core::FE::num_nodes<parent_distype> * Core::FE::dim<parent_distype>;
    static constexpr int num_parent_thermo_dof = Core::FE::num_nodes<parent_distype>;

    Core::LinAlg::Matrix<num_parent_thermo_dof, 1> rhs_t_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_thermo_dof, 1>> k_tt_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_thermo_dof, 1>> k_td_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_disp_dof, 1>> k_dt_;
  };

  template <Core::FE::CellType parent_distype>
  class ElementNitscheDataPoro
  {
   public:
    void clear()
    {
      rhs_p_.clear();
      k_pp_.clear();
      k_pd_.clear();
      k_dp_.clear();
    }

    static constexpr int num_parent_disp_dof =
        Core::FE::num_nodes<parent_distype> * Core::FE::dim<parent_distype>;
    static constexpr int num_parent_pf_dof =
        Core::FE::num_nodes<parent_distype> * (Core::FE::dim<parent_distype> + 1);

    Core::LinAlg::Matrix<num_parent_pf_dof, 1> rhs_p_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_pf_dof, 1>> k_pp_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_pf_dof, 1>> k_pd_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_disp_dof, 1>> k_dp_;
  };

  template <Core::FE::CellType parent_distype>
  class ElementNitscheDataSsi
  {
   public:
    void clear()
    {
      rhs_s_.clear();
      k_ss_.clear();
      k_sd_.clear();
      k_ds_.clear();
    }

    static constexpr int num_parent_disp_dof =
        Core::FE::num_nodes<parent_distype> * Core::FE::dim<parent_distype>;
    static constexpr int num_parent_scatra_dof = Core::FE::num_nodes<parent_distype>;

    Core::LinAlg::Matrix<num_parent_scatra_dof, 1> rhs_s_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_scatra_dof, 1>> k_ss_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_scatra_dof, 1>> k_sd_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_disp_dof, 1>> k_ds_;
  };

  template <Core::FE::CellType parent_distype>
  class ElementNitscheDataSsiElch
  {
   public:
    void clear()
    {
      rhs_e_.clear();
      k_ee_.clear();
      k_ed_.clear();
      k_de_.clear();
    }

    static constexpr int num_parent_disp_dof =
        Core::FE::num_nodes<parent_distype> * Core::FE::dim<parent_distype>;
    static constexpr int num_parent_elch_dof = Core::FE::num_nodes<parent_distype> * 2;

    Core::LinAlg::Matrix<num_parent_elch_dof, 1> rhs_e_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_elch_dof, 1>> k_ee_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_elch_dof, 1>> k_ed_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_disp_dof, 1>> k_de_;
  };

  template <Core::FE::CellType parent_distype>
  class ElementNitscheData : public ElementNitscheContainer
  {
    using VectorType =
        Core::LinAlg::Matrix<Core::FE::num_nodes<parent_distype> * Core::FE::dim<parent_distype>,
            1>;

   public:
    const VectorType& RhsVec() { return rhs_; }
    double* Rhs(int dof) override { return &rhs_(dof); }
    double* Rhs() override { return rhs_.data(); }
    double* K(int col) override { return k_[col].data(); }
    double* K(int col, int dof) override { return &k_[col](dof); }

    double* RhsT(int dof) override { return &tsi_data_.rhs_t_(dof); }
    double* RhsT() override { return tsi_data_.rhs_t_.data(); }
    double* Ktt(int col) override { return tsi_data_.k_tt_[col].data(); }
    double* Ktd(int col) override { return tsi_data_.k_td_[col].data(); }
    double* Kdt(int col) override { return tsi_data_.k_dt_[col].data(); }

    double* RhsP(int dof) override { return &poro_data_.rhs_p_(dof); }
    double* Kpp(int col) override { return poro_data_.k_pp_[col].data(); }
    double* Kpd(int col) override { return poro_data_.k_pd_[col].data(); }
    double* Kdp(int col) override { return poro_data_.k_dp_[col].data(); }

    double* RhsS(int dof) override { return &ssi_data_.rhs_s_(dof); }
    double* Kss(int col) override { return ssi_data_.k_ss_[col].data(); }
    double* Ksd(int col) override { return ssi_data_.k_sd_[col].data(); }
    double* Kds(int col) override { return ssi_data_.k_ds_[col].data(); }

    double* RhsE(int dof) override { return &ssi_elch_data_.rhs_e_(dof); }
    double* Kee(int col) override { return ssi_elch_data_.k_ee_[col].data(); }
    double* Ked(int col) override { return ssi_elch_data_.k_ed_[col].data(); }
    double* Ked(int col, int dof) override { return &ssi_elch_data_.k_ed_[col](dof); }
    double* Kde(int col) override { return ssi_elch_data_.k_de_[col].data(); }

    void AssembleRHS(Mortar::Element* mele, CONTACT::VecBlockType row,
        Teuchos::RCP<Epetra_FEVector> fc) const override;

    void AssembleMatrix(Mortar::Element* mele, CONTACT::MatBlockType block,
        Teuchos::RCP<Core::LinAlg::SparseMatrix> kc) const override;

    template <int num_dof_per_node>
    void AssembleRHS(Mortar::Element* mele,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<parent_distype> * num_dof_per_node, 1>& rhs,
        std::vector<int>& dofs, Teuchos::RCP<Epetra_FEVector> fc) const;

    template <int num_dof_per_node>
    void AssembleMatrix(Mortar::Element* mele,
        const std::unordered_map<int,
            Core::LinAlg::Matrix<Core::FE::num_nodes<parent_distype> * num_dof_per_node, 1>>& k,
        std::vector<int>& dofs, Teuchos::RCP<Core::LinAlg::SparseMatrix> kc) const;


    void clear() override
    {
      rhs_.clear();
      k_.clear();
      tsi_data_.clear();
      poro_data_.clear();
      ssi_data_.clear();
      ssi_elch_data_.clear();
    }

   private:
    VectorType rhs_;
    std::unordered_map<int, VectorType> k_;
    Mortar::ElementNitscheDataTsi<parent_distype> tsi_data_;
    Mortar::ElementNitscheDataPoro<parent_distype> poro_data_;
    Mortar::ElementNitscheDataSsi<parent_distype> ssi_data_;
    Mortar::ElementNitscheDataSsiElch<parent_distype> ssi_elch_data_;
  };

}  // namespace Mortar

FOUR_C_NAMESPACE_CLOSE

#endif