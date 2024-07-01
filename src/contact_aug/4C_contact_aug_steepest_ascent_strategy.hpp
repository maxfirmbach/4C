/*---------------------------------------------------------------------*/
/*! \file
\brief Steepest ascent solution strategy based on the augmented contact
       formulation.

\level 3

*/
/*---------------------------------------------------------------------*/

#ifndef FOUR_C_CONTACT_AUG_STEEPEST_ASCENT_STRATEGY_HPP
#define FOUR_C_CONTACT_AUG_STEEPEST_ASCENT_STRATEGY_HPP

#include "4C_config.hpp"

#include "4C_contact_aug_steepest_ascent_sp_strategy.hpp"

FOUR_C_NAMESPACE_OPEN

namespace CONTACT
{
  namespace Aug
  {
    class LagrangeMultiplierFunction;
    class PenaltyUpdate;
    namespace SteepestAscent
    {
      // forward declarations
      class Interface;

      /*--------------------------------------------------------------------------*/
      /** \brief Condensed variant of the modified Newton approach.
       *
       * \author hiermeier \date 03/17 */
      class Strategy : public CONTACT::Aug::SteepestAscentSaddlePoint::Strategy
      {
        /** The combo_strategy is a wrapper class for a set of augmented Lagrangian
         *  strategies and needs access to all methods. */
        friend class CONTACT::Aug::ComboStrategy;

       public:
        /// constructor
        Strategy(const Teuchos::RCP<CONTACT::AbstractStratDataContainer>& data_ptr,
            const Epetra_Map* dof_row_map, const Epetra_Map* NodeRowMap,
            const Teuchos::ParameterList& params, const plain_interface_set& interfaces, int dim,
            const Teuchos::RCP<const Epetra_Comm>& comm, int maxdof);

        Inpar::CONTACT::SolvingStrategy Type() const override
        {
          return Inpar::CONTACT::solution_steepest_ascent;
        }

       protected:
        // un-do changes from the base class
        void evaluate_str_contact_rhs() override { Aug::Strategy::evaluate_str_contact_rhs(); }

        /// derived
        Teuchos::RCP<const Epetra_Vector> get_rhs_block_ptr_for_norm_check(
            const enum CONTACT::VecBlockType& bt) const override;

        /// derived
        void add_contributions_to_constr_rhs(Epetra_Vector& augConstrRhs) const override;

        /// derived
        Teuchos::RCP<Core::LinAlg::SparseMatrix> get_matrix_block_ptr(
            const enum CONTACT::MatBlockType& bt,
            const CONTACT::ParamsInterface* cparams = nullptr) const override;

        /// derived
        void add_contributions_to_matrix_block_displ_displ(Core::LinAlg::SparseMatrix& kdd,
            const CONTACT::ParamsInterface* cparams = nullptr) const override;

        /// derived
        void run_post_apply_jacobian_inverse(const CONTACT::ParamsInterface& cparams,
            const Epetra_Vector& rhs, Epetra_Vector& result, const Epetra_Vector& xold,
            const NOX::Nln::Group& grp) override;

        /// derived
        void remove_condensed_contributions_from_rhs(Epetra_Vector& str_rhs) const override;

       private:
        void augment_direction(const CONTACT::ParamsInterface& cparams, const Epetra_Vector& xold,
            Epetra_Vector& dir_mutable);

        Teuchos::RCP<Epetra_Vector> compute_active_lagrange_incr_in_normal_direction(
            const Epetra_Vector& displ_incr);

        Teuchos::RCP<Epetra_Vector> compute_inactive_lagrange_incr_in_normal_direction(
            const Epetra_Vector& displ_incr, const Epetra_Vector& zold);

        void post_augment_direction(
            const CONTACT::ParamsInterface& cparams, const Epetra_Vector& xold, Epetra_Vector& dir);
      };  //  class Strategy

    }  // namespace SteepestAscent
  }    // namespace Aug
}  // namespace CONTACT


FOUR_C_NAMESPACE_CLOSE

#endif