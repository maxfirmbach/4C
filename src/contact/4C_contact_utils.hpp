/*-----------------------------------------------------------------------*/
/*! \file
\brief Contains a summary of contact utility functions


\level 2

*/
/*----------------------------------------------------------------------------*/

#ifndef FOUR_C_CONTACT_UTILS_HPP
#define FOUR_C_CONTACT_UTILS_HPP

#include "4C_config.hpp"

#include "4C_fem_condition.hpp"

#include <Teuchos_RCP.hpp>

#include <set>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Nodes
{
  class Node;
}

namespace Core::Elements
{
  class Element;
}

namespace Core::LinAlg
{
  class SerialDenseMatrix;
}  // namespace Core::LinAlg

namespace CONTACT
{
  /// enum, which specifies the desired matrix block for the different models
  enum class MatBlockType
  {
    displ_displ,          ///< Kdd block (structural block)
    displ_lm,             ///< Kdz block (of the corresponding model evaluator)
    lm_displ,             ///< Kzd block (of the corresponding model evaluator)
    lm_lm,                ///< Kzz block (of the corresponding model evaluator)
    temp_temp,            ///< Ktt block (thermal block)
    temp_displ,           ///< Ktd block (structure-thermo-coupling)
    displ_temp,           ///< Kdt block (thermo-structure-coupling)
    porofluid_porofluid,  ///< Kpp block (porofluid-porofluid)
    porofluid_displ,      ///< Kpd block (porofluid-structure)
    displ_porofluid,      ///< Kdp block (structure-porofluid)
    scatra_scatra,        ///< Kss block (scatra-scatra)
    scatra_displ,         ///< Ksd block (scatra-structure)
    displ_scatra,         ///< Kds block (structure-scatra)
    elch_elch,            ///< Kee block (elch-elch)
    elch_displ,           ///< Ked block (elch-structure)
    displ_elch            ///< Kde block (structure-elch)
  };

  //! enum, which specifies the desired vector blocks for the different models
  enum class VecBlockType
  {
    displ,       ///< displacement block (structural block)
    constraint,  ///< lagrange multiplier/constraint block of the corresponding model
    temp,        ///< temperature block (thermal block)
    porofluid,   ///< porofluid block (porofluid block)
    scatra,      ///< scalar transport block (scatra block)
    elch         ///< electrochemistry block (elch block)
  };

  std::string VecBlockTypeToStr(const VecBlockType bt);

  namespace UTILS
  {
    /// Get the solid to solid contact conditions
    int GetContactConditions(std::vector<Core::Conditions::Condition*>& contact_conditions,
        const std::vector<Core::Conditions::Condition*>& beamandsolidcontactconditions,
        const bool& throw_error = true);

    /// Find the solid to solid contact conditions and combine them to contact condition groups
    int GetContactConditionGroups(
        std::vector<std::vector<Core::Conditions::Condition*>>& ccond_grps,
        const Core::FE::Discretization& discret_wrapper, const bool& throw_error = true);

    /// Combine the solid to solid contact conditions to contact condition groups
    void GetContactConditionGroups(
        std::vector<std::vector<Core::Conditions::Condition*>>& ccond_grps,
        const std::vector<Core::Conditions::Condition*>& cconds);

    /// Gather information which side is master and which side is slave
    void GetMasterSlaveSideInfo(std::vector<bool>& isslave, std::vector<bool>& isself,
        const std::vector<Core::Conditions::Condition*>& cond_grp);

    /**
     * \brief Gather information on initialization (Active/Inactive)
     *
     * \param [in,out]  Two_half_pass: two half pass approach applied for current condition group
     * \param [in,out]  Check_nonsmooth_selfcontactsurface: reference configuration check for
     *                  non-smooth self contact shall be performed for current condition group
     * \param [in,out]  Searchele_AllProc: Search elements on all processors
     * \param [in,out]  isactive:  condition is set active
     * \param [in]      isslave:   condition is defined as slave side
     * \param [in]      isself:    condition is self contact condition
     * \param [in]      cond_grp: current contact condition group (i.e. conditions with same ID)
     *
     * \author cschmidt \date 11/18 */
    void GetInitializationInfo(bool& Two_half_pass, bool& Check_nonsmooth_selfcontactsurface,
        bool& Searchele_AllProc, std::vector<bool>& isactive, std::vector<bool>& isslave,
        std::vector<bool>& isself, const std::vector<Core::Conditions::Condition*>& cond_grp);

    /// write conservation data to an output file
    void WriteConservationDataToFile(const int mypid, const int interface_id, const int nln_iter,
        const Core::LinAlg::SerialDenseMatrix& conservation_data, const std::string& ofile_path,
        const std::string& prefix);

    /** \brief Detect DBC slave nodes and elements
     *
     *  Check all slave contact conditions. If the optional condition tag
     *  "RemoveDBCSlaveNodes" can be found in the slave condition line,
     *  all slave nodes and adjacent elements are added to the corresponding
     *  sets.
     *
     *  A possible condition line can look like
     *  E 7 - 1 Slave Inactive FrCoeffOrBound 0.0 AdhesionBound 0.0 Solidcontact RemoveDBCSlaveNodes
     *
     *  \author hiermeier \date 01/18 */
    class DbcHandler
    {
     public:
      /// remove constructor and destructor
      DbcHandler() = delete;
      ~DbcHandler() = delete;

      /** \brief Detect all slave nodes and elements which hold Dbc information
       *
       *  \param(in)  str_discret: structural discretization
       *  \param(in)  ccond_grps:  contact condition groups
       *  \param(out) dbc_slave_nodes: set containing all slave nodes which hold
       *                               DBC information
       *  \param(out) dbc_slave_eles: set containing all slave elements which
       *                              contain at least one DBC slave node
       *
       *  \author hiermeier \date 01/18 */
      static void detect_dbc_slave_nodes_and_elements(const Core::FE::Discretization& str_discret,
          const std::vector<std::vector<Core::Conditions::Condition*>>& ccond_grps,
          std::set<const Core::Nodes::Node*>& dbc_slave_nodes,
          std::set<const Core::Elements::Element*>& dbc_slave_eles);

     private:
      static void detect_dbc_slave_nodes(
          std::map<const Core::Nodes::Node*, int>& dbc_slave_node_map,
          const Core::FE::Discretization& str_discret,
          const std::vector<const Core::Conditions::Condition*>& sl_conds);

      static void detect_dbc_slave_elements(
          std::set<const Core::Elements::Element*>& dbc_slave_eles,
          const std::map<const Core::Nodes::Node*, int>& dbc_slave_nodes,
          const std::vector<const Core::Conditions::Condition*>& sl_conds);
    };  // class DbcHandler

  }  // namespace UTILS
}  // namespace CONTACT


FOUR_C_NAMESPACE_CLOSE

#endif