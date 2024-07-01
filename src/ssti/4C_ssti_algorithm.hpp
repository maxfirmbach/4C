/*----------------------------------------------------------------------*/
/*! \file
 \brief base class for all scalar structure thermo algorithms

 \level 2


 *------------------------------------------------------------------------------------------------*/

#ifndef FOUR_C_SSTI_ALGORITHM_HPP
#define FOUR_C_SSTI_ALGORITHM_HPP

#include "4C_config.hpp"

#include "4C_adapter_algorithmbase.hpp"

#include <Epetra_Vector.h>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Adapter
{
  class Coupling;
  class ScaTraBaseAlgorithm;
  class SSIStructureWrapper;
  class Structure;
  class StructureBaseAlgorithmNew;
}  // namespace Adapter

namespace Inpar
{
  namespace SSTI
  {
    enum class SolutionScheme;
  }
}  // namespace Inpar

namespace Core::LinAlg
{
  class MultiMapExtractor;
}

namespace ScaTra
{
  class MeshtyingStrategyS2I;
  class ScaTraTimIntImpl;
}  // namespace ScaTra

namespace SSI
{
  namespace UTILS
  {
    class SSIMeshTying;
  }
}  // namespace SSI

namespace SSTI
{
  //! Base class of all solid-scatra algorithms
  class SSTIAlgorithm : public Adapter::AlgorithmBase
  {
   public:
    /// create using a Epetra_Comm
    explicit SSTIAlgorithm(const Epetra_Comm& comm, const Teuchos::ParameterList& globaltimeparams);

    //! Setup of algorithm
    //! Clone Discretizations, init and setup subproblems, setup coupling adapters at interfaces,
    //! setup submatrices for coupling between fields
    //@{
    virtual void init(const Epetra_Comm& comm, const Teuchos::ParameterList& sstitimeparams,
        const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& thermoparams,
        const Teuchos::ParameterList& structparams) = 0;
    virtual void setup();
    virtual void SetupSystem() = 0;
    //@}

    //! increment the counter for Newton-Raphson iterations (monolithic algorithm)
    void IncrementIter() { ++iter_; }

    //! return the counter for Newton-Raphson iterations (monolithic algorithm)
    unsigned int Iter() const { return iter_; }

    //! reset the counter for Newton-Raphson iterations (monolithic algorithm)
    void ResetIter() { iter_ = 0; }

    //! return coupling
    //@{
    Teuchos::RCP<const ScaTra::MeshtyingStrategyS2I> meshtying_scatra() const
    {
      return meshtying_strategy_scatra_;
    }
    Teuchos::RCP<const ScaTra::MeshtyingStrategyS2I> meshtying_thermo() const
    {
      return meshtying_strategy_thermo_;
    }
    Teuchos::RCP<const SSI::UTILS::SSIMeshTying> ssti_structure_mesh_tying() const
    {
      return ssti_structure_meshtying_;
    }
    //@}

    //! return subproblems
    //@{
    Teuchos::RCP<Adapter::SSIStructureWrapper> structure_field() const { return structure_; };
    Teuchos::RCP<ScaTra::ScaTraTimIntImpl> ScaTraField() const;
    Teuchos::RCP<ScaTra::ScaTraTimIntImpl> ThermoField() const;
    Teuchos::RCP<Adapter::ScaTraBaseAlgorithm> ScaTraFieldBase() { return scatra_; };
    Teuchos::RCP<Adapter::ScaTraBaseAlgorithm> ThermoFieldBase() { return thermo_; };
    //@}

    //! get bool indicating if we have at least one ssi interface meshtying condition
    bool interface_meshtying() const { return interfacemeshtying_; };

    //! read restart
    void read_restart(int restart) override;

    //! timeloop of coupled problem
    virtual void Timeloop() = 0;

    //! test results (if necessary)
    virtual void TestResults(const Epetra_Comm& comm) const;

   protected:
    //! clone scatra from structure and then thermo from scatra
    virtual void clone_discretizations(const Epetra_Comm& comm);

    //! copies modified time step from scatra to structure and to this SSI algorithm
    void distribute_dt_from_sca_tra();

    //! distribute states between subproblems
    //@{
    void distribute_solution_all_fields();
    void distribute_scatra_solution();
    void distribute_structure_solution();
    void distribute_thermo_solution();
    //@}

   private:
    //! counter for Newton-Raphson iterations (monolithic algorithm)
    unsigned int iter_;

    //! exchange materials between discretizations
    void assign_material_pointers();

    void check_is_init();

    //! clone thermo parameters from scatra parameters and adjust where needed
    Teuchos::ParameterList clone_thermo_params(
        const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& thermoparams);

    //! Pointers to subproblems
    //@{
    Teuchos::RCP<Adapter::ScaTraBaseAlgorithm> scatra_;
    Teuchos::RCP<Adapter::SSIStructureWrapper> structure_;
    Teuchos::RCP<Adapter::StructureBaseAlgorithmNew> struct_adapterbase_ptr_;
    Teuchos::RCP<Adapter::ScaTraBaseAlgorithm> thermo_;
    //@}

    //! Pointers to coupling strategies
    //@{
    Teuchos::RCP<const ScaTra::MeshtyingStrategyS2I> meshtying_strategy_scatra_;
    Teuchos::RCP<const ScaTra::MeshtyingStrategyS2I> meshtying_strategy_thermo_;
    Teuchos::RCP<SSI::UTILS::SSIMeshTying> ssti_structure_meshtying_;
    //@}

    //! bool indicating if we have at least one ssi interface meshtying condition
    const bool interfacemeshtying_;

    //! flag indicating if class is initialized
    bool isinit_;

    //! flag indicating if class is setup
    bool issetup_;
  };  // SSTI_Algorithm

  //! Construct specific SSTI algorithm
  Teuchos::RCP<SSTI::SSTIAlgorithm> BuildSSTI(Inpar::SSTI::SolutionScheme coupling,
      const Epetra_Comm& comm, const Teuchos::ParameterList& sstiparams);
}  // namespace SSTI
FOUR_C_NAMESPACE_CLOSE

#endif