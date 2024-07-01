/*----------------------------------------------------------------------*/
/*! \file

 \brief General framework for monolithic fpsi solution schemes

\level 3

 */

/*----------------------------------------------------------------------*/

#ifndef FOUR_C_FPSI_MONOLITHIC_HPP
#define FOUR_C_FPSI_MONOLITHIC_HPP

// FPSI includes
#include "4C_config.hpp"

#include "4C_adapter_ale_fpsi.hpp"
#include "4C_adapter_fld_fluid_fpsi.hpp"
#include "4C_ale.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_fpsi.hpp"
#include "4C_fpsi_coupling.hpp"
#include "4C_fsi_monolithic.hpp"
#include "4C_inpar_fpsi.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_poroelast_base.hpp"

#include <Teuchos_Time.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | forward declarations                                                  |
 *----------------------------------------------------------------------*/
namespace Core::LinAlg
{
  class SparseMatrix;
  class MapExtractor;
}  // namespace Core::LinAlg

namespace PoroElast
{
  class Monolithic;
}

namespace Adapter
{
  class Coupling;
  class AleFpsiWrapper;
}  // namespace Adapter

namespace FSI
{
  namespace UTILS
  {
    class DebugWriter;
    class MonolithicDebugWriter;
    class MatrixColTransform;
  }  // namespace UTILS
}  // namespace FSI
/*----------------------------------------------------------------------*/

namespace FPSI
{
  class MonolithicBase : public FpsiBase
  {
   public:
    //! ctor
    explicit MonolithicBase(const Epetra_Comm& comm, const Teuchos::ParameterList& fpsidynparams,
        const Teuchos::ParameterList& poroelastdynparams);


    //! read restart data
    void read_restart(int step) override;

    //! start a new time step
    void prepare_time_step() override;

    //! take current results for converged and save for next time step
    void update() override;

    //! calculate stresses, strains, energies
    virtual void prepare_output(bool force_prepare);

    //! Output routine accounting for Lagrange multiplier at the interface
    void output() override;

    //! @name access sub-fields
    const Teuchos::RCP<PoroElast::Monolithic>& poro_field() { return poroelast_subproblem_; };
    const Teuchos::RCP<Adapter::FluidFPSI>& fluid_field() { return fluid_subproblem_; };
    const Teuchos::RCP<Adapter::AleFpsiWrapper>& ale_field() { return ale_; };

    //@}

    Teuchos::RCP<std::map<int, int>> Fluid_PoroFluid_InterfaceMap;
    Teuchos::RCP<std::map<int, int>> PoroFluid_Fluid_InterfaceMap;

   protected:
    //! underlying poroelast problem
    Teuchos::RCP<PoroElast::Monolithic> poroelast_subproblem_;
    //! underlying fluid of the FSI problem
    Teuchos::RCP<Adapter::FluidFPSI> fluid_subproblem_;
    //! underlying ale of the FSI problem
    Teuchos::RCP<Adapter::AleFpsiWrapper> ale_;

    //! flag defines if FSI Interface exists for this problem
    bool FSI_Interface_exists_;

   public:
    //! FPSI coupling object (does the interface evaluations)
    Teuchos::RCP<FPSI::FpsiCoupling>& FPSICoupl() { return fpsicoupl_; }

    //! @name Access General Couplings
    Core::Adapter::Coupling& fluid_ale_coupling() { return *coupfa_; }

    const Core::Adapter::Coupling& fluid_ale_coupling() const { return *coupfa_; }

    // Couplings for FSI
    Core::Adapter::Coupling& structure_fluid_coupling_fsi() { return *coupsf_fsi_; }
    Core::Adapter::Coupling& structure_ale_coupling_fsi() { return *coupsa_fsi_; }
    Core::Adapter::Coupling& interface_fluid_ale_coupling_fsi() { return *icoupfa_fsi_; }

    const Core::Adapter::Coupling& structure_fluid_coupling_fsi() const { return *coupsf_fsi_; }
    const Core::Adapter::Coupling& structure_ale_coupling_fsi() const { return *coupsa_fsi_; }
    const Core::Adapter::Coupling& interface_fluid_ale_coupling_fsi() const
    {
      return *icoupfa_fsi_;
    }

    //@}

   protected:
    //! @name Transfer helpers
    virtual Teuchos::RCP<Epetra_Vector> fluid_to_ale(Teuchos::RCP<const Epetra_Vector> iv) const;
    virtual Teuchos::RCP<Epetra_Vector> ale_to_fluid(Teuchos::RCP<const Epetra_Vector> iv) const;

    virtual Teuchos::RCP<Epetra_Vector> struct_to_fluid_fsi(
        Teuchos::RCP<const Epetra_Vector> iv) const;
    virtual Teuchos::RCP<Epetra_Vector> fluid_to_struct_fsi(
        Teuchos::RCP<const Epetra_Vector> iv) const;
    virtual Teuchos::RCP<Epetra_Vector> struct_to_ale_fsi(
        Teuchos::RCP<const Epetra_Vector> iv) const;
    virtual Teuchos::RCP<Epetra_Vector> ale_to_struct_fsi(
        Teuchos::RCP<const Epetra_Vector> iv) const;
    virtual Teuchos::RCP<Epetra_Vector> fluid_to_ale_fsi(
        Teuchos::RCP<const Epetra_Vector> iv) const;
    virtual Teuchos::RCP<Epetra_Vector> ale_to_fluid_fsi(
        Teuchos::RCP<const Epetra_Vector> iv) const;
    virtual Teuchos::RCP<Epetra_Vector> ale_to_fluid_interface_fsi(
        Teuchos::RCP<const Epetra_Vector> iv) const;

    //@}

   private:
    //! FPSI - COUPLING
    //! coupling of fluid and ale in the entire fluid volume
    Teuchos::RCP<Core::Adapter::Coupling> coupfa_;

    //! FSI - COUPLING
    //! coupling of structure and fluid at the interface
    Teuchos::RCP<Core::Adapter::Coupling> coupsf_fsi_;
    //! coupling of structure and ale at the interface
    Teuchos::RCP<Core::Adapter::Coupling> coupsa_fsi_;
    //! coupling of fluid and ale in the entire fluid volume
    Teuchos::RCP<Core::Adapter::Coupling> coupfa_fsi_;
    //! coupling of all interface fluid and ale dofs
    Teuchos::RCP<Core::Adapter::Coupling> icoupfa_fsi_;
    //! coupling of FPSI+FSI interface overlapping dofs of structure and freefluid
    Teuchos::RCP<Core::Adapter::Coupling> iffcoupsf_fsi_;

    //! FPSI Coupling Object
    Teuchos::RCP<FPSI::FpsiCoupling> fpsicoupl_;
  };
  // MonolithicBase

  //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  //<<<<<<<<<<<<<<<<<<<<<<  MonolithicBase -> Monolithic  >>>>>>>>>>>>>>>>>>>>>
  //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  class Monolithic : public MonolithicBase
  {
    friend class FSI::UTILS::MonolithicDebugWriter;

   public:
    //! ctor
    explicit Monolithic(const Epetra_Comm& comm, const Teuchos::ParameterList& fpsidynparams,
        const Teuchos::ParameterList& poroelastdynparams);

    //! setup fpsi system
    void SetupSystem() override;

    //! setup fsi part of the system
    virtual void SetupSystem_FSI();

    //! perform time loop
    void Timeloop() override;

    //! prepare time loop
    void PrepareTimeloop();

    //! solve one time step
    virtual void TimeStep();

    //! perform result test
    void TestResults(const Epetra_Comm& comm) override;

    //! build RHS vector from sub fields
    virtual void setup_rhs(bool firstcall = false) = 0;

    //! build system matrix form sub fields + coupling
    virtual void setup_system_matrix(Core::LinAlg::BlockSparseMatrixBase& mat) = 0;
    //! build system matrix form sub fields + coupling
    virtual void setup_system_matrix() { setup_system_matrix(*SystemMatrix()); }

    //! access system matrix
    virtual Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> SystemMatrix() const = 0;

    /// setup solver
    void SetupSolver() override;

    /// Recover the Lagrange multiplier at the interface   mayr.mt (03/2012)
    virtual void recover_lagrange_multiplier()
    {
      FOUR_C_THROW("recover_lagrange_multiplier: Not Implemented in Base Class!");
    }

    /// Extract specific columns from Sparse Matrix
    void extract_columnsfrom_sparse(Teuchos::RCP<Epetra_CrsMatrix> src,  ///< source Matrix
        const Teuchos::RCP<const Epetra_Map>&
            colmap,  ///< map with column gids to be extracted! (gid which are not in the source
                     ///< Matrix will be ignored!)
        Teuchos::RCP<Epetra_CrsMatrix> dst);  ///< destination Matrix (will be filled!)

    //! Evaluate all fields at x^n+1 with x^n+1 = x_n + stepinc
    virtual void evaluate(
        Teuchos::RCP<const Epetra_Vector> stepinc);  ///< increment between time step n and n+1

    //! setup of newton scheme
    void SetupNewton();

    //! finite difference check for fpsi systemmatrix
    void FPSIFDCheck();

    //! solve linear system
    void linear_solve();

    //! solve using line search method
    void LineSearch(Teuchos::RCP<Core::LinAlg::SparseMatrix>& sparse);

    //! create linear solver (setup of parameter lists, etc...)
    void create_linear_solver();

    //! build convergence norms after solve
    void build_convergence_norms();

    //! print header and results of newton iteration to screen
    void print_newton_iter();

    //! print header of newton iteration
    void print_newton_iter_header(FILE* ofile);

    //! print results of newton iteration
    void print_newton_iter_text(FILE* ofile);

    //! perform convergence check
    bool Converged();

    //! full monolithic dof row map
    Teuchos::RCP<const Epetra_Map> dof_row_map() const { return blockrowdofmap_.FullMap(); }

    //! map of all dofs on Dirichlet-Boundary
    virtual Teuchos::RCP<Epetra_Map> combined_dbc_map();

    //! extractor to communicate between full monolithic map and block maps
    const Core::LinAlg::MultiMapExtractor& Extractor() const { return blockrowdofmap_; }

    //! set conductivity (for fps3i)
    void SetConductivity(double conduct);

    //! external acces to rhs vector (used by xfpsi)
    Teuchos::RCP<Epetra_Vector>& RHS() { return rhs_; }  // TodoAge: will be removed again!

   protected:
    //! block systemmatrix
    Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> systemmatrix_;
    //! dof row map splitted in (field) blocks
    Core::LinAlg::MultiMapExtractor blockrowdofmap_;
    //! dof row map (not splitted)
    Teuchos::RCP<Epetra_Map> fullmap_;
    //! increment between Newton steps k and k+1
    Teuchos::RCP<Epetra_Vector> iterinc_;
    Teuchos::RCP<Epetra_Vector> iterincold_;
    //! zero vector of full length
    Teuchos::RCP<Epetra_Vector> zeros_;
    //! linear algebraic solver
    Teuchos::RCP<Core::LinAlg::Solver> solver_;
    //! rhs of FPSI system
    Teuchos::RCP<Epetra_Vector> rhs_;
    Teuchos::RCP<Epetra_Vector> rhsold_;

    Teuchos::RCP<const Epetra_Vector> meshdispold_;

    Teuchos::RCP<Epetra_Vector> porointerfacedisplacementsold_;

    //! adapt solver tolerancePoroField()->SystemSparseMatrix()
    bool solveradapttol_;
    int linesearch_;
    double linesearch_counter;
    double solveradaptolbetter_;

    bool active_FD_check_;  // indicates if evaluate() is called from fd_check (firstiter should not
                            // be added anymore!!!)

    /// extract the three field vectors from a given composed vector
    /*
     \param x  (i) composed vector that contains all field vectors
     \param sx (o) poroelast dofs
     \param fx (o) free fluid velocities and pressure
     \param ax (o) ale displacements
     \param firstiter_ (i) firstiteration? - how to evaluate FSI-velocities
     */
    virtual void extract_field_vectors(Teuchos::RCP<const Epetra_Vector> x,
        Teuchos::RCP<const Epetra_Vector>& sx, Teuchos::RCP<const Epetra_Vector>& pfx,
        Teuchos::RCP<const Epetra_Vector>& fx, Teuchos::RCP<const Epetra_Vector>& ax,
        bool firstiter_) = 0;

    /// setup list with default parameters
    void set_default_parameters(const Teuchos::ParameterList& fpsidynparams);

    //! block ids of the monolithic system
    int porofluid_block_;
    int structure_block_;
    int fluid_block_;
    int ale_i_block_;

   private:
    //! flag for direct solver of linear system
    bool directsolve_;

    enum Inpar::FPSI::ConvergenceNorm normtypeinc_;
    enum Inpar::FPSI::ConvergenceNorm normtypefres_;
    enum Inpar::FPSI::BinaryOp combinedconvergence_;

    double toleranceiterinc_;
    double toleranceresidualforces_;
    std::vector<double>
        toleranceresidualforceslist_;  // order of fields: porofluidvelocity, porofluidpressure,
                                       // porostructure, fluidvelocity, fluidpressure, ale
    std::vector<double>
        toleranceiterinclist_;  // order of fields: porofluidvelocity, porofluidpressure,
                                // porostructure, fluidvelocity, fluidpressure, ale

    int maximumiterations_;
    int minimumiterations_;
    double normofrhs_;
    double normofrhsold_;
    double normofiterinc_;
    double normofiterincold_;

    double normrhsfluidvelocity_;
    double normrhsfluidpressure_;
    double normrhsporofluidvelocity_;
    double normrhsporofluidpressure_;
    double normrhsporointerface_;
    double normrhsfluidinterface_;
    double normrhsporostruct_;
    double normrhsfluid_;
    double normrhsale_;

    double normofiterincporostruct_;
    double normofiterincporofluid_;
    double normofiterincfluid_;
    double normofiterincporofluidvelocity_;
    double normofiterincporofluidpressure_;
    double normofiterincfluidvelocity_;
    double normofiterincfluidpressure_;
    double normofiterincale_;
    double normofiterincfluidinterface_;
    double normofiterincporointerface_;

    double sqrtnall_;  //!< squareroot of lenght of all dofs
    double sqrtnfv_;   //!< squareroot of length of fluid velocity dofs
    double sqrtnfp_;   //!< squareroot of length of fluid pressure dofs
    double sqrtnpfv_;  //!< squareroot of length of porofluid velocity dofs
    double sqrtnpfp_;  //!< squareroot of length of porofluid pressure dofs
    double sqrtnps_;   //!< squareroot of length of porostruct dofs
    double sqrtna_;    //!< squareroot of length of ale dofs

    double norm1_alldof_;  //!< sum of absolute values of all dofs
    double norm1_fv_;      //!< sum of absolute fluid velocity values
    double norm1_fp_;      //!< sum of absolute fluid pressure values
    double norm1_pfv_;     //!< sum of absolute poro fluid velocity values
    double norm1_pfp_;     //!< sum of absolute poro fluid pressure values
    double norm1_ps_;      //!< sum of absolute poro structural displacements values
    double norm1_a_;       //!< sum of absolute ale displacements values

    //! iteration step
    int iter_;

    int printscreen_;  ///> print infos to standard out every printscreen_ steps
    bool printiter_;   ///> print intermediate iterations during solution
    //! timer for solution technique
    Teuchos::Time timer_;

    bool isfirsttimestep_;

    //! hydraulic conductivity (needed for coupling in case of probtype fps3i)
    double conductivity_;

   protected:
    bool islinesearch_;
    //!  flag is true if this is the first Newton iteration, false otherwise
    bool firstcall_;
  };
  // class Monolithic

}  // namespace FPSI

FOUR_C_NAMESPACE_CLOSE

#endif