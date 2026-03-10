// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linear_solver_preconditioner_teko.hpp"

#include "4C_comm_utils.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_linear_solver_thyra_utils.hpp"

#include <Stratimikos_DefaultLinearSolverBuilder.hpp>
#include <Stratimikos_MueLuHelpers.hpp>
#include <Teko_EpetraInverseOpWrapper.hpp>
#include <Teko_GaussSeidelPreconditionerFactory.hpp>
#include <Teko_InverseLibrary.hpp>
#include <Teko_JacobiPreconditionerFactory.hpp>
#include <Teko_LU2x2PreconditionerFactory.hpp>
#include <Teko_StratimikosFactory.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Thyra_EpetraOperatorWrapper.hpp>
#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_ThyraUtils.hpp>

#include <filesystem>

FOUR_C_NAMESPACE_OPEN

std::shared_ptr<Core::LinAlg::Vector<double>> kappa;

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
Core::LinearSolver::TekoPreconditioner::TekoPreconditioner(Teuchos::ParameterList& tekolist)
    : tekolist_(tekolist)
{
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Core::LinearSolver::TekoPreconditioner::setup(
    Core::LinAlg::SparseOperator& matrix, Core::LinAlg::MultiVector<double>& b)
{
  using EpetraMultiVector = Xpetra::EpetraMultiVectorT<GlobalOrdinal, Node>;
  using XpetraMultiVector = Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

  if (!tekolist_.sublist("Teko Parameters").isParameter("TEKO_XML_FILE"))
    FOUR_C_THROW("TEKO_XML_FILE parameter not set!");
  auto xmlFileName = tekolist_.sublist("Teko Parameters").get<std::string>("TEKO_XML_FILE");

  Teuchos::ParameterList tekoParams;
  auto comm = Core::Communication::to_teuchos_comm<int>(matrix.get_comm());
  Teuchos::updateParametersFromXmlFileAndBroadcast(xmlFileName, Teuchos::Ptr(&tekoParams), *comm);

  auto A = std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(
      Core::Utils::shared_ptr_from_ref(matrix));

  // TODO: Uncomment to get 3x3 block system
  auto maps = tekolist_.sublist("Teko Parameters")
                  .get<std::vector<std::shared_ptr<const Core::LinAlg::Map>>>("reorder: maps");
  auto dof_constraint_map = std::make_shared<Core::LinAlg::Map>(A->matrix(1, 1).domain_map());
  maps.emplace_back(dof_constraint_map);

  // TODO: Due to global ID fuck up, we shoot ourselves big omega time yikes pogchamp kekw
  const std::shared_ptr<Core::LinAlg::MultiMapExtractor> extractor(
      new Core::LinAlg::MultiMapExtractor(A->full_domain_map(), maps));
  auto Asparse = A->merge();
  Asparse->complete();

  A = Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
      *Asparse, *extractor, *extractor);
  A->complete();

  A = std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(A);
  A->complete();


  // TODO: Uncomment to get 2x2 system
  /*
  if (!A)
  {
    auto maps = tekolist_.sublist("Teko Parameters")
                    .get<std::vector<std::shared_ptr<const Core::LinAlg::Map>>>("reorder: maps");
    Core::LinAlg::MultiMapExtractor extractor;
    std::shared_ptr<Core::LinAlg::SparseMatrix> A_sparse;

    // If we have a sparse matrix at hand and given reorder maps, we try to split the linear
    // operator into a block matrix. If it is a block matrix, we first merge and split afterwards.
    if (!A)
    {
      A_sparse = std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(
          Core::Utils::shared_ptr_from_ref(matrix));
    }
    else
    {
      A_sparse = A->merge();
    }

    extractor = Core::LinAlg::MultiMapExtractor(A_sparse->row_map(), maps);
    A = Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
        *A_sparse, extractor, extractor);
    A->complete();
  }
  */


  // wrap linear operators
  if (!A)
  {
    auto A_crs = Teuchos::rcp_dynamic_cast<Core::LinAlg::SparseMatrix>(Teuchos::rcpFromRef(matrix));
    pmatrix_ = Teuchos::rcp_const_cast<Thyra::LinearOpBase<double>>(
        Utils::create_thyra_linear_op(*A_crs, LinAlg::DataAccess::Copy));
  }
  else
  {
    pmatrix_ = Teuchos::rcp_const_cast<Thyra::LinearOpBase<double>>(
        Utils::create_thyra_linear_op(*A, LinAlg::DataAccess::Copy));

    // check if multigrid is used as preconditioner for single field inverse approximation and
    // attach nullspace and coordinate information to the respective inverse parameter list.
    for (int block = 0; block < A->rows(); block++)
    {
      std::string inverse = "Inverse" + std::to_string(block + 1);

      if (tekolist_.isSublist(inverse))
      {
        // get the single field preconditioner sub-list of a matrix block hardwired under
        // "Inverse<1...n>".
        Teuchos::ParameterList& inverseList = tekolist_.sublist(inverse);

        if (tekoParams.sublist("Inverse Factory Library")
                .sublist(inverse)
                .get<std::string>("Type") == "MueLu")
        {
          // const int number_of_equations = inverseList.get<int>("PDE equations");

          auto nullspace_vector =
              inverseList.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("nullspace");

          // Check for boundary conditions
          const bool has_dirichlet =
              Core::LinAlg::has_dirichlet_boundary_condition(A->matrix(block, block));

          // If we have no Dirichlet rows, we need to do rank 1 correction, we have a pure Neumann
          // problem!
          if (!has_dirichlet)
          {
            if (Communication::my_mpi_rank(A->get_comm()) == 0)
              std::cout << "We have a Neumann problem at hand (maybe). If you want to factorize "
                           "this matrix either do a rank correction or go home. :)"
                        << std::endl;

            auto body_indices =
                inverseList.get<std::shared_ptr<Core::LinAlg::Vector<int>>>("bodyid");

            Core::LinAlg::MultiVector<double> blocked_nullspace(A->matrix(block, block).row_map(),
                nullspace_vector->num_vectors() * (body_indices->max_value() + 1), true);

            // Partition nullspace over individual fibers
            for (int row = 0; row < A->matrix(block, block).num_my_rows(); row++)
            {
              // Check to which bodyID this row belongs, this will determine the block column in
              // blocked_nullspace
              const int bodyID = body_indices->get_local_values()[row];
              const int block_col = bodyID * nullspace_vector->num_vectors();

              std::vector<int> indices;
              std::vector<double> values;

              for (int nsdim = 0; nsdim < nullspace_vector->num_vectors(); nsdim++)
              {
                const double value =
                    nullspace_vector->get_vector(nsdim).local_values_as_span()[row];

                values.push_back(value);
                indices.push_back(block_col + nsdim);
              }

              for (size_t index = 0; index < indices.size(); index++)
                blocked_nullspace.get_vector(indices[index]).local_values_as_span()[row] =
                    values[index];
            }

            // TODO: For now we skip orthonormalization
            auto projected_A = Core::LinAlg::matrix_rank_correction(A->matrix(block, block),
                blocked_nullspace, {.alpha = 1e-6, .orthonormalize = true});

            // Put the matrix into the thyra operator
            Teuchos::rcp_dynamic_cast<Thyra::PhysicallyBlockedLinearOpBase<double>>(pmatrix_)
                ->setBlock(block, block,
                    Utils::create_thyra_linear_op(*projected_A, LinAlg::DataAccess::Copy));
          }

          Teuchos::RCP<XpetraMultiVector> nullspace = Teuchos::make_rcp<EpetraMultiVector>(
              Teuchos::rcpFromRef(nullspace_vector->get_epetra_multi_vector()));

          Teuchos::RCP<XpetraMultiVector> coordinates =
              Teuchos::make_rcp<EpetraMultiVector>(Teuchos::rcpFromRef(inverseList
                      .get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("Coordinates")
                      ->get_epetra_multi_vector()));

          /*
          tekoParams.sublist("Inverse Factory Library")
              .sublist(inverse)
              .set("number of equations", number_of_equations);  // number_of_equations;
          */
          Teuchos::ParameterList& userParamList =
              tekoParams.sublist("Inverse Factory Library").sublist(inverse).sublist("user data");
          userParamList.set("Nullspace", nullspace);
          userParamList.set("Coordinates", coordinates);
        }
      }
    }
  }

  // setup preconditioner builder and enable relevant packages
  Stratimikos::LinearSolverBuilder<double> builder;

  // enable block preconditioning and multigrid
  Stratimikos::enableMueLu<Scalar, LocalOrdinal, GlobalOrdinal, Node>(builder);
  Teko::addTekoToStratimikosBuilder(builder);

  // add special in-house block preconditioning methods
  Teuchos::RCP<Teko::Cloneable> lu2x2_strategy =
      Teuchos::make_rcp<Teko::AutoClone<LU2x2SpaiStrategy>>();
  Teko::LU2x2PreconditionerFactory::addStrategy("Spai Strategy", lu2x2_strategy);

  // TODO: Here we could add something to the Jacobi one ...
  Teuchos::RCP<Teko::Cloneable> block_diagonal_strategy =
      Teuchos::make_rcp<Teko::AutoClone<InvFactoryDiagSpaiStrategy>>();
  Teko::JacobiPreconditionerFactory::addStrategy("Spai Strategy", block_diagonal_strategy);
  Teko::GaussSeidelPreconditionerFactory::addStrategy("Spai Strategy", block_diagonal_strategy);

  // get preconditioner parameter list
  Teuchos::RCP<Teuchos::ParameterList> stratimikos_params =
      Teuchos::make_rcp<Teuchos::ParameterList>(*builder.getValidParameters());
  Teuchos::ParameterList& tekoList =
      stratimikos_params->sublist("Preconditioner Types").sublist("Teko");
  tekoList.setParameters(tekoParams);
  builder.setParameterList(stratimikos_params);

  // construct preconditioning operator
  Teuchos::RCP<Thyra::PreconditionerFactoryBase<double>> precFactory =
      builder.createPreconditioningStrategy("Teko");
  Teuchos::RCP<Thyra::PreconditionerBase<double>> prec =
      Thyra::prec<double>(*precFactory, pmatrix_);
  Teko::LinearOp inverseOp = prec->getUnspecifiedPrecOp();

  p_ = std::make_shared<Teko::Epetra::EpetraInverseOpWrapper>(inverseOp);
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
const Teko::LinearOp Core::LinearSolver::LU2x2SpaiStrategy::getHatInvA00(
    const Teko::BlockedLinearOp& A, Teko::BlockPreconditionerState& state) const
{
  initialize_state(A, state);

  return state.getModifiableOp("invA00");
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
const Teko::LinearOp Core::LinearSolver::LU2x2SpaiStrategy::getTildeInvA00(
    const Teko::BlockedLinearOp& A, Teko::BlockPreconditionerState& state) const
{
  initialize_state(A, state);

  return state.getModifiableOp("invA00");
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
const Teko::LinearOp Core::LinearSolver::LU2x2SpaiStrategy::getInvS(
    const Teko::BlockedLinearOp& A, Teko::BlockPreconditionerState& state) const
{
  initialize_state(A, state);

  return state.getModifiableOp("invS");
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Core::LinearSolver::LU2x2SpaiStrategy::initialize_state(
    const Teko::BlockedLinearOp& A, Teko::BlockPreconditionerState& state) const
{
  if (state.isInitialized()) return;

  Teko::LinearOp F = Teko::getBlock(0, 0, A);
  Teko::LinearOp Bt = Teko::getBlock(0, 1, A);
  Teko::LinearOp B = Teko::getBlock(1, 0, A);
  Teko::LinearOp C = Teko::getBlock(1, 1, A);

  // build the Schur complement
  Teko::ModifiableLinearOp& S = state.getModifiableOp("S");
  {
    auto A_op = Teuchos::rcp_dynamic_cast<const Thyra::EpetraLinearOp>(F);
    auto A_crs = Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(A_op->epetra_op(), true);
    const Core::LinAlg::SparseMatrix A_sparse(
        Core::Utils::shared_ptr_from_ref(*Teuchos::rcp_const_cast<Epetra_CrsMatrix>(A_crs)),
        Core::LinAlg::DataAccess::Copy);

    // sparse inverse calculation
    std::shared_ptr<Core::LinAlg::SparseMatrix> A_thresh =
        Core::LinAlg::threshold_matrix(A_sparse, drop_tol_);
    std::shared_ptr<Core::LinAlg::Graph> sparsity_pattern_enriched =
        Core::LinAlg::enrich_matrix_graph(*A_thresh, fill_level_);
    std::shared_ptr<Core::LinAlg::SparseMatrix> A_inverse =
        Core::LinAlg::matrix_sparse_inverse(A_sparse, sparsity_pattern_enriched);
    A_thresh = Core::LinAlg::threshold_matrix(*A_inverse, drop_tol_);
    Teko::LinearOp H = Thyra::epetraLinearOp(Teuchos::rcpFromRef(A_thresh->epetra_matrix()));

    // build Schur-complement
    Teko::LinearOp HBt;
    Teko::ModifiableLinearOp& mHBt = state.getModifiableOp("HBt");
    Teko::ModifiableLinearOp& mhatS = state.getModifiableOp("hatS");
    Teko::ModifiableLinearOp& BHBt = state.getModifiableOp("BHBt");

    // build H*Bt
    mHBt = Teko::explicitMultiply(H, Bt, mHBt);
    HBt = mHBt;

    // build B*H*Bt
    BHBt = Teko::explicitMultiply(B, HBt, BHBt);

    // build C-B*H*Bt
    mhatS = Teko::explicitAdd(C, Teko::scale(-1.0, BHBt), mhatS);
    S = mhatS;
  }

  // build inverse S
  {
    Teko::ModifiableLinearOp& invS = state.getModifiableOp("invS");
    if (invS == Teuchos::null)
      invS = buildInverse(*inv_factory_s_, S);
    else
      rebuildInverse(*inv_factory_s_, S, invS);
  }

  // build inverse A00
  {
    Teko::ModifiableLinearOp& invA00 = state.getModifiableOp("invA00");
    if (invA00 == Teuchos::null)
      invA00 = buildInverse(*inv_factory_f_, F);
    else
      rebuildInverse(*inv_factory_f_, F, invA00);
  }

  state.setInitialized(true);
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Core::LinearSolver::LU2x2SpaiStrategy::initializeFromParameterList(
    const Teuchos::ParameterList& lulist, const Teko::InverseLibrary& invLib)
{
  std::string invStr = "", invA00Str = "", invSStr = "";

  // "parse" the parameter list
  if (lulist.isParameter("Inverse Type")) invStr = lulist.get<std::string>("Inverse Type");
  if (lulist.isParameter("Inverse A00 Type"))
    invA00Str = lulist.get<std::string>("Inverse A00 Type");
  if (lulist.isParameter("Inverse Schur Type"))
    invSStr = lulist.get<std::string>("Inverse Schur Type");

  // Spai parameters
  if (lulist.isParameter("Drop tolerance"))
  {
    drop_tol_ = lulist.get<double>("Drop tolerance");
  }

  if (lulist.isParameter("Fill-in level"))
  {
    fill_level_ = lulist.get<int>("Fill-in level");
  }

  // set defaults as needed
  if (invA00Str == "") invA00Str = invStr;
  if (invSStr == "") invSStr = invStr;

  inv_factory_f_ = invLib.getInverseFactory(invA00Str);

  if (invA00Str == invSStr)
    inv_factory_s_ = inv_factory_f_;
  else
    inv_factory_s_ = invLib.getInverseFactory(invSStr);
}


//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
// TODO: Add block diagonal strategy?



//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
Core::LinearSolver::InvFactoryDiagSpaiStrategy::InvFactoryDiagSpaiStrategy(
    const Teuchos::RCP<Teko::InverseFactory>& factory)
{
  // only one factory to use!
  invDiagFact_.resize(1, factory);
  defaultInvFact_ = factory;
}

Core::LinearSolver::InvFactoryDiagSpaiStrategy::InvFactoryDiagSpaiStrategy(
    const std::vector<Teuchos::RCP<Teko::InverseFactory>>& factories,
    const Teuchos::RCP<Teko::InverseFactory>& defaultFact)
{
  invDiagFact_ = factories;

  if (defaultFact == Teuchos::null)
    defaultInvFact_ = invDiagFact_[0];
  else
    defaultInvFact_ = defaultFact;
}

Core::LinearSolver::InvFactoryDiagSpaiStrategy::InvFactoryDiagSpaiStrategy(
    const std::vector<Teuchos::RCP<Teko::InverseFactory>>& inverseFactories,
    const std::vector<Teuchos::RCP<Teko::InverseFactory>>& preconditionerFactories,
    const Teuchos::RCP<Teko::InverseFactory>& defaultInverseFact,
    const Teuchos::RCP<Teko::InverseFactory>& defaultPreconditionerFact)
{
  invDiagFact_ = inverseFactories;
  precDiagFact_ = preconditionerFactories;

  if (defaultInverseFact == Teuchos::null)
    defaultInvFact_ = invDiagFact_[0];
  else
    defaultInvFact_ = defaultInverseFact;
  defaultPrecFact_ = defaultPreconditionerFact;
}

void Core::LinearSolver::InvFactoryDiagSpaiStrategy::initialize(
    const std::vector<Teuchos::RCP<Teko::InverseFactory>>& inverseFactories,
    const std::vector<Teuchos::RCP<Teko::InverseFactory>>& preconditionerFactories,
    const Teuchos::RCP<Teko::InverseFactory>& defaultInverseFact,
    const Teuchos::RCP<Teko::InverseFactory>& defaultPreconditionerFact)
{
  invDiagFact_ = inverseFactories;
  precDiagFact_ = preconditionerFactories;

  if (defaultInverseFact == Teuchos::null)
    defaultInvFact_ = invDiagFact_[0];
  else
    defaultInvFact_ = defaultInverseFact;
  defaultPrecFact_ = defaultPreconditionerFact;
}

/** returns an (approximate) inverse of the diagonal blocks of A
 * where A is closely related to the original source for invD0 and invD1
 * with the zero block being approximated by the respective Schur complement
 */
void Core::LinearSolver::InvFactoryDiagSpaiStrategy::getInvD(const Teko::BlockedLinearOp& A,
    Teko::BlockPreconditionerState& state, std::vector<Teko::LinearOp>& invDiag) const
{
  Teko_DEBUG_SCOPE("InvFactoryDiagSchurStrategy::getInvD", 10);

  // loop over diagonals, build an inverse operator for each
  size_t diagCnt = A->productRange()->numBlocks();

  const std::string opPrefix = "BlockDiagOp";
  for (size_t i = 0; i < diagCnt; i++)
  {
    auto precFact = ((i < precDiagFact_.size()) && (!precDiagFact_[i].is_null()))
                        ? precDiagFact_[i]
                        : defaultPrecFact_;
    auto invFact = (i < invDiagFact_.size()) ? invDiagFact_[i] : defaultInvFact_;

    // TODO: for 3x3 block system only!!!
    if (i == 2)
    {
      // 1. get the Schur complement contribution from the augmentation \epsilon\inv{W}
      /*
      auto A22 = Teko::getBlock(2, 2, A);
      auto A22_op = Teuchos::rcp_dynamic_cast<const Thyra::EpetraLinearOp>(A22);
      auto A22_crs = Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(A22_op->epetra_op(), true);

      auto scaling_matrix = std::make_shared<Core::LinAlg::SparseMatrix>(*kappa);
      scaling_matrix->complete();

      Teko::LinearOp schur_penalty = Thyra::epetraLinearOp(
          Teuchos::make_rcp<Epetra_CrsMatrix>(scaling_matrix->epetra_matrix()));

      auto schur_scaled_1 = Teko::explicitScale(-1.0, schur_penalty);
      */

      // 2. get the Schur complement contribution from the solid part (without augmentation?)
      auto A20 = Teko::getBlock(2, 0, A);
      auto A00 = Teko::getBlock(0, 0, A);
      auto A02 = Teko::getBlock(0, 2, A);

      auto diagonalType00 = Teko::getDiagonalType("Diagonal");
      auto invA00 = getInvDiagonalOp(A00, diagonalType00);

      auto triple00 = Teko::explicitMultiply(A20, Teko::explicitMultiply(invA00, A02));


      // 3. get the Schur complement contributino from the beam part (without augmentation?)
      auto A21 = Teko::getBlock(2, 1, A);
      auto A11 = Teko::getBlock(1, 1, A);
      auto A12 = Teko::getBlock(1, 2, A);

      // sparse inverse calculation
      double drop_tol = 1e-12;
      int fill_level = 64;

      auto A_op = Teuchos::rcp_dynamic_cast<const Thyra::EpetraLinearOp>(A11);
      auto A_crs = Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(A_op->epetra_op(), true);

      const Core::LinAlg::SparseMatrix A_sparse(
          Core::Utils::shared_ptr_from_ref(*Teuchos::rcp_const_cast<Epetra_CrsMatrix>(A_crs)),
          Core::LinAlg::DataAccess::Copy);

      std::shared_ptr<Core::LinAlg::SparseMatrix> A_thresh =
          Core::LinAlg::threshold_matrix(A_sparse, drop_tol);
      std::shared_ptr<Core::LinAlg::Graph> sparsity_pattern_enriched =
          Core::LinAlg::enrich_matrix_graph(*A_thresh, fill_level);
      std::shared_ptr<Core::LinAlg::SparseMatrix> A_inverse =
          Core::LinAlg::matrix_sparse_inverse(A_sparse, sparsity_pattern_enriched);
      A_thresh = Core::LinAlg::threshold_matrix(*A_inverse, drop_tol);

      auto invA11 =
          Thyra::epetraLinearOp(Teuchos::make_rcp<Epetra_CrsMatrix>(A_thresh->epetra_matrix()));

      auto triple11 = Teko::explicitMultiply(A21, Teko::explicitMultiply(invA11, A12));
      auto schur = Teko::explicitAdd(triple00, triple11);
      auto schur_scaled_2 = Teko::explicitScale(-1.0, schur);

      // TODO: Get A(2,2) if possible
      auto A22 = Teko::getBlock(2, 2, A);
      auto complete_schur = Teko::explicitAdd(A22, schur_scaled_2);

      // 4. Get Schur complement
      // auto inverse_1 = schur_scaled_1;
      auto inverse_2 = buildInverse(*invFact, precFact, complete_schur, state, opPrefix, i);

      // auto inverse_operator = Teko::add(inverse_1, inverse_2);

      // TODO: Check which inverse is used in here!
      invDiag.push_back(inverse_2);
    }
    // TODO: For 2x2 system only!
    /*
    if (i == 1)
    {
      auto scaling_matrix = std::make_shared<Core::LinAlg::SparseMatrix>(*kappa);
      scaling_matrix->complete();

      auto schur_penalty = Thyra::epetraLinearOp(
          Teuchos::make_rcp<Epetra_CrsMatrix>(scaling_matrix->epetra_matrix()));

      auto schur_scaled = Teko::explicitScale(-1.0, schur_penalty);

      invDiag.push_back(buildInverse(*invFact, precFact, schur_scaled, state, opPrefix, i));
    }
    */
    else
    {
      auto block = Teko::getBlock(i, i, A);
      invDiag.push_back(buildInverse(*invFact, precFact, block, state, opPrefix, i));
    }
  }
}

Teko::LinearOp Core::LinearSolver::InvFactoryDiagSpaiStrategy::buildInverse(
    const Teko::InverseFactory& invFact, Teuchos::RCP<Teko::InverseFactory>& precFact,
    const Teko::LinearOp& matrix, Teko::BlockPreconditionerState& state,
    const std::string& opPrefix, int i) const
{
  std::stringstream ss;
  ss << opPrefix << "_" << i;

  Teko::ModifiableLinearOp& invOp = state.getModifiableOp(ss.str());
  Teko::ModifiableLinearOp& precOp = state.getModifiableOp("prec_" + ss.str());

  if (precFact != Teuchos::null)
  {
    if (precOp == Teuchos::null)
    {
      precOp = precFact->buildInverse(matrix);
      state.addModifiableOp("prec_" + ss.str(), precOp);
    }
    else
    {
      Teko::rebuildInverse(*precFact, matrix, precOp);
    }
  }

  if (invOp == Teuchos::null)
    if (precOp.is_null())
      invOp = Teko::buildInverse(invFact, matrix);
    else
      invOp = Teko::buildInverse(invFact, matrix, precOp);
  else
  {
    if (precOp.is_null())
      Teko::rebuildInverse(invFact, matrix, invOp);
    else
      Teko::rebuildInverse(invFact, matrix, precOp, invOp);
  }

  return invOp;
}

FOUR_C_NAMESPACE_CLOSE
