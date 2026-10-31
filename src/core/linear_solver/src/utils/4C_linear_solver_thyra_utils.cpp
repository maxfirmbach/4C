// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linear_solver_thyra_utils.hpp"

#include <Teko_EpetraInverseOpWrapper.hpp>
#include <Thyra_EpetraLinearOp.hpp>
#include <Thyra_EpetraThyraWrappers.hpp>
#include <Thyra_PhysicallyBlockedLinearOpBase.hpp>

FOUR_C_NAMESPACE_OPEN

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
Teuchos::RCP<const Thyra::VectorSpaceBase<double>> Core::LinearSolver::Utils::create_thyra_map(
    const Core::LinAlg::Map& map)
{
  return Thyra::create_VectorSpace(Teuchos::rcpFromRef(map.get_epetra_map()));
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
Teuchos::RCP<const Thyra::MultiVectorBase<double>>
Core::LinearSolver::Utils::create_thyra_multi_vector(
    const Core::LinAlg::MultiVector<double>& multi_vector, const Core::LinAlg::Map& map)
{
  return Thyra::create_MultiVector(
      Teuchos::rcpFromRef(multi_vector.get_epetra_multi_vector()), create_thyra_map(map));
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
Teuchos::RCP<Thyra::MultiVectorBase<double>> Core::LinearSolver::Utils::create_thyra_multi_vector(
    Core::LinAlg::MultiVector<double>& multi_vector, const Core::LinAlg::Map& map)
{
  return Thyra::create_MultiVector(
      Teuchos::rcpFromRef(multi_vector.get_epetra_multi_vector()), create_thyra_map(map));
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
Teuchos::RCP<const Thyra::LinearOpBase<double>> Core::LinearSolver::Utils::create_thyra_linear_op(
    const Core::LinAlg::SparseOperator& matrix, Core::LinAlg::DataAccess access)
{
  if (const auto* sparse_matrix = dynamic_cast<const Core::LinAlg::SparseMatrix*>(&matrix))
  {
    return create_thyra_linear_op(*sparse_matrix, access);
  }

  if (const auto* block_matrix = dynamic_cast<const Core::LinAlg::BlockSparseMatrixBase*>(&matrix))
  {
    return create_thyra_linear_op(*block_matrix, access);
  }

  FOUR_C_THROW("Cannot create a Thyra operator from this SparseOperator implementation.");
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
Teuchos::RCP<const Thyra::LinearOpBase<double>> Core::LinearSolver::Utils::create_thyra_linear_op(
    const Core::LinAlg::SparseMatrix& matrix, Core::LinAlg::DataAccess access)
{
  Teuchos::RCP<const Epetra_CrsMatrix> A_crs;

  if (access == Core::LinAlg::DataAccess::Copy)
  {
    A_crs = Teuchos::make_rcp<Epetra_CrsMatrix>(matrix.epetra_matrix());
  }
  else
  {
    A_crs = Teuchos::rcpFromRef(matrix.epetra_matrix());
  }

  return Thyra::epetraLinearOp(A_crs);
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
Teuchos::RCP<const Thyra::LinearOpBase<double>> Core::LinearSolver::Utils::create_thyra_linear_op(
    const Core::LinAlg::BlockSparseMatrixBase& matrix, Core::LinAlg::DataAccess access)
{
  auto block_matrix = Thyra::defaultBlockedLinearOp<double>();

  block_matrix->beginBlockFill(matrix.rows(), matrix.cols());
  for (int row = 0; row < matrix.rows(); ++row)
  {
    for (int col = 0; col < matrix.cols(); ++col)
    {
      block_matrix->setBlock(row, col, create_thyra_linear_op(matrix(row, col), access));
    }
  }
  block_matrix->endBlockFill();

  return block_matrix;
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
std::shared_ptr<Epetra_Operator> Core::LinearSolver::Utils::get_epetra_inverse_operator_from_thyra(
    const Teuchos::RCP<const Thyra::LinearOpBase<double>> thyra_linear_op)
{
  return std::make_shared<Teko::Epetra::EpetraInverseOpWrapper>(thyra_linear_op);
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
Teuchos::RCP<Epetra_Vector> Core::LinearSolver::Utils::get_epetra_vector_from_thyra(
    const Core::LinAlg::Map& map, const Teuchos::RCP<::Thyra::VectorBase<double>>& thyra_vector)
{
  return ::Thyra::get_Epetra_Vector(map.get_epetra_map(), thyra_vector);
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
Teuchos::RCP<const Epetra_Vector> Core::LinearSolver::Utils::get_epetra_vector_from_thyra(
    const Core::LinAlg::Map& map,
    const Teuchos::RCP<const ::Thyra::VectorBase<double>>& thyra_vector)
{
  return ::Thyra::get_Epetra_Vector(map.get_epetra_map(), thyra_vector);
}

FOUR_C_NAMESPACE_CLOSE
