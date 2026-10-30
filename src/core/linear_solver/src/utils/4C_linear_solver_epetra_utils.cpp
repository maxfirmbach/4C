// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linear_solver_epetra_utils.hpp"

#include "4C_linalg_sparseoperator.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

Core::LinearSolver::EpetraOperatorAdapter::EpetraOperatorAdapter(
    std::shared_ptr<const Core::LinAlg::SparseOperator> sparse_operator)
    : sparse_operator_(std::move(sparse_operator))
{
  FOUR_C_ASSERT_ALWAYS(sparse_operator_ != nullptr, "Cannot adapt a null SparseOperator.");
}

int Core::LinearSolver::EpetraOperatorAdapter::SetUseTranspose(bool use_transpose)
{
  use_transpose_ = use_transpose;
  return 0;
}

int Core::LinearSolver::EpetraOperatorAdapter::Apply(
    const Epetra_MultiVector& x, Epetra_MultiVector& y) const
{
  sparse_operator_->multiply(use_transpose_, x_view_.sync(x), y_view_.sync(y));

  return 0;
}

int Core::LinearSolver::EpetraOperatorAdapter::ApplyInverse(
    const Epetra_MultiVector&, Epetra_MultiVector&) const
{
  FOUR_C_THROW("EpetraOperatorAdapter does not provide ApplyInverse().");
}

double Core::LinearSolver::EpetraOperatorAdapter::NormInf() const { return -1.0; }

const char* Core::LinearSolver::EpetraOperatorAdapter::Label() const
{
  return "Core::LinearSolver::EpetraOperatorAdapter";
}

bool Core::LinearSolver::EpetraOperatorAdapter::UseTranspose() const { return use_transpose_; }

bool Core::LinearSolver::EpetraOperatorAdapter::HasNormInf() const { return false; }

const Epetra_Comm& Core::LinearSolver::EpetraOperatorAdapter::Comm() const
{
  return OperatorDomainMap().Comm();
}

const Epetra_Map& Core::LinearSolver::EpetraOperatorAdapter::OperatorDomainMap() const
{
  return sparse_operator_->domain_map().get_epetra_map();
}

const Epetra_Map& Core::LinearSolver::EpetraOperatorAdapter::OperatorRangeMap() const
{
  return sparse_operator_->range_map().get_epetra_map();
}

FOUR_C_NAMESPACE_CLOSE
