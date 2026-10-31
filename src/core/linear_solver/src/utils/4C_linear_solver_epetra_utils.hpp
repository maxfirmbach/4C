// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINEAR_SOLVER_EPETRA_UTILS_HPP
#define FOUR_C_LINEAR_SOLVER_EPETRA_UTILS_HPP

#include "4C_config.hpp"

#include "4C_linalg_multi_vector.hpp"
#include "4C_linalg_view.hpp"

#include <Epetra_Operator.h>

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class SparseOperator;
}

namespace Core::LinearSolver
{
  /**
   * \brief Expose a Core::LinAlg::SparseOperator through the Epetra_Operator interface.
   *
   * This is a temporary compatibility adapter for Epetra-based solver packages.
   * The wrapped SparseOperator retains ownership-independent linear algebra semantics;
   * Epetra-specific functionality is confined to the solver boundary.
   */
  class EpetraOperatorAdapter final : public Epetra_Operator
  {
   public:
    explicit EpetraOperatorAdapter(
        std::shared_ptr<const Core::LinAlg::SparseOperator> sparse_operator);

    int SetUseTranspose(bool use_transpose) override;

    int Apply(const Epetra_MultiVector& x, Epetra_MultiVector& y) const override;

    int ApplyInverse(const Epetra_MultiVector& x, Epetra_MultiVector& y) const override;

    double NormInf() const override;

    const char* Label() const override;

    bool UseTranspose() const override;

    bool HasNormInf() const override;

    const Epetra_Comm& Comm() const override;

    const Epetra_Map& OperatorDomainMap() const override;

    const Epetra_Map& OperatorRangeMap() const override;

   private:
    std::shared_ptr<const Core::LinAlg::SparseOperator> sparse_operator_;

    bool use_transpose_{false};

    mutable Core::LinAlg::View<const Core::LinAlg::MultiVector<double>> x_view_;
    mutable Core::LinAlg::View<Core::LinAlg::MultiVector<double>> y_view_;
  };
}  // namespace Core::LinearSolver

FOUR_C_NAMESPACE_CLOSE

#endif
