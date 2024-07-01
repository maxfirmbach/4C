/*---------------------------------------------------------------------*/
/*! \file

\brief Implementation of least squares by Sudhakar for Moment-fitting

\level 3


*----------------------------------------------------------------------*/

#include "4C_cut_least_squares.hpp"

#include <cmath>
#include <iostream>

FOUR_C_NAMESPACE_OPEN

// solve the rectangular system with linear least squares
Core::LinAlg::SerialDenseVector Core::Geo::Cut::LeastSquares::linear_least_square()
{
  Core::LinAlg::SerialDenseMatrix sqr(matri_[0].size(), matri_[0].size());
  Core::LinAlg::SerialDenseVector rhs(matri_[0].size());
  sqr = get_square_matrix(rhs);
  unknown_.size(matri_[0].size());

  using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
  using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
  Teuchos::SerialDenseSolver<ordinalType, scalarType> solve_for_GPweights;
  solve_for_GPweights.setMatrix(Teuchos::rcpFromRef(sqr));
  solve_for_GPweights.setVectors(Teuchos::rcpFromRef(unknown_), Teuchos::rcpFromRef(rhs));
  solve_for_GPweights.factorWithEquilibration(true);
  int err2 = solve_for_GPweights.factor();
  int err = solve_for_GPweights.solve();
  if ((err != 0) && (err2 != 0))
    FOUR_C_THROW(
        "Computation of Gauss weights failed, Ill"
        "conditioned matrix in least square");

  return unknown_;
}

// premultiplying the matrix with its transpose to get the square matrix
// the source terms also get multiplied
Core::LinAlg::SerialDenseMatrix Core::Geo::Cut::LeastSquares::get_square_matrix(
    Core::LinAlg::SerialDenseVector &rhs)
{
  Core::LinAlg::SerialDenseMatrix sqr(matri_[0].size(), matri_[0].size());

  for (unsigned i = 0; i < matri_[0].size(); i++)
  {
    for (unsigned j = 0; j < matri_[0].size(); j++)
    {
      sqr(j, i) = 0.0;

      // it is sqr(j,i) because the Epetra elements are column ordered first
      for (unsigned k = 0; k < matri_.size(); k++) sqr(j, i) += matri_[k][i] * matri_[k][j];
    }
  }

  for (unsigned i = 0; i < matri_[0].size(); i++)
  {
    rhs(i) = 0.0;
    for (unsigned j = 0; j < matri_.size(); j++) rhs(i) += matri_[j][i] * sourc_(j);
  }

  return sqr;
}

FOUR_C_NAMESPACE_CLOSE