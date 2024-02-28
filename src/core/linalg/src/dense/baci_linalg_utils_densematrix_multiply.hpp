/*----------------------------------------------------------------------*/
/*! \file

\brief A collection of multiplication methods for namespace LINALG

\level 0
*/
/*----------------------------------------------------------------------*/

#ifndef BACI_LINALG_UTILS_DENSEMATRIX_MULTIPLY_HPP
#define BACI_LINALG_UTILS_DENSEMATRIX_MULTIPLY_HPP

#include "baci_config.hpp"

#include "baci_linalg_serialdensematrix.hpp"
#include "baci_linalg_serialdensevector.hpp"
#include "baci_utils_exceptions.hpp"

BACI_NAMESPACE_OPEN

namespace CORE::LINALG::INTERNAL
{
  /*!
   \brief Utility function to get type string of a matrix/vector object
   */
  template <typename VectorOrMatrix>
  inline std::string GetMatrixOrVectorString();

  template <>
  inline std::string GetMatrixOrVectorString<CORE::LINALG::SerialDenseMatrix::Base>()
  {
    return "Matrix";
  }

  template <>
  inline std::string GetMatrixOrVectorString<CORE::LINALG::SerialDenseVector::Base>()
  {
    return "Vector";
  }

  /*!
   \brief Utility function to get the correct case string of a matrix/vector object
   */
  template <typename VectorOrMatrix>
  inline std::string GetMatrixOrVectorCase(char ch);

  template <>
  inline std::string GetMatrixOrVectorCase<CORE::LINALG::SerialDenseMatrix::Base>(char ch)
  {
    char c = std::toupper(ch);
    std::string s;
    s = c;
    return s;
  }

  template <>
  inline std::string GetMatrixOrVectorCase<CORE::LINALG::SerialDenseVector::Base>(char ch)
  {
    char c = std::tolower(ch);
    std::string s;
    s = c;
    return s;
  }

  /*!
   \brief Utility function to get type string of transposition operation
   */
  template <bool Transpose>
  inline std::string GetTransposeString();

  template <>
  inline std::string GetTransposeString<false>()
  {
    return "";
  }

  template <>
  inline std::string GetTransposeString<true>()
  {
    return "^T";
  }

  /*!
   \brief Utility function to check for error code of LINALG multiplication
   */
  template <bool Transpose1, bool Transpose2, typename VectorOrMatrix1, typename VectorOrMatrix2>
  inline void CheckErrorCodeInDebug(
      int errorCode, const VectorOrMatrix1& a, const VectorOrMatrix2& b, const VectorOrMatrix2& c)
  {
    dsassert(errorCode == 0,
        std::string(
            "Error code (" + std::to_string(errorCode) +
            ") is returned. Something goes wrong with the " +
            GetMatrixOrVectorString<VectorOrMatrix1>() + "-" +
            GetMatrixOrVectorString<VectorOrMatrix2>() + " multiplication " +
            GetMatrixOrVectorCase<VectorOrMatrix2>('c') + " = " +
            GetMatrixOrVectorCase<VectorOrMatrix1>('a') + GetTransposeString<Transpose1>() + " * " +
            GetMatrixOrVectorCase<VectorOrMatrix2>('b') + GetTransposeString<Transpose2>() +
            ". Dimensions of " + GetMatrixOrVectorCase<VectorOrMatrix1>('a') + ", " +
            GetMatrixOrVectorCase<VectorOrMatrix2>('b') + " and " +
            GetMatrixOrVectorCase<VectorOrMatrix2>('c') + " are (" + std::to_string(a.numRows()) +
            "x" + std::to_string(a.numCols()) + "), (" + std::to_string(b.numRows()) + "x" +
            std::to_string(b.numCols()) + ") and (" + std::to_string(c.numRows()) + "x" +
            std::to_string(c.numCols()) + ") respectively.")
            .c_str());
  }
}  // namespace CORE::LINALG::INTERNAL

namespace CORE::LINALG
{
  /*!
   \brief Matrix-vector multiplication c = A*b

   \param A (in):        Matrix A
   \param b (in):        Vector b
   \param c (out):       Vector c
   */
  inline int multiply(SerialDenseVector::Base& c, const SerialDenseMatrix::Base& A,
      const SerialDenseVector::Base& b)
  {
    const int err = c.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, A, b, 0.0);
    INTERNAL::CheckErrorCodeInDebug<false, false>(err, A, b, c);
    return err;
  }

  /*!
   \brief Matrix-vector multiplication c = alpha*A*b + beta*c

   \param A (in):        Matrix A
   \param b (in):        Vector b
   \param c (out):       Vector c
   */
  inline int multiply(double beta, SerialDenseVector::Base& c, double alpha,
      const SerialDenseMatrix::Base& A, const SerialDenseVector::Base& b)
  {
    const int err = c.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, alpha, A, b, beta);
    INTERNAL::CheckErrorCodeInDebug<false, false>(err, A, b, c);
    return err;
  }

  /*!
   \brief Matrix-vector multiplication c = A^T*b

   \param A (in):        Matrix A
   \param b (in):        Vector b
   \param c (out):       Vector c
   */
  inline int multiplyTN(SerialDenseVector::Base& c, const SerialDenseMatrix::Base& A,
      const SerialDenseVector::Base& b)
  {
    const int err = c.multiply(Teuchos::TRANS, Teuchos::NO_TRANS, 1.0, A, b, 0.0);
    INTERNAL::CheckErrorCodeInDebug<true, false>(err, A, b, c);
    return err;
  }

  /*!
   \brief Matrix-vector multiplication c = alpha*A^T*b + beta*c

   \param A (in):        Matrix A
   \param b (in):        Vector b
   \param c (out):       Vector c
   */
  inline int multiplyTN(double beta, SerialDenseVector::Base& c, double alpha,
      const SerialDenseMatrix::Base& A, const SerialDenseVector::Base& b)
  {
    const int err = c.multiply(Teuchos::TRANS, Teuchos::NO_TRANS, alpha, A, b, beta);
    INTERNAL::CheckErrorCodeInDebug<true, false>(err, A, b, c);
    return err;
  }

  /*!
   \brief Matrix-matrix multiplication C = A*B

   \param A (in):        Matrix A
   \param b (in):        Matrix B
   \param c (out):       Matrix C
   */
  inline int multiply(SerialDenseMatrix::Base& C, const SerialDenseMatrix::Base& A,
      const SerialDenseMatrix::Base& B)
  {
    const int err = C.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, A, B, 0.0);
    INTERNAL::CheckErrorCodeInDebug<false, false>(err, A, B, C);
    return err;
  }

  /*!
   \brief Matrix-matrix multiplication C = alpha*A*B + beta*C

   \param A (in):        Matrix A
   \param b (in):        Matrix B
   \param c (out):       Matrix C
   */
  inline int multiply(double beta, SerialDenseMatrix::Base& C, double alpha,
      const SerialDenseMatrix::Base& A, const SerialDenseMatrix::Base& B)
  {
    const int err = C.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, alpha, A, B, beta);
    INTERNAL::CheckErrorCodeInDebug<false, false>(err, A, B, C);
    return err;
  }

  /*!
   \brief Matrix-matrix multiplication C = A^T*B

   \param A (in):        Matrix A
   \param b (in):        Matrix B
   \param c (out):       Matrix C
   */
  inline int multiplyTN(SerialDenseMatrix::Base& C, const SerialDenseMatrix::Base& A,
      const SerialDenseMatrix::Base& B)
  {
    const int err = C.multiply(Teuchos::TRANS, Teuchos::NO_TRANS, 1.0, A, B, 0.0);
    INTERNAL::CheckErrorCodeInDebug<true, false>(err, A, B, C);
    return err;
  }

  /*!
   \brief Matrix-matrix multiplication C = alpha*A^T*B + beta*C

   \param A (in):        Matrix A
   \param b (in):        Matrix B
   \param c (out):       Matrix C
   */
  inline int multiplyTN(double beta, SerialDenseMatrix::Base& C, double alpha,
      const SerialDenseMatrix::Base& A, const SerialDenseMatrix::Base& B)
  {
    const int err = C.multiply(Teuchos::TRANS, Teuchos::NO_TRANS, alpha, A, B, beta);
    INTERNAL::CheckErrorCodeInDebug<true, false>(err, A, B, C);
    return err;
  }

  /*!
   \brief Matrix-matrix multiplication C = A*B^T

   \param A (in):        Matrix A
   \param b (in):        Matrix B
   \param c (out):       Matrix C
   */
  inline int multiplyNT(SerialDenseMatrix::Base& C, const SerialDenseMatrix::Base& A,
      const SerialDenseMatrix::Base& B)
  {
    const int err = C.multiply(Teuchos::NO_TRANS, Teuchos::TRANS, 1.0, A, B, 0.0);
    INTERNAL::CheckErrorCodeInDebug<false, true>(err, A, B, C);
    return err;
  }

  /*!
   \brief Matrix-matrix multiplication C = alpha*A*B^T + beta*C

   \param A (in):        Matrix A
   \param b (in):        Matrix B
   \param c (out):       Matrix C
   */
  inline int multiplyNT(double beta, SerialDenseMatrix::Base& C, double alpha,
      const SerialDenseMatrix::Base& A, const SerialDenseMatrix::Base& B)
  {
    const int err = C.multiply(Teuchos::NO_TRANS, Teuchos::TRANS, alpha, A, B, beta);
    INTERNAL::CheckErrorCodeInDebug<false, true>(err, A, B, C);
    return err;
  }

  /*!
   \brief Matrix-matrix multiplication C = A^T*B^T

   \param A (in):        Matrix A
   \param b (in):        Matrix B
   \param c (out):       Matrix C
   */
  inline int multiplyTT(SerialDenseMatrix::Base& C, const SerialDenseMatrix::Base& A,
      const SerialDenseMatrix::Base& B)
  {
    const int err = C.multiply(Teuchos::TRANS, Teuchos::TRANS, 1.0, A, B, 0.0);
    INTERNAL::CheckErrorCodeInDebug<true, true>(err, A, B, C);
    return err;
  }

  /*!
   \brief Matrix-matrix multiplication C = alpha*A^T*B^T + beta*C

   \param A (in):        Matrix A
   \param b (in):        Matrix B
   \param c (out):       Matrix C
   */
  inline int multiplyTT(double beta, SerialDenseMatrix::Base& C, double alpha,
      const SerialDenseMatrix::Base& A, const SerialDenseMatrix::Base& B)
  {
    const int err = C.multiply(Teuchos::TRANS, Teuchos::TRANS, alpha, A, B, beta);
    INTERNAL::CheckErrorCodeInDebug<true, true>(err, A, B, C);
    return err;
  }
}  // namespace CORE::LINALG

BACI_NAMESPACE_CLOSE

#endif  // BACI_LINALG_UTILS_DENSEMATRIX_MULTIPLY_H