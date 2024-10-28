// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cardiovascular0d_mor_pod.hpp"

#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_utils_string.hpp"

#include <iostream>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Cardiovascular0D::ProperOrthogonalDecomposition::ProperOrthogonalDecomposition(
    Teuchos::RCP<const Epetra_Map> full_model_dof_row_map, const std::string& pod_matrix_file_name,
    const std::string& absolute_path_to_input_file)
    : full_model_dof_row_map_(full_model_dof_row_map)
{
  // check if model order reduction is given in input file
  {
    std::vector<std::string> components_of_absolute_path =
        Core::Utils::split_string_list(pod_matrix_file_name, "/");
    if (components_of_absolute_path.back() != std::string("none")) havemor_ = true;
  }

  // no mor? -> exit
  if (not havemor_) return;

  // A multi-vector to store basis vectors to be read from file
  Teuchos::RCP<Core::LinAlg::MultiVector<double>> reduced_basis = Teuchos::null;

  // read projection matrix from binary file
  {
    std::string absolute_path_to_pod_file = pod_matrix_file_name;

    // Make sure that we have the absolute path to the file
    if (pod_matrix_file_name[0] != '/')
    {
      std::string::size_type pos = absolute_path_to_input_file.rfind('/');
      if (pos != std::string::npos)
      {
        std::string path = absolute_path_to_input_file.substr(0, pos + 1);
        absolute_path_to_pod_file.insert(
            absolute_path_to_pod_file.begin(), path.begin(), path.end());
      }
    }

    read_pod_basis_vectors_from_file(absolute_path_to_pod_file, reduced_basis);
  }

  // build an importer
  Epetra_Import dofrowimporter(*full_model_dof_row_map_, (reduced_basis->Map()));
  projmatrix_ = Teuchos::make_rcp<Core::LinAlg::MultiVector<double>>(
      *full_model_dof_row_map_, reduced_basis->NumVectors(), true);
  int err = projmatrix_->Import(*reduced_basis, dofrowimporter, Insert, nullptr);
  if (err != 0) FOUR_C_THROW("POD projection matrix could not be mapped onto the dof map");

  // check row dimension
  if (projmatrix_->GlobalLength() != full_model_dof_row_map_->NumGlobalElements())
    FOUR_C_THROW("Projection matrix does not match discretization.");

  // check orthogonality
  if (not is_pod_basis_orthogonal(*projmatrix_))
    FOUR_C_THROW("Projection matrix is not orthogonal.");

  // maps for reduced system
  structmapr_ =
      Teuchos::make_rcp<Epetra_Map>(projmatrix_->NumVectors(), 0, full_model_dof_row_map_->Comm());
  redstructmapr_ = Teuchos::make_rcp<Epetra_Map>(
      projmatrix_->NumVectors(), projmatrix_->NumVectors(), 0, full_model_dof_row_map_->Comm());
  // Core::LinAlg::allreduce_e_map cant't be used here, because NumGlobalElements will be choosen
  // wrong

  // importers for reduced system
  structrimpo_ = Teuchos::make_rcp<Epetra_Import>(*structmapr_, *redstructmapr_);
  structrinvimpo_ = Teuchos::make_rcp<Epetra_Import>(*redstructmapr_, *structmapr_);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::SparseMatrix>
Cardiovascular0D::ProperOrthogonalDecomposition::reduce_diagnoal(Core::LinAlg::SparseMatrix& M)
{
  // right multiply M * V
  Core::LinAlg::MultiVector<double> M_tmp(M.row_map(), projmatrix_->NumVectors(), true);
  int err = M.multiply(false, *projmatrix_, M_tmp);
  if (err) FOUR_C_THROW("Multiplication M * V failed.");

  // left multiply V^T * (M * V)
  Teuchos::RCP<Core::LinAlg::MultiVector<double>> M_red_mvec =
      Teuchos::make_rcp<Core::LinAlg::MultiVector<double>>(*structmapr_, M_tmp.NumVectors(), true);
  multiply_epetra_multi_vectors(
      *projmatrix_, 'T', M_tmp, 'N', *redstructmapr_, *structrimpo_, *M_red_mvec);

  // convert Core::LinAlg::MultiVector<double> to Core::LinAlg::SparseMatrix
  Teuchos::RCP<Core::LinAlg::SparseMatrix> M_red =
      Teuchos::make_rcp<Core::LinAlg::SparseMatrix>(*structmapr_, 0, false, true);
  epetra_multi_vector_to_linalg_sparse_matrix(*M_red_mvec, *structmapr_, Teuchos::null, *M_red);

  return M_red;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::SparseMatrix>
Cardiovascular0D::ProperOrthogonalDecomposition::reduce_off_diagonal(Core::LinAlg::SparseMatrix& M)
{
  // right multiply M * V
  Teuchos::RCP<Core::LinAlg::MultiVector<double>> M_tmp =
      Teuchos::make_rcp<Core::LinAlg::MultiVector<double>>(
          M.domain_map(), projmatrix_->NumVectors(), true);
  int err = M.multiply(true, *projmatrix_, *M_tmp);
  if (err) FOUR_C_THROW("Multiplication V^T * M failed.");

  // convert Core::LinAlg::MultiVector<double> to Core::LinAlg::SparseMatrix
  Teuchos::RCP<Epetra_Map> rangemap = Teuchos::make_rcp<Epetra_Map>(M.domain_map());
  Teuchos::RCP<Core::LinAlg::SparseMatrix> M_red =
      Teuchos::make_rcp<Core::LinAlg::SparseMatrix>(*rangemap, 0, false, true);
  epetra_multi_vector_to_linalg_sparse_matrix(*M_tmp, *rangemap, structmapr_, *M_red);

  return M_red;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::MultiVector<double>>
Cardiovascular0D::ProperOrthogonalDecomposition::reduce_rhs(Core::LinAlg::MultiVector<double>& v)
{
  Teuchos::RCP<Core::LinAlg::MultiVector<double>> v_red =
      Teuchos::make_rcp<Core::LinAlg::MultiVector<double>>(*structmapr_, 1, true);
  multiply_epetra_multi_vectors(*projmatrix_, 'T', v, 'N', *redstructmapr_, *structrimpo_, *v_red);

  return v_red;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::Vector<double>>
Cardiovascular0D::ProperOrthogonalDecomposition::reduce_residual(Core::LinAlg::Vector<double>& v)
{
  Core::LinAlg::Vector<double> v_tmp(*redstructmapr_);
  int err = v_tmp.Multiply('T', 'N', 1.0, *projmatrix_, v, 0.0);
  if (err) FOUR_C_THROW("Multiplication V^T * v failed.");

  Teuchos::RCP<Core::LinAlg::Vector<double>> v_red =
      Teuchos::make_rcp<Core::LinAlg::Vector<double>>(*structmapr_);
  v_red->Import(v_tmp, *structrimpo_, Insert, nullptr);

  return v_red;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::Vector<double>>
Cardiovascular0D::ProperOrthogonalDecomposition::extend_solution(
    Core::LinAlg::Vector<double>& v_red)
{
  Core::LinAlg::Vector<double> v_tmp(*redstructmapr_, true);
  v_tmp.Import(v_red, *structrinvimpo_, Insert, nullptr);
  Teuchos::RCP<Core::LinAlg::Vector<double>> v =
      Teuchos::make_rcp<Core::LinAlg::Vector<double>>(*full_model_dof_row_map_);
  int err = v->Multiply('N', 'N', 1.0, *projmatrix_, v_tmp, 0.0);
  if (err) FOUR_C_THROW("Multiplication V * v_red failed.");

  return v;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Cardiovascular0D::ProperOrthogonalDecomposition::multiply_epetra_multi_vectors(
    Core::LinAlg::MultiVector<double>& multivect1, char multivect1Trans,
    Core::LinAlg::MultiVector<double>& multivect2, char multivect2Trans, Epetra_Map& redmap,
    Epetra_Import& impo, Core::LinAlg::MultiVector<double>& result)
{
  // initialize temporary Core::LinAlg::MultiVector<double> (redmap: all procs hold all
  // elements/rows)
  Core::LinAlg::MultiVector<double> multivect_temp(redmap, multivect2.NumVectors(), true);

  // do the multiplication: (all procs hold the full result)
  int err =
      multivect_temp.Multiply(multivect1Trans, multivect2Trans, 1.0, multivect1, multivect2, 0.0);
  if (err) FOUR_C_THROW("Multiplication failed.");

  // import the result to a Core::LinAlg::MultiVector<double> whose elements/rows are distributed
  // over all procs
  result.Import(multivect_temp, impo, Insert, nullptr);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Cardiovascular0D::ProperOrthogonalDecomposition::epetra_multi_vector_to_linalg_sparse_matrix(
    Core::LinAlg::MultiVector<double>& multivect, Epetra_Map& rangemap,
    Teuchos::RCP<Epetra_Map> domainmap, Core::LinAlg::SparseMatrix& sparsemat)
{
  // pointer to values of the Core::LinAlg::MultiVector<double>
  double* Values;
  Values = multivect.Values();

  // loop over columns of the Core::LinAlg::MultiVector<double>
  for (int i = 0; i < multivect.NumVectors(); i++)
  {
    // loop over rows of the Core::LinAlg::MultiVector<double>
    for (int j = 0; j < multivect.MyLength(); j++)
    {
      // assemble the values into the Core::LinAlg::SparseMatrix (value by value)
      sparsemat.assemble(Values[i * multivect.MyLength() + j], rangemap.GID(j), i);
    }
  }

  // Complete the Core::LinAlg::SparseMatrix
  if (domainmap == Teuchos::null)
    sparsemat.complete();
  else
    sparsemat.complete(*domainmap, rangemap);
  return;
}

/*----------------------------------------------------------------------*
 | read binary projection matrix from file                pfaller Oct17 |
 |                                                                      |
 | MATLAB code to write projmatrix(DIM x dim):                          |
 |                                                                      |
 |   fid=fopen(filename, 'w');                                          |
 |   fwrite(fid, size(projmatrix, 1), 'int');                           |
 |   fwrite(fid, size(projmatrix, 2), 'int');                           |
 |   fwrite(fid, projmatrix', 'single');                                |
 |   fclose(fid);                                                       |
 |                                                                      |
 *----------------------------------------------------------------------*/
void Cardiovascular0D::ProperOrthogonalDecomposition::read_pod_basis_vectors_from_file(
    const std::string& absolute_path_to_pod_file,
    Teuchos::RCP<Core::LinAlg::MultiVector<double>>& projmatrix)
{
  // ***************************
  // PART1: Read in Matrix Sizes
  // ***************************

  // open binary file
  std::ifstream file1(absolute_path_to_pod_file.c_str(),
      std::ifstream::in | std::ifstream::binary | std::ifstream::ate);

  if (!file1.good())
    FOUR_C_THROW("File containing the matrix could not be opened. Check Input-File.");

  // allocation of a memory-block to read the data
  char* sizeblock = new char[8];

  // jump to beginning of file
  file1.seekg(0, std::ifstream::beg);

  // read the first 8 byte into the memory-block
  file1.read(sizeblock, std::streamoff(8));

  // close the file
  file1.close();

  // union for conversion of 4 bytes to int or float
  union CharIntFloat
  {
    char ValueAsChars[4];
    int ValueAsInt;
    float ValueAsFloat;
  } NumRows, NumCols;

  // number of rows
  for (int k = 0; k < 4; k++) NumRows.ValueAsChars[k] = sizeblock[k];

  // number of columns
  for (int k = 0; k < 4; k++) NumCols.ValueAsChars[k] = sizeblock[k + 4];

  delete[] sizeblock;

  // allocate multivector according to matrix size:
  Teuchos::RCP<Epetra_Map> mymap =
      Teuchos::make_rcp<Epetra_Map>(NumRows.ValueAsInt, 0, full_model_dof_row_map_->Comm());
  projmatrix = Teuchos::make_rcp<Core::LinAlg::MultiVector<double>>(*mymap, NumCols.ValueAsInt);


  // ***************************
  // PART2: Read In Matrix
  // ***************************

  // select the format we want to import matrices in
  const int formatfactor = 4;  // 8 is double 4 is single

  // union for conversion of some bytes to int or float
  union NewCharIntFloat
  {
    char VAsChar[formatfactor];
    double VAsDbl;
    float VAsFlt;
  } Val;

  // calculate a number of bytes that are needed to be reserved in order to fill the multivector
  int mysize = projmatrix->NumVectors() * projmatrix->MyLength() * formatfactor;

  // open binary file (again)
  std::ifstream file2(absolute_path_to_pod_file.c_str(),
      std::ifstream::in | std::ifstream::binary | std::ifstream::ate);

  // allocation of a memory-block to read the data
  char* memblock = new char[mysize];

  // calculation of starting points in matrix for each processor
  const Epetra_Comm& comm(full_model_dof_row_map_->Comm());
  const int numproc(comm.NumProc());
  const int mypid(comm.MyPID());
  std::vector<int> localnumbers(numproc, 0);
  std::vector<int> globalnumbers(numproc, 0);
  localnumbers[mypid] = mymap->NumMyElements();
  comm.SumAll(localnumbers.data(), globalnumbers.data(), numproc);

  int factor(0);
  for (int i = 0; i < mypid; i++) factor += globalnumbers[i];

  comm.Barrier();

  // 64 bit number necessary, as integer can overflow for large matrices
  long long start =
      (long long)factor * (long long)projmatrix->NumVectors() * (long long)formatfactor;

  // leads to a starting point:
  file2.seekg(8 + start, std::ifstream::beg);

  // read into memory
  file2.read(memblock, std::streamoff(mysize));

  // close the file
  file2.close();

  // loop over columns and fill double array
  for (int i = 0; i < projmatrix->NumVectors(); i++)
  {
    // loop over all rows owned by the calling processor
    for (int j = 0; j < projmatrix->MyLength(); j++)
    {
      // current value
      for (int k = 0; k < formatfactor; k++)
        Val.VAsChar[k] = memblock[j * formatfactor * NumCols.ValueAsInt + i * formatfactor + k];

      // write current value to Multivector
      projmatrix->ReplaceMyValue(j, i, double(Val.VAsFlt));
    }
  }

  // delete memory-block
  delete[] memblock;

  // all procs wait until proc number 0 did finish the stuff before
  comm.Barrier();

  // Inform user
  if (comm.MyPID() == 0) std::cout << " --> Successful\n" << std::endl;

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Cardiovascular0D::ProperOrthogonalDecomposition::is_pod_basis_orthogonal(
    const Core::LinAlg::MultiVector<double>& M)
{
  const int n = M.NumVectors();

  // calculate V^T * V (should be an nxn identity matrix)
  Epetra_Map map = Epetra_Map(n, n, 0, full_model_dof_row_map_->Comm());
  Core::LinAlg::MultiVector<double> identity = Core::LinAlg::MultiVector<double>(map, n, true);
  identity.Multiply('T', 'N', 1.0, M, M, 0.0);

  // subtract one from diagonal
  for (int i = 0; i < n; ++i) identity.SumIntoGlobalValue(i, i, -1.0);

  // inf norm of columns
  double* norms = new double[n];
  identity.NormInf(norms);

  for (int i = 0; i < n; ++i)
    if (norms[i] > 1.0e-7) return false;

  return true;
}

FOUR_C_NAMESPACE_CLOSE