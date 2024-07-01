/*----------------------------------------------------------------------*/
/*! \file

\brief Nstet element nodal strain implementation


\level 3

*----------------------------------------------------------------------*/

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_mat_aaaneohooke.hpp"
#include "4C_mat_elasthyper.hpp"
#include "4C_mat_micromaterial.hpp"
#include "4C_mat_service.hpp"
#include "4C_mat_stvenantkirchhoff.hpp"
#include "4C_so3_nstet5.hpp"
#include "4C_utils_exceptions.hpp"

#include <Epetra_FECrsMatrix.h>
#include <Teuchos_SerialDenseSolver.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |                                                             gee 03/12|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::NStet5Type::element_deformation_gradient(Core::FE::Discretization& dis)
{
  // current displacement
  Teuchos::RCP<const Epetra_Vector> disp = dis.GetState("displacement");
  if (disp == Teuchos::null) FOUR_C_THROW("Cannot get state vector 'displacement'");
  // loop elements
  std::map<int, Discret::ELEMENTS::NStet5*>::iterator ele;
  for (ele = elecids_.begin(); ele != elecids_.end(); ++ele)
  {
    Discret::ELEMENTS::NStet5* e = ele->second;
    std::vector<int> lm;
    std::vector<int> lmowner;
    std::vector<int> lmstride;
    e->LocationVector(dis, lm, lmowner, lmstride);
    std::vector<double> mydisp(lm.size());
    Core::FE::ExtractMyValues(*disp, mydisp, lm);

    //------------------------------------subelement F
    Core::LinAlg::Matrix<5, 3> subdisp(false);
    for (int j = 0; j < 3; ++j)
      for (int i = 0; i < 5; ++i) subdisp(i, j) = mydisp[i * 3 + j];

    Core::LinAlg::Matrix<4, 3> disp(false);
    for (int k = 0; k < 4; ++k)  // subelement k
    {
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 3; ++j) disp(i, j) = subdisp(e->sub_lm(k)[i], j);

      e->sub_f(k) = e->build_f(disp, e->sub_nxyz(k));
      double J = e->sub_f(k).determinant();
      if (J <= 0.0)
        FOUR_C_THROW("det(F) of Element %d / Subelement %d %10.5e <= 0 !!\n", e->Id(), k, J);
    }  // for (int k=0; k<4; ++k)

  }  // ele
  return;
}


/*----------------------------------------------------------------------*
 |  pre-evaluation of elements (public)                        gee 03/12|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::NStet5Type::pre_evaluate(Core::FE::Discretization& dis,
    Teuchos::ParameterList& p, Teuchos::RCP<Core::LinAlg::SparseOperator> systemmatrix1,
    Teuchos::RCP<Core::LinAlg::SparseOperator> systemmatrix2,
    Teuchos::RCP<Epetra_Vector> systemvector1, Teuchos::RCP<Epetra_Vector> systemvector2,
    Teuchos::RCP<Epetra_Vector> systemvector3)
{
  TEUCHOS_FUNC_TIME_MONITOR("Discret::ELEMENTS::NStet5Type::pre_evaluate");

  // nodal integration for nlnstiff and internal forces only
  // (this method does not compute stresses/strains/element updates/mass matrix)
  auto& action = p.get<std::string>("action", "none");
  if (action != "calc_struct_nlnstiffmass" && action != "calc_struct_nlnstifflmass" &&
      action != "calc_struct_nlnstiff" && action != "calc_struct_stress" &&
      action != "calc_struct_internalforce")
    return;

  // These get filled in here, so remove old stuff
  if (action == "calc_struct_stress")
  {
    nstress_ = Teuchos::rcp(new Epetra_MultiVector(*dis.NodeRowMap(), 6, false));
    nstrain_ = Teuchos::rcp(new Epetra_MultiVector(*dis.NodeRowMap(), 6, false));
  }
  else
  {
    nstress_ = Teuchos::null;
    nstrain_ = Teuchos::null;
  }

  // see what we have for input
  bool assemblemat1 = systemmatrix1 != Teuchos::null;
  bool assemblevec1 = systemvector1 != Teuchos::null;
  bool assemblevec2 = systemvector2 != Teuchos::null;
  bool assemblevec3 = systemvector3 != Teuchos::null;
  if (assemblevec2 || assemblevec3) FOUR_C_THROW("Wrong assembly expectations");

  //-----------------------------------------------------------------
  // nodal stiffness and force (we don't do mass here)
  Core::LinAlg::SerialDenseMatrix stiff;
  Core::LinAlg::SerialDenseVector force;

  //-------------------------------------- construct F for each NStet5
  element_deformation_gradient(dis);

  //-----------------------------------------------------------------
  // create a temporary matrix to assemble to in a 4C-unusual way
  // (across-parallel-interface assembly)
  const Epetra_Map* rmap = nullptr;
  const Epetra_Map* dmap = nullptr;

  Teuchos::RCP<Epetra_FECrsMatrix> stifftmp;
  Teuchos::RCP<Core::LinAlg::SparseMatrix> systemmatrix;
  if (systemmatrix1 != Teuchos::null)
  {
    rmap = &(systemmatrix1->OperatorRangeMap());
    dmap = rmap;
    systemmatrix = Teuchos::rcp_dynamic_cast<Core::LinAlg::SparseMatrix>(systemmatrix1);
    if (systemmatrix != Teuchos::null && systemmatrix->Filled())
      stifftmp =
          Teuchos::rcp(new Epetra_FECrsMatrix(::Copy, systemmatrix->EpetraMatrix()->Graph()));
    else
      stifftmp = Teuchos::rcp(new Epetra_FECrsMatrix(::Copy, *rmap, 256, false));
  }

  //-----------------------------------------------------------------
  // make some tests for fast assembly
  if (systemmatrix != Teuchos::null && systemmatrix->Filled())
  {
    Epetra_CrsMatrix& matrix = *(systemmatrix->EpetraMatrix());
    if (!matrix.StorageOptimized()) FOUR_C_THROW("Matrix must be StorageOptimized() when Filled()");
  }

  //-----------------------------------------------------------------
  // create temporary vector in column map to assemble to
  Epetra_Vector forcetmp1(*dis.DofColMap(), true);

  //-----------------------------------------------------------------
  // current displacements
  Teuchos::RCP<const Epetra_Vector> disp = dis.GetState("displacement");

  //================================================== do nodal stiffness
  std::map<int, Core::Nodes::Node*>::iterator node;
  for (node = noderids_.begin(); node != noderids_.end(); ++node)
  {
    Core::Nodes::Node* nodeL = node->second;  // row node
    const int nodeLid = nodeL->Id();

    // standard quantities for all nodes
    std::vector<Discret::ELEMENTS::NStet5*>& adjele = adjele_[nodeLid];
    std::map<int, std::vector<int>>& adjsubele = adjsubele_[nodeLid];
    std::map<int, Core::Nodes::Node*>& adjnode = adjnode_[nodeLid];
    std::vector<int>& lm = adjlm_[nodeLid];
    std::vector<std::vector<std::vector<int>>>& lmlm = lmlm_[nodeLid];
    const auto ndofperpatch = (int)lm.size();

    if (action == "calc_struct_nlnstiffmass" || action == "calc_struct_nlnstifflmass" ||
        action == "calc_struct_nlnstiff" || action == "calc_struct_internalforce")
    {
      // do nodal integration of stiffness and internal force
      stiff.shape(ndofperpatch, ndofperpatch);
      force.size(ndofperpatch);
      Core::LinAlg::SerialDenseMatrix* stiffptr = &stiff;
      if (action == "calc_struct_internalforce") stiffptr = nullptr;
      TEUCHOS_FUNC_TIME_MONITOR("Discret::ELEMENTS::NStet5Type::nodal_integration");
      nodal_integration(stiffptr, &force, adjnode, adjele, adjsubele, lm, lmlm, *disp, dis, nullptr,
          nullptr, Inpar::STR::stress_none, Inpar::STR::strain_none);
    }
    else if (action == "calc_struct_stress")
    {
      auto iostress =
          Core::UTILS::GetAsEnum<Inpar::STR::StressType>(p, "iostress", Inpar::STR::stress_none);
      auto iostrain =
          Core::UTILS::GetAsEnum<Inpar::STR::StrainType>(p, "iostrain", Inpar::STR::strain_none);
      std::vector<double> nodalstress(6);
      std::vector<double> nodalstrain(6);
      nodal_integration(nullptr, nullptr, adjnode, adjele, adjsubele, lm, lmlm, *disp, dis,
          &nodalstress, &nodalstrain, iostress, iostrain);

      const int lid = dis.NodeRowMap()->LID(nodeLid);
      if (lid == -1) FOUR_C_THROW("Cannot find local id for row node");
      for (int i = 0; i < 6; ++i)
      {
        (*(*nstress_)(i))[lid] = nodalstress[i];
        (*(*nstrain_)(i))[lid] = nodalstrain[i];
      }
    }
    else
      FOUR_C_THROW("Unknown action");


    //---------------------- do assembly of stiffness and internal force
    // (note: this is non-standard-4C assembly and therefore a do it all yourself version!)
    // there is no guarantee that systemmatrix exists
    // (e.g. if systemmatrix1 is actually a BlockSparseMatrix)
    bool fastassemble = false;
    if (systemmatrix != Teuchos::null) fastassemble = true;

    if (assemblemat1)
    {
      TEUCHOS_FUNC_TIME_MONITOR("Discret::ELEMENTS::NStet5Type::pre_evaluate Assembly");
      std::vector<int> lrlm;
      std::vector<int> lclm;

      const Epetra_Map& dofrowmap = systemmatrix1->OperatorRangeMap();
      lrlm.resize(ndofperpatch);
      for (int i = 0; i < ndofperpatch; ++i) lrlm[i] = dofrowmap.LID(lm[i]);
      if (fastassemble)
      {
        const Epetra_Map& dofcolmap = systemmatrix->ColMap();
        lclm.resize(ndofperpatch);
        for (int i = 0; i < ndofperpatch; ++i) lclm[i] = dofcolmap.LID(lm[i]);
      }

      for (int i = 0; i < ndofperpatch; ++i)
      {
        if (lrlm[i] == -1)  // off-processor row
        {
          for (int j = 0; j < ndofperpatch; ++j)
          {
            int errone = stifftmp->SumIntoGlobalValues(1, &lm[i], 1, &lm[j], &stiff(i, j));
            if (errone > 0)
            {
              int errtwo = stifftmp->InsertGlobalValues(1, &lm[i], 1, &lm[j], &stiff(i, j));
              if (errtwo < 0)
                FOUR_C_THROW(
                    "Epetra_FECrsMatrix::InsertGlobalValues returned error code %d", errtwo);
            }
            else if (errone)
              FOUR_C_THROW(
                  "Epetra_FECrsMatrix::SumIntoGlobalValues returned error code %d", errone);
          }
        }
        else  // local row
        {
          if (systemmatrix != Teuchos::null && systemmatrix->Filled())  // matrix is SparseMatrix
          {
            Epetra_CrsMatrix& matrix = *(systemmatrix->EpetraMatrix());
            int length;
            double* values;
            int* indices;
            matrix.ExtractMyRowView(lrlm[i], length, values, indices);
            for (int j = 0; j < ndofperpatch; ++j)
            {
              int* loc = std::lower_bound(indices, indices + length, lclm[j]);
              // #ifdef FOUR_C_ENABLE_ASSERTIONS
              if (*loc != lclm[j]) FOUR_C_THROW("Cannot find local column entry %d", lclm[j]);
              // #endif
              int pos = loc - indices;

              // test physical continuity of nodal values inside the Epetra_CrsMatrix
              bool continuous = true;
              for (int k = 1; k < 3; ++k)
                if (indices[pos + k] == lclm[j + k])
                  continue;
                else
                {
                  continuous = false;
                  break;
                }

              if (continuous)
              {
                values[pos++] += stiff(i, j++);
                values[pos++] += stiff(i, j++);
                values[pos] += stiff(i, j);
              }
              else
              {
                int err = matrix.SumIntoMyValues(lrlm[i], 1, &stiff(i, j), &lclm[j]);
                j++;
                err += matrix.SumIntoMyValues(lrlm[i], 1, &stiff(i, j), &lclm[j]);
                j++;
                err += matrix.SumIntoMyValues(lrlm[i], 1, &stiff(i, j), &lclm[j]);
                if (err) FOUR_C_THROW("Epetra_CrsMatrix::SumIntoMyValues returned err=%d", err);
              }
            }
          }
          else  // matrix not SparseMatrix (e.g. BlockMatrix) -> fall back to standard assembly
          {
            for (int j = 0; j < ndofperpatch; ++j)
              systemmatrix1->Assemble(stiff(i, j), lm[i], lm[j]);
          }
        }
      }
    }

    //-----------------------------------------------------------------------------------
    if (assemblevec1)
    {
      for (int i = 0; i < ndofperpatch; ++i)
      {
        const int rgid = lm[i];
        const int lid = forcetmp1.Map().LID(rgid);
        if (lid < 0) FOUR_C_THROW("global row %d does not exist in column map", rgid);
        forcetmp1[lid] += force[i];
      }
    }

    //=========================================================================
  }  // for (node=noderids_.begin(); node != noderids_.end(); ++node)

  //-------------------------------------------------------------------------
  if (action == "calc_struct_stress")
  {
    // we have to export the nodal stresses and strains to column map
    // so they can be written by the elements
    Teuchos::RCP<Epetra_MultiVector> tmp =
        Teuchos::rcp(new Epetra_MultiVector(*dis.NodeColMap(), 6, false));
    Core::LinAlg::Export(*nstress_, *tmp);
    nstress_ = tmp;
    tmp = Teuchos::rcp(new Epetra_MultiVector(*dis.NodeColMap(), 6, false));
    Core::LinAlg::Export(*nstrain_, *tmp);
    nstrain_ = tmp;
  }


  //-------------------------------------------------------------------------
  // need to export forcetmp to systemvector1 and insert stiffnesses from stifftmp
  // into systemmatrix1
  // Note that fillComplete is never called on stifftmp
  if (assemblevec1)
  {
    Epetra_Vector tmp(systemvector1->Map(), false);
    Epetra_Export exporter(forcetmp1.Map(), tmp.Map());
    int err = tmp.Export(forcetmp1, exporter, Add);
    if (err) FOUR_C_THROW("Export using exporter returned err=%d", err);
    systemvector1->Update(1.0, tmp, 1.0);
  }
  if (assemblemat1)
  {
    int err = stifftmp->GlobalAssemble(*dmap, *rmap, false);
    if (err) FOUR_C_THROW("Epetra_FECrsMatrix::GlobalAssemble returned err=%d", err);
    const Epetra_Map& cmap = stifftmp->ColMap();
    for (int lrow = 0; lrow < stifftmp->NumMyRows(); ++lrow)
    {
      int numentries;
      double* values;
      if (!stifftmp->Filled())
      {
        const int grow = stifftmp->RowMap().GID(lrow);
        int* gindices;
        int err = stifftmp->ExtractGlobalRowView(grow, numentries, values, gindices);
        if (err) FOUR_C_THROW("Epetra_FECrsMatrix::ExtractGlobalRowView returned err=%d", err);
        for (int j = 0; j < numentries; ++j) systemmatrix1->Assemble(values[j], grow, gindices[j]);
      }
      else
      {
        int* lindices;
        int err = stifftmp->ExtractMyRowView(lrow, numentries, values, lindices);
        if (err) FOUR_C_THROW("Epetra_FECrsMatrix::ExtractMyRowView returned err=%d", err);
        if (systemmatrix != Teuchos::null && systemmatrix->Filled())
        {
          Epetra_CrsMatrix& matrix = *systemmatrix->EpetraMatrix();
          for (int j = 0; j < numentries; ++j)
          {
            int err = matrix.SumIntoMyValues(lrow, 1, &values[j], &lindices[j]);
            if (err) FOUR_C_THROW("Epetra_CrsMatrix::SumIntoMyValues returned err=%d", err);
          }
        }
        else
        {
          const int grow = stifftmp->RowMap().GID(lrow);
          for (int j = 0; j < numentries; ++j)
            systemmatrix1->Assemble(values[j], grow, cmap.GID(lindices[j]));
        }
      }
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  do nodal integration (public)                              gee 03/12|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::NStet5Type::nodal_integration(Core::LinAlg::SerialDenseMatrix* stiff,
    Core::LinAlg::SerialDenseVector* force, std::map<int, Core::Nodes::Node*>& adjnode,
    std::vector<Discret::ELEMENTS::NStet5*>& adjele, std::map<int, std::vector<int>>& adjsubele,
    std::vector<int>& lm, std::vector<std::vector<std::vector<int>>>& lmlm,
    const Epetra_Vector& disp, Core::FE::Discretization& dis, std::vector<double>* nodalstress,
    std::vector<double>* nodalstrain, const Inpar::STR::StressType iostress,
    const Inpar::STR::StrainType iostrain)
{
  TEUCHOS_FUNC_TIME_MONITOR("Discret::ELEMENTS::NStet5Type::nodal_integration");
  //  typedef Sacado::Fad::DFad<double> FAD; // for first derivs
  //  typedef Sacado::Fad::DFad<Sacado::Fad::DFad<double> > FADFAD; // for second derivs

  //-------------------------------------------------- standard quantities
  const auto ndofinpatch = (int)lm.size();
  const auto neleinpatch = (int)adjele.size();

  //------------------------------ see whether materials in patch are equal
  bool matequal = true;
  {
    int mat = adjele[0]->material_;
    for (int i = 1; i < neleinpatch; ++i)
      if (mat != adjele[i]->material_)
      {
        matequal = false;
        break;
      }
  }

  //-----------------------------------------------------------------------
  // get displacements of this patch
  std::vector<double> patchdisp(ndofinpatch);
  //  std::vector<FADFAD> tpatchdisp(ndofinpatch);
  for (int i = 0; i < ndofinpatch; ++i)
  {
    int lid = disp.Map().LID(lm[i]);
    if (lid == -1) FOUR_C_THROW("Cannot find degree of freedom on this proc");
    patchdisp[i] = disp[disp.Map().LID(lm[i])];
    //    tpatchdisp[i] = patchdisp[i];
    //    tpatchdisp[i].diff(i,ndofinpatch);
    //    tpatchdisp[i].val().diff(i,ndofinpatch);
  }

  //-----------------------------------------------------------------------
  // build averaged F and volume of node
  double Vnode = 0.0;
  Core::LinAlg::Matrix<3, 3> Fnode(true);
  //  Core::LinAlg::Matrix<3,3,FADFAD> tFnode(true);

  for (int i = 0; i < neleinpatch; ++i)
  {
    Discret::ELEMENTS::NStet5* ele = adjele[i];
    std::vector<int>& subele = adjsubele[ele->Id()];

    for (int subeleid : subele)
    {
      // copy subelement displacements to 4x3 format
      //      Core::LinAlg::Matrix<4,3> eledispmat(false);
      //      Core::LinAlg::Matrix<4,3,FADFAD> teledispmat(false);
      //      for (int k=0; k<4; ++k)
      //        for (int l=0; l<3; ++l)
      //        {
      //          eledispmat(k,l) = patchdisp[lmlm[i][j][k*3+l]];
      //          teledispmat(k,l) = tpatchdisp[lmlm[i][j][k*3+l]];
      //        }

      // build F from this subelement
      //     Core::LinAlg::Matrix<3,3> F(false);
      //     Core::LinAlg::Matrix<3,3,FADFAD> tF(true);
      //     F = ele->BuildF(eledispmat,ele->SubNxyz(subeleid));
      Core::LinAlg::Matrix<3, 3> F = ele->sub_f(subeleid);
      //     tF = t_build_f(teledispmat,ele->SubNxyz(subeleid));

      // add 1/3 of subelement material volume to this node
      const double V = ele->sub_v(subeleid) / 3.0;
      Vnode += V;

      // add to nodal deformation gradient
      F.scale(V);
      //     tF.scale(V);
      Fnode += F;
      //     tFnode += tF;

    }  // for (unsigned j=0; j<subele.size(); ++j)
  }    // for (int i=0; i<neleinpatch; ++i)

  // do the actual averaging
  Fnode.scale(1.0 / Vnode);
  //  tFnode.scale(1.0/Vnode);


  //-----------------------------------------------------------------------
  // build B operator

  Core::LinAlg::SerialDenseMatrix bopbar(6, ndofinpatch);
  Core::LinAlg::SerialDenseMatrix nxyzbar(9, ndofinpatch);

  // loop elements in patch
  for (int ele = 0; ele < neleinpatch; ++ele)
  {
    // current element
    Discret::ELEMENTS::NStet5* actele = adjele[ele];
    std::vector<int>& subele = adjsubele[actele->Id()];
    // loop subelements in this element
    for (unsigned j = 0; j < subele.size(); ++j)
    {
      const int subeleid = subele[j];
      double V = actele->sub_v(subeleid) / 3;
      V = V / Vnode;

      // get derivatives with respect to X
      const Core::LinAlg::Matrix<4, 3>& nxyz = actele->sub_nxyz(subeleid);
      for (int k = 0; k < 4; ++k)
      {
        for (int l = 0; l < 3; ++l)
        {
          int index = lmlm[ele][j][k * 3 + l];
          if (index % 3 == 0)
          {
            nxyzbar(l * 3 + 0, index + 0) += V * nxyz(k, l);
            nxyzbar(l * 3 + 1, index + 1) += V * nxyz(k, l);
            nxyzbar(l * 3 + 2, index + 2) += V * nxyz(k, l);
          }
          else if (index % 3 == 1)
          {
            nxyzbar(l * 3 + 0, index - 1) += V * nxyz(k, l);
            nxyzbar(l * 3 + 1, index + 0) += V * nxyz(k, l);
            nxyzbar(l * 3 + 2, index + 1) += V * nxyz(k, l);
          }
          else
          {
            nxyzbar(l * 3 + 0, index - 2) += V * nxyz(k, l);
            nxyzbar(l * 3 + 1, index - 1) += V * nxyz(k, l);
            nxyzbar(l * 3 + 2, index + 0) += V * nxyz(k, l);
          }
        }  // for (int l=0; l<3; ++l)
      }    // for (int k=0; k<4; ++k)
    }      // for (unsigned j=0; j<subele.size();++j)
  }        // for (int ele=0; ele<neleinpatch; ++ele)



  for (int j = 0; j < ndofinpatch; ++j)
  {
    bopbar(0, j) =
        Fnode(0, 0) * nxyzbar(0, j) + Fnode(1, 0) * nxyzbar(1, j) + Fnode(2, 0) * nxyzbar(2, j);
    bopbar(1, j) =
        Fnode(0, 1) * nxyzbar(3, j) + Fnode(1, 1) * nxyzbar(4, j) + Fnode(2, 1) * nxyzbar(5, j);
    bopbar(2, j) =
        Fnode(0, 2) * nxyzbar(6, j) + Fnode(1, 2) * nxyzbar(7, j) + Fnode(2, 2) * nxyzbar(8, j);
    bopbar(3, j) = Fnode(0, 1) * nxyzbar(0, j) + Fnode(0, 0) * nxyzbar(3, j) +
                   Fnode(1, 1) * nxyzbar(1, j) + Fnode(1, 0) * nxyzbar(4, j) +
                   Fnode(2, 1) * nxyzbar(2, j) + Fnode(2, 0) * nxyzbar(5, j);
    bopbar(4, j) = Fnode(0, 2) * nxyzbar(3, j) + Fnode(0, 1) * nxyzbar(6, j) +
                   Fnode(1, 2) * nxyzbar(4, j) + Fnode(1, 1) * nxyzbar(7, j) +
                   Fnode(2, 2) * nxyzbar(5, j) + Fnode(2, 1) * nxyzbar(8, j);
    bopbar(5, j) = Fnode(0, 2) * nxyzbar(0, j) + Fnode(0, 0) * nxyzbar(6, j) +
                   Fnode(1, 2) * nxyzbar(1, j) + Fnode(1, 0) * nxyzbar(7, j) +
                   Fnode(2, 2) * nxyzbar(2, j) + Fnode(2, 0) * nxyzbar(8, j);
  }



  //-------------------------------------------------------------- averaged strain
  // right cauchy green

  Core::LinAlg::Matrix<3, 3> cauchygreen;
  //  Core::LinAlg::Matrix<3,3,FADFAD> tcauchygreen;
  cauchygreen.multiply_tn(Fnode, Fnode);
  //  tcauchygreen.multiply_tn(tFnode,tFnode);

  // Green-Lagrange strains matrix E = 0.5 * (Cauchygreen - Identity)
  // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
  Core::LinAlg::Matrix<6, 1> glstrain(true);
  glstrain(0) = 0.5 * (cauchygreen(0, 0) - 1.0);
  glstrain(1) = 0.5 * (cauchygreen(1, 1) - 1.0);
  glstrain(2) = 0.5 * (cauchygreen(2, 2) - 1.0);
  glstrain(3) = cauchygreen(0, 1);
  glstrain(4) = cauchygreen(1, 2);
  glstrain(5) = cauchygreen(2, 0);


  //-------------------------------------------------------- output of strain
  if (iostrain != Inpar::STR::strain_none)
  {
#ifndef PUSO_NSTET5
    strain_output(iostrain, *nodalstrain, Fnode, Fnode.determinant(), 1.0, 1.0 - ALPHA_NSTET5);
#else
    strain_output(iostrain, *nodalstrain, Fnode, glstrain, 1.0 - ALPHA_NSTET5);
#endif
  }

  //----------------------------------------- averaged material and stresses
  Core::LinAlg::Matrix<6, 6> cmat(true);
  Core::LinAlg::Matrix<6, 1> stress(true);

  //-----------------------------------------------------------------------
  // material law
  if (matequal)  // element patch has single material
  {
    double density;  // just a dummy density
    Teuchos::RCP<Core::Mat::Material> mat = adjele[0]->Material();
    // EleGID is set to -1 errorcheck is performed in
    // Mat::Evaluate. I.e if we have elementwise mat params you will catch an error
    select_material(mat, stress, cmat, density, glstrain, Fnode, 0, -1);
  }
  else
  {
    double density;  // just a dummy density
    Core::LinAlg::Matrix<6, 6> cmatele;
    Core::LinAlg::Matrix<6, 1> stressele;
    for (int ele = 0; ele < neleinpatch; ++ele)
    {
      cmatele = 0.0;
      stressele = 0.0;
      // current element
      Discret::ELEMENTS::NStet5* actele = adjele[ele];
      // volume of that element assigned to node L
      double V = 0.0;
      for (unsigned j = 0; j < adjsubele[actele->Id()].size(); ++j)
        V += (actele->sub_v(adjsubele[actele->Id()][j]) / 3.0);
      // material of the element
      Teuchos::RCP<Core::Mat::Material> mat = actele->Material();
      // EleGID is set to -1 errorcheck is performed in
      // Mat::Evaluate. I.e if we have elementwise mat params you will catch an error
      select_material(mat, stressele, cmatele, density, glstrain, Fnode, 0, -1);
      cmat.update(V, cmatele, 1.0);
      stress.update(V, stressele, 1.0);
    }  // for (int ele=0; ele<neleinpatch; ++ele)
    stress.scale(1.0 / Vnode);
    cmat.scale(1.0 / Vnode);
  }

  //-----------------------------------------------------------------------
  // stress is split as follows:
  // stress = vol_node + (1-alpha) * dev_node + alpha * dev_ele
#ifndef PUSO_NSTET5
  {
    Core::LinAlg::Matrix<6, 1> stressdev(true);
    Core::LinAlg::Matrix<6, 6> cmatdev(true);
    Core::LinAlg::Matrix<6, 1> stressvol(true);
    Core::LinAlg::Matrix<6, 6> cmatvol(true);

    // compute deviatoric stress and tangent from total stress and tangent
    dev_stress_tangent(stressdev, cmatdev, cmat, stress, cauchygreen);

    // compute volumetric stress and tangent
    stressvol.update(-1.0, stressdev, 1.0, stress, 0.0);
    cmatvol.update(-1.0, cmatdev, 1.0, cmat, 0.0);

    // compute nodal stress
    stress.update(1.0, stressvol, 1 - ALPHA_NSTET5, stressdev, 0.0);
    cmat.update(1.0, cmatvol, 1 - ALPHA_NSTET5, cmatdev, 0.0);
  }
#else
  {
    stress.scale(1. - ALPHA_NSTET5);
    cmat.scale(1. - ALPHA_NSTET5);
  }
#endif
  //-----------------------------------------------------------------------
  // stress output
  if (iostress != Inpar::STR::stress_none)
  {
    stress_output(iostress, *nodalstress, stress, Fnode, Fnode.determinant());
  }
  //----------------------------------------------------- internal forces
  if (force)
  {
    Core::LinAlg::SerialDenseVector stress_epetra(Teuchos::View, stress.data(), stress.numRows());
    Core::LinAlg::multiply_tn(0.0, *force, Vnode, bopbar, stress_epetra);
  }
  //--------------------------------------------------- elastic stiffness
  if (stiff)
  {
    Core::LinAlg::SerialDenseMatrix cmat_epetra(
        Teuchos::View, cmat.data(), cmat.numRows(), cmat.numRows(), cmat.numCols());
    Core::LinAlg::SerialDenseMatrix cb(6, ndofinpatch);
    Core::LinAlg::multiply(cb, cmat_epetra, bopbar);
    Core::LinAlg::multiply_tn(0.0, *stiff, Vnode, bopbar, cb);
  }

  //----------------------------------------------------- geom. stiffness
  if (stiff)
  {
    if (!force) FOUR_C_THROW("Cannot compute stiffness matrix without computing internal force");

    Core::LinAlg::SerialDenseMatrix kg(ndofinpatch, ndofinpatch, true);
    for (int m = 0; m < ndofinpatch; ++m)
    {
      for (int n = 0; n < ndofinpatch; ++n)
      {
        for (int i = 0; i < 3; ++i)
        {
          double sum = Vnode * (stress(0) * nxyzbar(i, n) * nxyzbar(i, m) +
                                   stress(1) * nxyzbar(i + 1 * 3, n) * nxyzbar(i + 1 * 3, m) +
                                   stress(2) * nxyzbar(i + 2 * 3, n) * nxyzbar(i + 2 * 3, m) +
                                   stress(3) * (nxyzbar(i, n) * nxyzbar(i + 1 * 3, m) +
                                                   nxyzbar(i + 1 * 3, n) * nxyzbar(i, m)) +
                                   stress(4) * (nxyzbar(i + 1 * 3, n) * nxyzbar(i + 2 * 3, m) +
                                                   nxyzbar(i + 2 * 3, n) * nxyzbar(i + 1 * 3, m)) +
                                   stress(5) * (nxyzbar(i, n) * nxyzbar(i + 2 * 3, m) +
                                                   nxyzbar(i + 2 * 3, n) * nxyzbar(i, m)));
          (*stiff)(m, n) += sum;
          kg(m, n) += sum;

        }  // for (int i=0; i<3; ++i)
      }    // for (int n=0; n<ndofinpatch; ++n)
    }      // for (int m=0; m<ndofinpatch; ++m)
  }        // if (stiff)

  return;
}



/*----------------------------------------------------------------------*
 | material laws for NStet5 (protected)                        gee 03/12|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::NStet5Type::select_material(const Teuchos::RCP<Core::Mat::Material>& mat,
    Core::LinAlg::Matrix<6, 1>& stress, Core::LinAlg::Matrix<6, 6>& cmat, double& density,
    Core::LinAlg::Matrix<6, 1>& glstrain, Core::LinAlg::Matrix<3, 3>& defgrd, const int gp,
    const int eleGID)
{
  switch (mat->MaterialType())
  {
    case Core::Materials::m_stvenant: /*------------------ st.venant-kirchhoff-material */
    {
      auto* stvk = dynamic_cast<Mat::StVenantKirchhoff*>(mat.get());
      Teuchos::ParameterList params;
      Core::LinAlg::Matrix<3, 3> defgrd(true);
      stvk->evaluate(&defgrd, &glstrain, params, &stress, &cmat, gp, eleGID);
      density = stvk->Density();
    }
    break;
    case Core::Materials::m_aaaneohooke: /*-- special case of generalised NeoHookean material see
                                       Raghavan, Vorp */
    {
      auto* aaa = dynamic_cast<Mat::AAAneohooke*>(mat.get());
      Teuchos::ParameterList params;
      aaa->evaluate(&defgrd, &glstrain, params, &stress, &cmat, gp, eleGID);
      density = aaa->Density();
    }
    break;
    case Core::Materials::m_elasthyper: /*----------- general hyperelastic matrial */
    {
      auto* hyper = dynamic_cast<Mat::ElastHyper*>(mat.get());
      Teuchos::ParameterList params;
      hyper->evaluate(&defgrd, &glstrain, params, &stress, &cmat, gp, eleGID);
      density = hyper->Density();
      return;
      break;
    }
    default:
      FOUR_C_THROW("Illegal type %d of material for element NStet5 tet4", mat->MaterialType());
      break;
  }

  /*--------------------------------------------------------------------*/
  return;
}  // Discret::ELEMENTS::NStet5::select_material

/*----------------------------------------------------------------------*
 |  compute deviatoric tangent and stresses (private/static)   gee 03/12|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::NStet5Type::dev_stress_tangent(Core::LinAlg::Matrix<6, 1>& Sdev,
    Core::LinAlg::Matrix<6, 6>& CCdev, Core::LinAlg::Matrix<6, 6>& CC,
    const Core::LinAlg::Matrix<6, 1>& S, const Core::LinAlg::Matrix<3, 3>& C)
{
  //---------------------------------- things that we'll definitely need
  // inverse of C
  Core::LinAlg::Matrix<3, 3> Cinv;
  const double detC = Cinv.invert(C);
  // J = det(F) = sqrt(detC)
  const double J = sqrt(detC);

  // S as a 3x3 matrix
  Core::LinAlg::Matrix<3, 3> Smat;
  Smat(0, 0) = S(0);
  Smat(0, 1) = S(3);
  Smat(0, 2) = S(5);
  Smat(1, 0) = Smat(0, 1);
  Smat(1, 1) = S(1);
  Smat(1, 2) = S(4);
  Smat(2, 0) = Smat(0, 2);
  Smat(2, 1) = Smat(1, 2);
  Smat(2, 2) = S(2);

  //--------------------------------------------- pressure p = -1/(3J) S:C
  double p = 0.0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) p += Smat(i, j) * C(i, j);
  p *= (-1. / (3. * J));

  //-------------------------------- compute volumetric PK2 Svol = -p J Cinv
  //-------------------------------------------------------- Sdev = S - Svol
  const double fac = -p * J;
  Sdev(0) = Smat(0, 0) - fac * Cinv(0, 0);
  Sdev(1) = Smat(1, 1) - fac * Cinv(1, 1);
  Sdev(2) = Smat(2, 2) - fac * Cinv(2, 2);
  Sdev(3) = Smat(0, 1) - fac * Cinv(0, 1);
  Sdev(4) = Smat(1, 2) - fac * Cinv(1, 2);
  Sdev(5) = Smat(0, 2) - fac * Cinv(0, 2);

  //======================================== volumetric tangent matrix CCvol
  Core::LinAlg::Matrix<6, 6> CCvol(true);  // fill with zeros

  //--------------------------------------- CCvol += 2pJ (Cinv boeppel Cinv)
  Mat::add_kronecker_tensor_product(CCvol, -2.0 * fac, Cinv, Cinv, 0.0);

  //------------------------------------------ CCvol += 2/3 * Cinv dyad S
  Mat::add_elasticity_tensor_product(CCvol, 2.0 / 3.0, Cinv, Smat, 1.0);

  //-------------------------------------- CCvol += 1/3 Cinv dyad ( CC : C )
  {
    // C as Voigt vector
    Core::LinAlg::Matrix<6, 1> Cvec;
    Cvec(0) = C(0, 0);
    Cvec(1) = C(1, 1);
    Cvec(2) = C(2, 2);
    Cvec(3) = 2.0 * C(0, 1);
    Cvec(4) = 2.0 * C(1, 2);
    Cvec(5) = 2.0 * C(0, 2);

    Core::LinAlg::Matrix<6, 1> CCcolonC;
    CCcolonC.multiply(CC, Cvec);

    Core::LinAlg::Matrix<3, 3> CCcC;
    CCcC(0, 0) = CCcolonC(0);
    CCcC(0, 1) = CCcolonC(3);
    CCcC(0, 2) = CCcolonC(5);
    CCcC(1, 0) = CCcC(0, 1);
    CCcC(1, 1) = CCcolonC(1);
    CCcC(1, 2) = CCcolonC(4);
    CCcC(2, 0) = CCcC(0, 2);
    CCcC(2, 1) = CCcC(1, 2);
    CCcC(2, 2) = CCcolonC(2);
    Mat::add_elasticity_tensor_product(CCvol, 1. / 3., Cinv, CCcC, 1.0);
  }

  //----------------------------------------------------- CCdev = CC - CCvol
  CCdev.update(1.0, CC, -1.0, CCvol);

  return;
}

/*----------------------------------------------------------------------*
 |                                                             gee 03/12|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::NStet5Type::strain_output(const Inpar::STR::StrainType iostrain,
    std::vector<double>& nodalstrain, Core::LinAlg::Matrix<3, 3>& F, const double& detF,
    const double volweight, const double devweight)
{
  Core::LinAlg::Matrix<3, 3> Fiso = F;
  Fiso.scale(pow(detF, -1.0 / 3.0));

  Core::LinAlg::Matrix<3, 3> Fvol(true);
  Fvol(0, 0) = 1.0;
  Fvol(1, 1) = 1.0;
  Fvol(2, 2) = 1.0;
  Fvol.scale(pow(detF, 1.0 / 3.0));

  Core::LinAlg::Matrix<3, 3> cauchygreeniso(false);
  cauchygreeniso.multiply_tn(Fiso, Fiso);

  Core::LinAlg::Matrix<3, 3> cauchygreenvol(false);
  cauchygreenvol.multiply_tn(Fvol, Fvol);

  Core::LinAlg::Matrix<3, 3> glstrainiso(false);
  glstrainiso(0, 0) = 0.5 * (cauchygreeniso(0, 0) - 1.0);
  glstrainiso(0, 1) = 0.5 * cauchygreeniso(0, 1);
  glstrainiso(0, 2) = 0.5 * cauchygreeniso(0, 2);
  glstrainiso(1, 0) = glstrainiso(0, 1);
  glstrainiso(1, 1) = 0.5 * (cauchygreeniso(1, 1) - 1.0);
  glstrainiso(1, 2) = 0.5 * cauchygreeniso(1, 2);
  glstrainiso(2, 0) = glstrainiso(0, 2);
  glstrainiso(2, 1) = glstrainiso(1, 2);
  glstrainiso(2, 2) = 0.5 * (cauchygreeniso(2, 2) - 1.0);

  Core::LinAlg::Matrix<3, 3> glstrainvol(false);
  glstrainvol(0, 0) = 0.5 * (cauchygreenvol(0, 0) - 1.0);
  glstrainvol(0, 1) = 0.5 * cauchygreenvol(0, 1);
  glstrainvol(0, 2) = 0.5 * cauchygreenvol(0, 2);
  glstrainvol(1, 0) = glstrainvol(0, 1);
  glstrainvol(1, 1) = 0.5 * (cauchygreenvol(1, 1) - 1.0);
  glstrainvol(1, 2) = 0.5 * cauchygreenvol(1, 2);
  glstrainvol(2, 0) = glstrainvol(0, 2);
  glstrainvol(2, 1) = glstrainvol(1, 2);
  glstrainvol(2, 2) = 0.5 * (cauchygreenvol(2, 2) - 1.0);

  Core::LinAlg::Matrix<3, 3> glstrainout = glstrainiso;
  glstrainout.update(volweight, glstrainvol, devweight);

  switch (iostrain)
  {
    case Inpar::STR::strain_gl:
    {
      nodalstrain[0] = glstrainout(0, 0);
      nodalstrain[1] = glstrainout(1, 1);
      nodalstrain[2] = glstrainout(2, 2);
      nodalstrain[3] = glstrainout(0, 1);
      nodalstrain[4] = glstrainout(1, 2);
      nodalstrain[5] = glstrainout(0, 2);
    }
    break;
    case Inpar::STR::strain_ea:
    {
      // inverse of deformation gradient
      Core::LinAlg::Matrix<3, 3> invdefgrd;
      invdefgrd.invert(F);
      Core::LinAlg::Matrix<3, 3> temp;
      Core::LinAlg::Matrix<3, 3> euler_almansi;
      temp.multiply(glstrainout, invdefgrd);
      euler_almansi.multiply_tn(invdefgrd, temp);
      nodalstrain[0] = euler_almansi(0, 0);
      nodalstrain[1] = euler_almansi(1, 1);
      nodalstrain[2] = euler_almansi(2, 2);
      nodalstrain[3] = euler_almansi(0, 1);
      nodalstrain[4] = euler_almansi(1, 2);
      nodalstrain[5] = euler_almansi(0, 2);
    }
    break;
    case Inpar::STR::strain_none:
      break;
    default:
      FOUR_C_THROW("requested strain type not available");
  }

  return;
}


/*----------------------------------------------------------------------*
 |                                                             gee 03/12|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::NStet5Type::strain_output(const Inpar::STR::StrainType iostrain,
    std::vector<double>& nodalstrain, Core::LinAlg::Matrix<3, 3>& F,
    Core::LinAlg::Matrix<6, 1>& glstrain, const double weight)
{
  Core::LinAlg::Matrix<3, 3> glstrainout;

  glstrainout(0, 0) = weight * glstrain(0);
  glstrainout(1, 1) = weight * glstrain(1);
  glstrainout(2, 2) = weight * glstrain(2);
  glstrainout(0, 1) = weight * glstrain(3);
  glstrainout(1, 2) = weight * glstrain(4);
  glstrainout(0, 2) = weight * glstrain(5);


  switch (iostrain)
  {
    case Inpar::STR::strain_gl:
    {
      nodalstrain[0] = glstrainout(0, 0);
      nodalstrain[1] = glstrainout(1, 1);
      nodalstrain[2] = glstrainout(2, 2);
      nodalstrain[3] = glstrainout(0, 1);
      nodalstrain[4] = glstrainout(1, 2);
      nodalstrain[5] = glstrainout(0, 2);
    }
    break;
    case Inpar::STR::strain_ea:
    {
      // inverse of deformation gradient
      Core::LinAlg::Matrix<3, 3> invdefgrd;
      invdefgrd.invert(F);
      Core::LinAlg::Matrix<3, 3> temp;
      Core::LinAlg::Matrix<3, 3> euler_almansi;
      temp.multiply(glstrainout, invdefgrd);
      euler_almansi.multiply_tn(invdefgrd, temp);
      nodalstrain[0] = euler_almansi(0, 0);
      nodalstrain[1] = euler_almansi(1, 1);
      nodalstrain[2] = euler_almansi(2, 2);
      nodalstrain[3] = euler_almansi(0, 1);
      nodalstrain[4] = euler_almansi(1, 2);
      nodalstrain[5] = euler_almansi(0, 2);
    }
    break;
    case Inpar::STR::strain_none:
      break;
    default:
      FOUR_C_THROW("requested strain type not available");
  }

  return;
}


/*----------------------------------------------------------------------*
 |                                                             gee 03/12|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::NStet5Type::stress_output(const Inpar::STR::StressType iostress,
    std::vector<double>& nodalstress, Core::LinAlg::Matrix<6, 1>& stress,
    Core::LinAlg::Matrix<3, 3>& F, const double& detF)
{
  switch (iostress)
  {
    case Inpar::STR::stress_2pk:
    {
      for (int i = 0; i < 6; ++i) nodalstress[i] = stress(i);
    }
    break;
    case Inpar::STR::stress_cauchy:
    {
      Core::LinAlg::Matrix<3, 3> pkstress;
      pkstress(0, 0) = stress(0);
      pkstress(0, 1) = stress(3);
      pkstress(0, 2) = stress(5);
      pkstress(1, 0) = pkstress(0, 1);
      pkstress(1, 1) = stress(1);
      pkstress(1, 2) = stress(4);
      pkstress(2, 0) = pkstress(0, 2);
      pkstress(2, 1) = pkstress(1, 2);
      pkstress(2, 2) = stress(2);
      Core::LinAlg::Matrix<3, 3> temp;
      Core::LinAlg::Matrix<3, 3> cauchystress;
      temp.multiply(1.0 / detF, F, pkstress);
      cauchystress.multiply_nt(temp, F);
      nodalstress[0] = cauchystress(0, 0);
      nodalstress[1] = cauchystress(1, 1);
      nodalstress[2] = cauchystress(2, 2);
      nodalstress[3] = cauchystress(0, 1);
      nodalstress[4] = cauchystress(1, 2);
      nodalstress[5] = cauchystress(0, 2);
    }
    break;
    case Inpar::STR::stress_none:
      break;
    default:
      FOUR_C_THROW("requested stress type not available");
  }
  return;
}

FOUR_C_NAMESPACE_CLOSE