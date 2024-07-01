/*---------------------------------------------------------------------*/
/*! \file
\brief a derived strategy handling the Lagrange multiplier based TSI contact

\level 3


*/
/*---------------------------------------------------------------------*/

#include "4C_contact_lagrange_strategy_tsi.hpp"

#include "4C_contact_defines.hpp"
#include "4C_contact_friction_node.hpp"
#include "4C_contact_interface.hpp"
#include "4C_contact_tsi_interface.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_inpar_contact.hpp"
#include "4C_inpar_thermo.hpp"
#include "4C_io.hpp"
#include "4C_linalg_matrixtransform.hpp"
#include "4C_linalg_multiply.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_mortar_utils.hpp"

#include <Epetra_SerialComm.h>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | ctor (public)                                             seitz 08/15|
 *----------------------------------------------------------------------*/
CONTACT::LagrangeStrategyTsi::LagrangeStrategyTsi(
    const Teuchos::RCP<CONTACT::AbstractStratDataContainer>& data_ptr,
    const Epetra_Map* dof_row_map, const Epetra_Map* NodeRowMap, Teuchos::ParameterList params,
    std::vector<Teuchos::RCP<CONTACT::Interface>> interface, int dim,
    Teuchos::RCP<const Epetra_Comm> comm, double alphaf, int maxdof)
    : LagrangeStrategy(
          data_ptr, dof_row_map, NodeRowMap, params, interface, dim, comm, alphaf, maxdof),
      tsi_alpha_(1.)
{
  return;
}

/*------------------------------------------------------------------------*
 | Assign general thermo contact state                         seitz 08/15|
 *------------------------------------------------------------------------*/
void CONTACT::LagrangeStrategyTsi::set_state(
    const enum Mortar::StateType& statetype, const Epetra_Vector& vec)
{
  switch (statetype)
  {
    case Mortar::state_temperature:
    {
      for (int j = 0; j < (int)interface_.size(); ++j)
      {
        Core::FE::Discretization& idiscr = interface_[j]->Discret();
        Teuchos::RCP<Epetra_Vector> global =
            Teuchos::rcp(new Epetra_Vector(*idiscr.DofColMap(), false));
        Core::LinAlg::Export(vec, *global);

        for (int i = 0; i < idiscr.NumMyColNodes(); ++i)
        {
          CONTACT::Node* node = dynamic_cast<CONTACT::Node*>(idiscr.lColNode(i));
          if (node == nullptr) FOUR_C_THROW("cast failed");
          std::vector<double> mytemp(1);
          std::vector<int> lm(1, node->Dofs()[0]);

          Core::FE::ExtractMyValues(*global, mytemp, lm);
          if (node->HasTSIData())  // in case the interface has not been initialized yet
            node->TSIData().Temp() = mytemp[0];
        }
      }
      break;
    }
    case Mortar::state_thermo_lagrange_multiplier:
    {
      for (int j = 0; j < (int)interface_.size(); ++j)
      {
        Core::FE::Discretization& idiscr = interface_[j]->Discret();

        Teuchos::RCP<Epetra_Vector> global =
            Teuchos::rcp(new Epetra_Vector(*idiscr.DofColMap(), false));
        Core::LinAlg::Export(vec, *global);

        for (int i = 0; i < idiscr.NumMyColNodes(); ++i)
        {
          CONTACT::Node* node = dynamic_cast<CONTACT::Node*>(idiscr.lColNode(i));
          std::vector<int> lm(1, node->Dofs()[0]);
          std::vector<double> myThermoLM(1, 0.);
          Core::FE::ExtractMyValues(*global, myThermoLM, lm);
          node->TSIData().ThermoLM() = myThermoLM[0];
        }
      }
      break;
    }
    default:
    {
      CONTACT::AbstractStrategy::set_state(statetype, vec);
      break;
    }
  }

  return;
}


void CONTACT::LagrangeStrategyTsi::evaluate(
    Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> sysmat,
    Teuchos::RCP<Epetra_Vector>& combined_RHS, Teuchos::RCP<Core::Adapter::Coupling> coupST,
    Teuchos::RCP<const Epetra_Vector> dis, Teuchos::RCP<const Epetra_Vector> temp)
{
  if (thr_s_dofs_ == Teuchos::null) thr_s_dofs_ = coupST->MasterToSlaveMap(gsdofrowmap_);

  // set the new displacements
  set_state(Mortar::state_new_displacement, *dis);

  for (unsigned i = 0; i < interface_.size(); ++i) interface_[i]->initialize();

  // set new temperatures
  Teuchos::RCP<Epetra_Vector> temp2 = coupST()->SlaveToMaster(temp);
  set_state(Mortar::state_temperature, *temp2);

  // error checks
  if (Core::UTILS::IntegralValue<Inpar::CONTACT::SystemType>(Params(), "SYSTEM") !=
      Inpar::CONTACT::system_condensed)
    FOUR_C_THROW("only condensed system implemented");

  // First, we need to evaluate all the interfaces
  initialize_mortar();                  // initialize mortar matrices and vectors
  initialize_and_evaluate_interface();  // evaluate mortar terms (integrate...)
  assemble_mortar();

  // get the relative movement for frictional contact
  evaluate_relative_movement();

  // update active set
  update_active_set_semi_smooth();

  // init lin-matrices
  initialize();

  // get the necessary maps on the thermo dofs
  Teuchos::RCP<Epetra_Map> gactive_themo_dofs = coupST->MasterToSlaveMap(gactivedofs_);
  Teuchos::RCP<Epetra_Map> master_thermo_dofs = coupST->MasterToSlaveMap(gmdofrowmap_);
  Teuchos::RCP<Epetra_Map> thr_act_dofs = coupST->MasterToSlaveMap(gactivedofs_);
  Teuchos::RCP<Epetra_Map> thr_m_dofs = coupST->MasterToSlaveMap(gmdofrowmap_);
  Teuchos::RCP<Epetra_Map> thr_sm_dofs = coupST->MasterToSlaveMap(gsmdofrowmap_);
  Teuchos::RCP<Epetra_Map> thr_all_dofs = Teuchos::rcp(new Epetra_Map(*coupST->SlaveDofMap()));

  // assemble the constraint lines for the active contact nodes
  Teuchos::RCP<Core::LinAlg::SparseMatrix> dcsdd = Teuchos::rcp(new Core::LinAlg::SparseMatrix(
      *gactivedofs_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX));
  Core::LinAlg::SparseMatrix dcsdT(
      *gactivedofs_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Teuchos::RCP<Core::LinAlg::SparseMatrix> dcsdLMc = Teuchos::rcp(new Core::LinAlg::SparseMatrix(
      *gactivedofs_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX));
  Teuchos::RCP<Epetra_Vector> rcsa = Core::LinAlg::CreateVector(*gactivedofs_, true);
  Teuchos::RCP<Epetra_Vector> g_all;
  if (constr_direction_ == Inpar::CONTACT::constr_xyz)
    g_all = Core::LinAlg::CreateVector(*gsdofrowmap_, true);
  else
    g_all = Core::LinAlg::CreateVector(*gsnoderowmap_, true);

  // assemble linearization of heat conduction (thermal contact)
  Core::LinAlg::SparseMatrix dcTdd(
      *gactivedofs_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::SparseMatrix dcTdT(
      *gactivedofs_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::SparseMatrix dcTdLMc(
      *gactivedofs_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::SparseMatrix dcTdLMt(
      *gactivedofs_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Teuchos::RCP<Epetra_Vector> rcTa = Core::LinAlg::CreateVector(*gactivedofs_, true);

  // D and M matrix for the active nodes
  Teuchos::RCP<Core::LinAlg::SparseMatrix> dInv =
      Teuchos::rcp(new Core::LinAlg::SparseMatrix(*gsdofrowmap_, 100, true, false));
  Core::LinAlg::SparseMatrix s(
      *gactivedofs_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);

  Core::LinAlg::SparseMatrix m_LinDissDISP(
      *gmdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::SparseMatrix m_LinDissContactLM(
      *gmdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);

  // setup some linearizations
  Core::LinAlg::SparseMatrix linDcontactLM(
      *gsdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::SparseMatrix linMcontactLM(
      *gmdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::SparseMatrix linMdiss(
      *gmdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::SparseMatrix linMThermoLM(
      *gmdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::SparseMatrix linDThermoLM(
      *gsdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::SparseMatrix m_LinDissContactLM_thrRow(
      *thr_m_dofs, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);

  // stick / slip linearization
  for (unsigned i = 0; i < interface_.size(); ++i)
  {
    CONTACT::TSIInterface* tsi_interface = dynamic_cast<CONTACT::TSIInterface*>(&(*interface_[i]));
    if (!tsi_interface) FOUR_C_THROW("in TSI contact, this should be a TSIInterface!");

    // linearized normal contact
    interface_[i]->AssembleS(s);
    interface_[i]->AssembleG(*g_all);

    // linearized tangential contact (friction)
    if (friction_)
    {
      tsi_interface->AssembleLinSlip(*dcsdLMc, *dcsdd, dcsdT, *rcsa);
      tsi_interface->AssembleLinStick(*dcsdLMc, *dcsdd, dcsdT, *rcsa);
    }
    else
    {
      tsi_interface->AssembleTN(dcsdLMc, Teuchos::null);
      tsi_interface->AssembleTNderiv(dcsdd, Teuchos::null);
      tsi_interface->AssembleTangrhs(*rcsa);
    }

    // linearized thermal contact (heat conduction)
    tsi_interface->AssembleLinConduct(dcTdd, dcTdT, dcTdLMt, dcTdLMc);

    tsi_interface->AssembleDM_linDiss(nullptr, &m_LinDissDISP, nullptr, &m_LinDissContactLM, 1.);

    tsi_interface->AssembleLinDM(linDcontactLM, linMcontactLM);
    tsi_interface->AssembleLinDM_X(
        nullptr, &linMdiss, 1., CONTACT::TSIInterface::LinDM_Diss, gsnoderowmap_);
    tsi_interface->AssembleLinDM_X(
        &linDThermoLM, &linMThermoLM, 1., CONTACT::TSIInterface::LinDM_ThermoLM, gsnoderowmap_);
  }

  // complete all those linearizations
  //                             colmap        rowmap
  linDcontactLM.Complete(*gsmdofrowmap_, *gsdofrowmap_);
  linMcontactLM.Complete(*gsmdofrowmap_, *gmdofrowmap_);
  m_LinDissDISP.Complete(*gsmdofrowmap_, *gmdofrowmap_);
  linMdiss.Complete(*gsmdofrowmap_, *gmdofrowmap_);
  linMThermoLM.Complete(*gsmdofrowmap_, *gmdofrowmap_);
  linDThermoLM.Complete(*gsmdofrowmap_, *gsdofrowmap_);
  m_LinDissContactLM.Complete(*gactivedofs_, *gmdofrowmap_);
  s.Complete(*gsmdofrowmap_, *gactivedofs_);

  dcsdd->Add(s, false, 1., -1.);
  dcsdLMc->Scale(-1.);
  dcsdT.Scale(-1.);
  rcsa->Scale(1.);

  // normal contact
  Teuchos::RCP<Epetra_Vector> gact;
  if (constr_direction_ == Inpar::CONTACT::constr_xyz)
  {
    gact = Core::LinAlg::CreateVector(*gactivedofs_, true);
    if (gact->GlobalLength()) Core::LinAlg::Export(*g_all, *gact);
  }
  else
  {
    gact = Core::LinAlg::CreateVector(*gactivenodes_, true);
    if (gact->GlobalLength())
    {
      Core::LinAlg::Export(*g_all, *gact);
      if (gact->ReplaceMap(*gactiven_)) FOUR_C_THROW("replaceMap went wrong");
    }
  }
  CONTACT::UTILS::add_vector(*gact, *rcsa);
  rcsa->Norm2(&mech_contact_res_);

  // complete all the new matrix blocks
  // Note: since the contact interace assemled them, they are all based
  //       on displacement row and col maps. Hence, some still need to be transformed
  dcsdd->Complete(*gsmdofrowmap_, *gactivedofs_);
  dcsdT.Complete(*gsmdofrowmap_, *gactivedofs_);
  dcsdLMc->Complete(*gactivedofs_, *gactivedofs_);
  dcTdd.Complete(*gsmdofrowmap_, *gactivedofs_);
  dcTdT.Complete(*gsmdofrowmap_, *gactivedofs_);
  dcTdLMc.Complete(*gactivedofs_, *gactivedofs_);

  // get the seperate blocks of the 2x2 TSI block system
  // View mode!!! Since we actually want to add things there
  Teuchos::RCP<Core::LinAlg::SparseMatrix> kss =
      Teuchos::rcp(new Core::LinAlg::SparseMatrix(sysmat->Matrix(0, 0), Core::LinAlg::Copy));
  Teuchos::RCP<Core::LinAlg::SparseMatrix> kst =
      Teuchos::rcp(new Core::LinAlg::SparseMatrix(sysmat->Matrix(0, 1), Core::LinAlg::Copy));
  Teuchos::RCP<Core::LinAlg::SparseMatrix> kts =
      Teuchos::rcp(new Core::LinAlg::SparseMatrix(sysmat->Matrix(1, 0), Core::LinAlg::Copy));
  Teuchos::RCP<Core::LinAlg::SparseMatrix> ktt =
      Teuchos::rcp(new Core::LinAlg::SparseMatrix(sysmat->Matrix(1, 1), Core::LinAlg::Copy));
  kss->UnComplete();
  kts->UnComplete();

  // split rhs
  Teuchos::RCP<Epetra_Vector> rs = Teuchos::rcp(new Epetra_Vector(*gdisprowmap_, true));
  Teuchos::RCP<Epetra_Vector> rt = Teuchos::rcp(new Epetra_Vector(*coupST->SlaveDofMap(), true));
  Core::LinAlg::Export(*combined_RHS, *rs);
  Core::LinAlg::Export(*combined_RHS, *rt);

  // we don't want the rhs but the residual
  rs->Scale(-1.);
  rt->Scale(-1.);

  // add last time step contact forces to rhs
  if (fscn_ != Teuchos::null)  // in the first time step, we don't have any history of the
                               // contact force, after that, fscn_ should be initialized propperly
  {
    Epetra_Vector tmp(*gdisprowmap_);
    Core::LinAlg::Export(*fscn_, tmp);
    if (rs->Update(alphaf_, tmp, 1.) != 0)  // fscn already scaled with alphaf_ in update
      FOUR_C_THROW("update went wrong");
  }

  if (ftcn_ != Teuchos::null)
  {
    Epetra_Vector tmp(*coupST->SlaveDofMap());
    Core::LinAlg::Export(*ftcn_, tmp);
    if (rt->Update((1. - tsi_alpha_), tmp, 1.) != 0) FOUR_C_THROW("update went wrong");
  }

  // map containing the inactive and non-contact structural dofs
  Teuchos::RCP<Epetra_Map> str_gni_dofs =
      Core::LinAlg::SplitMap(*Core::LinAlg::SplitMap(*gdisprowmap_, *gmdofrowmap_), *gactivedofs_);
  // map containing the inactive and non-contact thermal dofs
  Teuchos::RCP<Epetra_Map> thr_gni_dofs = coupST->MasterToSlaveMap(str_gni_dofs);

  // add to kss
  kss->Add(linDcontactLM, false, 1. - alphaf_, 1.);
  kss->Add(linMcontactLM, false, 1. - alphaf_, 1.);

  // transform and add to kts
  Core::LinAlg::MatrixRowTransform()(
      m_LinDissDISP, +tsi_alpha_, Core::Adapter::CouplingMasterConverter(*coupST), *kts, true);
  Core::LinAlg::MatrixRowTransform()(linMdiss,
      -tsi_alpha_,  // this minus sign is there, since assemble linM does not actually
      Core::Adapter::CouplingMasterConverter(*coupST), *kts,
      true);  // assemble the linearization of M but the negative linearization of M
  Core::LinAlg::MatrixRowTransform()(
      linMThermoLM, tsi_alpha_, Core::Adapter::CouplingMasterConverter(*coupST), *kts, true);
  Core::LinAlg::MatrixRowTransform()(
      linDThermoLM, tsi_alpha_, Core::Adapter::CouplingMasterConverter(*coupST), *kts, true);

  Core::LinAlg::MatrixRowTransform().operator()(m_LinDissContactLM, 1.,
      Core::Adapter::CouplingMasterConverter(*coupST), m_LinDissContactLM_thrRow, false);
  m_LinDissContactLM_thrRow.Complete(*gactivedofs_, *thr_m_dofs);

  // complete the matrix blocks again, now that we have added
  // the additional displacement linearizations
  kss->Complete();
  kts->Complete(*gdisprowmap_, *coupST->SlaveDofMap());

  // split matrix blocks in 3 rows: Active, Master and (Inactive+others)
  Teuchos::RCP<Core::LinAlg::SparseMatrix> kss_ni, kss_m, kss_a, kst_ni, kst_m, kst_a, kts_ni,
      kts_m, kts_a, ktt_ni, ktt_m, ktt_a, dummy1, dummy2, dummy3;

  // temporary matrix
  Teuchos::RCP<Core::LinAlg::SparseMatrix> tmp;
  Teuchos::RCP<Epetra_Vector> tmpv;

  // an empty dummy map
  Teuchos::RCP<Epetra_Map> dummy_map1, dummy_map2;

  // ****************************************************
  // split kss block*************************************
  // ****************************************************
  // split first row
  Core::LinAlg::SplitMatrix2x2(
      kss, str_gni_dofs, dummy_map1, gdisprowmap_, dummy_map2, kss_ni, dummy1, tmp, dummy2);

  // this shoud be a split in rows, so that two blocks should have zero columns
  if (dummy1->DomainMap().NumGlobalElements() != 0 || dummy2->DomainMap().NumGlobalElements() != 0)
    FOUR_C_THROW("this split should only split rows, no columns expected for this matrix blocks");

  // reset
  dummy1 = Teuchos::null;
  dummy2 = Teuchos::null;
  dummy_map1 = Teuchos::null;
  dummy_map2 = Teuchos::null;

  // split the remaining two rows
  Core::LinAlg::SplitMatrix2x2(
      tmp, gmdofrowmap_, dummy_map1, gdisprowmap_, dummy_map2, kss_m, dummy1, kss_a, dummy2);

  // this shoud be a split in rows, so that two blocks should have zero columns
  if (dummy1->DomainMap().NumGlobalElements() != 0 || dummy2->DomainMap().NumGlobalElements() != 0)
    FOUR_C_THROW("this split should only split rows, no columns expected for this matrix blocks");

  // reset
  dummy1 = Teuchos::null;
  dummy2 = Teuchos::null;
  dummy_map1 = Teuchos::null;
  dummy_map2 = Teuchos::null;
  tmp = Teuchos::null;
  // ****************************************************
  // split kss block*************************************
  // ****************************************************

  // ****************************************************
  // split kst block*************************************
  // ****************************************************
  // split first row
  Core::LinAlg::SplitMatrix2x2(
      kst, str_gni_dofs, dummy_map1, thr_all_dofs, dummy_map2, kst_ni, dummy1, tmp, dummy2);

  // this shoud be a split in rows, so that two blocks should have zero columns
  if (dummy1->DomainMap().NumGlobalElements() != 0 || dummy2->DomainMap().NumGlobalElements() != 0)
    FOUR_C_THROW("this split should only split rows, no columns expected for this matrix blocks");

  // reset
  dummy1 = Teuchos::null;
  dummy2 = Teuchos::null;
  dummy_map1 = Teuchos::null;
  dummy_map2 = Teuchos::null;

  // split the remaining two rows
  Core::LinAlg::SplitMatrix2x2(
      tmp, gmdofrowmap_, dummy_map1, thr_all_dofs, dummy_map2, kst_m, dummy1, kst_a, dummy2);

  // this shoud be a split in rows, so that two blocks should have zero columns
  if (dummy1->DomainMap().NumGlobalElements() != 0 || dummy2->DomainMap().NumGlobalElements() != 0)
    FOUR_C_THROW("this split should only split rows, no columns expected for this matrix blocks");

  // reset
  dummy1 = Teuchos::null;
  dummy2 = Teuchos::null;
  dummy_map1 = Teuchos::null;
  dummy_map2 = Teuchos::null;
  tmp = Teuchos::null;
  // ****************************************************
  // split kst block*************************************
  // ****************************************************

  // ****************************************************
  // split kts block*************************************
  // ****************************************************
  // split first row
  Core::LinAlg::SplitMatrix2x2(
      kts, thr_gni_dofs, dummy_map1, gdisprowmap_, dummy_map2, kts_ni, dummy1, tmp, dummy2);

  // this shoud be a split in rows, so that two blocks should have zero columns
  if (dummy1->DomainMap().NumGlobalElements() != 0 || dummy2->DomainMap().NumGlobalElements() != 0)
    FOUR_C_THROW("this split should only split rows, no columns expected for this matrix blocks");

  // reset
  dummy1 = Teuchos::null;
  dummy2 = Teuchos::null;
  dummy_map1 = Teuchos::null;
  dummy_map2 = Teuchos::null;

  // split the remaining two rows
  Core::LinAlg::SplitMatrix2x2(
      tmp, thr_m_dofs, dummy_map1, gdisprowmap_, dummy_map2, kts_m, dummy1, kts_a, dummy2);

  // this shoud be a split in rows, so that two blocks should have zero columns
  if (dummy1->DomainMap().NumGlobalElements() != 0 || dummy2->DomainMap().NumGlobalElements() != 0)
    FOUR_C_THROW("this split should only split rows, no columns expected for this matrix blocks");

  // reset
  dummy1 = Teuchos::null;
  dummy2 = Teuchos::null;
  dummy_map1 = Teuchos::null;
  dummy_map2 = Teuchos::null;
  tmp = Teuchos::null;
  // ****************************************************
  // split kts block*************************************
  // ****************************************************

  // ****************************************************
  // split ktt block*************************************
  // ****************************************************
  // split first row
  Core::LinAlg::SplitMatrix2x2(
      ktt, thr_gni_dofs, dummy_map1, thr_all_dofs, dummy_map2, ktt_ni, dummy1, tmp, dummy2);

  // this shoud be a split in rows, so that two blocks should have zero columns
  if (dummy1->DomainMap().NumGlobalElements() != 0 || dummy2->DomainMap().NumGlobalElements() != 0)
    FOUR_C_THROW("this split should only split rows, no columns expected for this matrix blocks");

  // reset
  dummy1 = Teuchos::null;
  dummy2 = Teuchos::null;
  dummy_map1 = Teuchos::null;
  dummy_map2 = Teuchos::null;

  // split the remaining two rows
  Core::LinAlg::SplitMatrix2x2(
      tmp, thr_m_dofs, dummy_map1, thr_all_dofs, dummy_map2, ktt_m, dummy1, ktt_a, dummy2);

  // this shoud be a split in rows, so that two blocks should have zero columns
  if (dummy1->DomainMap().NumGlobalElements() != 0 || dummy2->DomainMap().NumGlobalElements() != 0)
    FOUR_C_THROW("this split should only split rows, no columns expected for this matrix blocks");

  // reset
  dummy1 = Teuchos::null;
  dummy2 = Teuchos::null;
  dummy_map1 = Teuchos::null;
  dummy_map2 = Teuchos::null;
  tmp = Teuchos::null;
  // ****************************************************
  // split ktt block*************************************
  // ****************************************************

  // ****************************************************
  // split rhs vectors***********************************
  // ****************************************************
  // split structural rhs
  Epetra_Vector rsni(*str_gni_dofs);
  Core::LinAlg::Export(*rs, rsni);
  Epetra_Vector rsm(*gmdofrowmap_);
  Core::LinAlg::Export(*rs, rsm);
  Teuchos::RCP<Epetra_Vector> rsa = Teuchos::rcp(new Epetra_Vector(*gactivedofs_));
  Core::LinAlg::Export(*rs, *rsa);

  // split thermal rhs
  Epetra_Vector rtni(*thr_gni_dofs);
  Core::LinAlg::Export(*rt, rtni);
  Epetra_Vector rtm(*thr_m_dofs);
  Core::LinAlg::Export(*rt, rtm);
  Teuchos::RCP<Epetra_Vector> rta = Teuchos::rcp(new Epetra_Vector(*thr_act_dofs));
  Core::LinAlg::Export(*rt, *rta);
  // ****************************************************
  // split rhs vectors***********************************
  // ****************************************************

  // D and M matrix for the active nodes
  Teuchos::RCP<Core::LinAlg::SparseMatrix> dInvA =
      Teuchos::rcp(new Core::LinAlg::SparseMatrix(*gactivedofs_, 100, true, false));
  Teuchos::RCP<Core::LinAlg::SparseMatrix> mA =
      Teuchos::rcp(new Core::LinAlg::SparseMatrix(*gactivedofs_, 100, true, false));

  dummy_map1 = dummy_map2 = Teuchos::null;
  dummy1 = dummy2 = dummy3 = Teuchos::null;
  Core::LinAlg::SplitMatrix2x2(
      dmatrix_, gactivedofs_, dummy_map1, gactivedofs_, dummy_map2, dInvA, dummy1, dummy2, dummy3);
  dummy_map1 = dummy_map2 = Teuchos::null;
  dummy1 = dummy2 = dummy3 = Teuchos::null;
  Core::LinAlg::SplitMatrix2x2(
      mmatrix_, gactivedofs_, dummy_map1, gmdofrowmap_, dummy_map2, mA, dummy1, dummy2, dummy3);

  // now we have added the additional linearizations.
  // if there are no active nodes, we can leave now

  if (gactivenodes_->NumGlobalElements() == 0)
  {
    sysmat->reset();
    sysmat->Assign(0, 0, Core::LinAlg::Copy, *kss);
    sysmat->Assign(0, 1, Core::LinAlg::Copy, *kst);
    sysmat->Assign(1, 0, Core::LinAlg::Copy, *kts);
    sysmat->Assign(1, 1, Core::LinAlg::Copy, *ktt);
    return;
  }


  // we need to add another term, since AssembleLinStick/Slip assumes that we solve
  // for the Lagrange multiplier increments. However, we solve for the LM directly.
  // We can do that, since the system is linear in the LMs.
  tmpv = Teuchos::rcp(new Epetra_Vector(*gactivedofs_));
  Teuchos::RCP<Epetra_Vector> tmpv2 = Teuchos::rcp(new Epetra_Vector(*gactivedofs_));
  Core::LinAlg::Export(*z_, *tmpv2);
  dcsdLMc->Multiply(false, *tmpv2, *tmpv);
  tmpv->Scale(-1.);
  CONTACT::UTILS::add_vector(*tmpv, *rcsa);
  tmpv = Teuchos::null;
  tmpv2 = Teuchos::null;

  dcTdLMt.Complete(*gsdofrowmap_, *gactivedofs_);
  Core::LinAlg::SparseMatrix test(
      *gactivedofs_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Teuchos::RCP<Core::LinAlg::SparseMatrix> a1(&dcTdLMt, false);
  Teuchos::RCP<Core::LinAlg::SparseMatrix> a2;
  dummy_map1 = dummy_map2 = Teuchos::null;
  dummy1 = dummy2 = dummy3 = Teuchos::null;
  Core::LinAlg::SplitMatrix2x2(
      a1, gactivedofs_, dummy_map1, gactivedofs_, dummy_map2, a2, dummy1, dummy2, dummy3);
  dcTdLMt = *a2;

  dcTdLMt.Complete(*gactivedofs_, *gactivedofs_);
  dInvA->Complete(*gactivedofs_, *gactivedofs_);
  mA->Complete(*gmdofrowmap_, *gactivedofs_);

  Core::LinAlg::SparseMatrix dcTdLMc_thr(
      *thr_act_dofs, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::SparseMatrix dcTdLMt_thr(
      *thr_act_dofs, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::MatrixRowTransform()(
      dcTdLMc, 1., Core::Adapter::CouplingMasterConverter(*coupST), dcTdLMc_thr, true);
  Core::LinAlg::MatrixRowColTransform()(dcTdLMt, 1.,
      Core::Adapter::CouplingMasterConverter(*coupST),
      Core::Adapter::CouplingMasterConverter(*coupST), dcTdLMt_thr, true, false);
  dcTdLMc_thr.Complete(*gactivedofs_, *thr_act_dofs);
  dcTdLMt_thr.Complete(*thr_act_dofs, *thr_act_dofs);

  // invert D-matrix
  Epetra_Vector dDiag(*gactivedofs_);
  dInvA->ExtractDiagonalCopy(dDiag);
  if (dDiag.Reciprocal(dDiag)) FOUR_C_THROW("inversion of diagonal D matrix failed");
  dInvA->replace_diagonal_values(dDiag);

  // get dinv on thermal dofs
  Teuchos::RCP<Core::LinAlg::SparseMatrix> dInvaThr = Teuchos::rcp(new Core::LinAlg::SparseMatrix(
      *thr_act_dofs, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX));
  Core::LinAlg::MatrixRowColTransform()(*dInvA, 1., Core::Adapter::CouplingMasterConverter(*coupST),
      Core::Adapter::CouplingMasterConverter(*coupST), *dInvaThr, false, false);
  dInvaThr->Complete(*thr_act_dofs, *thr_act_dofs);

  // save some matrix blocks for recovery
  dinvA_ = dInvA;
  dinvAthr_ = dInvaThr;
  kss_a_ = kss_a;
  kst_a_ = kst_a;
  kts_a_ = kts_a;
  ktt_a_ = ktt_a;
  rs_a_ = rsa;
  rt_a_ = rta;
  thr_act_dofs_ = thr_act_dofs;

  // get dinv * M
  Teuchos::RCP<Core::LinAlg::SparseMatrix> dInvMa =
      Core::LinAlg::MLMultiply(*dInvA, false, *mA, false, false, false, true);

  // get dinv * M on the thermal dofs
  Core::LinAlg::SparseMatrix dInvMaThr(
      *thr_act_dofs, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::MatrixRowColTransform()(*dInvMa, 1.,
      Core::Adapter::CouplingMasterConverter(*coupST),
      Core::Adapter::CouplingMasterConverter(*coupST), dInvMaThr, false, false);
  dInvMaThr.Complete(*thr_m_dofs, *thr_act_dofs);

  // apply contact symmetry conditions
  if (constr_direction_ == Inpar::CONTACT::constr_xyz)
  {
    double haveDBC = 0;
    pgsdirichtoggle_->Norm1(&haveDBC);
    if (haveDBC > 0.)
    {
      Teuchos::RCP<Epetra_Vector> diag = Core::LinAlg::CreateVector(*gactivedofs_, true);
      dInvA->ExtractDiagonalCopy(*diag);
      Teuchos::RCP<Epetra_Vector> lmDBC = Core::LinAlg::CreateVector(*gactivedofs_, true);
      Core::LinAlg::Export(*pgsdirichtoggle_, *lmDBC);
      Teuchos::RCP<Epetra_Vector> tmp = Core::LinAlg::CreateVector(*gactivedofs_, true);
      tmp->Multiply(1., *diag, *lmDBC, 0.);
      diag->Update(-1., *tmp, 1.);
      dInvA->replace_diagonal_values(*diag);
      dInvMa = Core::LinAlg::MLMultiply(*dInvA, false, *mA, false, false, false, true);
    }
  }

  // reset the tangent stiffness
  // (for the condensation we have constructed copies above)
  sysmat->reset();
  sysmat->UnComplete();

  // need diagonal block kss with explicitdirichtlet_=true
  // to be able to apply dirichlet values for contact symmetry condition
  Core::LinAlg::SparseMatrix tmpkss(
      *gdisprowmap_, 100, false, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  sysmat->Assign(0, 0, Core::LinAlg::Copy, tmpkss);

  // get references to the blocks (just for convenience)
  Core::LinAlg::SparseMatrix& kss_new = sysmat->Matrix(0, 0);
  Core::LinAlg::SparseMatrix& kst_new = sysmat->Matrix(0, 1);
  Core::LinAlg::SparseMatrix& kts_new = sysmat->Matrix(1, 0);
  Core::LinAlg::SparseMatrix& ktt_new = sysmat->Matrix(1, 1);

  // reset rhs
  combined_RHS->PutScalar(0.0);

  // **********************************************************************
  // **********************************************************************
  // BUILD CONDENSED SYSTEM
  // **********************************************************************
  // **********************************************************************

  // (1) add the blocks, we do nothing with (i.e. (Inactive+others))
  kss_new.Add(*kss_ni, false, 1., 1.);
  kst_new.Add(*kst_ni, false, 1., 1.);
  kts_new.Add(*kts_ni, false, 1., 1.);
  ktt_new.Add(*ktt_ni, false, 1., 1.);
  CONTACT::UTILS::add_vector(rsni, *combined_RHS);
  CONTACT::UTILS::add_vector(rtni, *combined_RHS);

  // (2) add the 'uncondensed' blocks (i.e. everything w/o a D^-1
  // (2)a actual stiffness blocks of the master-rows
  kss_new.Add(*kss_m, false, 1., 1.);
  kst_new.Add(*kst_m, false, 1., 1.);
  kts_new.Add(*kts_m, false, 1., 1.);
  ktt_new.Add(*ktt_m, false, 1., 1.);
  CONTACT::UTILS::add_vector(rsm, *combined_RHS);
  CONTACT::UTILS::add_vector(rtm, *combined_RHS);

  // (2)b active constraints in the active slave rows
  kss_new.Add(*dcsdd, false, 1., 1.);

  Core::LinAlg::MatrixColTransform()(*gactivedofs_, *gsmdofrowmap_, dcsdT, 1.,
      Core::Adapter::CouplingMasterConverter(*coupST), kst_new, false, true);
  Core::LinAlg::MatrixRowTransform()(
      dcTdd, 1., Core::Adapter::CouplingMasterConverter(*coupST), kts_new, true);
  Core::LinAlg::MatrixRowColTransform()(dcTdT, 1., Core::Adapter::CouplingMasterConverter(*coupST),
      Core::Adapter::CouplingMasterConverter(*coupST), ktt_new, true, true);
  CONTACT::UTILS::add_vector(*rcsa, *combined_RHS);

  // (3) condensed parts
  // second row
  kss_new.Add(
      *Core::LinAlg::MLMultiply(*dInvMa, true, *kss_a, false, false, false, true), false, 1., 1.);
  kst_new.Add(
      *Core::LinAlg::MLMultiply(*dInvMa, true, *kst_a, false, false, false, true), false, 1., 1.);
  tmpv = Teuchos::rcp(new Epetra_Vector(*gmdofrowmap_));
  dInvMa->Multiply(true, *rsa, *tmpv);
  CONTACT::UTILS::add_vector(*tmpv, *combined_RHS);
  tmpv = Teuchos::null;

  // third row
  Teuchos::RCP<Core::LinAlg::SparseMatrix> wDinv =
      Core::LinAlg::MLMultiply(*dcsdLMc, false, *dInvA, true, false, false, true);
  kss_new.Add(*Core::LinAlg::MLMultiply(*wDinv, false, *kss_a, false, false, false, true), false,
      -1. / (1. - alphaf_), 1.);
  kst_new.Add(*Core::LinAlg::MLMultiply(*wDinv, false, *kst_a, false, false, false, true), false,
      -1. / (1. - alphaf_), 1.);
  tmpv = Teuchos::rcp(new Epetra_Vector(*gactivedofs_));
  wDinv->Multiply(false, *rsa, *tmpv);
  tmpv->Scale(-1. / (1. - alphaf_));
  CONTACT::UTILS::add_vector(*tmpv, *combined_RHS);
  tmpv = Teuchos::null;
  wDinv = Teuchos::null;

  // fourth row: no condensation. Terms already added in (1)

  // fifth row
  tmp = Teuchos::null;
  tmp =
      Core::LinAlg::MLMultiply(m_LinDissContactLM_thrRow, false, *dInvA, false, false, false, true);
  kts_new.Add(*Core::LinAlg::MLMultiply(*tmp, false, *kss_a, false, false, false, true), false,
      -tsi_alpha_ / (1. - alphaf_), 1.);
  ktt_new.Add(*Core::LinAlg::MLMultiply(*tmp, false, *kst_a, false, false, false, true), false,
      -tsi_alpha_ / (1. - alphaf_), 1.);
  tmpv = Teuchos::rcp(new Epetra_Vector(*thr_m_dofs));
  tmp->Multiply(false, *rsa, *tmpv);
  tmpv->Scale(-tsi_alpha_ / (1. - alphaf_));
  CONTACT::UTILS::add_vector(*tmpv, *combined_RHS);
  tmpv = Teuchos::null;

  kts_new.Add(
      *Core::LinAlg::MLMultiply(dInvMaThr, true, *kts_a, false, false, false, true), false, 1., 1.);
  ktt_new.Add(
      *Core::LinAlg::MLMultiply(dInvMaThr, true, *ktt_a, false, false, false, true), false, 1., 1.);
  tmpv = Teuchos::rcp(new Epetra_Vector(*thr_m_dofs));
  dInvMaThr.Multiply(true, *rta, *tmpv);
  CONTACT::UTILS::add_vector(*tmpv, *combined_RHS);
  tmp = Teuchos::null;

  // sixth row
  Teuchos::RCP<Core::LinAlg::SparseMatrix> yDinv =
      Core::LinAlg::MLMultiply(dcTdLMc_thr, false, *dInvA, false, false, false, true);
  kts_new.Add(*Core::LinAlg::MLMultiply(*yDinv, false, *kss_a, false, false, false, true), false,
      -1. / (1. - alphaf_), 1.);
  ktt_new.Add(*Core::LinAlg::MLMultiply(*yDinv, false, *kst_a, false, false, false, true), false,
      -1. / (1. - alphaf_), 1.);
  tmpv = Teuchos::rcp(new Epetra_Vector(*thr_act_dofs));
  yDinv->Multiply(false, *rsa, *tmpv);
  tmpv->Scale(-1. / (1. - alphaf_));
  CONTACT::UTILS::add_vector(*tmpv, *combined_RHS);
  tmpv = Teuchos::null;

  Teuchos::RCP<Core::LinAlg::SparseMatrix> gDinv =
      Core::LinAlg::MLMultiply(dcTdLMt_thr, false, *dInvaThr, false, false, false, true);
  kts_new.Add(*Core::LinAlg::MLMultiply(*gDinv, false, *kts_a, false, false, false, true), false,
      -1. / (tsi_alpha_), 1.);
  ktt_new.Add(*Core::LinAlg::MLMultiply(*gDinv, false, *ktt_a, false, false, false, true), false,
      -1. / (tsi_alpha_), 1.);
  tmpv = Teuchos::rcp(new Epetra_Vector(*thr_act_dofs));
  gDinv->Multiply(false, *rta, *tmpv);
  tmpv->Scale(-1. / tsi_alpha_);
  CONTACT::UTILS::add_vector(*tmpv, *combined_RHS);

  // and were done with the system matrix
  sysmat->Complete();

  // we need to return the rhs, not the residual
  combined_RHS->Scale(-1.);

  return;
}


void CONTACT::UTILS::add_vector(Epetra_Vector& src, Epetra_Vector& dst)
{
  // return if src has no elements
  if (src.GlobalLength() == 0) return;

#ifdef FOUR_C_ENABLE_ASSERTIONS
  for (int i = 0; i < src.Map().NumMyElements(); ++i)
    if ((dst.Map().LID(src.Map().GID(i))) < 0)
      FOUR_C_THROW("src is not a vector on a sub-map of dst");
#endif

  Epetra_Vector tmp = Epetra_Vector(dst.Map(), true);
  Core::LinAlg::Export(src, tmp);
  if (dst.Update(1., tmp, 1.)) FOUR_C_THROW("vector update went wrong");
  return;
}

void CONTACT::LagrangeStrategyTsi::RecoverCoupled(Teuchos::RCP<Epetra_Vector> sinc,
    Teuchos::RCP<Epetra_Vector> tinc, Teuchos::RCP<Core::Adapter::Coupling> coupST)
{
  Teuchos::RCP<Epetra_Vector> z_old = Teuchos::null;
  if (z_ != Teuchos::null) z_old = Teuchos::rcp(new Epetra_Vector(*z_));
  Teuchos::RCP<Epetra_Vector> z_thr_old = Teuchos::null;
  if (z_thr_ != Teuchos::null) z_thr_old = Teuchos::rcp(new Epetra_Vector(*z_thr_));

  // recover contact LM
  if (gactivedofs_->NumGlobalElements() > 0)
  {
    // do we have everything we need?
    if (rs_a_ == Teuchos::null || kss_a_ == Teuchos::null || kst_a_ == Teuchos::null ||
        dinvA_ == Teuchos::null)
      FOUR_C_THROW("some data for LM recovery is missing");

    Epetra_Vector lmc_a_new(*gactivedofs_, false);
    Epetra_Vector tmp(*gactivedofs_, false);
    lmc_a_new.Update(1., *rs_a_, 0.);
    kss_a_->Multiply(false, *sinc, tmp);
    lmc_a_new.Update(1., tmp, 1.);
    kst_a_->Multiply(false, *tinc, tmp);
    lmc_a_new.Update(1., tmp, 1.);
    dinvA_->Multiply(false, lmc_a_new, tmp);
    tmp.Scale(-1. / (1. - alphaf_));
    z_ = Teuchos::rcp(new Epetra_Vector(*gsdofrowmap_));
    Core::LinAlg::Export(tmp, *z_);

    // recover thermo LM
    // do we have everything we need?
    if (rt_a_ == Teuchos::null || kts_a_ == Teuchos::null || ktt_a_ == Teuchos::null ||
        dinvAthr_ == Teuchos::null)
      FOUR_C_THROW("some data for LM recovery is missing");

    Epetra_Vector lmt_a_new(*thr_act_dofs_, false);
    Epetra_Vector tmp2(*thr_act_dofs_, false);
    lmt_a_new.Update(1., *rt_a_, 0.);
    kts_a_->Multiply(false, *sinc, tmp2);
    lmt_a_new.Update(1., tmp2, 1.);
    ktt_a_->Multiply(false, *tinc, tmp2);
    lmt_a_new.Update(1., tmp2, 1.);
    dinvAthr_->Multiply(false, lmt_a_new, tmp2);
    tmp2.Scale(-1. / (tsi_alpha_));
    z_thr_ = Teuchos::rcp(new Epetra_Vector(*thr_s_dofs_));
    Core::LinAlg::Export(tmp2, *z_thr_);
  }

  else
  {
    z_ = Teuchos::rcp(new Epetra_Vector(*gsdofrowmap_));
    z_thr_ = Teuchos::rcp(new Epetra_Vector(*thr_s_dofs_));
  }

  if (z_old != Teuchos::null)
  {
    z_old->Update(-1., *z_, 1.);
    z_old->Norm2(&mech_contact_incr_);
  }
  if (z_thr_old != Teuchos::null)
  {
    z_thr_old->Update(-1., *z_thr_, 1.);
    z_thr_old->Norm2(&thr_contact_incr_);
  }

  // store updated LM into nodes
  store_nodal_quantities(Mortar::StrategyBase::lmupdate, Teuchos::null);
  store_nodal_quantities(Mortar::StrategyBase::lmThermo, coupST);

  return;
};

void CONTACT::LagrangeStrategyTsi::store_nodal_quantities(
    Mortar::StrategyBase::QuantityType type, Teuchos::RCP<Core::Adapter::Coupling> coupST)
{
  Teuchos::RCP<Epetra_Vector> vectorglobal = Teuchos::null;
  // start type switch
  switch (type)
  {
    case Mortar::StrategyBase::lmThermo:
    {
      Teuchos::RCP<Epetra_Vector> tmp = Teuchos::rcp(new Epetra_Vector(*coupST->SlaveDofMap()));

      Core::LinAlg::Export(*z_thr_, *tmp);
      vectorglobal = z_thr_;
      vectorglobal = coupST->SlaveToMaster(tmp);
      Teuchos::RCP<Epetra_Map> sdofmap, snodemap;
      // loop over all interfaces
      for (int i = 0; i < (int)interface_.size(); ++i)
      {
        sdofmap = interface_[i]->SlaveColDofs();
        snodemap = interface_[i]->SlaveColNodes();
        Teuchos::RCP<Epetra_Vector> vectorinterface = Teuchos::null;
        vectorinterface = Teuchos::rcp(new Epetra_Vector(*sdofmap));
        if (vectorglobal != Teuchos::null) Core::LinAlg::Export(*vectorglobal, *vectorinterface);

        // loop over all slave nodes (column or row) on the current interface
        for (int j = 0; j < snodemap->NumMyElements(); ++j)
        {
          int gid = snodemap->GID(j);
          Core::Nodes::Node* node = interface_[i]->Discret().gNode(gid);
          if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
          Node* cnode = dynamic_cast<Node*>(node);

          cnode->TSIData().ThermoLM() =
              (*vectorinterface)[(vectorinterface->Map()).LID(cnode->Dofs()[0])];
        }
      }
      break;
    }
    default:
      CONTACT::AbstractStrategy::store_nodal_quantities(type);
      break;
  }
}

void CONTACT::LagrangeStrategyTsi::Update(Teuchos::RCP<const Epetra_Vector> dis)
{
  if (fscn_ == Teuchos::null) fscn_ = Teuchos::rcp(new Epetra_Vector(*gsmdofrowmap_));
  fscn_->PutScalar(0.0);

  if (ftcnp_ == Teuchos::null)
    ftcnp_ = Teuchos::rcp(new Epetra_Vector(*coupST_->MasterToSlaveMap(gsmdofrowmap_)));
  ftcnp_->PutScalar(0.0);

  Teuchos::RCP<Epetra_Vector> tmp = Teuchos::rcp(new Epetra_Vector(*gsdofrowmap_));
  dmatrix_->Multiply(false, *z_, *tmp);
  CONTACT::UTILS::add_vector(*tmp, *fscn_);

  tmp = Teuchos::rcp(new Epetra_Vector(*gmdofrowmap_));
  mmatrix_->Multiply(true, *z_, *tmp);
  tmp->Scale(-1.);
  CONTACT::UTILS::add_vector(*tmp, *fscn_);

  CONTACT::AbstractStrategy::Update(dis);

  Core::LinAlg::SparseMatrix dThr(*coupST_->MasterToSlaveMap(gsdofrowmap_), 100, true, false,
      Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::MatrixRowColTransform()(*dmatrix_, 1.,
      Core::Adapter::CouplingMasterConverter(*coupST_),
      Core::Adapter::CouplingMasterConverter(*coupST_), dThr, false, false);
  dThr.Complete();
  tmp = Teuchos::rcp(new Epetra_Vector(*coupST_->MasterToSlaveMap(gsdofrowmap_)));
  if (dThr.Apply(*z_thr_, *tmp) != 0) FOUR_C_THROW("apply went wrong");
  CONTACT::UTILS::add_vector(*tmp, *ftcnp_);

  Core::LinAlg::SparseMatrix mThr(*coupST_->MasterToSlaveMap(gsdofrowmap_), 100, true, false,
      Core::LinAlg::SparseMatrix::FE_MATRIX);
  Core::LinAlg::MatrixRowColTransform()(*mmatrix_, 1.,
      Core::Adapter::CouplingMasterConverter(*coupST_),
      Core::Adapter::CouplingMasterConverter(*coupST_), mThr, false, false);
  mThr.Complete(*coupST_->MasterToSlaveMap(gmdofrowmap_), *coupST_->MasterToSlaveMap(gsdofrowmap_));
  mThr.UseTranspose();
  tmp = Teuchos::rcp(new Epetra_Vector(*coupST_->MasterToSlaveMap(gmdofrowmap_)));
  if (mThr.Multiply(true, *z_thr_, *tmp) != 0) FOUR_C_THROW("multiply went wrong");
  tmp->Scale(-1.);
  CONTACT::UTILS::add_vector(*tmp, *ftcnp_);

  Core::LinAlg::SparseMatrix m_LinDissContactLM(
      *gmdofrowmap_, 100, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  for (unsigned i = 0; i < interface_.size(); ++i)
    dynamic_cast<CONTACT::TSIInterface*>(&(*interface_[i]))
        ->AssembleDM_linDiss(nullptr, nullptr, nullptr, &m_LinDissContactLM, 1.);
  m_LinDissContactLM.Complete(*gactivedofs_, *gmdofrowmap_);
  Teuchos::RCP<Epetra_Vector> z_act = Teuchos::rcp(new Epetra_Vector(*gactivedofs_));
  Core::LinAlg::Export(*z_, *z_act);
  tmp = Teuchos::rcp(new Epetra_Vector(*gmdofrowmap_));
  if (m_LinDissContactLM.Multiply(false, *z_act, *tmp) != 0) FOUR_C_THROW("multiply went wrong");
  Teuchos::RCP<Epetra_Vector> tmp2 = Teuchos::rcp(new Epetra_Vector(*coupST_->MasterDofMap()));
  Core::LinAlg::Export(*tmp, *tmp2);
  Teuchos::RCP<Epetra_Vector> tmp3 = coupST_->MasterToSlave(tmp2);
  Teuchos::RCP<Epetra_Vector> tmp4 =
      Teuchos::rcp(new Epetra_Vector(*coupST_->MasterToSlaveMap(gmdofrowmap_)));
  Core::LinAlg::Export(*tmp3, *tmp4);
  CONTACT::UTILS::add_vector(*tmp4, *ftcnp_);

  ftcn_ = ftcnp_;
}

void CONTACT::LagrangeStrategyTsi::SetAlphafThermo(const Teuchos::ParameterList& tdyn)
{
  Inpar::THR::DynamicType dyn_type =
      Core::UTILS::IntegralValue<Inpar::THR::DynamicType>(tdyn, "DYNAMICTYP");
  switch (dyn_type)
  {
    case Inpar::THR::dyna_genalpha:
      tsi_alpha_ = tdyn.sublist("GENALPHA").get<double>("ALPHA_F");
      break;
    case Inpar::THR::dyna_onesteptheta:
      tsi_alpha_ = tdyn.sublist("ONESTEPTHETA").get<double>("THETA");
      break;
    case Inpar::THR::dyna_statics:
      tsi_alpha_ = 1.;
      break;
    default:
      FOUR_C_THROW("unknown thermal time integration type");
  }
  return;
}


/*----------------------------------------------------------------------*
 |  write restart information for contact                     popp 03/08|
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategyTsi::DoWriteRestart(
    std::map<std::string, Teuchos::RCP<Epetra_Vector>>& restart_vectors, bool forcedrestart) const
{
  CONTACT::AbstractStrategy::DoWriteRestart(restart_vectors, forcedrestart);

  if (fscn_ != Teuchos::null)
  {
    Teuchos::RCP<Epetra_Vector> tmp = Teuchos::rcp(new Epetra_Vector(*gsmdofrowmap_));
    Core::LinAlg::Export(*fscn_, *tmp);
    restart_vectors["last_contact_force"] = tmp;
  }
  if (ftcn_ != Teuchos::null)
  {
    Teuchos::RCP<Epetra_Vector> tmp = Teuchos::rcp(new Epetra_Vector(*coupST_->SlaveDofMap()));
    Core::LinAlg::Export(*ftcn_, *tmp);
    restart_vectors["last_thermo_force"] = coupST_->SlaveToMaster(tmp);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::LagrangeStrategyTsi::DoReadRestart(Core::IO::DiscretizationReader& reader,
    Teuchos::RCP<const Epetra_Vector> dis, Teuchos::RCP<CONTACT::ParamsInterface> cparams_ptr)
{
  bool restartwithcontact = Core::UTILS::IntegralValue<int>(Params(), "RESTART_WITH_CONTACT");

  CONTACT::AbstractStrategy::DoReadRestart(reader, dis);
  fscn_ = Teuchos::rcp(new Epetra_Vector(*gsmdofrowmap_));
  if (!restartwithcontact) reader.read_vector(fscn_, "last_contact_force");

  Teuchos::RCP<Epetra_Vector> tmp = Teuchos::rcp(new Epetra_Vector(*coupST_->MasterDofMap()));
  if (!restartwithcontact) reader.read_vector(tmp, "last_thermo_force");
  ftcn_ = coupST_->MasterToSlave(tmp);
  tmp = Teuchos::rcp(new Epetra_Vector(*coupST_->MasterToSlaveMap(gsmdofrowmap_)));
  Core::LinAlg::Export(*ftcn_, *tmp);
  ftcn_ = tmp;
}

FOUR_C_NAMESPACE_CLOSE