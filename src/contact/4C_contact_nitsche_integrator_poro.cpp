/*---------------------------------------------------------------------*/
/*! \file
\brief A class to perform integrations of nitsche related terms for the poro contact case

\level 3


*/
/*---------------------------------------------------------------------*/
#include "4C_contact_nitsche_integrator_poro.hpp"

#include "4C_contact_element.hpp"
#include "4C_contact_integrator.hpp"
#include "4C_contact_nitsche_utils.hpp"
#include "4C_contact_node.hpp"
#include "4C_contact_paramsinterface.hpp"
#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_mat_elasthyper.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_so3_base.hpp"
#include "4C_so3_hex8.hpp"
#include "4C_so3_poro.hpp"
#include "4C_solid_3D_ele.hpp"
#include "4C_utils_exceptions.hpp"

#include <Epetra_FEVector.h>
#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
CONTACT::IntegratorNitschePoro::IntegratorNitschePoro(
    Teuchos::ParameterList& params, Core::FE::CellType eletype, const Epetra_Comm& comm)
    : IntegratorNitsche(params, eletype, comm),
      no_penetration_(params.get<bool>("CONTACTNOPEN")),
      dv_dd_(params.get<double>("porotimefac"))
{
  if (fabs(theta_) > 1e-16) FOUR_C_THROW("Poro Contact just implemented Adjoint free ...");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::IntegratorNitschePoro::integrate_gp_3_d(Mortar::Element& sele, Mortar::Element& mele,
    Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& lmval,
    Core::LinAlg::SerialDenseVector& mval, Core::LinAlg::SerialDenseMatrix& sderiv,
    Core::LinAlg::SerialDenseMatrix& mderiv, Core::LinAlg::SerialDenseMatrix& lmderiv,
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap, double& wgt,
    double& jac, Core::Gen::Pairedvector<int, double>& derivjac, double* normal,
    std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit, double& gap,
    Core::Gen::Pairedvector<int, double>& deriv_gap, double* sxi, double* mxi,
    std::vector<Core::Gen::Pairedvector<int, double>>& derivsxi,
    std::vector<Core::Gen::Pairedvector<int, double>>& derivmxi)
{
  // TEUCHOS_FUNC_TIME_MONITOR("CONTACT::IntegratorNitsche::integrate_gp_3_d");
  // We use the consistent element normal for poro contact!
  // if (nit_normal_==Inpar::CONTACT::NitNor_ele)
  {
    double n[3];
    sele.compute_unit_normal_at_xi(sxi, n);
    std::vector<Core::Gen::Pairedvector<int, double>> dn(3, sele.num_node() * 3);
    dynamic_cast<CONTACT::Element&>(sele).DerivUnitNormalAtXi(sxi, dn);

    gpts_forces<3>(sele, mele, sval, sderiv, derivsxi, mval, mderiv, derivmxi, jac, derivjac, wgt,
        gap, deriv_gap, n, dn, sxi, mxi);
  }
  //  else if (nit_normal_==Inpar::CONTACT::NitNor_sm)
  //    FOUR_C_THROW("Want to use the element normal!");
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::IntegratorNitschePoro::integrate_gp_2_d(Mortar::Element& sele, Mortar::Element& mele,
    Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& lmval,
    Core::LinAlg::SerialDenseVector& mval, Core::LinAlg::SerialDenseMatrix& sderiv,
    Core::LinAlg::SerialDenseMatrix& mderiv, Core::LinAlg::SerialDenseMatrix& lmderiv,
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap, double& wgt,
    double& jac, Core::Gen::Pairedvector<int, double>& derivjac, double* normal,
    std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit, double& gap,
    Core::Gen::Pairedvector<int, double>& deriv_gap, double* sxi, double* mxi,
    std::vector<Core::Gen::Pairedvector<int, double>>& derivsxi,
    std::vector<Core::Gen::Pairedvector<int, double>>& derivmxi)
{
  FOUR_C_THROW("2D is not implemented!");
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void CONTACT::IntegratorNitschePoro::gpts_forces(Mortar::Element& sele, Mortar::Element& mele,
    const Core::LinAlg::SerialDenseVector& sval, const Core::LinAlg::SerialDenseMatrix& sderiv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dsxi,
    const Core::LinAlg::SerialDenseVector& mval, const Core::LinAlg::SerialDenseMatrix& mderiv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dmxi, const double jac,
    const Core::Gen::Pairedvector<int, double>& jacintcellmap, const double wgt, const double gap,
    const Core::Gen::Pairedvector<int, double>& dgapgp, const double* gpn,
    std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit, double* sxi, double* mxi)
{
  if (sele.Owner() != Comm_.MyPID()) return;

  static const bool do_fast_checks = true;
  // first rough check
  if (do_fast_checks)
    if (abs(theta_) < 1.e-12 && gap > std::max(sele.MaxEdgeSize(), mele.MaxEdgeSize())) return;

  const Core::LinAlg::Matrix<dim, 1> normal(gpn, true);

  if (dim != Dim()) FOUR_C_THROW("dimension inconsistency");

  Core::LinAlg::Matrix<dim, 1> slave_normal, master_normal;
  std::vector<Core::Gen::Pairedvector<int, double>> deriv_slave_normal(0, 0);
  std::vector<Core::Gen::Pairedvector<int, double>> deriv_master_normal(0, 0);
  sele.compute_unit_normal_at_xi(sxi, slave_normal.data());
  mele.compute_unit_normal_at_xi(mxi, master_normal.data());
  sele.DerivUnitNormalAtXi(sxi, deriv_slave_normal);
  mele.DerivUnitNormalAtXi(mxi, deriv_master_normal);

  double pen = ppn_;
  double pet = ppt_;

  double ws = 0.;
  double wm = 0.;
  CONTACT::UTILS::NitscheWeightsAndScaling(sele, mele, nit_wgt_, dt_, ws, wm, pen, pet);

  double cauchy_nn_weighted_average = 0.;
  Core::Gen::Pairedvector<int, double> cauchy_nn_weighted_average_deriv_d(
      sele.num_node() * 3 * 12 + sele.MoData().ParentDisp().size() +
      mele.MoData().ParentDisp().size());
  Core::Gen::Pairedvector<int, double> cauchy_nn_weighted_average_deriv_p(
      sele.MoData().ParentPFPres().size() + mele.MoData().ParentPFPres().size());

  so_ele_cauchy<dim>(sele, sxi, dsxi, wgt, slave_normal, deriv_slave_normal, normal, dnmap_unit, ws,
      cauchy_nn_weighted_average, cauchy_nn_weighted_average_deriv_d,
      cauchy_nn_weighted_average_deriv_p);
  so_ele_cauchy<dim>(mele, mxi, dmxi, wgt, master_normal, deriv_master_normal, normal, dnmap_unit,
      -wm, cauchy_nn_weighted_average, cauchy_nn_weighted_average_deriv_d,
      cauchy_nn_weighted_average_deriv_p);

  const double snn_av_pen_gap = cauchy_nn_weighted_average + pen * gap;
  Core::Gen::Pairedvector<int, double> d_snn_av_pen_gap(
      cauchy_nn_weighted_average_deriv_d.size() + dgapgp.size());
  for (const auto& p : cauchy_nn_weighted_average_deriv_d) d_snn_av_pen_gap[p.first] += p.second;
  for (const auto& p : dgapgp) d_snn_av_pen_gap[p.first] += pen * p.second;

  if (snn_av_pen_gap < 0.)
  {
    // test in normal contact direction
    integrate_test<dim>(-1., sele, sval, sderiv, dsxi, jac, jacintcellmap, wgt, snn_av_pen_gap,
        d_snn_av_pen_gap, cauchy_nn_weighted_average_deriv_p, normal, dnmap_unit);
    integrate_test<dim>(+1., mele, mval, mderiv, dmxi, jac, jacintcellmap, wgt, snn_av_pen_gap,
        d_snn_av_pen_gap, cauchy_nn_weighted_average_deriv_p, normal, dnmap_unit);

    integrate_poro_no_out_flow<dim>(
        -1, sele, sxi, sval, sderiv, jac, jacintcellmap, wgt, normal, dnmap_unit, mele, mval);
    integrate_poro_no_out_flow<dim>(
        +1, mele, mxi, mval, mderiv, jac, jacintcellmap, wgt, normal, dnmap_unit, sele, sval);
  }
}


template <int dim>
void CONTACT::IntegratorNitschePoro::so_ele_cauchy(Mortar::Element& moEle, double* boundary_gpcoord,
    std::vector<Core::Gen::Pairedvector<int, double>> boundary_gpcoord_lin, const double gp_wgt,
    const Core::LinAlg::Matrix<dim, 1>& normal,
    std::vector<Core::Gen::Pairedvector<int, double>>& normal_deriv,
    const Core::LinAlg::Matrix<dim, 1>& direction,
    std::vector<Core::Gen::Pairedvector<int, double>>& direction_deriv, const double w,
    double& cauchy_nt, Core::Gen::Pairedvector<int, double>& deriv_sigma_nt_d,
    Core::Gen::Pairedvector<int, double>& deriv_sigma_nt_p)
{
  Core::LinAlg::Matrix<dim, 1> pxsi(true);
  Core::LinAlg::Matrix<dim, dim> derivtravo_slave;
  CONTACT::UTILS::MapGPtoParent<dim>(moEle, boundary_gpcoord, gp_wgt, pxsi, derivtravo_slave);

  double sigma_nt;
  Core::LinAlg::SerialDenseMatrix dsntdd, dsntdp;
  Core::LinAlg::Matrix<dim, 1> dsntdn, dsntdt, dsntdpxi;

  if (!moEle.MoData().ParentPFPres().size())
  {
    // The element can be either an old so3 element or a new solid element
    if (auto* solid_ele = dynamic_cast<Discret::ELEMENTS::SoBase*>(moEle.parent_element());
        solid_ele != nullptr)
    {
      solid_ele->get_cauchy_n_dir_and_derivatives_at_xi(pxsi, moEle.MoData().ParentDisp(), normal,
          direction, sigma_nt, &dsntdd, nullptr, nullptr, nullptr, nullptr, &dsntdn, &dsntdt,
          &dsntdpxi, nullptr, nullptr, nullptr, nullptr, nullptr);
    }
    else if (auto* solid_ele = dynamic_cast<Discret::ELEMENTS::Solid*>(moEle.parent_element());
             solid_ele != nullptr)
    {
      Discret::ELEMENTS::CauchyNDirLinearizations<3> cauchy_linearizations{};
      cauchy_linearizations.d_cauchyndir_dd = &dsntdd;
      cauchy_linearizations.d_cauchyndir_dn = &dsntdn;
      cauchy_linearizations.d_cauchyndir_ddir = &dsntdt;
      cauchy_linearizations.d_cauchyndir_dxi = &dsntdpxi;

      sigma_nt = solid_ele->get_normal_cauchy_stress_at_xi<3>(
          moEle.MoData().ParentDisp(), pxsi, normal, direction, cauchy_linearizations);
    }
    else
    {
      FOUR_C_THROW("Unsupported solid element type");
    }
  }
  else
  {
    dynamic_cast<Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::SoHex8, Core::FE::CellType::hex8>*>(
        moEle.parent_element())
        ->get_cauchy_n_dir_and_derivatives_at_xi(pxsi, moEle.MoData().ParentDisp(),
            moEle.MoData().ParentPFPres(), normal, direction, sigma_nt, &dsntdd, &dsntdp, &dsntdn,
            &dsntdt, &dsntdpxi);
  }

  cauchy_nt += w * sigma_nt;

  for (int i = 0; i < moEle.parent_element()->num_node() * dim; ++i)
    deriv_sigma_nt_d[moEle.MoData().ParentDof().at(i)] += w * dsntdd(i, 0);

  for (int d = 0; d < dim; ++d)
  {
    for (Core::Gen::Pairedvector<int, double>::const_iterator p = normal_deriv[d].begin();
         p != normal_deriv[d].end(); ++p)
      deriv_sigma_nt_d[p->first] += dsntdn(d) * p->second * w;
  }

  for (int d = 0; d < dim; ++d)
  {
    for (Core::Gen::Pairedvector<int, double>::const_iterator p = direction_deriv[d].begin();
         p != direction_deriv[d].end(); ++p)
      deriv_sigma_nt_d[p->first] += dsntdt(d) * p->second * w;
  }

  if (moEle.MoData().ParentPFPres().size())
  {
    for (int i = 0; i < moEle.parent_element()->num_node(); ++i)
      deriv_sigma_nt_p[moEle.MoData().ParentPFDof()[i * (dim + 1) + 3]] += dsntdp(i, 0) * w;
  }
}

template <int dim>
void CONTACT::IntegratorNitschePoro::integrate_test(const double fac, Mortar::Element& ele,
    const Core::LinAlg::SerialDenseVector& shape, const Core::LinAlg::SerialDenseMatrix& deriv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dxi, const double jac,
    const Core::Gen::Pairedvector<int, double>& jacintcellmap, const double wgt,
    const double test_val, const Core::Gen::Pairedvector<int, double>& test_deriv_d,
    const Core::Gen::Pairedvector<int, double>& test_deriv_p,
    const Core::LinAlg::Matrix<dim, 1>& test_dir,
    const std::vector<Core::Gen::Pairedvector<int, double>>& test_dir_deriv)
{
  if (abs(fac) < 1.e-16) return;

  CONTACT::IntegratorNitsche::integrate_test<dim>(fac, ele, shape, deriv, dxi, jac, jacintcellmap,
      wgt, test_val, test_deriv_d, test_dir, test_dir_deriv);

  for (const auto& p : test_deriv_p)
  {
    double* row = ele.GetNitscheContainer().Kdp(p.first);
    for (int s = 0; s < ele.num_node(); ++s)
    {
      for (int d = 0; d < Dim(); ++d)
      {
        row[Core::FE::getParentNodeNumberFromFaceNodeNumber(
                ele.parent_element()->Shape(), ele.FaceParentNumber(), s) *
                dim +
            d] -= fac * jac * wgt * test_dir(d) * p.second * shape(s);
      }
    }
  }
}

template <int dim>
void CONTACT::IntegratorNitschePoro::integrate_poro_no_out_flow(const double fac,
    Mortar::Element& ele, double* xi, const Core::LinAlg::SerialDenseVector& shape,
    const Core::LinAlg::SerialDenseMatrix& deriv, const double jac,
    const Core::Gen::Pairedvector<int, double>& jacintcellmap, const double wgt,
    const Core::LinAlg::Matrix<dim, 1>& normal,
    const std::vector<Core::Gen::Pairedvector<int, double>>& normal_deriv,
    Mortar::Element& otherele, const Core::LinAlg::SerialDenseVector& othershape)
{
  if (abs(fac) < 1e-16) return;
  if (!no_penetration_) return;

  if (!ele.MoData().ParentPFDof().size()) return;

  // weighting for poro pressure depenent if two or onesided porocontact
  double sweight = 1;
  double oweight = 0;

  double spresgp = 0;
  double srelveln = 0;
  for (int j = 0; j < ele.num_node(); ++j)
  {
    int pj = Core::FE::getParentNodeNumberFromFaceNodeNumber(
        ele.parent_element()->Shape(), ele.FaceParentNumber(), j);
    spresgp += ele.MoData().ParentPFPres()[pj] * shape(j);
    for (int d = 0; d < dim; ++d)
    {
      srelveln +=
          (ele.MoData().ParentPFVel()[pj * dim + d] - ele.MoData().ParentVel()[pj * dim + d]) *
          shape(j) * normal(d);
    }
  }

  double sporosity = -1;
  double sJ = -1;
  std::map<int, double> sJLin;
  double sdphi_dp;
  double sdphi_dJ;
  get_poro_quantitiesat_gp(ele, xi, spresgp, sJ, sJLin, sporosity, sdphi_dp, sdphi_dJ);

  if (otherele.MoData().ParentPFDof().size())  // two sided poro contact case
  {
    sweight = 0.5;
    oweight = 1 - sweight;
  }

  double val = fac * jac * wgt / dv_dd_;  //*1./dv_dd_;
  for (int i = 0; i < ele.num_node(); ++i)
  {
    int pi = Core::FE::getParentNodeNumberFromFaceNodeNumber(
        ele.parent_element()->Shape(), ele.FaceParentNumber(), i);
    for (int j = 0; j < ele.num_node(); ++j)
    {
      int pj = Core::FE::getParentNodeNumberFromFaceNodeNumber(
          ele.parent_element()->Shape(), ele.FaceParentNumber(), j);
      for (int d = 0; d < dim; ++d)
      {
        (*ele.GetNitscheContainer().RhsP(pi * (dim + 1) + d)) +=
            shape(i) * ele.MoData().ParentPFPres()[pj] * shape(j) * normal(d) * val *
            sweight;  //(v,k1 p1 n)
        ele.GetNitscheContainer().Kpp(
            ele.MoData().ParentPFDof()[pj * (dim + 1) + dim])[(pi * (dim + 1) + d)] -=
            shape(i) * shape(j) * normal(d) * val * sweight;  //(v,k1 dp1/dp n)
        for (auto p = normal_deriv[d].begin(); p != normal_deriv[d].end(); ++p)
        {
          ele.GetNitscheContainer().Kpd(p->first)[(pi * (dim + 1) + d)] -=
              shape(i) * shape(j) * ele.MoData().ParentPFPres()[pj] * p->second * val *
              sweight;  //(v,k1 p1 dn/dd)

          ele.GetNitscheContainer().Kpd(p->first)[(pi * (dim + 1) + dim)] +=
              shape(i) * sporosity * p->second *
              (ele.MoData().ParentPFVel()[pj * dim + d] - ele.MoData().ParentVel()[pj * dim + d]) *
              shape(j) * val * sweight;  // (k1 q1, phi (vF-vS) dn/dd)
        }
        ele.GetNitscheContainer().Kpd(ele.MoData().ParentDof()[pj * dim + d])[(
            pi * (dim + 1) + dim)] -=  // (k1 q1, phi (vF-dvS/dd) n)
            shape(i) * sporosity * normal(d) * shape(j) * val * dv_dd_ * sweight;

        ele.GetNitscheContainer().Kpp(ele.MoData().ParentPFDof()[pj * (dim + 1) + d])[(
            pi * (dim + 1) + dim)] +=  // (k1 q1, phi (dvF/dvF-vS) n)
            shape(i) * sporosity * normal(d) * shape(j) * val * sweight;

        (*ele.GetNitscheContainer().RhsP(pi * (dim + 1) + (dim))) -=
            shape(i) * sporosity * normal(d) *
            (ele.MoData().ParentPFVel()[pj * dim + d] - ele.MoData().ParentVel()[pj * dim + d]) *
            shape(j) * val * sweight;  // (k1 q1, phi (vF-vS) n)
      }
      ele.GetNitscheContainer().Kpp(
          ele.MoData().ParentPFDof()[pj * (dim + 1) + dim])[(pi * (dim + 1) + dim)] +=
          val * srelveln * shape(i) * sdphi_dp * shape(j) * sweight;  // (k1 q1, phi/dp (vF-vS) n)
    }

    for (auto& dJit : sJLin)
    {
      ele.GetNitscheContainer().Kpd(dJit.first)[(pi * (dim + 1) + dim)] +=
          val * srelveln * shape(i) * sdphi_dJ * dJit.second *
          sweight;  // (k1 q1, phi/dd (vF-vS) n)
    }
  }

  if (oweight > 1e-16)
  {
    for (int i = 0; i < ele.num_node(); ++i)
    {
      int pi = Core::FE::getParentNodeNumberFromFaceNodeNumber(
          ele.parent_element()->Shape(), ele.FaceParentNumber(), i);
      for (int j = 0; j < otherele.num_node(); ++j)
      {
        int pj = Core::FE::getParentNodeNumberFromFaceNodeNumber(
            otherele.parent_element()->Shape(), otherele.FaceParentNumber(), j);
        for (int d = 0; d < dim; ++d)
        {
          (*ele.GetNitscheContainer().RhsP(pi * (dim + 1) + d)) +=
              shape(i) * otherele.MoData().ParentPFPres()[pj] * othershape(j) * normal(d) * val *
              oweight;  //(v,k2 p2 n)
          ele.GetNitscheContainer().Kpp(
              otherele.MoData().ParentPFDof()[pj * (dim + 1) + dim])[(pi * (dim + 1) + d)] -=
              shape(i) * othershape(j) * normal(d) * val * oweight;  //(v,k2 dp2/dp n)
          for (auto p = normal_deriv[d].begin(); p != normal_deriv[d].end(); ++p)
          {
            ele.GetNitscheContainer().Kpd(p->first)[(pi * (dim + 1) + d)] -=
                shape(i) * othershape(j) * otherele.MoData().ParentPFPres()[pj] * p->second * val *
                oweight;  //(v,k2 p2 dn/dd)

            otherele.GetNitscheContainer().Kpd(p->first)[(pj * (dim + 1) + dim)] +=
                othershape(j) * sporosity * p->second *
                (ele.MoData().ParentPFVel()[pi * dim + d] -
                    ele.MoData().ParentVel()[pi * dim + d]) *
                shape(i) * val * oweight;  // (k2 q2, phi (vF-vS) dn/dd)
          }

          otherele.GetNitscheContainer().Kpd(ele.MoData().ParentDof()[pi * dim + d])[(
              pj * (dim + 1) + dim)] -=  // (k2 q2, phi (vF-dvS/dd) n)
              othershape(j) * sporosity * normal(d) * shape(i) * val * dv_dd_ * oweight;

          otherele.GetNitscheContainer().Kpp(ele.MoData().ParentPFDof()[pi * (dim + 1) + d])[(
              pj * (dim + 1) + dim)] +=  // (k2 q2, phi (dvF/dvF-vS) n)
              othershape(j) * sporosity * normal(d) * shape(i) * val * oweight;

          (*otherele.GetNitscheContainer().RhsP(pj * (dim + 1) + (dim))) -=
              othershape(j) * sporosity * normal(d) *
              (ele.MoData().ParentPFVel()[pi * dim + d] - ele.MoData().ParentVel()[pi * dim + d]) *
              shape(i) * val * oweight;  // (k2 q2, phi (vF-vS) n)
        }
        otherele.GetNitscheContainer().Kpp(
            ele.MoData().ParentPFDof()[pi * (dim + 1) + dim])[(pj * (dim + 1) + dim)] +=
            val * srelveln * othershape(j) * sdphi_dp * shape(i) *
            oweight;  // (k2 q2, phi/dp (vF-vS) n)
      }
    }
    for (int j = 0; j < otherele.num_node(); ++j)
    {
      int pj = Core::FE::getParentNodeNumberFromFaceNodeNumber(
          otherele.parent_element()->Shape(), otherele.FaceParentNumber(), j);
      for (auto& dJit : sJLin)
      {
        otherele.GetNitscheContainer().Kpd(dJit.first)[(pj * (dim + 1) + dim)] +=
            val * srelveln * othershape(j) * sdphi_dJ * dJit.second *
            oweight;  // (k2 q2, phi/dd (vF-vS) n)
      }
    }
  }
}

bool CONTACT::IntegratorNitschePoro::get_poro_pressure(Mortar::Element& ele,
    const Core::LinAlg::SerialDenseVector& shape, Mortar::Element& otherele,
    const Core::LinAlg::SerialDenseVector& othershape, double& poropressure)
{
  if (!ele.MoData().ParentPFDof().size() && !otherele.MoData().ParentPFDof().size()) return false;

  double w1 = 1;
  if (ele.MoData().ParentPFDof().size() && otherele.MoData().ParentPFDof().size())
    w1 = 0.5;
  else if (ele.MoData().ParentPFDof().size())
    w1 = 1.0;
  else if (otherele.MoData().ParentPFDof().size())
    w1 = 0.0;
  else
    FOUR_C_THROW("Thats not exptected...!");
  double w2 = 1.0 - w1;

  poropressure = 0.0;
  if (ele.MoData().ParentPFDof().size())
  {
    for (int j = 0; j < ele.num_node(); ++j)
    {
      int pj = Core::FE::getParentNodeNumberFromFaceNodeNumber(
          ele.parent_element()->Shape(), ele.FaceParentNumber(), j);
      poropressure += w1 * ele.MoData().ParentPFPres()[pj] * shape(j);
    }
  }

  if (otherele.MoData().ParentPFDof().size())
  {
    for (int j = 0; j < otherele.num_node(); ++j)
    {
      int pj = Core::FE::getParentNodeNumberFromFaceNodeNumber(
          otherele.parent_element()->Shape(), otherele.FaceParentNumber(), j);
      poropressure += w2 * otherele.MoData().ParentPFPres()[pj] * othershape(j);
    }
  }
  return true;
}


void CONTACT::IntegratorNitschePoro::get_poro_quantitiesat_gp(Mortar::Element& ele, double* xi,
    double& spresgp,  //(in)
    double& sJ, std::map<int, double>& sJLin, double& sporosity, double& sdphi_dp,
    double& sdphi_dJ)  // out
{
  static double dummy = 1.0;
  sJ = det_deformation_gradient(ele, dummy, xi, sJLin);
  Teuchos::ParameterList sparams;  // empty parameter list;

  Teuchos::RCP<Mat::StructPoro> sstructmat =
      Teuchos::rcp_dynamic_cast<Mat::StructPoro>(ele.parent_element()->Material(0));
  if (sstructmat == Teuchos::null)
    sstructmat = Teuchos::rcp_dynamic_cast<Mat::StructPoro>(ele.parent_element()->Material(1));
  if (sstructmat == Teuchos::null) FOUR_C_THROW("Cast to StructPoro failed!");
  sstructmat->ComputeSurfPorosity(sparams, spresgp, sJ, ele.FaceParentNumber(), 1, sporosity,
      &sdphi_dp, &sdphi_dJ, nullptr, nullptr, nullptr, false);
}

template void CONTACT::IntegratorNitschePoro::integrate_test<2>(const double, Mortar::Element&,
    const Core::LinAlg::SerialDenseVector&, const Core::LinAlg::SerialDenseMatrix&,
    const std::vector<Core::Gen::Pairedvector<int, double>>& i, const double,
    const Core::Gen::Pairedvector<int, double>&, const double, const double,
    const Core::Gen::Pairedvector<int, double>&, const Core::Gen::Pairedvector<int, double>&,
    const Core::LinAlg::Matrix<2, 1>& test_dir,
    const std::vector<Core::Gen::Pairedvector<int, double>>& test_dir_deriv);
template void CONTACT::IntegratorNitschePoro::integrate_test<3>(const double, Mortar::Element&,
    const Core::LinAlg::SerialDenseVector&, const Core::LinAlg::SerialDenseMatrix&,
    const std::vector<Core::Gen::Pairedvector<int, double>>& i, const double,
    const Core::Gen::Pairedvector<int, double>&, const double, const double,
    const Core::Gen::Pairedvector<int, double>&, const Core::Gen::Pairedvector<int, double>&,
    const Core::LinAlg::Matrix<3, 1>& test_dir,
    const std::vector<Core::Gen::Pairedvector<int, double>>& test_dir_deriv);

template void CONTACT::IntegratorNitschePoro::integrate_poro_no_out_flow<3>(const double fac,
    Mortar::Element& ele, double* xi, const Core::LinAlg::SerialDenseVector& shape,
    const Core::LinAlg::SerialDenseMatrix& deriv, const double jac,
    const Core::Gen::Pairedvector<int, double>& jacintcellmap, const double wgt,
    const Core::LinAlg::Matrix<3, 1>& normal,
    const std::vector<Core::Gen::Pairedvector<int, double>>& normal_deriv,
    Mortar::Element& otherele, const Core::LinAlg::SerialDenseVector& othershape);

template void CONTACT::IntegratorNitschePoro::so_ele_cauchy<3>(Mortar::Element& moEle,
    double* boundary_gpcoord,
    std::vector<Core::Gen::Pairedvector<int, double>> boundary_gpcoord_lin, const double gp_wgt,
    const Core::LinAlg::Matrix<3, 1>& normal,
    std::vector<Core::Gen::Pairedvector<int, double>>& normal_deriv,
    const Core::LinAlg::Matrix<3, 1>& direction,
    std::vector<Core::Gen::Pairedvector<int, double>>& direction_deriv, const double w,
    double& cauchy_nt, Core::Gen::Pairedvector<int, double>& deriv_sigma_nt_d,
    Core::Gen::Pairedvector<int, double>& deriv_sigma_nt_p);

FOUR_C_NAMESPACE_CLOSE