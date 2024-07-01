/*-----------------------------------------------------------*/
/*! \file

\brief Utility routines for the structural non-linear solver classes.


\level 3

*/
/*-----------------------------------------------------------*/

#include "4C_structure_new_nln_solver_utils.hpp"

#include "4C_global_data.hpp"
#include "4C_inpar_contact.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_XMLParameterListCoreHelpers.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::Nln::SOLVER::is_xml_status_test_file(const Teuchos::ParameterList& pstatus)
{
  bool check = false;

  Teuchos::ParameterList pxml;
  const std::string& xmlfilename = pstatus.get<std::string>("XML File");

  // check the input: path to the "Status Test" xml-file
  if (xmlfilename != "none")
  {
    if (xmlfilename.length() and xmlfilename.rfind(".xml"))
    {
      pxml = *(Teuchos::getParametersFromXmlFile(xmlfilename));
      if (pxml.isSublist("Outer Status Test"))
        if (pxml.sublist("Outer Status Test").numParams() > 0) check = true;
    }
    else
      FOUR_C_THROW("The file name \"%s\" is not a valid XML file name.", xmlfilename.c_str());
  }

  return check;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::Nln::SOLVER::create_quantity_types(
    std::set<enum NOX::Nln::StatusTest::QuantityType>& qtypes,
    const STR::TimeInt::BaseDataSDyn& datasdyn)
{
  // ---------------------------------------------------------------------------
  // get the model types
  // ---------------------------------------------------------------------------
  const std::set<enum Inpar::STR::ModelType>& mtypes = datasdyn.GetModelTypes();
  std::vector<enum NOX::Nln::StatusTest::QuantityType> qt_vec;

  std::set<enum Inpar::STR::ModelType>::const_iterator miter;
  for (miter = mtypes.begin(); miter != mtypes.end(); ++miter)
  {
    convert_model_type_to_quantity_type(*miter, qt_vec);

    // check if the corresponding enum could be found.
    // Note: We do not throw an error, since there are model types,
    //       which have no representation in the quantity list
    //       (e.g. spring dashpot).
    if (qt_vec.size() == 0) continue;

    qtypes.insert(qt_vec.begin(), qt_vec.end());
  }
  // ---------------------------------------------------------------------------
  // get the element technologies
  // ---------------------------------------------------------------------------
  const std::set<enum Inpar::STR::EleTech>& eletechs = datasdyn.get_element_technologies();

  std::set<enum Inpar::STR::EleTech>::const_iterator etiter;
  for (etiter = eletechs.begin(); etiter != eletechs.end(); ++etiter)
  {
    convert_ele_tech_to_quantity_type(*etiter, qt_vec);

    // check if the corresponding enum could be found.
    // Note: We do not throw an error, since there are element technologies,
    //       which have no representation in the quantity list
    //       (e.g. fbar).
    if (qt_vec.size() == 0) continue;

    qtypes.insert(qt_vec.begin(), qt_vec.end());
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::Nln::SOLVER::convert_model_type_to_quantity_type(
    const enum Inpar::STR::ModelType& mt, std::vector<enum NOX::Nln::StatusTest::QuantityType>& qt)
{
  // clear quantity type vector
  qt.clear();

  switch (mt)
  {
    // --- Structural case -----------------------------------------------------
    case Inpar::STR::model_structure:
    {
      qt.push_back(NOX::Nln::StatusTest::quantity_structure);
      break;
    }
    // --- Contact case --------------------------------------------------------
    case Inpar::STR::model_contact:
    {
      // add the normal/frictionless case
      qt.push_back(NOX::Nln::StatusTest::quantity_contact_normal);
      // check for friction
      const Teuchos::ParameterList& p_contact =
          Global::Problem::Instance()->contact_dynamic_params();
      enum Inpar::CONTACT::FrictionType frictiontype =
          Core::UTILS::IntegralValue<Inpar::CONTACT::FrictionType>(p_contact, "FRICTION");
      switch (frictiontype)
      {
        case Inpar::CONTACT::friction_none:
        {
          break;
        }
        default:
        {
          qt.push_back(NOX::Nln::StatusTest::quantity_contact_friction);
          break;
        }
      }
      // ToDo add wear etc.
      break;
    }
    // --- MeshTying case ------------------------------------------------------
    case Inpar::STR::model_meshtying:
    {
      qt.push_back(NOX::Nln::StatusTest::quantity_meshtying);
      break;
    }
    // --- constraint model case -----------------------------------------------
    case Inpar::STR::model_constraints:
    {
      qt.push_back(NOX::Nln::StatusTest::quantity_structure);
      break;
    }
    // --- 0D cardiovascular model case ----------------------------------------
    case Inpar::STR::model_cardiovascular0d:
    {
      qt.push_back(NOX::Nln::StatusTest::quantity_cardiovascular0d);
      break;
    }
    // --- Lagrangian/penalty case constraint ----------------------------------
    case Inpar::STR::model_lag_pen_constraint:
    {
      qt.push_back(NOX::Nln::StatusTest::quantity_lag_pen_constraint);
      break;
    }
    default:
      // no representation in the quantity type list
      break;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::Nln::SOLVER::convert_ele_tech_to_quantity_type(
    const enum Inpar::STR::EleTech& et, std::vector<enum NOX::Nln::StatusTest::QuantityType>& qt)
{
  // clear quantity type vector
  qt.clear();

  switch (et)
  {
    // --- EAS case ------------------------------------------------------------
    case Inpar::STR::EleTech::eas:
    {
      qt.push_back(NOX::Nln::StatusTest::quantity_eas);
      break;
    }
    // --- Plasticity case -----------------------------------------------------
    case Inpar::STR::EleTech::plasticity:
    {
      qt.push_back(NOX::Nln::StatusTest::quantity_plasticity);
      break;
    }
    // --- Pressure case -------------------------------------------------------
    case Inpar::STR::EleTech::pressure:
    {
      qt.push_back(NOX::Nln::StatusTest::quantity_pressure);
      break;
    }
    default:
    {
      // no representation in the quantity type list
      break;
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::Nln::SOLVER::set_status_test_params(Teuchos::ParameterList& pstatus,
    const STR::TimeInt::BaseDataSDyn& datasdyn,
    const std::set<enum NOX::Nln::StatusTest::QuantityType>& qt)
{
  // ------ outer status test ------
  // Combo test: OR-combination:
  // Combines the maximum iteration and the remaining tests
  Teuchos::ParameterList& postatus = pstatus.sublist("Outer Status Test");
  postatus.set("Test Type", "Combo");
  postatus.set("Combo Type", "OR");

  // ---------------------------------------------------------------------------
  // lvl. 0: combo OR - Test 0:
  // Combo test: AND
  //             combination of increment AND/OR residual
  //    AND      constraints
  //    AND      0D cardiovascular model
  //    AND      0D cardiovascular model increment
  //    AND      contact    (active set)
  //    AND      plasticity (active set)
  // ---------------------------------------------------------------------------
  Teuchos::ParameterList& pcombo_incr_fres_constr = postatus.sublist("Test 0");
  pcombo_incr_fres_constr.set("Test Type", "Combo");
  pcombo_incr_fres_constr.set("Combo Type", "AND");

  // ---------------------------------------------------------------------------
  // | lvl. 1: combo AND - Test 0:
  // | Combo test: AND/OR
  // | Combines increment (i.e. displacement and others) AND/OR
  // | force residual tests
  // ---------------------------------------------------------------------------
  Teuchos::ParameterList& pcombo_incr_fres = pcombo_incr_fres_constr.sublist("Test 0");
  pcombo_incr_fres.set("Test Type", "Combo");
  switch (datasdyn.GetResIncrComboType(
      NOX::Nln::StatusTest::quantity_structure, NOX::Nln::StatusTest::quantity_structure))
  {
    case Inpar::STR::bop_and:
      pcombo_incr_fres.set("Combo Type", "AND");
      break;
    case Inpar::STR::bop_or:
      pcombo_incr_fres.set("Combo Type", "OR");
      break;
    default:
      FOUR_C_THROW("Unknown structural combination type enum!");
      break;
  }

  // ---------------------------------------------------------------------------
  // || lvl. 2: combo AND/OR - Test 0:
  // || Tests the combination of the different solution increment norms
  // ---------------------------------------------------------------------------
  set_combo_quantity_test_params(pcombo_incr_fres, datasdyn, 0, "NormUpdate", qt);

  // ---------------------------------------------------------------------------
  // || lvl. 2: combo AND/OR - Test 1:
  // || Tests the combination of the different solution force norms
  // ---------------------------------------------------------------------------
  set_combo_quantity_test_params(pcombo_incr_fres, datasdyn, 1, "NormF", qt);

  // *** BEGIN: OPTIONAL STATUS TESTS ******************************************
  int opt_count = 1;
  // ---------------------------------------------------------------------------
  // | lvl. 1: combo AND - Test 1:
  // | Enforces a minimal number of non-linear solution steps
  // ---------------------------------------------------------------------------
  if (datasdyn.get_iter_min() > 0)
  {
    std::ostringstream test_string;
    test_string << "Test " << opt_count;
    Teuchos::ParameterList& pnstep = pcombo_incr_fres_constr.sublist(test_string.str());
    pnstep.set("Test Type", "NStep");
    pnstep.set("Number of Nonlinear Iterations", datasdyn.get_iter_min());
    ++opt_count;
  }
  // ---------------------------------------------------------------------------
  // | lvl. 1: combo AND - Test OPTIONAL:
  // | Tests the constraint NormF
  // ---------------------------------------------------------------------------
  if (qt.find(NOX::Nln::StatusTest::quantity_lag_pen_constraint) != qt.end())
  {
    set_quantity_test_params(pcombo_incr_fres_constr, datasdyn,
        NOX::Nln::StatusTest::quantity_lag_pen_constraint, opt_count, "NormF");
    ++opt_count;

    // ---------------------------------------------------------------------------
    // | lvl. 1: combo AND - Test OPTIONAL:
    // | Tests the constraint model NormUpdate
    // ---------------------------------------------------------------------------
    set_quantity_test_params(pcombo_incr_fres_constr, datasdyn,
        NOX::Nln::StatusTest::quantity_lag_pen_constraint, opt_count, "NormUpdate");
    ++opt_count;
  }

  // ---------------------------------------------------------------------------
  // | lvl. 1: combo AND - Test OPTIONAL:
  // | Tests the 0D cardiovascular model NormF
  // ---------------------------------------------------------------------------
  if (qt.find(NOX::Nln::StatusTest::quantity_cardiovascular0d) != qt.end())
  {
    set_quantity_test_params(pcombo_incr_fres_constr, datasdyn,
        NOX::Nln::StatusTest::quantity_cardiovascular0d, opt_count, "NormF");
    ++opt_count;

    // ---------------------------------------------------------------------------
    // | lvl. 1: combo AND - Test OPTIONAL:
    // | Tests the 0D cardiovascular model NormUpdate
    // ---------------------------------------------------------------------------
    set_quantity_test_params(pcombo_incr_fres_constr, datasdyn,
        NOX::Nln::StatusTest::quantity_cardiovascular0d, opt_count, "NormUpdate");
    ++opt_count;
  }

  // ---------------------------------------------------------------------------
  // | lvl. 1: combo AND - Test OPTIONAL:
  // | Tests the semi-smooth contact active set
  // ---------------------------------------------------------------------------
  if (qt.find(NOX::Nln::StatusTest::quantity_contact_normal) != qt.end())
  {
    set_quantity_test_params(pcombo_incr_fres_constr, datasdyn,
        NOX::Nln::StatusTest::quantity_contact_normal, opt_count, "ActiveSet");
    ++opt_count;
  }
  if (qt.find(NOX::Nln::StatusTest::quantity_contact_friction) != qt.end())
  {
    set_quantity_test_params(pcombo_incr_fres_constr, datasdyn,
        NOX::Nln::StatusTest::quantity_contact_friction, opt_count, "ActiveSet");
    ++opt_count;
  }

  // ---------------------------------------------------------------------------
  // | lvl. 1: combo AND - Test OPTIONAL:
  // | Tests the semi-smooth plasticity active set
  // ---------------------------------------------------------------------------
  if (qt.find(NOX::Nln::StatusTest::quantity_plasticity) != qt.end())
  {
    set_quantity_test_params(pcombo_incr_fres_constr, datasdyn,
        NOX::Nln::StatusTest::quantity_plasticity, opt_count, "ActiveSet");
    ++opt_count;
  }
  // *** END: OPTIONAL STATUS TESTS ********************************************

  // ---------------------------------------------------------------------------
  // lvl. 0: combo OR - Test 1:
  // Tests the maximal iteration counter
  // ---------------------------------------------------------------------------
  Teuchos::ParameterList& pmaxiters = postatus.sublist("Test 1");
  pmaxiters.set("Test Type", "MaxIters");
  pmaxiters.set<int>("Maximum Iterations", datasdyn.get_iter_max());
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::Nln::SOLVER::set_combo_quantity_test_params(Teuchos::ParameterList& p,
    const STR::TimeInt::BaseDataSDyn& datasdyn, const std::size_t& count,
    const std::string& testname, const std::set<enum NOX::Nln::StatusTest::QuantityType>& qtypes)
{
  std::vector<enum NOX::Nln::StatusTest::QuantityType> combo_or(0);
  std::vector<enum NOX::Nln::StatusTest::QuantityType> combo_and(0);
  split_and_or_combo(combo_or, combo_and, datasdyn, testname, qtypes);
  std::ostringstream test_string;
  test_string << "Test " << count;
  Teuchos::ParameterList& ptest = p.sublist(test_string.str());

  std::vector<enum NOX::Nln::StatusTest::QuantityType>::const_iterator qtiter;
  // if there are any OR combinations
  std::size_t count_or = 0;
  std::size_t count_and = 0;
  if (combo_or.size() > 0)
  {
    ptest.set("Test Type", "Combo");
    ptest.set("Combo Type", "OR");

    std::ostringstream test_string;
    for (qtiter = combo_or.begin(); qtiter != combo_or.end(); ++qtiter)
    {
      set_quantity_test_params(ptest, datasdyn, *qtiter, count_or, testname);
      ++count_or;
    }
    // if there are only OR combinations
    if (combo_and.size() == 0)
      set_quantity_test_params(
          ptest, datasdyn, NOX::Nln::StatusTest::quantity_structure, count_or, testname);
  }
  // if there are any AND combinations
  if (combo_and.size() > 0)
  {
    Teuchos::ParameterList& ptest_and = ptest;
    if (count_or > 0)
    {
      test_string.str("");
      test_string << "Test " << count_or;
      ptest_and = ptest.sublist(test_string.str());
    }
    ptest_and.set("Test Type", "Combo");
    ptest_and.set("Combo Type", "AND");
    for (qtiter = combo_and.begin(); qtiter != combo_and.end(); ++qtiter)
    {
      set_quantity_test_params(ptest_and, datasdyn, *qtiter, count_and, testname);
      ++count_and;
    }
    set_quantity_test_params(
        ptest_and, datasdyn, NOX::Nln::StatusTest::quantity_structure, count_and, testname);
  }
  // if there are neither AND nor OR combinations
  if (count_or == 0 and count_and == 0)
    set_quantity_test_params(ptest, datasdyn, NOX::Nln::StatusTest::quantity_structure, testname);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::Nln::SOLVER::set_quantity_test_params(Teuchos::ParameterList& p,
    const STR::TimeInt::BaseDataSDyn& datasdyn,
    const enum NOX::Nln::StatusTest::QuantityType& qtype, const std::size_t& count,
    const std::string& testname)
{
  std::ostringstream test_string;
  test_string << "Test " << count;
  Teuchos::ParameterList& ptest = p.sublist(test_string.str());
  set_quantity_test_params(ptest, datasdyn, qtype, testname);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::Nln::SOLVER::set_quantity_test_params(Teuchos::ParameterList& p,
    const STR::TimeInt::BaseDataSDyn& datasdyn,
    const enum NOX::Nln::StatusTest::QuantityType& qtype, const std::string& testname)
{
  if (testname == "NormUpdate")
    set_norm_update_params(p, qtype, datasdyn.get_incr_tolerance_type(qtype),
        datasdyn.GetIncrTolerance(qtype), datasdyn.GetNormType());
  else if (testname == "NormF")
    set_norm_f_params(p, qtype, datasdyn.GetResToleranceType(qtype),
        datasdyn.GetResTolerance(qtype), datasdyn.GetNormType());
  else if (testname == "ActiveSet")
    set_active_set_params(p, qtype);
  else
    FOUR_C_THROW("Unknown/Unsupported status test name: %s", testname.c_str());
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::Nln::SOLVER::split_and_or_combo(
    std::vector<enum NOX::Nln::StatusTest::QuantityType>& combo_or,
    std::vector<enum NOX::Nln::StatusTest::QuantityType>& combo_and,
    const STR::TimeInt::BaseDataSDyn& datasdyn, const std::string& testname,
    const std::set<enum NOX::Nln::StatusTest::QuantityType>& qtypes)
{
  std::set<enum NOX::Nln::StatusTest::QuantityType>::const_iterator qtiter;

  for (qtiter = qtypes.begin(); qtiter != qtypes.end(); ++qtiter)
  {
    if (*qtiter == NOX::Nln::StatusTest::quantity_structure) continue;

    enum Inpar::STR::BinaryOp combotype = datasdyn.GetIncrComboType(*qtiter);
    if (testname == "NormF")
      combotype = datasdyn.GetResComboType(*qtiter);
    else if (testname != "NormUpdate")
      FOUR_C_THROW("The given test \"%s\" name is not supported!", testname.c_str());

    switch (combotype)
    {
      case Inpar::STR::bop_or:
        combo_or.push_back(*qtiter);
        break;
      case Inpar::STR::bop_and:
        combo_and.push_back(*qtiter);
        break;
      default:
        FOUR_C_THROW(
            "Unknown combination type. See list of valid "
            "\"Inpar::STR::BinaryOp\" enums for more information.");
        break;
    }  // switch case
  }    // loop over the model type vector
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::Nln::SOLVER::set_norm_update_params(Teuchos::ParameterList& qlist,
    const enum NOX::Nln::StatusTest::QuantityType& qtype, const enum Inpar::STR::ConvNorm& toltype,
    const double& tol, const enum Inpar::STR::VectorNorm& normtype)
{
  set_norm_update_params(qlist, qtype, 1.0, 0.5, toltype, tol, normtype, false);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::Nln::SOLVER::set_norm_update_params(Teuchos::ParameterList& qlist,
    const enum NOX::Nln::StatusTest::QuantityType& qtype, const double& alpha, const double& beta,
    const enum Inpar::STR::ConvNorm& toltype, const double& tol,
    const enum Inpar::STR::VectorNorm& normtype, const bool& isscaled)
{
  /* Set the tolerance type
   * Be careful: This has to be done in first place because of the special
   *             treatment of the mixed tolerance type!
   */
  switch (toltype)
  {
    // ABSOLUTE TOLERANCE TYPE
    case Inpar::STR::convnorm_abs:
    {
      qlist.set("Test Type", "NormUpdate");
      qlist.set("Quantity Type", NOX::Nln::StatusTest::QuantityType2String(qtype).c_str());
      qlist.set("Tolerance Type", "Absolute");
      break;
    }
    // RELATIVE TOLERANCE TYPE
    case Inpar::STR::convnorm_rel:
    {
      qlist.set("Test Type", "NormUpdate");
      qlist.set("Quantity Type", NOX::Nln::StatusTest::QuantityType2String(qtype).c_str());
      qlist.set("Tolerance Type", "Relative");
      break;
    }
    // MIX TOLERANCE TYPE
    /* Actually, this should not be part of the tolerance type decision, but due
     * to historical reasons, we still support it. If you consider to implement a
     * related status test, please use a xml file in combination with the combo
     * status test instead.
     */
    case Inpar::STR::convnorm_mix:
    {
      qlist.set("Test Type", "Combo");
      qlist.set("Combo Type", "OR");
      Teuchos::ParameterList& qlist_abs = qlist.sublist("Test 0");
      // first recursive call
      set_norm_update_params(qlist_abs, qtype, Inpar::STR::convnorm_abs, tol, normtype);
      Teuchos::ParameterList& qlist_rel = qlist.sublist("Test 1");
      // second recursive call
      set_norm_update_params(qlist_rel, qtype, Inpar::STR::convnorm_rel, tol, normtype);
      break;
    }
    default:
      FOUR_C_THROW("Unknown \"Tolerance Type\" defined!");
      break;
  }  // switch case toltype

  // set tolerance
  qlist.set<double>("Tolerance", tol);

  // set norm type
  switch (normtype)
  {
    case Inpar::STR::norm_l2:
      qlist.set("Norm Type", "Two Norm");
      break;
    case Inpar::STR::norm_l1:
      qlist.set("Norm Type", "One Norm");
      break;
    case Inpar::STR::norm_inf:
      qlist.set("Norm Type", "Max Norm");
      break;
    case Inpar::STR::norm_rms:
      FOUR_C_THROW(
          "The norm type \"Root Mean Square\" is no longer supported! "
          "Consider to use the \"NOX::Nln::StatusTest::NormWRMS\" test instead!");
      break;
    case Inpar::STR::norm_vague:
    default:
      FOUR_C_THROW("Unknown vector norm type!");
      break;
  }  // switch case normtype

  // set alpha
  qlist.set<double>("Alpha", alpha);

  // set beta
  qlist.set<double>("Beta", beta);

  // set scaling by the vector length
  if (isscaled) qlist.set("Scale Type", "Scaled");
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::Nln::SOLVER::set_norm_f_params(Teuchos::ParameterList& qlist,
    const enum NOX::Nln::StatusTest::QuantityType& qtype, const enum Inpar::STR::ConvNorm& toltype,
    const double& tol, const enum Inpar::STR::VectorNorm& normtype)
{
  set_norm_f_params(qlist, qtype, toltype, tol, normtype, false);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::Nln::SOLVER::set_norm_f_params(Teuchos::ParameterList& qlist,
    const enum NOX::Nln::StatusTest::QuantityType& qtype, const enum Inpar::STR::ConvNorm& toltype,
    const double& tol, const enum Inpar::STR::VectorNorm& normtype, const bool& isscaled)
{
  /* Set the tolerance type
   * Be careful: This has to be done in first place because of the special
   *             treatment of the mixed tolerance type!
   */
  switch (toltype)
  {
    // ABSOLUTE TOLERANCE TYPE
    case Inpar::STR::convnorm_abs:
    {
      qlist.set("Test Type", "NormF");
      qlist.set("Quantity Type", NOX::Nln::StatusTest::QuantityType2String(qtype).c_str());
      qlist.set("Tolerance Type", "Absolute");
      break;
    }
    // RELATIVE TOLERANCE TYPE
    case Inpar::STR::convnorm_rel:
    {
      qlist.set("Test Type", "NormF");
      qlist.set("Quantity Type", NOX::Nln::StatusTest::QuantityType2String(qtype).c_str());
      qlist.set("Tolerance Type", "Relative");
      break;
    }
    // MIX TOLERANCE TYPE
    /* Actually, this should not be part of the tolerance type decision, but due
     * to historical reasons, we still support it. If you consider to implement a
     * related status test, please use a xml file in combination with the combo
     * status test instead.
     */
    case Inpar::STR::convnorm_mix:
    {
      qlist.set("Test Type", "Combo");
      qlist.set("Combo Type", "OR");
      Teuchos::ParameterList& qlist_abs = qlist.sublist("Test 0");
      // first recursive call
      set_norm_f_params(qlist_abs, qtype, Inpar::STR::convnorm_abs, tol, normtype);
      Teuchos::ParameterList& qlist_rel = qlist.sublist("Test 1");
      // second recursive call
      set_norm_f_params(qlist_rel, qtype, Inpar::STR::convnorm_rel, tol, normtype);
      break;
    }
    default:
      FOUR_C_THROW("Unknown \"Tolerance Type\" defined!");
      break;
  }  // switch case toltype

  // set tolerance
  qlist.set<double>("Tolerance", tol);

  // set norm type
  switch (normtype)
  {
    case Inpar::STR::norm_l2:
      qlist.set("Norm Type", "Two Norm");
      break;
    case Inpar::STR::norm_l1:
      qlist.set("Norm Type", "One Norm");
      break;
    case Inpar::STR::norm_inf:
      qlist.set("Norm Type", "Max Norm");
      break;
    case Inpar::STR::norm_rms:
      FOUR_C_THROW(
          "This norm type is no longer supported! "
          "Consider to use the \"NOX::Nln::StatusTest::NormWRMS\" test instead!");
      break;
    case Inpar::STR::norm_vague:
    default:
      FOUR_C_THROW("Unknown vector norm type!");
      break;
  }  // switch case normtype

  // set scaling by the vector length
  if (isscaled) qlist.set("Scale Type", "Scaled");
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::Nln::SOLVER::set_active_set_params(
    Teuchos::ParameterList& qlist, const enum NOX::Nln::StatusTest::QuantityType& qtype)
{
  qlist.set("Test Type", "ActiveSet");
  // set the quantity type
  qlist.set("Quantity Type", NOX::Nln::StatusTest::QuantityType2String(qtype).c_str());
  qlist.set<int>("Max Cycle Size", 3);
}

FOUR_C_NAMESPACE_CLOSE