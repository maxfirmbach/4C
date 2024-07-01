/*-----------------------------------------------------------------------------------------------*/
/*! \file

\brief three dimensional nonlinear Kirchhoff beam element based on a C1 curve

\level 2

*/
/*-----------------------------------------------------------------------------------------------*/

#include "4C_beam3_kirchhoff.hpp"
#include "4C_fem_general_largerotations.hpp"
#include "4C_io_linedefinition.hpp"
#include "4C_legacy_enum_definitions_materials.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
bool Discret::ELEMENTS::Beam3k::ReadElement(
    const std::string& eletype, const std::string& distype, Input::LineDefinition* linedef)
{
  // read number of material model and cross-sections specs
  int material = 0;
  linedef->extract_int("MAT", material);
  SetMaterial(0, Mat::Factory(material));

  const auto mat_type = Material()->Parameter()->Type();
  FOUR_C_THROW_UNLESS(mat_type == Core::Materials::m_beam_kirchhoff_elast_hyper ||
                          mat_type == Core::Materials::m_beam_kirchhoff_elast_hyper_bymodes,
      "The material parameter definition '%s' is not supported by Beam3k element! "
      "Choose MAT_BeamKirchhoffElastHyper or MAT_BeamKirchhoffElastHyper_ByModes!",
      to_string(mat_type).data());


  int rotvec = 0;
  linedef->extract_int("ROTVEC", rotvec);

  int wk = 0;
  linedef->extract_int("WK", wk);

  if (rotvec == 0)
    rotvec_ = false;
  else if (rotvec == 1)
    rotvec_ = true;
  else
    FOUR_C_THROW(
        "The variable ROTVEC can only take on the values 0 (tangent vectors as nodal DoFs) and "
        "1 (rotation vectors as nodal DoFs)!");

  if (wk == 0)
    weakkirchhoff_ = false;
  else if (wk == 1)
  {
    weakkirchhoff_ = true;
#ifdef CONSISTENTSPINSK
    FOUR_C_THROW(
        "The flag CONSISTENTSPINSK is only possible for strong Kirchhoff constraint enforcement "
        "(weakkirchhoff_=false)");
#endif
  }
  else
    FOUR_C_THROW(
        "The variable WK can only take on the values 0 (Kirchhoff constraint enforced in a strong "
        "manner) and "
        "1 (Kirchhoff constraint enforced in a weak manner)!");


  // extract triads at element nodes in reference configuration as rotation vectors and save them as
  // quaternions at each node, respectively
  std::vector<double> nodal_thetas;
  linedef->extract_double_vector("TRIADS", nodal_thetas);
  this->set_up_initial_rotations(nodal_thetas);

  // read whether automatic differentiation via Sacado::Fad package shall be used
  use_fad_ = linedef->has_named("FAD") ? true : false;

  return true;
}

FOUR_C_NAMESPACE_CLOSE