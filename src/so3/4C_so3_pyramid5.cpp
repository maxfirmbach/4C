/*----------------------------------------------------------------------*/
/*! \file

\brief pyramid shaped solid element

\level 1


*----------------------------------------------------------------------*/

#include "4C_so3_pyramid5.hpp"

#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_global_data.hpp"
#include "4C_io_linedefinition.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_so3_line.hpp"
#include "4C_so3_nullspace.hpp"
#include "4C_so3_prestress.hpp"
#include "4C_so3_prestress_service.hpp"
#include "4C_so3_pyramid5fbar.hpp"
#include "4C_so3_surface.hpp"
#include "4C_so3_utils.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

Discret::ELEMENTS::SoPyramid5Type Discret::ELEMENTS::SoPyramid5Type::instance_;

Discret::ELEMENTS::SoPyramid5Type& Discret::ELEMENTS::SoPyramid5Type::Instance()
{
  return instance_;
}


Core::Communication::ParObject* Discret::ELEMENTS::SoPyramid5Type::Create(
    const std::vector<char>& data)
{
  auto* object = new Discret::ELEMENTS::SoPyramid5(-1, -1);
  object->unpack(data);
  return object;
}


Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::SoPyramid5Type::Create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    Teuchos::RCP<Core::Elements::Element> ele =
        Teuchos::rcp(new Discret::ELEMENTS::SoPyramid5(id, owner));
    return ele;
  }
  return Teuchos::null;
}


Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::SoPyramid5Type::Create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele =
      Teuchos::rcp(new Discret::ELEMENTS::SoPyramid5(id, owner));
  return ele;
}


void Discret::ELEMENTS::SoPyramid5Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  numdf = 3;
  dimns = 6;
  nv = 3;
}

Core::LinAlg::SerialDenseMatrix Discret::ELEMENTS::SoPyramid5Type::ComputeNullSpace(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return ComputeSolid3DNullSpace(node, x0);
}

void Discret::ELEMENTS::SoPyramid5Type::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, Input::LineDefinition>& defs = definitions[get_element_type_string()];

  defs["PYRAMID5"] = Input::LineDefinition::Builder()
                         .add_int_vector("PYRAMID5", 5)
                         .add_named_int("MAT")
                         .add_named_string("KINEM")
                         .add_optional_named_double_vector("RAD", 3)
                         .add_optional_named_double_vector("AXI", 3)
                         .add_optional_named_double_vector("CIR", 3)
                         .add_optional_named_double_vector("FIBER1", 3)
                         .add_optional_named_double_vector("FIBER2", 3)
                         .add_optional_named_double_vector("FIBER3", 3)
                         .add_optional_named_double("STRENGTH")
                         .add_optional_named_double("GROWTHTRIG")
                         .build();
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                                       |
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::SoPyramid5::SoPyramid5(int id, int owner)
    : SoBase(id, owner), pstype_(Inpar::STR::PreStress::none), pstime_(0.0), time_(0.0)
{
  kintype_ = Inpar::STR::KinemType::nonlinearTotLag;
  invJ_.resize(NUMGPT_SOP5, Core::LinAlg::Matrix<NUMDIM_SOP5, NUMDIM_SOP5>(true));
  detJ_.resize(NUMGPT_SOP5, 0.0);

  Teuchos::RCP<const Teuchos::ParameterList> params =
      Global::Problem::Instance()->getParameterList();
  if (params != Teuchos::null)
  {
    pstype_ = Prestress::GetType();
    pstime_ = Prestress::GetPrestressTime();

    Discret::ELEMENTS::UTILS::ThrowErrorFDMaterialTangent(
        Global::Problem::Instance()->structural_dynamic_params(), get_element_type_string());
  }
  if (Prestress::IsMulf(pstype_))
    prestress_ = Teuchos::rcp(new Discret::ELEMENTS::PreStress(NUMNOD_SOP5, NUMGPT_SOP5));

  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                                  |
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::SoPyramid5::SoPyramid5(const Discret::ELEMENTS::SoPyramid5& old)
    : SoBase(old),
      kintype_(old.kintype_),
      detJ_(old.detJ_),
      pstype_(old.pstype_),
      pstime_(old.pstime_),
      time_(old.time_)
{
  invJ_.resize(old.invJ_.size());
  for (int i = 0; i < (int)invJ_.size(); ++i)
  {
    // can this size be anything but NUMDIM_SOP5 x NUMDIM_SOP5?
    invJ_[i] = old.invJ_[i];
  }

  if (Prestress::IsMulf(pstype_))
    prestress_ = Teuchos::rcp(new Discret::ELEMENTS::PreStress(*(old.prestress_)));

  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of Solid3 and return pointer to it (public) |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::ELEMENTS::SoPyramid5::Clone() const
{
  auto* newelement = new Discret::ELEMENTS::SoPyramid5(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |                                                             (public) |
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::ELEMENTS::SoPyramid5::Shape() const
{
  return Core::FE::CellType::pyramid5;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoPyramid5::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::PackBuffer::SizeMarker sm(data);

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  add_to_pack(data, type);
  // add base class Element
  Element::pack(data);
  // kintype_
  add_to_pack(data, kintype_);

  // detJ_
  add_to_pack(data, detJ_);

  // invJ_
  const auto size = (int)invJ_.size();
  add_to_pack(data, size);
  for (int i = 0; i < size; ++i) add_to_pack(data, invJ_[i]);

  // Pack prestress_
  add_to_pack(data, static_cast<int>(pstype_));
  add_to_pack(data, pstime_);
  add_to_pack(data, time_);
  if (Prestress::IsMulf(pstype_))
  {
    Core::Communication::ParObject::add_to_pack(data, *prestress_);
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoPyramid5::unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;

  Core::Communication::ExtractAndAssertId(position, data, UniqueParObjectId());

  // extract base class Element
  std::vector<char> basedata(0);
  extract_from_pack(position, data, basedata);
  Element::unpack(basedata);
  // kintype_
  kintype_ = static_cast<Inpar::STR::KinemType>(extract_int(position, data));

  // detJ_
  extract_from_pack(position, data, detJ_);
  // invJ_
  int size = 0;
  extract_from_pack(position, data, size);
  invJ_.resize(size, Core::LinAlg::Matrix<NUMDIM_SOP5, NUMDIM_SOP5>(true));
  for (int i = 0; i < size; ++i) extract_from_pack(position, data, invJ_[i]);

  // Extract prestress_
  pstype_ = static_cast<Inpar::STR::PreStress>(extract_int(position, data));
  extract_from_pack(position, data, pstime_);
  extract_from_pack(position, data, time_);
  if (Prestress::IsMulf(pstype_))
  {
    std::vector<char> tmpprestress(0);
    extract_from_pack(position, data, tmpprestress);
    if (prestress_ == Teuchos::null)
    {
      int numgpt = NUMGPT_SOP5;
      // see whether I am actually a So_pyramid5fbar element
      auto* me = dynamic_cast<Discret::ELEMENTS::SoPyramid5fbar*>(this);
      if (me) numgpt += 1;  // one more history entry for centroid data in pyramid5fbar
      prestress_ = Teuchos::rcp(new Discret::ELEMENTS::PreStress(NUMNOD_SOP5, numgpt));
    }
    prestress_->unpack(tmpprestress);
    // end
  }

  if (position != data.size())
    FOUR_C_THROW("Mismatch in size of data %d <-> %d", (int)data.size(), position);
  return;
}



/*----------------------------------------------------------------------*
 |  print this element (public)                                         |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoPyramid5::print(std::ostream& os) const
{
  os << "So_pyramid5 ";
  Element::print(os);
  std::cout << std::endl;
  return;
}

/*====================================================================*/
/* 5-node pyramid node topology*/
/*--------------------------------------------------------------------*/
/* parameter coordinates (r,s,t) of nodes
 * of biunit pyramid [-1,1]x[-1,1]x[0,1]
 * 5-node pyramid: node 1,2,3,4,5


 *                /(5)\
 *              / //\\ \
 *            // //  \\ \\
 *          //  //    \\  \\
 *        //   //  t   \\   \\
 *      //    //   |    \\    \\
 *    //     //    |     \\     \\
 *  (4)-----//------------\\-----(3)
 *  ||     //      |       \\     ||
 *  ||    //       |        \\    ||
 *  ||   //        o---------\\---------s
 *  ||  //         |          \\  ||
 *  || //          |           \\ ||
 *  ||//           |            \\||
 *  (1)============|=============(2)
 *                 |
 *                 r
 *
 */
/*====================================================================*/

/*----------------------------------------------------------------------*
|  get vector of surfaces (public)                                      |
|  surface normals always point outward                                 |
*----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<Core::Elements::Element>> Discret::ELEMENTS::SoPyramid5::Surfaces()
{
  return Core::Communication::ElementBoundaryFactory<StructuralSurface>(
      Core::Communication::buildSurfaces, *this);
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                                        |
 *----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<Core::Elements::Element>> Discret::ELEMENTS::SoPyramid5::Lines()
{
  return Core::Communication::ElementBoundaryFactory<StructuralLine, Core::Elements::Element>(
      Core::Communication::buildLines, *this);
}

/*----------------------------------------------------------------------*
 |  Return names of visualization data (public)                         |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoPyramid5::VisNames(std::map<std::string, int>& names)
{
  SolidMaterial()->VisNames(names);
  return;
}

/*----------------------------------------------------------------------*
 |  Return visualization data (public)                                  |
 *----------------------------------------------------------------------*/
bool Discret::ELEMENTS::SoPyramid5::VisData(const std::string& name, std::vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if (Core::Elements::Element::VisData(name, data)) return true;

  return SolidMaterial()->VisData(name, data, NUMGPT_SOP5, this->Id());
}

FOUR_C_NAMESPACE_CLOSE