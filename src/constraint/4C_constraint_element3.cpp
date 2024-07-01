/*----------------------------------------------------------------------*/
/*! \file
\brief A 3D constraint element with no physics attached
\level 2


*----------------------------------------------------------------------*/

#include "4C_constraint_element3.hpp"

FOUR_C_NAMESPACE_OPEN


Discret::ELEMENTS::ConstraintElement3Type Discret::ELEMENTS::ConstraintElement3Type::instance_;


Discret::ELEMENTS::ConstraintElement3Type& Discret::ELEMENTS::ConstraintElement3Type::Instance()
{
  return instance_;
}


Core::Communication::ParObject* Discret::ELEMENTS::ConstraintElement3Type::Create(
    const std::vector<char>& data)
{
  Discret::ELEMENTS::ConstraintElement3* object = new Discret::ELEMENTS::ConstraintElement3(-1, -1);
  object->unpack(data);
  return object;
}


Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::ConstraintElement3Type::Create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "CONSTRELE3")
  {
    Teuchos::RCP<Core::Elements::Element> ele =
        Teuchos::rcp(new Discret::ELEMENTS::ConstraintElement3(id, owner));
    return ele;
  }
  return Teuchos::null;
}


Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::ConstraintElement3Type::Create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele =
      Teuchos::rcp(new Discret::ELEMENTS::ConstraintElement3(id, owner));
  return ele;
}


void Discret::ELEMENTS::ConstraintElement3Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
}

Core::LinAlg::SerialDenseMatrix Discret::ELEMENTS::ConstraintElement3Type::ComputeNullSpace(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  Core::LinAlg::SerialDenseMatrix nullspace;
  FOUR_C_THROW("method ComputeNullSpace not implemented!");
  return nullspace;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::ConstraintElement3::ConstraintElement3(int id, int owner)
    : Core::Elements::Element(id, owner)
{
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::ConstraintElement3::ConstraintElement3(
    const Discret::ELEMENTS::ConstraintElement3& old)
    : Core::Elements::Element(old)
{
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::ELEMENTS::ConstraintElement3::Clone() const
{
  Discret::ELEMENTS::ConstraintElement3* newelement =
      new Discret::ELEMENTS::ConstraintElement3(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::ConstraintElement3::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::PackBuffer::SizeMarker sm(data);

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  add_to_pack(data, type);
  // add base class Element
  Element::pack(data);

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::ConstraintElement3::unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;

  Core::Communication::ExtractAndAssertId(position, data, UniqueParObjectId());

  // extract base class Element
  std::vector<char> basedata(0);
  extract_from_pack(position, data, basedata);
  Element::unpack(basedata);

  if (position != data.size())
    FOUR_C_THROW("Mismatch in size of data %d <-> %d", (int)data.size(), position);
  return;
}

FOUR_C_NAMESPACE_CLOSE