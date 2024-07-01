#ifndef FOUR_C_SO3_PORO_FWD_HPP
#define FOUR_C_SO3_PORO_FWD_HPP

/*----------------------------------------------------------------------*/
/*! \file
\brief forward declarations (explicit instantiation) of the templated poro elements

\level 2
*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_OPEN

// template classes
template class Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::SoHex8, Core::FE::CellType::hex8>;
template class Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::SoTet4, Core::FE::CellType::tet4>;
template class Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::SoHex27, Core::FE::CellType::hex27>;
template class Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::SoTet10, Core::FE::CellType::tet10>;
template class Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::Nurbs::SoNurbs27,
    Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE

#endif