/*! \file

\brief Helper functions for shell7p element

\level 1
*/

#include "4C_shell7p_utils.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_io_linedefinition.hpp"
#include "4C_shell7p_ele.hpp"
#include "4C_shell7p_ele_scatra.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  enum class ShellEasTypes
  {
    none,
    N_1,
    N_2,
    N_3,
    N_4,
    N_5,
    N_6,
    N_7,
    N_8,
    N_9,
    N_11,
    N_undefined
  };

  template <ShellEasTypes eastype>
  struct EasTypeToNumEas
  {
  };
  template <>
  struct EasTypeToNumEas<ShellEasTypes::N_1>
  {
    static const int num_eas = 1;
  };
  template <>
  struct EasTypeToNumEas<ShellEasTypes::N_2>
  {
    static const int num_eas = 2;
  };
  template <>
  struct EasTypeToNumEas<ShellEasTypes::N_3>
  {
    static const int num_eas = 3;
  };
  template <>
  struct EasTypeToNumEas<ShellEasTypes::N_4>
  {
    static const int num_eas = 4;
  };
  template <>
  struct EasTypeToNumEas<ShellEasTypes::N_5>
  {
    static const int num_eas = 5;
  };
  template <>
  struct EasTypeToNumEas<ShellEasTypes::N_6>
  {
    static const int num_eas = 6;
  };
  template <>
  struct EasTypeToNumEas<ShellEasTypes::N_7>
  {
    static const int num_eas = 7;
  };
  template <>
  struct EasTypeToNumEas<ShellEasTypes::N_8>
  {
    static const int num_eas = 8;
  };
  template <>
  struct EasTypeToNumEas<ShellEasTypes::N_9>
  {
    static const int num_eas = 9;
  };
  template <>
  struct EasTypeToNumEas<ShellEasTypes::N_11>
  {
    static const int num_eas = 11;
  };
  template <>
  struct EasTypeToNumEas<ShellEasTypes::N_undefined>
  {
  };

  void SetMembraneLockingSizeQuad4(int& num_eas, const std::string& type)
  {
    if (type == "N_1")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_1>::num_eas;
    }
    else if (type == "N_2")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_1>::num_eas;
    }
    else if (type == "N_3")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_3>::num_eas;
    }
    else if (type == "N_4")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_4>::num_eas;
    }
    else if (type == "N_5")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_5>::num_eas;
    }
    else if (type == "N_7")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_7>::num_eas;
    }
    else if (type == "none")
    {
      num_eas = 0;
    }
    else
      FOUR_C_THROW(
          "Unrecognized EAS type for quad4 to alleviate membrane locking. Only none, N_1, N_2, "
          "N_3, N_4, N_5, N_7 are allowed. Given: %s",
          type.c_str());
  }

  void SetBendingLockingSizeQuad4(int& num_eas, const std::string& type)
  {
    if (type == "N_4")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_4>::num_eas;
    }
    else if (type == "N_5")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_5>::num_eas;
    }
    else if (type == "N_6")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_6>::num_eas;
    }
    else if (type == "N_7")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_7>::num_eas;
    }
    else if (type == "none")
    {
      num_eas = 0;
    }
    else
      FOUR_C_THROW(
          "Unrecognized EAS type for quad4 to alleviate bending locking. Only none, N_4, N_5, N_6, "
          "N_7 are allowed. Given: %s",
          type.c_str());
  }

  void SetThicknessLockingSizeQuad4(int& num_eas, const std::string& type)
  {
    if (type == "N_1")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_1>::num_eas;
    }
    else if (type == "N_3")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_3>::num_eas;
    }
    else if (type == "N_4")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_4>::num_eas;
    }
    else if (type == "N_6")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_6>::num_eas;
    }
    else if (type == "N_8")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_8>::num_eas;
    }
    else if (type == "N_9")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_9>::num_eas;
    }
    else if (type == "none")
    {
      num_eas = 0;
    }
    else
      FOUR_C_THROW(
          "Unrecognized EAS type for quad4 to alleviate thickness locking. Only none, N_1, N_3, "
          "N_4, "
          "N_6, N_8, N_9 are allowed. Given: %s",
          type.c_str());
  }

  void SetShearStrainLockingSizeQuad4(int& num_eas, const std::string& type)
  {
    if (type == "N_2")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_2>::num_eas;
    }
    else if (type == "N_4")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_4>::num_eas;
    }
    else if (type == "none")
    {
      num_eas = 0;
    }
    else
      FOUR_C_THROW(
          "Unrecognized EAS type for quad4 to alleviate transverse shear strain locking: Only "
          "none, N_2, N_4 are allowed. Given: %s",
          type.c_str());
  }

  void SetMembraneLockingSizeQuad9(int& num_eas, const std::string& type)
  {
    if (type == "N_7")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_7>::num_eas;
    }
    else if (type == "N_9")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_9>::num_eas;
    }
    else if (type == "none")
    {
      num_eas = 0;
    }
    else
      FOUR_C_THROW(
          "Unrecognized EAS type for quad9 to alleviate membrane locking. Only none, N_7, N_9 are "
          "allowed. Given: %s",
          type.c_str());
  }

  void SetBendingLockingSizeQuad9(int& num_eas, const std::string& type)
  {
    if (type == "N_9")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_9>::num_eas;
    }
    else if (type == "N_11")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_11>::num_eas;
    }
    else if (type == "none")
    {
      num_eas = 0;
    }
    else
      FOUR_C_THROW(
          "Unrecognized EAS type for quad9 to alleviate bending locking. Only none, N_9, N_11 are "
          "allowed. Given: %s",
          type.c_str());
  }

  void SetThicknessLockingSizeQuad9(int& num_eas, const std::string& type)
  {
    if (type == "N_1")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_1>::num_eas;
    }
    else if (type == "N_3")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_3>::num_eas;
    }
    else if (type == "N_4")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_4>::num_eas;
    }
    else if (type == "N_6")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_6>::num_eas;
    }
    else if (type == "N_8")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_8>::num_eas;
    }
    else if (type == "N_9")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_9>::num_eas;
    }
    else if (type == "none")
    {
      num_eas = 0;
    }
    else
      FOUR_C_THROW(
          "Unrecognized EAS type for quad9 to alleviate thickness locking. Only none, N_1, N_3, "
          "N_4, N_6, N_8, N_9 are allowed. Given: %s",
          type.c_str());
  }

  void SetShearStrainLockingSizeQuad9(int& num_eas, const std::string& type)
  {
    if (type == "N_2")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_2>::num_eas;
    }
    else if (type == "N_4")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_4>::num_eas;
    }
    else if (type == "N_6")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_6>::num_eas;
    }
    else if (type == "none")
    {
      num_eas = 0;
    }
    else
      FOUR_C_THROW(
          "Unrecognized EAS type for quad9 to alleviate transverse shear strain locking. Only "
          "none, N_2, N_4, N_6 are allowed. Given: %s",
          type.c_str());
  }

  inline auto SquareValue = [](auto a) { return a * a; };

}  // namespace

Teuchos::SerialDenseMatrix<int, double> STR::UTILS::Shell::ComputeShellNullSpace(
    Core::Nodes::Node& node, const double* x0, const Core::LinAlg::Matrix<3, 1>& dir)
{
  const auto& x = node.X();

  Teuchos::SerialDenseMatrix<int, double> nullspace(6, 6);
  // x-modes
  nullspace(0, 0) = 1.0;
  nullspace(0, 1) = 0.0;
  nullspace(0, 2) = 0.0;
  nullspace(0, 3) = 0.0;
  nullspace(0, 4) = x[2] - x0[2];
  nullspace(0, 5) = -x[1] + x0[1];
  // y-modes
  nullspace(1, 0) = 0.0;
  nullspace(1, 1) = 1.0;
  nullspace(1, 2) = 0.0;
  nullspace(1, 3) = -x[2] + x0[2];
  nullspace(1, 4) = 0.0;
  nullspace(1, 5) = x[0] - x0[0];
  // z-modes
  nullspace(2, 0) = 0.0;
  nullspace(2, 1) = 0.0;
  nullspace(2, 2) = 1.0;
  nullspace(2, 3) = x[1] - x0[1];
  nullspace(2, 4) = -x[0] + x0[0];
  nullspace(2, 5) = 0.0;
  // dx-modes
  nullspace(3, 0) = 0.0;
  nullspace(3, 1) = 0.0;
  nullspace(3, 2) = 0.0;
  nullspace(3, 3) = 0.0;
  nullspace(3, 4) = dir(2, 0);
  nullspace(3, 5) = -dir(1, 0);
  // dy-modes
  nullspace(4, 0) = 0.0;
  nullspace(4, 1) = 0.0;
  nullspace(4, 2) = 0.0;
  nullspace(4, 3) = -dir(2, 0);
  nullspace(4, 4) = 0.0;
  nullspace(4, 5) = dir(0, 0);
  // dz-modes
  nullspace(5, 0) = 0.0;
  nullspace(5, 1) = 0.0;
  nullspace(5, 2) = 0.0;
  nullspace(5, 3) = dir(1, 0);
  nullspace(5, 4) = -dir(0, 0);
  nullspace(5, 5) = 0.0;

  return nullspace;
}

void STR::UTILS::Shell::NodalBlockInformationShell(
    Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  numdf = 6;
  dimns = 6;
  nv = 3;
}

void STR::UTILS::Shell::Director::SetupDirectorForElement(
    const Core::Elements::Element& ele, Core::LinAlg::SerialDenseMatrix& nodal_directors)
{
  constexpr auto num_dim = Discret::ELEMENTS::Shell::DETAIL::num_dim;
  const int num_node = ele.num_node();
  Core::LinAlg::SerialDenseMatrix xrefe(num_node, num_dim);
  for (int i = 0; i < num_node; ++i)
  {
    for (int dim = 0; dim < num_dim; ++dim) xrefe(i, dim) = ele.Nodes()[i]->X()[dim];
  }
  // allocate matrix for kovariant metric vectors
  Core::LinAlg::SerialDenseMatrix metrics_kovariant(num_dim, num_dim);
  for (int i = 0; i < num_node; ++i)
  {
    // get shape functions and derivatives at nodes
    Core::LinAlg::Matrix<num_dim, 1> nodal_coordinates =
        Core::FE::GetNodeCoordinates(i, ele.Shape());
    Core::LinAlg::SerialDenseMatrix derivatives(num_dim, num_node);
    Core::FE::shape_function_2D_deriv1(
        derivatives, nodal_coordinates(0), nodal_coordinates(1), ele.Shape());

    // get a1, a2 direction derivatives in r and s direction
    metrics_kovariant.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, derivatives, xrefe, 0.0);

    // get thickness direction derivative perpendicular to a1 and a2
    // -> a3 = (a1 x a2) / (|a1 x a2 |)
    Core::LinAlg::Matrix<num_dim, 1> a1a2crossprod(true);
    a1a2crossprod(0) = metrics_kovariant(0, 1) * metrics_kovariant(1, 2) -
                       metrics_kovariant(0, 2) * metrics_kovariant(1, 1);
    a1a2crossprod(1) = metrics_kovariant(0, 2) * metrics_kovariant(1, 0) -
                       metrics_kovariant(0, 0) * metrics_kovariant(1, 2);
    a1a2crossprod(2) = metrics_kovariant(0, 0) * metrics_kovariant(1, 1) -
                       metrics_kovariant(0, 1) * metrics_kovariant(1, 0);
    double a1a2crossnorm = a1a2crossprod.norm2();
    if (a1a2crossnorm > 1.0e-14) a1a2crossprod.scale(1.0 / a1a2crossnorm);

    // set nodal director matrix for each node row vice
    for (int j = 0; j < num_dim; j++) nodal_directors(i, j) = a1a2crossprod(j);
  }
}

void STR::UTILS::Shell::Director::AverageDirector(const Core::LinAlg::Matrix<3, 8>& dir_list,
    const int num_directors, Core::LinAlg::Matrix<3, 1>& nodal_director)
{
  Core::LinAlg::Matrix<3, 1> davn(true);
  Core::LinAlg::Matrix<3, 1> averdir(true);
  for (int dim = 0; dim < Discret::ELEMENTS::Shell::DETAIL::num_dim; ++dim)
    averdir(dim) = dir_list(dim, 0);

  for (int i = 1; i < num_directors; ++i)
  {
    // make cross product of two directors
    Core::LinAlg::Matrix<3, 1> normal(true);
    normal(0) = averdir(1) * dir_list(2, i) - averdir(2) * dir_list(1, i);
    normal(1) = averdir(2) * dir_list(0, i) - averdir(0) * dir_list(2, i);
    normal(2) = averdir(0) * dir_list(1, i) - averdir(1) * dir_list(0, i);
    const double length = normal.dot(normal);

    // if the length is small, both directors point nearly in the same direction
    if (length <= 1.e-12)
    {
      for (int dim = 0; dim < Discret::ELEMENTS::Shell::DETAIL::num_dim; ++dim)
        davn(dim) = 0.5 * (averdir(dim) + dir_list(dim, i));
    }
    // if not average the nodal directors
    else
    {
      const double denom =
          (SquareValue(dir_list(0, i)) + SquareValue(dir_list(2, i))) * SquareValue(averdir(1)) +
          (-2. * dir_list(0, i) * averdir(0) * dir_list(1, i) -
              2. * dir_list(2, i) * averdir(2) * dir_list(1, i)) *
              averdir(1) +
          (SquareValue(dir_list(2, i)) + SquareValue(dir_list(1, i))) * SquareValue(averdir(0)) -
          2. * averdir(2) * averdir(0) * dir_list(2, i) * dir_list(0, i) +
          (SquareValue(dir_list(0, i)) + SquareValue(dir_list(1, i))) * SquareValue(averdir(2));

      if (std::abs(denom) <= 1.e-13) FOUR_C_THROW("Making of modified directors failed");

      const double alpha = (averdir(2) * dir_list(2, i) - SquareValue(dir_list(0, i)) +
                               averdir(0) * dir_list(0, i) - SquareValue(dir_list(1, i)) +
                               dir_list(1, i) * averdir(1) - SquareValue(dir_list(2, i))) /
                           denom;

      davn(0, 0) = -alpha * SquareValue(averdir(1)) * dir_list(0, i) +
                   alpha * averdir(1) * averdir(0) * dir_list(1, i) + averdir(0) +
                   alpha * averdir(2) * averdir(0) * dir_list(2, i) -
                   alpha * SquareValue(averdir(2)) * dir_list(0, i);

      davn(1, 0) = alpha * averdir(0) * averdir(1) * dir_list(0, i) + averdir(1) +
                   alpha * averdir(2) * averdir(1) * dir_list(2, i) -
                   alpha * SquareValue(averdir(0)) * dir_list(1, i) -
                   alpha * SquareValue(averdir(2)) * dir_list(1, i);

      davn(2, 0) = -alpha * SquareValue(averdir(1)) * dir_list(2, i) +
                   alpha * averdir(1) * averdir(2) * dir_list(1, i) -
                   alpha * SquareValue(averdir(0)) * dir_list(2, i) +
                   alpha * averdir(0) * averdir(2) * dir_list(0, i) + averdir(2);
    }
    for (int dim = 0; dim < Discret::ELEMENTS::Shell::DETAIL::num_dim; ++dim)
    {
      averdir(dim) = davn(dim);
      nodal_director(dim) = davn(dim);
    }
  }
}

void STR::UTILS::Shell::Director::ExportDirectorMapFromRowToColMap(
    const Core::Elements::ElementType& eletype, const Core::FE::Discretization& dis,
    std::map<int, std::vector<double>>& director_map)
{
  // export this map from nodal row map to nodal col map
  const Epetra_Map* noderowmap = dis.NodeRowMap();
  const Epetra_Map* nodecolmap = dis.NodeColMap();
  Core::Communication::Exporter exporter(*noderowmap, *nodecolmap, dis.Comm());
  exporter.Export(director_map);

  // loop through column nodes and put directors back into discretization
  for (const auto& actnode : dis.MyColNodeRange())
  {
    auto curr = director_map.find(actnode->Id());
    FOUR_C_ASSERT(curr != director_map.end(), "Cannot find director map entry");
    for (int j = 0; j < actnode->NumElement(); ++j)
    {
      Core::Elements::Element* tmpele = actnode->Elements()[j];
      if (!tmpele) continue;
      if (tmpele->ElementType() != eletype) continue;
      if (auto* scatra_ele = dynamic_cast<Discret::ELEMENTS::Shell7pScatra*>(tmpele))
      {
        for (int k = 0; k < scatra_ele->num_node(); ++k)
        {
          if (scatra_ele->Nodes()[k] == actnode)
          {
            scatra_ele->SetNodalDirector(k, curr->second);
            break;
          }
        }
      }
      else if (auto* shell_ele = dynamic_cast<Discret::ELEMENTS::Shell7p*>(tmpele))
      {
        for (int k = 0; k < shell_ele->num_node(); ++k)
        {
          if (shell_ele->Nodes()[k] == actnode)
          {
            shell_ele->SetNodalDirector(k, curr->second);
            break;
          }
        }
      }
      else
        FOUR_C_THROW("Element is not a shell element");
    }
  }
}


void STR::UTILS::Shell::Director::AverageDirectorsAtNodes(
    const Core::Elements::ElementType& eletype, const Core::FE::Discretization& dis,
    std::map<int, std::vector<double>>& director_map)
{
  const int max_ele = 8;
  static constexpr int num_dim = Discret::ELEMENTS::Shell::DETAIL::num_dim;
  Core::LinAlg::Matrix<num_dim, max_ele> collaverdir(true);

  // loop through all row nodes and build director map
  for (const auto& act_node : dis.MyRowNodeRange())
  {
    int num_directors = 0;
    for (int j = 0; j < act_node->NumElement(); ++j)
    {
      Core::Elements::Element* tmpele = act_node->Elements()[j];
      if (tmpele->ElementType() != eletype) continue;
      if (auto* scatra_ele = dynamic_cast<Discret::ELEMENTS::Shell7pScatra*>(tmpele))
      {
        for (int k = 0; k < scatra_ele->num_node(); ++k)
        {
          if (scatra_ele->Nodes()[k] == act_node)
          {
            const auto nodal_directors = scatra_ele->GetDirectors();
            for (int dim = 0; dim < num_dim; ++dim)
              collaverdir(dim, num_directors) = nodal_directors(k, dim);
            ++num_directors;
            FOUR_C_ASSERT(num_directors <= max_ele, "max_ele too small");
            break;
          }
        }
      }
      else if (auto* shell_ele = dynamic_cast<Discret::ELEMENTS::Shell7p*>(tmpele))
      {
        for (int k = 0; k < shell_ele->num_node(); ++k)
        {
          if (shell_ele->Nodes()[k] == act_node)
          {
            const auto nodal_directors = shell_ele->GetDirectors();
            for (int dim = 0; dim < num_dim; ++dim)
              collaverdir(dim, num_directors) = nodal_directors(k, dim);
            ++num_directors;
            FOUR_C_ASSERT(num_directors <= max_ele, "max_ele too small");
            break;
          }
        }
      }
      else
        FOUR_C_THROW("Element is not a shell element");
    }
    FOUR_C_ASSERT(num_directors, "Number of neighboring nodes to a node is zero");

    if (num_directors == 1)  // no averaging if number of neighboring elements to a node is one
    {
      director_map[act_node->Id()].resize(num_dim);
      for (int dim = 0; dim < num_dim; ++dim)
        director_map[act_node->Id()][dim] = collaverdir(dim, 0);
    }
    else  // average director at node actnode
    {
      Core::LinAlg::Matrix<num_dim, 1> nodal_director(true);
      AverageDirector(collaverdir, num_directors, nodal_director);
      director_map[act_node->Id()].resize(num_dim);
      for (int dim = 0; dim < num_dim; ++dim)
        director_map[act_node->Id()][dim] = nodal_director(dim);
    }
  }
}

void STR::UTILS::Shell::Director::SetupShellElementDirectors(
    const Core::Elements::ElementType& eletype, const Core::FE::Discretization& dis)
{
  for (const auto& actele : dis.MyColElementRange())
  {
    if (actele->ElementType() != eletype) return;
    if (auto* scatra_ele = dynamic_cast<Discret::ELEMENTS::Shell7pScatra*>(actele))
    {
      // create matrix nodal_directors for nodal basis vector in thickness direction in material
      // configuration
      const int num_node = scatra_ele->num_node();
      Core::LinAlg::SerialDenseMatrix nodal_directors(
          num_node, Discret::ELEMENTS::Shell::DETAIL::num_dim);
      SetupDirectorForElement(*scatra_ele, nodal_directors);
      scatra_ele->set_all_nodal_directors(nodal_directors);
    }
    else if (auto* shell_ele = dynamic_cast<Discret::ELEMENTS::Shell7p*>(actele))
    {
      // create matrix nodal_directors for nodal basis vector in thickness direction in material
      // configuration
      const int num_node = shell_ele->num_node();
      Core::LinAlg::SerialDenseMatrix nodal_directors(
          num_node, Discret::ELEMENTS::Shell::DETAIL::num_dim);
      SetupDirectorForElement(*shell_ele, nodal_directors);
      shell_ele->set_all_nodal_directors(nodal_directors);
    }
    else
      FOUR_C_THROW("Element is not a shell element");
  }

  std::map<int, std::vector<double>> director_map;
  AverageDirectorsAtNodes(eletype, dis, director_map);

  ExportDirectorMapFromRowToColMap(eletype, dis, director_map);
}



void STR::UTILS::Shell::LumpMassMatrix(Core::LinAlg::SerialDenseMatrix& mass_matrix)
{
  // lump mass matrix
  FOUR_C_ASSERT(mass_matrix.numRows() == mass_matrix.numCols(),
      "The provided mass matrix is not a square matrix!");

  // we assume mass is a square matrix
  for (int c = 0; c < mass_matrix.numCols(); ++c)  // parse columns
  {
    double d = 0.0;
    for (int r = 0; r < mass_matrix.numRows(); ++r)  // parse rows
    {
      d += mass_matrix(r, c);  // accumulate row entries
      mass_matrix(r, c) = 0.0;
    }
    mass_matrix(c, c) = d;  // apply sum of row entries on diagonal
  }
}


void STR::UTILS::Shell::ReadElement::ReadAndSetLockingTypes(const Core::FE::CellType& distype,
    Input::LineDefinition* linedef, STR::ELEMENTS::ShellLockingTypes& locking_types)
{
  std::string type;
  switch (distype)
  {
    case Core::FE::CellType::quad4:
    {
      linedef->extract_string("EAS", type);
      SetMembraneLockingSizeQuad4(locking_types.membrane, type);
      linedef->extract_string("EAS2", type);
      SetBendingLockingSizeQuad4(locking_types.bending, type);
      linedef->extract_string("EAS3", type);
      SetThicknessLockingSizeQuad4(locking_types.thickness, type);
      linedef->extract_string("EAS4", type);
      SetShearStrainLockingSizeQuad4(locking_types.transverse_shear_strain_const, type);
      linedef->extract_string("EAS5", type);
      SetShearStrainLockingSizeQuad4(locking_types.transverse_shear_strain_lin, type);
      break;
    }
    case Core::FE::CellType::quad9:
    {
      linedef->extract_string("EAS", type);
      SetMembraneLockingSizeQuad9(locking_types.membrane, type);
      linedef->extract_string("EAS2", type);
      SetBendingLockingSizeQuad9(locking_types.bending, type);
      linedef->extract_string("EAS3", type);
      SetThicknessLockingSizeQuad9(locking_types.thickness, type);
      linedef->extract_string("EAS4", type);
      SetShearStrainLockingSizeQuad9(locking_types.transverse_shear_strain_const, type);
      linedef->extract_string("EAS5", type);
      SetShearStrainLockingSizeQuad9(locking_types.transverse_shear_strain_lin, type);
      break;
    }
    default:
      FOUR_C_THROW("EAS is not supported with %s", distype);
  }
  locking_types.total = locking_types.membrane + locking_types.bending + locking_types.thickness +
                        locking_types.transverse_shear_strain_const +
                        locking_types.transverse_shear_strain_lin;
}

int STR::UTILS::Shell::ReadElement::ReadAndSetElementMaterial(Input::LineDefinition* linedef)
{
  int material = 0;
  linedef->extract_int("MAT", material);
  return material;
}

int STR::UTILS::Shell::ReadElement::ReadAndSetNumANS(const Core::FE::CellType& distype)
{
  switch (distype)
  {
    case Core::FE::CellType::quad4:
    {
      return 2;
    }
    case Core::FE::CellType::quad9:
    {
      return 6;
    }
    default:
      FOUR_C_THROW("ANS is not supported with %s", distype);
  }
}
FOUR_C_NAMESPACE_CLOSE