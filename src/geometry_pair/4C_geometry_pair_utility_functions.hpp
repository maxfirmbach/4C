// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GEOMETRY_PAIR_UTILITY_FUNCTIONS_HPP
#define FOUR_C_GEOMETRY_PAIR_UTILITY_FUNCTIONS_HPP


#include "4C_config.hpp"

#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_utility_classes.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_utils_fad.hpp"

#include <string>

FOUR_C_NAMESPACE_OPEN

// Forward declarations.
namespace GeometryPair
{
  enum class DiscretizationTypeGeometry;
}


namespace GeometryPair
{
  /**
   * @brief Maps the number of nodes of a line element to its finite-element cell type.
   *
   * The mapping is evaluated at compile time. Line elements with two, three, or four
   * nodes are mapped to @c line2, @c line3, and @c line4, respectively. Unsupported
   * node counts are mapped to @c Core::FE::CellType::dis_none.
   *
   * @tparam n_nodes Number of nodes of the line element.
   */
  template <unsigned int n_nodes>
  inline constexpr Core::FE::CellType line_n_nodes_to_cell_type = Core::FE::CellType::dis_none;

  template <>
  inline constexpr auto line_n_nodes_to_cell_type<2> = Core::FE::CellType::line2;

  template <>
  inline constexpr auto line_n_nodes_to_cell_type<3> = Core::FE::CellType::line3;

  template <>
  inline constexpr auto line_n_nodes_to_cell_type<4> = Core::FE::CellType::line4;

  /**
   * \brief Convert the enum DiscretizationTypeGeometry to a human readable string.
   * @param discretization_type (in)
   * @return Human readable string representation of the enum.
   */
  std::string discretization_type_geometry_to_string(
      const DiscretizationTypeGeometry discretization_type);

  /**
   * \brief Set a line-to-xxx segment from a segment with a different scalar type (all dependencies
   * of FAD variables will be deleted).
   *
   * @param vector_scalar_type_ptr (in) Pointer to a vector of arbitrary scalar type.
   * @param vector_double (out) Temp reference to double vector (this has to come from the outside
   * scope).
   * @return Pointer on the double vector.
   *
   * @tparam A Scalar type of segment_in.
   * @tparam B Scalar type of segment_out.
   */
  template <typename A, typename B>
  void copy_segment(const LineSegment<A>& segment_in, LineSegment<B>& segment_out)
  {
    // Add the start and end points.
    segment_out.get_start_point().set_from_other_point_double(segment_in.get_start_point());
    segment_out.get_end_point().set_from_other_point_double(segment_in.get_end_point());

    // Add the projection points.
    const auto n_points = segment_in.get_number_of_projection_points();
    const std::vector<ProjectionPoint1DTo3D<A>>& projection_points_in =
        segment_in.get_projection_points();
    std::vector<ProjectionPoint1DTo3D<B>>& projection_points_out =
        segment_out.get_projection_points();
    projection_points_out.resize(n_points);
    for (unsigned int i_point = 0; i_point < n_points; i_point++)
      projection_points_out[i_point].set_from_other_point_double(projection_points_in[i_point]);
  }

  /**
   * @brief Add the current displacement state of a pair to an output stream
   */
  template <typename PairType, typename ScalarType, typename... OptionalType>
  void print_pair_information(std::ostream& out, const PairType* pair,
      const ElementData<typename PairType::t_line, ScalarType>& element_data_line,
      const ElementData<typename PairType::t_other, ScalarType>& element_data_other)
  {
    using line = typename PairType::t_line;
    using other = typename PairType::t_other;

    constexpr auto max_precision{std::numeric_limits<double>::digits10 + 1};
    out << std::setprecision(max_precision);

    const auto* face_element = dynamic_cast<const Core::Elements::FaceElement*>(pair->element2());
    if (face_element == nullptr)
    {
      out << "Pair consisting of the line with GID " << pair->element1()->id()
          << " and the other element GID " << pair->element2()->id() << "\n";
    }
    else
    {
      out << "Pair consisting of the line with GID " << pair->element1()->id()
          << " and the other element " << pair->element2()->id() << " with the parent element GID "
          << face_element->parent_element_id() << "\n";
    }
    out << "Line:";
    PrintElementData<line>::print(element_data_line, out);
    out << "Other:";
    PrintElementData<other>::print(element_data_other, out);
  }

  /**
   * \brief Create the element ID to element pointer map.
   */
  std::unordered_map<int, const Core::Elements::Element*> condition_to_element_id_map(
      const Core::Conditions::Condition& condition);
}  // namespace GeometryPair


FOUR_C_NAMESPACE_CLOSE

#endif
