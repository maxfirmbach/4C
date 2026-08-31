// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// Shared test utilities for reduced_lung tests
#ifndef FOUR_C_REDUCED_LUNG_TEST_UTILS_TEST_HPP
#define FOUR_C_REDUCED_LUNG_TEST_UTILS_TEST_HPP

#include <gtest/gtest.h>

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_rebalance.hpp"
#include "4C_reduced_lung_airways.hpp"
#include "4C_reduced_lung_helpers.hpp"
#include "4C_reduced_lung_terminal_unit.hpp"

#include <mpi.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung::TestUtils
{
  /**
   * Build a line2 discretization from nodal coordinates and connectivity, using the same routine
   * as the production code. Nodes and elements get 0-based ids in the order they are passed;
   * @p element_nodes is the connectivity [node_in, node_out] referring to those node ids.
   */
  inline std::unique_ptr<Core::FE::Discretization> make_line2_discretization(
      const std::string& name, const std::vector<std::array<double, 3>>& node_coordinates,
      const std::vector<std::array<int, 2>>& element_nodes)
  {
    auto discretization = std::make_unique<Core::FE::Discretization>(name, MPI_COMM_WORLD, 3);
    const Core::Rebalance::RebalanceParameters rebalance_parameters{};
    build_discretization_from_nodes_and_elements(
        *discretization, node_coordinates, element_nodes, rebalance_parameters);
    discretization->fill_complete(Core::FE::OptionsFillComplete{
        .assign_degrees_of_freedom = true,
        .init_elements = true,
        .do_boundary_conditions = false,
    });

    return discretization;
  }

  /**
   * Build a straight line discretization along the x-axis with 0-based node ids.
   * The discretization is bifurcation-free consisting of @p num_elements line2 elements of unit
   * length.
   */
  inline std::unique_ptr<Core::FE::Discretization> make_chain_discretization(
      const std::string& name, const int num_elements)
  {
    std::vector<std::array<double, 3>> node_coordinates;
    std::vector<std::array<int, 2>> element_nodes;
    for (int node_id = 0; node_id <= num_elements; ++node_id)
    {
      node_coordinates.push_back({static_cast<double>(node_id), 0.0, 0.0});
      if (node_id > 0) element_nodes.push_back({node_id - 1, node_id});
    }
    return make_line2_discretization(name, node_coordinates, element_nodes);
  }

  //! A single element splitting into two elements at its outlet node.
  inline std::unique_ptr<Core::FE::Discretization> make_bifurcation_discretization(
      const std::string& name)
  {
    const std::vector<std::array<double, 3>> node_coordinates{
        {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 1.0, 0.0}, {2.0, -1.0, 0.0}};
    const std::vector<std::array<int, 2>> element_nodes{{0, 1}, {1, 2}, {1, 3}};
    return make_line2_discretization(name, node_coordinates, element_nodes);
  }

  // Generalized helper to check a Jacobian column against a central finite-difference
  // approximation of the residual. Works for both TerminalUnitModel and AirwayModel as long
  // as the model exposes `data`, `residual_evaluator` and the data structure
  // provides `number_of_elements()` and per-element lid vectors passed in `dof_lids`.
  template <typename Model>
  void check_jacobian_column_against_fd(const std::vector<int>& dof_lids, int jac_col, Model& model,
      Core::LinAlg::SparseMatrix& jac, Core::LinAlg::Vector<double>& locally_relevant_dofs,
      double dt, double eps, const Core::LinAlg::Map& row_map)
  {
    SCOPED_TRACE(
        std::string("Comparing FD approximation with Jacobian column ") + std::to_string(jac_col));

    // Perturb in +epsilon direction
    for (int lid : dof_lids) locally_relevant_dofs.get_values()[lid] += eps;
    Core::LinAlg::Vector<double> res_plus(row_map, true);
    model.internal_state_updater(model.data, locally_relevant_dofs, dt);
    model.residual_evaluator(model.data, res_plus, locally_relevant_dofs, dt);

    // Perturb in -epsilon direction
    for (int lid : dof_lids) locally_relevant_dofs.get_values()[lid] -= 2 * eps;
    Core::LinAlg::Vector<double> res_minus(row_map, true);
    model.internal_state_updater(model.data, locally_relevant_dofs, dt);
    model.residual_evaluator(model.data, res_minus, locally_relevant_dofs, dt);

    // Restore original state
    for (int lid : dof_lids) locally_relevant_dofs.get_values()[lid] += eps;

    // Compute FD approximation
    Core::LinAlg::Vector<double> fd_derivative(row_map, true);
    fd_derivative.update(1.0 / (2 * eps), res_plus, -1.0 / (2 * eps), res_minus, 0.0);

    // Compare with Jacobian column
    const int n_rows_per_element = []<typename DataType>(const DataType& data)
    {
      if constexpr (requires { data.n_state_equations; })
      {
        return data.n_state_equations;
      }
      else
      {
        return 1;
      }
    }(model.data);

    for (size_t i = 0; i < model.data.number_of_elements(); ++i)
    {
      const int row_id = model.data.local_row_id[i];
      for (int row_offset = 0; row_offset < n_rows_per_element; ++row_offset)
      {
        int n_entries = 0;
        double* jac_vals = nullptr;
        int* col_indices = nullptr;
        const int row = row_id + row_offset;
        jac.extract_my_row_view(row, n_entries, jac_vals, col_indices);

        int col_index = -1;
        for (int entry = 0; entry < n_entries; ++entry)
        {
          if (col_indices[entry] == dof_lids[i])
          {
            col_index = entry;
            break;
          }
        }

        ASSERT_NE(col_index, -1) << "Column for dof " << dof_lids[i] << " not found in row " << row;
        EXPECT_NEAR(jac_vals[col_index], fd_derivative.local_values_as_span()[row], eps)
            << "Mismatch at row " << row << ", col " << jac_col;
      }
    }
  }
}  // namespace ReducedLung::TestUtils

FOUR_C_NAMESPACE_CLOSE

#endif
