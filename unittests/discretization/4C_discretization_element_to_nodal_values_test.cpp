// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_utils.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_global_data.hpp"
#include "4C_io_gridgenerator.hpp"
#include "4C_io_pstream.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_stvenantkirchhoff.hpp"
#include "4C_utils_singleton_owner.hpp"

namespace
{
  using namespace FourC;

  void create_material_in_global_problem()
  {
    Core::IO::InputParameterContainer mat_stvenant;
    mat_stvenant.add("YOUNG", 20.0);
    mat_stvenant.add("NUE", 0.3);
    mat_stvenant.add("DENS", 1.0);

    Global::Problem::instance()->materials()->insert(
        1, Mat::make_parameter(1, Core::Materials::MaterialType::m_stvenant, mat_stvenant));
  }

  // Serial discretization tests
  class ElementToNodalValuesTest : public testing::Test
  {
   public:
    ElementToNodalValuesTest()
    {
      create_material_in_global_problem();

      comm_ = MPI_COMM_WORLD;
      test_discretization_ = std::make_shared<Core::FE::Discretization>("structure", comm_, 3);

      Core::IO::cout.setup(false, false, false, Core::IO::standard, comm_, 0, 0, "dummyFilePrefix");

      // results in 27 nodes
      inputData_.bottom_corner_point_ = std::array<double, 3>{0.0, 0.0, 0.0};
      inputData_.top_corner_point_ = std::array<double, 3>{1.0, 1.0, 1.0};
      inputData_.interval_ = std::array<int, 3>{2, 2, 2};
      inputData_.node_gid_of_first_new_node_ = 0;

      inputData_.elementtype_ = "SOLID";
      inputData_.distype_ = "HEX8";
      inputData_.elearguments_ = "MAT 1 KINEM nonlinear";

      Core::IO::GridGenerator::create_rectangular_cuboid_discretization(
          *test_discretization_, inputData_, true);

      test_discretization_->fill_complete(false, false, false);
    }

    void TearDown() override { Core::IO::cout.close(); }

   protected:
    Core::IO::GridGenerator::RectangularCuboidInputs inputData_{};
    std::shared_ptr<Core::FE::Discretization> test_discretization_;
    MPI_Comm comm_;

    Core::Utils::SingletonOwnerRegistry::ScopeGuard guard;
  };

  // Routine to average uniform scalar element material values into adjacent nodes
  TEST_F(ElementToNodalValuesTest, ElementToNodalUniformScalarMaterialValues)
  {
    Core::LinAlg::Vector<double> overlapping_element_material_vector =
        Core::LinAlg::Vector<double>(*test_discretization_->element_col_map(), true);

    auto get_element_material_vector = [&](Core::Elements::Element& ele)
    {
      auto material = std::dynamic_pointer_cast<Mat::StVenantKirchhoff>(ele.material());
      const double material_parameter = material->youngs();

      overlapping_element_material_vector.ReplaceGlobalValue(ele.id(), 0, material_parameter);
    };

    test_discretization_->evaluate(get_element_material_vector);

    Core::LinAlg::MultiVector<double> nodal_value_vector =
        Core::FE::Utils::average_element_to_nodal_values(
            *test_discretization_, overlapping_element_material_vector);

    std::array<double, 27> nodal_values;
    nodal_value_vector.ExtractCopy(nodal_values.data(), nodal_value_vector.MyLength());

    // vector holds a constant value of 20 at each node, we test the first and last one
    EXPECT_NEAR(nodal_values[0], 20.0, 1e-12);
    EXPECT_NEAR(nodal_values[26], 20.0, 1e-12);
  }

  // Test for a heterogeneous scalar distribution with a sharp boundary
  TEST_F(ElementToNodalValuesTest, ElementToNodalHeterogeneousScalarValue)
  {
    Core::LinAlg::Vector<double> overlapping_element_material_vector =
        Core::LinAlg::Vector<double>(*test_discretization_->element_col_map(), true);

    // first half of the domain gets value 1.0
    overlapping_element_material_vector.ReplaceGlobalValue(0, 0, 1.0);
    overlapping_element_material_vector.ReplaceGlobalValue(1, 0, 1.0);
    overlapping_element_material_vector.ReplaceGlobalValue(2, 0, 1.0);
    overlapping_element_material_vector.ReplaceGlobalValue(3, 0, 1.0);

    // second half of the domain gets value 100.0
    overlapping_element_material_vector.ReplaceGlobalValue(4, 0, 100.0);
    overlapping_element_material_vector.ReplaceGlobalValue(5, 0, 100.0);
    overlapping_element_material_vector.ReplaceGlobalValue(6, 0, 100.0);
    overlapping_element_material_vector.ReplaceGlobalValue(7, 0, 100.0);

    Core::LinAlg::MultiVector<double> nodal_value_vector =
        Core::FE::Utils::average_element_to_nodal_values(
            *test_discretization_, overlapping_element_material_vector);

    std::array<double, 27> nodal_values;
    nodal_value_vector.ExtractCopy(nodal_values.data(), nodal_value_vector.MyLength());

    // nodes inside the first half of domain hold value 1.0
    EXPECT_NEAR(nodal_values[0], 1.0, 1e-12);
    // nodes on the boundary hold the average so 50.5
    EXPECT_NEAR(nodal_values[9], 50.5, 1e-12);
    // nodes inside the second half of domain hold value 100.0
    EXPECT_NEAR(nodal_values[18], 100.0, 1e-12);
  }

  // Test for a heterogeneous anisotropic tensor distribution with a sharp boundary
  TEST_F(ElementToNodalValuesTest, ElementToNodalHeterogeneousTensorValue)
  {
    Core::LinAlg::MultiVector<double> overlapping_element_material_vector =
        Core::LinAlg::MultiVector<double>(*test_discretization_->element_col_map(), 3, true);

    // first half of the domain gets value 1.0
    overlapping_element_material_vector.ReplaceGlobalValue(0, 0, 1.0);
    overlapping_element_material_vector.ReplaceGlobalValue(0, 1, 100.0);
    overlapping_element_material_vector.ReplaceGlobalValue(0, 2, 1.0);

    overlapping_element_material_vector.ReplaceGlobalValue(1, 0, 1.0);
    overlapping_element_material_vector.ReplaceGlobalValue(1, 1, 100.0);
    overlapping_element_material_vector.ReplaceGlobalValue(1, 2, 1.0);

    overlapping_element_material_vector.ReplaceGlobalValue(2, 0, 1.0);
    overlapping_element_material_vector.ReplaceGlobalValue(2, 1, 100.0);
    overlapping_element_material_vector.ReplaceGlobalValue(2, 2, 1.0);

    overlapping_element_material_vector.ReplaceGlobalValue(3, 0, 1.0);
    overlapping_element_material_vector.ReplaceGlobalValue(3, 1, 100.0);
    overlapping_element_material_vector.ReplaceGlobalValue(3, 2, 1.0);

    // second half of the domain gets value 100.0
    overlapping_element_material_vector.ReplaceGlobalValue(4, 0, 100.0);
    overlapping_element_material_vector.ReplaceGlobalValue(4, 1, 1.0);
    overlapping_element_material_vector.ReplaceGlobalValue(4, 2, 100.0);

    overlapping_element_material_vector.ReplaceGlobalValue(5, 0, 100.0);
    overlapping_element_material_vector.ReplaceGlobalValue(5, 1, 1.0);
    overlapping_element_material_vector.ReplaceGlobalValue(5, 2, 100.0);

    overlapping_element_material_vector.ReplaceGlobalValue(6, 0, 100.0);
    overlapping_element_material_vector.ReplaceGlobalValue(6, 1, 1.0);
    overlapping_element_material_vector.ReplaceGlobalValue(6, 2, 100.0);

    overlapping_element_material_vector.ReplaceGlobalValue(7, 0, 100.0);
    overlapping_element_material_vector.ReplaceGlobalValue(7, 1, 1.0);
    overlapping_element_material_vector.ReplaceGlobalValue(7, 2, 100.0);

    Core::LinAlg::MultiVector<double> nodal_value_vector =
        Core::FE::Utils::average_element_to_nodal_values(
            *test_discretization_, overlapping_element_material_vector);

    std::array<double, 81> nodal_values;
    nodal_value_vector.ExtractCopy(nodal_values.data(), nodal_value_vector.MyLength());

    // First multivector column
    {
      // nodes inside the first half of domain hold value 1.0
      EXPECT_NEAR(nodal_values[0], 1.0, 1e-12);
      // nodes on the boundary hold the average so 50.5
      EXPECT_NEAR(nodal_values[9], 50.5, 1e-12);
      // nodes inside the second half of domain hold value 100.0
      EXPECT_NEAR(nodal_values[18], 100.0, 1e-12);
    }

    // Second multivector column
    {
      // nodes inside the first half of domain hold value 100.0
      EXPECT_NEAR(nodal_values[0 + nodal_value_vector.MyLength()], 100.0, 1e-12);
      // nodes on the boundary hold the average so 50.5
      EXPECT_NEAR(nodal_values[9 + nodal_value_vector.MyLength()], 50.5, 1e-12);
      // nodes inside the second half of domain hold value 1.0
      EXPECT_NEAR(nodal_values[18 + nodal_value_vector.MyLength()], 1.0, 1e-12);
    }

    // Third multivector column
    {
      // nodes inside the first half of domain hold value 1.0
      EXPECT_NEAR(nodal_values[0 + 2 * nodal_value_vector.MyLength()], 1.0, 1e-12);
      // nodes on the boundary hold the average so 50.5
      EXPECT_NEAR(nodal_values[9 + 2 * nodal_value_vector.MyLength()], 50.5, 1e-12);
      // nodes inside the second half of domain hold value 100.0
      EXPECT_NEAR(nodal_values[18 + 2 * nodal_value_vector.MyLength()], 100.0, 1e-12);
    }
  }
}  // namespace
