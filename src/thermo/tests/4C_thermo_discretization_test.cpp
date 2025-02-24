// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_global_data.hpp"
#include "4C_global_legacy_module.hpp"
#include "4C_io_gridgenerator.hpp"
#include "4C_io_pstream.hpp"
#include "4C_mat_fourier.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_thermo_element.hpp"
#include "4C_utils_singleton_owner.hpp"


namespace
{
  using namespace FourC;

  void create_material_in_global_problem()
  {
    Core::IO::InputParameterContainer mat_fourier;
    mat_fourier.add("CAPA", 420.0);
    mat_fourier.add("CONDUCT", std::vector<double>{1.0});
    mat_fourier.add("CONDUCT_PARA_NUM", 1);

    Global::Problem::instance()->materials()->insert(
        1, Mat::make_parameter(1, Core::Materials::MaterialType::m_thermo_fourier, mat_fourier));
  }

  class BuildSerialThermoDiscretizationTest : public testing::Test
  {
   public:
    BuildSerialThermoDiscretizationTest()
    {
      global_legacy_module_callbacks().RegisterParObjectTypes();

      create_material_in_global_problem();

      comm_ = MPI_COMM_WORLD;
      test_discretization_ = std::make_shared<Core::FE::Discretization>("thermo", comm_, 3);

      Core::IO::cout.setup(false, false, false, Core::IO::standard, comm_, 0, 0, "thermo");

      // results in 27 nodes
      inputData_.bottom_corner_point_ = std::array<double, 3>{0.0, 0.0, 0.0};
      inputData_.top_corner_point_ = std::array<double, 3>{1.0, 1.0, 1.0};
      inputData_.interval_ = std::array<int, 3>{2, 2, 2};
      inputData_.node_gid_of_first_new_node_ = 0;

      inputData_.elementtype_ = "THERMO";
      inputData_.distype_ = "HEX8";
      inputData_.elearguments_ = "MAT 1";

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

  // Small test checking the correct setup of the thermo discretization
  TEST_F(BuildSerialThermoDiscretizationTest, SetupThermoDiscretization)
  {
    // check for correct amount of nodes and elements
    EXPECT_EQ(test_discretization_->node_row_map()->NumGlobalElements(), 27);
    EXPECT_EQ(test_discretization_->element_row_map()->NumGlobalElements(), 8);

    // check if material is set correctly
    auto thermo_material =
        std::dynamic_pointer_cast<Mat::Fourier>(test_discretization_->l_row_element(0)->material());
    EXPECT_FALSE(thermo_material == nullptr);

    // check if element is set correctly
    auto* thermo_element =
        dynamic_cast<Discret::Elements::Thermo*>(test_discretization_->l_row_element(0));
    EXPECT_FALSE(thermo_element == nullptr);
  }
}  // namespace
