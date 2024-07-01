/*----------------------------------------------------------------------*/
/*! \file

\brief unit testing functionality for the power constact constitutivelaw

\level 2

*/
/*----------------------------------------------------------------------*/

#include <gtest/gtest.h>

#include "4C_contact_constitutivelaw_power_contactconstitutivelaw.hpp"
#include "4C_contact_node.hpp"

namespace
{
  using namespace FourC;

  class PowerConstitutiveLawTest : public ::testing::Test
  {
   public:
    PowerConstitutiveLawTest()
    {
      /// initialize container for material parameters
      const Teuchos::RCP<CONTACT::CONSTITUTIVELAW::Container> container =
          Teuchos::rcp(new CONTACT::CONSTITUTIVELAW::Container(
              1, Inpar::CONTACT::ConstitutiveLawType::colaw_power, "Power Constitutivelaw"));

      // add parameters to container
      container->Add("A", 3.0);
      container->Add("B", 3.0);
      container->Add("Offset", 0.5);

      const Teuchos::RCP<CONTACT::CONSTITUTIVELAW::ConstitutiveLaw> powercoconstlaw =
          CONTACT::CONSTITUTIVELAW::ConstitutiveLaw::Factory(container);
      coconstlaw_ = powercoconstlaw;
    }

    Teuchos::RCP<CONTACT::CONSTITUTIVELAW::ConstitutiveLaw> coconstlaw_;

    Teuchos::RCP<CONTACT::Node> cnode;
  };

  //! test member function Evaluate
  TEST_F(PowerConstitutiveLawTest, TestEvaluate)
  {
    // gap < 0
    EXPECT_ANY_THROW(coconstlaw_->evaluate(1.0, cnode.get()));
    // 0< gap < offset
    EXPECT_ANY_THROW(coconstlaw_->evaluate(-0.25, cnode.get()));
    // offset < gap
    EXPECT_NEAR(coconstlaw_->evaluate(-0.75, cnode.get()), -0.046875, 1.e-15);
  }

  //! test member function EvaluateDeriv
  TEST_F(PowerConstitutiveLawTest, TestEvaluateDeriv)
  {
    EXPECT_NEAR(coconstlaw_->EvaluateDeriv(-0.75, cnode.get()), 0.5625, 1.e-15);
    EXPECT_ANY_THROW(coconstlaw_->EvaluateDeriv(-0.25, cnode.get()));
  }
}  // namespace