// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include <gmock/gmock.h>

#include "4C_utils_singleton_owner.hpp"

namespace
{
  using namespace FourC;

  class DummySingleton
  {
   private:
    //! Private constructor to mimic typical use case of singletons.
    DummySingleton() = default;

    FRIEND_TEST(TestSingletonOwner, CreatesSingleton);
    FRIEND_TEST(TestSingletonOwner, DestructsSingleton);
    FRIEND_TEST(TestSingletonOwner, ReturnsExistingInstance);
    FRIEND_TEST(TestSingletonMap, DifferentKeys);
  };

  TEST(TestSingletonOwner, CreatesSingleton)
  {
    auto singleton_owner = Core::Utils::make_singleton_owner(
        []() { return std::unique_ptr<DummySingleton>(new DummySingleton()); });

    // Expect that the returned object is of DummySingleton type
    EXPECT_TRUE(dynamic_cast<DummySingleton*>(
        singleton_owner.instance(Core::Utils::SingletonAction::create)));
  }

  TEST(TestSingletonOwner, DestructsSingleton)
  {
    Core::Utils::SingletonOwner<DummySingleton> singleton_owner(
        []() { return std::unique_ptr<DummySingleton>(new DummySingleton()); });

    // Create a singleton to destruct it in the following
    singleton_owner.instance(Core::Utils::SingletonAction::create);

    // Expect that a nullptr is returned at destruction
    EXPECT_EQ(singleton_owner.instance(Core::Utils::SingletonAction::destruct), nullptr);
  }

  TEST(TestSingletonOwner, ReturnsExistingInstance)
  {
    struct Creator
    {
      MOCK_METHOD((std::unique_ptr<DummySingleton>), create, (), (const));
    };
    Creator creator;

    // Return a new DummySingleton exactly once, otherwise the test fails
    EXPECT_CALL(creator, create)
        .WillOnce([]() { return std::unique_ptr<DummySingleton>(new DummySingleton()); });

    Core::Utils::SingletonOwner<DummySingleton> singleton_owner{
        [&creator]() { return creator.create(); }};

    DummySingleton* ptr_1 = singleton_owner.instance(Core::Utils::SingletonAction::create);
    DummySingleton* ptr_2 = singleton_owner.instance(Core::Utils::SingletonAction::create);

    // Expect that both pointers point to the same object
    EXPECT_EQ(ptr_1, ptr_2);
  }

  TEST(TestSingletonMap, DifferentKeys)
  {
    struct Creator
    {
      MOCK_METHOD((std::unique_ptr<DummySingleton>), create, (), (const));
    };
    Creator creator;

    // Return a new DummySingleton exactly twice (for the two keys tested below)
    EXPECT_CALL(creator, create)
        .Times(2)
        .WillRepeatedly([]() { return std::unique_ptr<DummySingleton>(new DummySingleton()); });

    auto singleton_map =
        Core::Utils::make_singleton_map<std::string>([&creator]() { return creator.create(); });


    auto* a = singleton_map["a"].instance(Core::Utils::SingletonAction::create);
    auto* b = singleton_map["b"].instance(Core::Utils::SingletonAction::create);

    EXPECT_NE(a, b);
    EXPECT_EQ(singleton_map["a"].instance(Core::Utils::SingletonAction::create), a);
    EXPECT_EQ(singleton_map["b"].instance(Core::Utils::SingletonAction::create), b);
  }

  TEST(TestSingletonMap, ForwardConstructorArgs)
  {
    struct DummyWithArgs
    {
      DummyWithArgs(int a) : a(a) {}

      int a;
    };
    auto singleton_map = Core::Utils::make_singleton_map<std::string>(
        [](int input) { return std::make_unique<DummyWithArgs>(input); });

    EXPECT_EQ(singleton_map["a"].instance(Core::Utils::SingletonAction::create, 2)->a, 2);
  }
}  // namespace
