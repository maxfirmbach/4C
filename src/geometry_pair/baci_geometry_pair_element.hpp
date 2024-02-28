/*----------------------------------------------------------------------*/
/*! \file

\brief Element types that can be part of a pair. This types can be used as a template argument.

\level 1
*/
// End doxygen header.


#ifndef BACI_GEOMETRY_PAIR_ELEMENT_HPP
#define BACI_GEOMETRY_PAIR_ELEMENT_HPP


#include "baci_config.hpp"

#include "baci_beam3_base.hpp"
#include "baci_discretization_fem_general_utils_fem_shapefunctions.hpp"
#include "baci_discretization_fem_general_utils_nurbs_shapefunctions.hpp"
#include "baci_geometry_pair_element_classes.hpp"
#include "baci_global_data.hpp"
#include "baci_lib_element.hpp"
#include "baci_nurbs_discret_nurbs_utils.hpp"

BACI_NAMESPACE_OPEN

namespace GEOMETRYPAIR
{
  /**
   * \brief Shortcut to typed that are used to overload the get shape function methods.
   */
  using t_1D_element = std::integral_constant<unsigned int, 1>;
  using t_2D_element = std::integral_constant<unsigned int, 2>;
  using t_3D_element = std::integral_constant<unsigned int, 3>;

  /**
   * \brief Base class for the geometry pair element type.
   *
   * The template parameters are stored in static const member, so they can be accessed from derived
   * classes.
   * @tparam discretization Type of shape function.
   * @tparam values_per_node Number of nodal values per node (standard elements have 1, Hermitian
   * shape functions have 2)
   */
  template <CORE::FE::CellType discretization, unsigned int values_per_node>
  class ElementDiscretizationBase
  {
   public:
    //! Type of shape function that will be used when evaluating the shape functions.
    static constexpr CORE::FE::CellType discretization_ = discretization;

    //! Dimension of element (curve=1, surface=2, volume=3).
    static constexpr unsigned int dim_ = CORE::FE::dim<discretization_>;

    //! Number of values per node.
    static constexpr unsigned int n_val_ = values_per_node;

    //! Number of nodes for this element.
    static constexpr unsigned int n_nodes_ = CORE::FE::num_nodes<discretization_>;

    //! Number of unknowns for this element.
    static constexpr unsigned int n_dof_ = 3 * n_val_ * n_nodes_;

    //! Geometry type of the element.
    static constexpr GEOMETRYPAIR::DiscretizationTypeGeometry geometry_type_ =
        ElementDiscretizationToGeometryType<discretization_>::geometry_type_;
  };


  /**
   * \brief Discretization based on standard (in most cases Lagrangian) shape functions.
   * @tparam base Type of base class for this class. By doing it this way, member of the base class
   * can easily be accessed with base::member_.
   */
  template <typename base>
  class ElementDiscretizationStandard : public base
  {
   public:
    /**
     * \brief Evaluate the 1D shape functions of the element.
     * @param N (out) Array to store shape function values in.
     * @param xi (in) Local coordinate on the element.
     * @param element (in) Pointer to the element. In this case it is not needed.
     * @param dimension (in) A type to overload the method, in order to evaluate the correct shape
     * functions for the dimension of the element.
     */
    template <typename V, typename T>
    inline static void EvaluateShapeFunction(
        V& N, const T& xi, t_1D_element dimension, const DRT::Element* element = nullptr)
    {
      // Throw a compiler error if this function is called from an element with dim_ != 1.
      static_assert(base::dim_ == 1, "EvaluateShapeFunction1D can only be called for 1D elements!");

      // We need to redefine the discretization here, as otherwise the compiler has problems
      // passing the static const member of the base class as a reference.
      CORE::FE::CellType discretization = base::discretization_;

      // Evaluate the shape functions.
      CORE::FE::shape_function_1D(N, xi, discretization);
    };

    /**
     * \brief Evaluate the first derivative of the 1D shape functions of the element.
     * @param dN (out) Array to store shape function derivatives in.
     * @param xi (in) Local coordinates on the element.
     * @param element (in) Pointer to the element. In this case it is not needed.
     * @param dimension (in) A type to overload the method, in order to evaluate the correct shape
     * functions for the dimension of the element.
     */
    template <typename V, typename T>
    inline static void EvaluateShapeFunctionDeriv1(
        V& dN, const T& xi, t_1D_element dimension, const DRT::Element* element = nullptr)
    {
      // Throw a compiler error if this function is called from an element with dim_ != 1.
      static_assert(
          base::dim_ == 1, "EvaluateShapeFunctionDeriv1 can only be called for 1D elements!");

      // We need to redefine the discretization here, as otherwise the compiler has problems
      // passing the static const member of the base class as a reference.
      CORE::FE::CellType discretization = base::discretization_;

      // Evaluate the shape functions.
      CORE::FE::shape_function_1D_deriv1(dN, xi, discretization);
    };

    /**
     * \brief Evaluate the 2D shape functions of the element.
     * @param N (out) Array to store shape function values in.
     * @param xi (in) Local coordinates on the element.
     * @param element (in) Pointer to the element. In this case it is not needed.
     * @param dimension (in) A type to overload the method, in order to evaluate the correct shape
     * functions for the dimension of the element.
     */
    template <typename V, typename T>
    inline static void EvaluateShapeFunction(
        V& N, const T& xi, t_2D_element dimension, const DRT::Element* element = nullptr)
    {
      // Throw a compiler error if this function is called from an element with dim_ != 2.
      static_assert(base::dim_ == 2, "EvaluateShapeFunction2D can only be called for 2D elements!");

      // We need to redefine the discretization here, as otherwise the compiler has problems
      // passing the static const member of the base class as a reference.
      CORE::FE::CellType discretization = base::discretization_;

      // Evaluate the shape functions.
      CORE::FE::shape_function_2D(N, xi(0), xi(1), discretization);
    };

    /**
     * \brief Evaluate the first derivative of the 2D shape functions of the element.
     * @param dN (out) Array to store shape function derivatives in.
     * @param xi (in) Local coordinates on the element.
     * @param element (in) Pointer to the element. In this case it is not needed.
     * @param dimension (in) A type to overload the method, in order to evaluate the correct shape
     * functions for the dimension of the element.
     */
    template <typename V, typename T>
    inline static void EvaluateShapeFunctionDeriv1(
        V& dN, const T& xi, t_2D_element dimension, const DRT::Element* element = nullptr)
    {
      // Throw a compiler error if this function is called from an element with dim_ != 2.
      static_assert(
          base::dim_ == 2, "EvaluateShapeFunctionDeriv1 can only be called for 2D elements!");

      // We need to redefine the discretization here, as otherwise the compiler has problems
      // passing the static const member of the base class as a reference.
      CORE::FE::CellType discretization = base::discretization_;

      // Evaluate the shape functions.
      CORE::FE::shape_function_2D_deriv1(dN, xi(0), xi(1), discretization);
    };

    /**
     * \brief Evaluate the 3D shape functions of the element.
     * @param N (out) Array to store shape function values in.
     * @param xi (in) Local coordinates on the element.
     * @param element (in) Pointer to the element. In this case it is not needed.
     * @param dimension (in) A type to overload the method, in order to evaluate the correct shape
     * functions for the dimension of the element.
     */
    template <typename V, typename T>
    inline static void EvaluateShapeFunction(
        V& N, const T& xi, t_3D_element dimension, const DRT::Element* element = nullptr)
    {
      // Throw a compiler error if this function is called from an element with dim_ != 3.
      static_assert(base::dim_ == 3, "EvaluateShapeFunction3D can only be called for 3D elements!");

      // We need to redefine the discretization here, as otherwise the compiler has problems
      // passing the static const member of the base class as a reference.
      CORE::FE::CellType discretization = base::discretization_;

      // Evaluate the shape functions.
      CORE::FE::shape_function_3D(N, xi(0), xi(1), xi(2), discretization);
    };

    /**
     * \brief Evaluate the first derivative of the 3D shape functions of the element.
     * @param dN (out) Array to store shape function derivatives in.
     * @param xi (in) Local coordinates on the element.
     * @param element (in) Pointer to the element. In this case it is not needed.
     * @param dimension (in) A type to overload the method, in order to evaluate the correct shape
     * functions for the dimension of the element.
     */
    template <typename V, typename T>
    inline static void EvaluateShapeFunctionDeriv1(
        V& dN, const T& xi, t_3D_element dimension, const DRT::Element* element = nullptr)
    {
      // Throw a compiler error if this function is called from an element with dim_ != 3.
      static_assert(base::dim_ == 3, "EvaluateShapeFunction3D can only be called for 3D elements!");

      // We need to redefine the discretization here, as otherwise the compiler has problems
      // passing the static const member of the base class as a reference.
      CORE::FE::CellType discretization = base::discretization_;

      // Evaluate the shape functions.
      CORE::FE::shape_function_3D_deriv1(dN, xi(0), xi(1), xi(2), discretization);
    };
  };  // namespace GEOMETRYPAIR


  /**
   * \brief Discretization based on Hermitian shape functions.
   * @tparam base Type of base class for this class. By doing it this way, member of the base class
   * can easily be accessed with base::member_.
   */
  template <typename base>
  class ElementDiscretizationHermite : public base
  {
   public:
    /**
     * \brief Evaluate the 1D shape functions of the element.
     * @param N (out) Array to store shape function values in.
     * @param xi (in) Local coordinate on the element.
     * @param element (in) Pointer to the element. This is needed for beam elements, as the
     * reference length goes into the shape functions.
     * @param dimension (in) A type to overload the method, in order to evaluate the correct shape
     * functions for the dimension of the element.
     */
    template <typename V, typename T>
    inline static void EvaluateShapeFunction(
        V& N, const T& xi, t_1D_element dimension, const DRT::Element* element)
    {
      // Throw a compiler error if this function is called from an element with dim_ != 1.
      static_assert(base::dim_ == 1, "EvaluateShapeFunction1D can only be called for 1D elements!");

      // We need to redefine the discretization here, as otherwise the compiler has problems
      // passing the static const member of the base class as a reference.
      CORE::FE::CellType discretization = base::discretization_;

      // Get the reference length of the beam element.
      const auto* beam_element = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(element);
      if (beam_element == nullptr)
        dserror(
            "The element pointer has to point to a valid beam element when evaluating the shape "
            "functions of a beam, as we need to get RefLength()!");
      const double length = beam_element->RefLength();

      // Evaluate the shape functions.
      CORE::FE::shape_function_hermite_1D(N, xi, length, discretization);
    };

    /**
     * \brief Evaluate the first derivative of the 1D shape functions of the element.
     * @param dN (out) Array to store shape function derivatives in.
     * @param xi (in) Local coordinate on the element.
     * @param element (in) Pointer to the element. This is needed for beam elements, as the
     * reference length goes into the shape functions.
     * @param dimension (in) A type to overload the method, in order to evaluate the correct shape
     * functions for the dimension of the element.
     */
    template <typename V, typename T>
    inline static void EvaluateShapeFunctionDeriv1(
        V& dN, const T& xi, t_1D_element dimension, const DRT::Element* element)
    {
      // Throw a compiler error if this function is called from an element with dim_ != 1.
      static_assert(base::dim_ == 1, "EvaluateShapeFunction1D can only be called for 1D elements!");

      // We need to redefine the discretization here, as otherwise the compiler has problems
      // passing the static const member of the base class as a reference.
      CORE::FE::CellType discretization = base::discretization_;

      // Get the reference length of the beam element.
      const auto* beam_element = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(element);
      if (beam_element == nullptr)
        dserror(
            "The element pointer has to point to a valid beam element when evaluating the shape "
            "functions of a beam, as we need to get RefLength()!");
      const double length = beam_element->RefLength();

      // Evaluate the shape functions.
      CORE::FE::shape_function_hermite_1D_deriv1(dN, xi, length, discretization);
    };
  };


  /**
   * \brief Discretization based on nurbs.
   * @tparam base Type of base class for this class. By doing it this way, member of the base class
   * can easily be accessed with base::member_.
   */
  template <typename base>
  class ElementDiscretizationNurbs : public base
  {
   public:
    /**
     * \brief Evaluate the 2D shape functions of the element.
     * @param N (out) Array to store shape function values in.
     * @param xi (in) Local coordinates on the element.
     * @param element (in) Pointer to the element.
     * @param dimension (in) A type to overload the method, in order to evaluate the correct shape
     * functions for the dimension of the element.
     */
    template <typename V, typename T>
    inline static void EvaluateShapeFunction(
        V& N, const T& xi, t_2D_element dimension, const DRT::Element* element)
    {
      // Throw a compiler error if this function is called from an element with dim_ != 2.
      static_assert(base::dim_ == 2, "nurbs_get_2D_funct can only be called for 2D elements!");

      // The element pointer has to be a face element.
      auto face_element = dynamic_cast<const DRT::FaceElement*>(element);
      if (face_element == nullptr)
        dserror("EvaluateShapeFunction<nurbs9> needs a face element pointer.");

      // Factor for surface orientation.
      double normalfac = 1.0;

      // Get the knots and weights for this element.
      using type_weights = CORE::LINALG::Matrix<base::n_nodes_, 1, double>;
      type_weights weights(true);
      std::vector<CORE::LINALG::SerialDenseVector> mypknots(3);
      std::vector<CORE::LINALG::SerialDenseVector> myknots(2);
      const bool zero_size = DRT::NURBS::GetKnotVectorAndWeightsForNurbsBoundary(face_element,
          face_element->FaceMasterNumber(), face_element->ParentElementId(),
          *(GLOBAL::Problem::Instance()->GetDis("structure")), mypknots, myknots, weights,
          normalfac);
      if (zero_size)
        dserror("GetKnotVectorAndWeightsForNurbsBoundary has to return a non zero size.");

      // We need to redefine the discretization here, as otherwise the compiler has problems
      // passing the static const member of the base class as a reference.
      CORE::FE::CellType discretization = base::discretization_;

      // Evaluate the shape functions.
      CORE::FE::NURBS::nurbs_get_2D_funct<V, T, type_weights, typename V::scalar_type>(
          N, xi, myknots, weights, discretization);
    };

    /**
     * \brief Evaluate the 3D shape functions of the element.
     * @param N (out) Array to store shape function values in.
     * @param xi (in) Local coordinates on the element.
     * @param element (in) Pointer to the element.
     * @param dimension (in) A type to overload the method, in order to evaluate the correct shape
     * functions for the dimension of the element.
     */
    template <typename V, typename T>
    inline static void EvaluateShapeFunction(
        V& N, const T& xi, t_3D_element dimension, const DRT::Element* element)
    {
      // Throw a compiler error if this function is called from an element with dim_ != 3.
      static_assert(base::dim_ == 3, "nurbs_get_3D_funct can only be called for 3D elements!");

      if (element == nullptr)
        dserror("EvaluateShapeFunction for nurbs needs a valid element pointer!");

      // Get the knots and weights for this element.
      CORE::LINALG::Matrix<base::n_nodes_, 1, double> weights(true);
      std::vector<CORE::LINALG::SerialDenseVector> myknots(true);
      const bool zero_size = DRT::NURBS::GetMyNurbsKnotsAndWeights(
          *(GLOBAL::Problem::Instance()->GetDis("structure")), element, myknots, weights);
      if (zero_size) dserror("GetMyNurbsKnotsAndWeights has to return a non zero size.");

      // We need to redefine the discretization here, as otherwise the compiler has problems
      // passing the static const member of the base class as a reference.
      CORE::FE::CellType discretization = base::discretization_;

      // Evaluate the shape functions.
      CORE::FE::NURBS::nurbs_get_3D_funct(N, xi, myknots, weights, discretization);
    };

    /**
     * \brief Evaluate the first derivative of the 2D shape functions of the element.
     * @param dN (out) Array to store shape function derivatives in.
     * @param xi (in) Local coordinates on the element.
     * @param element (in) Pointer to the element.
     * @param dimension (in) A type to overload the method, in order to evaluate the correct shape
     * functions for the dimension of the element.
     */
    template <typename V, typename T>
    inline static void EvaluateShapeFunctionDeriv1(
        V& dN, const T& xi, t_2D_element dimension, const DRT::Element* element)
    {
      // Throw a compiler error if this function is called from an element with dim_ != 2.
      static_assert(base::dim_ == 2, "nurbs_get_2D_funct can only be called for 2D elements!");

      // The element pointer has to be a face element.
      auto face_element = dynamic_cast<const DRT::FaceElement*>(element);
      if (face_element == nullptr)
        dserror("EvaluateShapeFunction<nurbs9> needs a face element pointer.");

      // Factor for surface orientation.
      double normalfac = 1.0;

      // Get the knots and weights for this element.
      using type_weights = CORE::LINALG::Matrix<base::n_nodes_, 1, double>;
      type_weights weights(true);
      std::vector<CORE::LINALG::SerialDenseVector> mypknots(3);
      std::vector<CORE::LINALG::SerialDenseVector> myknots(2);
      const bool zero_size = DRT::NURBS::GetKnotVectorAndWeightsForNurbsBoundary(face_element,
          face_element->FaceMasterNumber(), face_element->ParentElementId(),
          *(GLOBAL::Problem::Instance()->GetDis("structure")), mypknots, myknots, weights,
          normalfac);
      if (zero_size)
        dserror("GetKnotVectorAndWeightsForNurbsBoundary has to return a non zero size.");

      // We need to redefine the discretization here, as otherwise the compiler has problems
      // passing the static const member of the base class as a reference.
      CORE::FE::CellType discretization = base::discretization_;

      // Evaluate the shape functions.
      using type_dummy = CORE::LINALG::Matrix<base::n_nodes_, 1, typename V::scalar_type>;
      type_dummy N_dummy;
      CORE::FE::NURBS::nurbs_get_2D_funct_deriv<type_dummy, V, T, type_weights,
          typename V::scalar_type>(N_dummy, dN, xi, myknots, weights, discretization);
    };

    /**
     * \brief Evaluate the first derivative of the 3D shape functions of the element.
     * @param dN (out) Array to store shape function derivatives in.
     * @param xi (in) Local coordinates on the element.
     * @param element (in) Pointer to the element.
     * @param dimension (in) A type to overload the method, in order to evaluate the correct shape
     * functions for the dimension of the element.
     */
    template <typename V, typename T>
    inline static void EvaluateShapeFunctionDeriv1(
        V& dN, const T& xi, t_3D_element dimension, const DRT::Element* element)
    {
      // Throw a compiler error if this function is called from an element with dim_ != 3.
      static_assert(base::dim_ == 3, "EvaluateShapeFunction3D can only be called for 3D elements!");

      if (element == nullptr)
        dserror("EvaluateShapeFunctionDeriv1 for nurbs needs a valid element pointer!");

      // Get the knots and weights for this element.
      CORE::LINALG::Matrix<base::n_nodes_, 1, double> weights(true);
      std::vector<CORE::LINALG::SerialDenseVector> myknots(true);
      const bool zero_size = DRT::NURBS::GetMyNurbsKnotsAndWeights(
          *(GLOBAL::Problem::Instance()->GetDis("structure")), element, myknots, weights);
      if (zero_size) dserror("GetMyNurbsKnotsAndWeights has to return a non zero size.");

      // We need to redefine the discretization here, as otherwise the compiler has problems
      // passing the static const member of the base class as a reference.
      CORE::FE::CellType discretization = base::discretization_;

      // Evaluate the shape functions.
      CORE::LINALG::Matrix<base::n_nodes_, 1, typename V::scalar_type> dummy;
      CORE::FE::NURBS::nurbs_get_3D_funct_deriv(dummy, dN, xi, myknots, weights, discretization);
    };
  };


  /**
   * Shortcuts to element types are created here, so the explicit template initialisations are
   * better readable.
   */

  //! 1D elements
  using t_hermite =
      ElementDiscretizationHermite<ElementDiscretizationBase<CORE::FE::CellType::line2, 2>>;
  using t_line2 =
      ElementDiscretizationStandard<ElementDiscretizationBase<CORE::FE::CellType::line2, 1>>;
  using t_line3 =
      ElementDiscretizationStandard<ElementDiscretizationBase<CORE::FE::CellType::line3, 1>>;
  using t_line4 =
      ElementDiscretizationStandard<ElementDiscretizationBase<CORE::FE::CellType::line4, 1>>;

  //! 2D elements
  using t_tri3 =
      ElementDiscretizationStandard<ElementDiscretizationBase<CORE::FE::CellType::tri3, 1>>;
  using t_tri6 =
      ElementDiscretizationStandard<ElementDiscretizationBase<CORE::FE::CellType::tri6, 1>>;
  using t_quad4 =
      ElementDiscretizationStandard<ElementDiscretizationBase<CORE::FE::CellType::quad4, 1>>;
  using t_quad8 =
      ElementDiscretizationStandard<ElementDiscretizationBase<CORE::FE::CellType::quad8, 1>>;
  using t_quad9 =
      ElementDiscretizationStandard<ElementDiscretizationBase<CORE::FE::CellType::quad9, 1>>;
  using t_nurbs9 =
      ElementDiscretizationNurbs<ElementDiscretizationBase<CORE::FE::CellType::nurbs9, 1>>;

  //! 3D elements
  using t_hex8 =
      ElementDiscretizationStandard<ElementDiscretizationBase<CORE::FE::CellType::hex8, 1>>;
  using t_hex20 =
      ElementDiscretizationStandard<ElementDiscretizationBase<CORE::FE::CellType::hex20, 1>>;
  using t_hex27 =
      ElementDiscretizationStandard<ElementDiscretizationBase<CORE::FE::CellType::hex27, 1>>;
  using t_tet4 =
      ElementDiscretizationStandard<ElementDiscretizationBase<CORE::FE::CellType::tet4, 1>>;
  using t_tet10 =
      ElementDiscretizationStandard<ElementDiscretizationBase<CORE::FE::CellType::tet10, 1>>;
  using t_nurbs27 =
      ElementDiscretizationNurbs<ElementDiscretizationBase<CORE::FE::CellType::nurbs27, 1>>;

}  // namespace GEOMETRYPAIR

BACI_NAMESPACE_CLOSE

#endif