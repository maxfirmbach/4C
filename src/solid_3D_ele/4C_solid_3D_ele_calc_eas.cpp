/*! \file

\brief Implementation of routines for calculation of solid element with EAS element technology

\level 1
*/

#include "4C_solid_3D_ele_calc_eas.hpp"

#include "4C_comm_parobject.hpp"
#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_fixedsizematrix_generators.hpp"
#include "4C_linalg_utils_densematrix_eigen.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_solid_3D_ele.hpp"
#include "4C_solid_3D_ele_calc_lib.hpp"
#include "4C_solid_3D_ele_calc_lib_integration.hpp"
#include "4C_solid_3D_ele_calc_lib_io.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_structure_new_gauss_point_data_output_manager.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_dyn_cast.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <optional>

FOUR_C_NAMESPACE_OPEN

namespace
{
  template <Core::FE::CellType celltype>
  inline static constexpr int num_nodes = Core::FE::num_nodes<celltype>;

  template <Core::FE::CellType celltype>
  inline static constexpr int num_dim = Core::FE::dim<celltype>;

  template <Core::FE::CellType celltype>
  inline static constexpr int num_str = num_dim<celltype>*(num_dim<celltype> + 1) / 2;

  template <Core::FE::CellType celltype>
  inline static constexpr int num_dof_per_ele = num_nodes<celltype>* num_dim<celltype>;

  /*!
   * @brief Solve for the inverse of a matrix and ignore any errors
   *
   * @tparam dim : matrix dimensions
   * @param matrix(in/out) : matrix to be inverted
   */
  template <unsigned int dim>
  void solve_for_inverse_ignoring_errors(Core::LinAlg::Matrix<dim, dim>& matrix)
  {
    Core::LinAlg::FixedSizeSerialDenseSolver<dim, dim, 1> solve_for_inverse;
    solve_for_inverse.SetMatrix(matrix);

    solve_for_inverse.invert();
  }

  /*!
   * @brief Solve for the inverse of a matrix and throw errors if unsuccessful
   *
   * @tparam dim : matrix dimensions
   * @param matrix(in/out) : matrix to be inverted
   */
  template <unsigned int dim>
  void solve_for_inverse(Core::LinAlg::Matrix<dim, dim>& matrix)
  {
    Core::LinAlg::FixedSizeSerialDenseSolver<dim, dim, 1> solve_for_inverse;
    solve_for_inverse.SetMatrix(matrix);

    int err_inv = solve_for_inverse.invert();
    if (err_inv != 0) FOUR_C_THROW("Inversion of matrix failed with LAPACK error code %d", err_inv);
  }

  template <Core::FE::CellType celltype>
  struct CentroidTransformation
  {
    // transformation matrix T0^{-T}, which maps the matrix M from parameter space to the material
    // configuration see Andelfinger et al., EAS-elements, 1993, doi: 10.1002/nme.1620360805
    Core::LinAlg::Matrix<num_str<celltype>, num_str<celltype>> T0invT_;

    // Jacobi determinant evaluated at the element centroid
    double detJ0_;
  };

  /*!
   * @brief Evaluates and returns the transformation matrix T0^{-T} which maps the matrix M from
   * parameter space to the material configuration
   *
   * For details, see Andelfinger et al., EAS-elements, 1993, doi: 10.1002/nme.1620360805.
   *
   * @tparam celltype : Cell type
   * @param jacobian_centroid(in) : Jacobian mapping evaluated at the element centroid
   * @return double : transformation matrix
   */
  template <Core::FE::CellType celltype>
  Core::LinAlg::Matrix<num_str<celltype>, num_str<celltype>> evaluate_T0invT(
      const Discret::ELEMENTS::JacobianMapping<celltype>& jacobian_centroid)
  {
    // build T0^T (based on strain-like Voigt notation: xx,yy,zz,xy,yz,xz)
    // currently only works in 3D
    Core::LinAlg::Matrix<num_str<celltype>, num_str<celltype>> T0invT(false);
    T0invT(0, 0) = jacobian_centroid.jacobian_(0, 0) * jacobian_centroid.jacobian_(0, 0);
    T0invT(1, 0) = jacobian_centroid.jacobian_(1, 0) * jacobian_centroid.jacobian_(1, 0);
    T0invT(2, 0) = jacobian_centroid.jacobian_(2, 0) * jacobian_centroid.jacobian_(2, 0);
    T0invT(3, 0) = 2 * jacobian_centroid.jacobian_(0, 0) * jacobian_centroid.jacobian_(1, 0);
    T0invT(4, 0) = 2 * jacobian_centroid.jacobian_(1, 0) * jacobian_centroid.jacobian_(2, 0);
    T0invT(5, 0) = 2 * jacobian_centroid.jacobian_(0, 0) * jacobian_centroid.jacobian_(2, 0);

    T0invT(0, 1) = jacobian_centroid.jacobian_(0, 1) * jacobian_centroid.jacobian_(0, 1);
    T0invT(1, 1) = jacobian_centroid.jacobian_(1, 1) * jacobian_centroid.jacobian_(1, 1);
    T0invT(2, 1) = jacobian_centroid.jacobian_(2, 1) * jacobian_centroid.jacobian_(2, 1);
    T0invT(3, 1) = 2 * jacobian_centroid.jacobian_(0, 1) * jacobian_centroid.jacobian_(1, 1);
    T0invT(4, 1) = 2 * jacobian_centroid.jacobian_(1, 1) * jacobian_centroid.jacobian_(2, 1);
    T0invT(5, 1) = 2 * jacobian_centroid.jacobian_(0, 1) * jacobian_centroid.jacobian_(2, 1);

    T0invT(0, 2) = jacobian_centroid.jacobian_(0, 2) * jacobian_centroid.jacobian_(0, 2);
    T0invT(1, 2) = jacobian_centroid.jacobian_(1, 2) * jacobian_centroid.jacobian_(1, 2);
    T0invT(2, 2) = jacobian_centroid.jacobian_(2, 2) * jacobian_centroid.jacobian_(2, 2);
    T0invT(3, 2) = 2 * jacobian_centroid.jacobian_(0, 2) * jacobian_centroid.jacobian_(1, 2);
    T0invT(4, 2) = 2 * jacobian_centroid.jacobian_(1, 2) * jacobian_centroid.jacobian_(2, 2);
    T0invT(5, 2) = 2 * jacobian_centroid.jacobian_(0, 2) * jacobian_centroid.jacobian_(2, 2);

    T0invT(0, 3) = jacobian_centroid.jacobian_(0, 0) * jacobian_centroid.jacobian_(0, 1);
    T0invT(1, 3) = jacobian_centroid.jacobian_(1, 0) * jacobian_centroid.jacobian_(1, 1);
    T0invT(2, 3) = jacobian_centroid.jacobian_(2, 0) * jacobian_centroid.jacobian_(2, 1);
    T0invT(3, 3) = jacobian_centroid.jacobian_(0, 0) * jacobian_centroid.jacobian_(1, 1) +
                   jacobian_centroid.jacobian_(1, 0) * jacobian_centroid.jacobian_(0, 1);
    T0invT(4, 3) = jacobian_centroid.jacobian_(1, 0) * jacobian_centroid.jacobian_(2, 1) +
                   jacobian_centroid.jacobian_(2, 0) * jacobian_centroid.jacobian_(1, 1);
    T0invT(5, 3) = jacobian_centroid.jacobian_(0, 0) * jacobian_centroid.jacobian_(2, 1) +
                   jacobian_centroid.jacobian_(2, 0) * jacobian_centroid.jacobian_(0, 1);

    T0invT(0, 4) = jacobian_centroid.jacobian_(0, 1) * jacobian_centroid.jacobian_(0, 2);
    T0invT(1, 4) = jacobian_centroid.jacobian_(1, 1) * jacobian_centroid.jacobian_(1, 2);
    T0invT(2, 4) = jacobian_centroid.jacobian_(2, 1) * jacobian_centroid.jacobian_(2, 2);
    T0invT(3, 4) = jacobian_centroid.jacobian_(0, 1) * jacobian_centroid.jacobian_(1, 2) +
                   jacobian_centroid.jacobian_(1, 1) * jacobian_centroid.jacobian_(0, 2);
    T0invT(4, 4) = jacobian_centroid.jacobian_(1, 1) * jacobian_centroid.jacobian_(2, 2) +
                   jacobian_centroid.jacobian_(2, 1) * jacobian_centroid.jacobian_(1, 2);
    T0invT(5, 4) = jacobian_centroid.jacobian_(0, 1) * jacobian_centroid.jacobian_(2, 2) +
                   jacobian_centroid.jacobian_(2, 1) * jacobian_centroid.jacobian_(0, 2);

    T0invT(0, 5) = jacobian_centroid.jacobian_(0, 0) * jacobian_centroid.jacobian_(0, 2);
    T0invT(1, 5) = jacobian_centroid.jacobian_(1, 0) * jacobian_centroid.jacobian_(1, 2);
    T0invT(2, 5) = jacobian_centroid.jacobian_(2, 0) * jacobian_centroid.jacobian_(2, 2);
    T0invT(3, 5) = jacobian_centroid.jacobian_(0, 0) * jacobian_centroid.jacobian_(1, 2) +
                   jacobian_centroid.jacobian_(1, 0) * jacobian_centroid.jacobian_(0, 2);
    T0invT(4, 5) = jacobian_centroid.jacobian_(1, 0) * jacobian_centroid.jacobian_(2, 2) +
                   jacobian_centroid.jacobian_(2, 0) * jacobian_centroid.jacobian_(1, 2);
    T0invT(5, 5) = jacobian_centroid.jacobian_(0, 0) * jacobian_centroid.jacobian_(2, 2) +
                   jacobian_centroid.jacobian_(2, 0) * jacobian_centroid.jacobian_(0, 2);

    // evaluate the inverse T0^{-T} with solver
    solve_for_inverse(T0invT);

    return T0invT;
  }

  /*!
   * @brief Evaluates and returns the centroid transformation quantities, i.e., the jacobi
   * determinant at the element centroid and the transformation matrix T0^{-T}
   *
   * @tparam celltype : Cell type
   * @param nodal_coordinates(in) : reference and current coordinates of the nodes of the element
   * @return CentroidTransformation<celltype> : Jacobi determinant at the element centroid and
   * transformation matrix T0^{-T}
   */
  template <Core::FE::CellType celltype>
  CentroidTransformation<celltype> evaluate_centroid_transformation(
      const Discret::ELEMENTS::ElementNodes<celltype>& nodal_coordinates)
  {
    CentroidTransformation<celltype> centroid_transformation;

    // 1) compute jacobian at element centroid
    const Discret::ELEMENTS::JacobianMapping<celltype> jacobian_mapping_centroid =
        Discret::ELEMENTS::evaluate_jacobian_mapping_centroid(nodal_coordinates);

    centroid_transformation.detJ0_ = jacobian_mapping_centroid.determinant_;

    // 2) compute matrix T0^{-T}: T0^{-T} maps the matrix M from local to global coordinates
    centroid_transformation.T0invT_ = evaluate_T0invT(jacobian_mapping_centroid);

    return centroid_transformation;
  }

  /*!
   * @brief Extracts and returns the residual displacement
   *
   * @tparam celltype : Cell type
   * @param discretization(in) : reference to the discretization
   * @param lm(in) : Location vector of the element, i.e., global dof numbers of elemental dofs
   * @return double : residual displacement or displacement increment
   */
  template <Core::FE::CellType celltype>
  Core::LinAlg::Matrix<num_dof_per_ele<celltype>, 1> get_displacement_increment(
      const Core::FE::Discretization& discretization, const std::vector<int>& lm)
  {
    auto residual_from_dis = discretization.GetState("residual displacement");
    std::vector<double> residual(lm.size());
    Core::FE::ExtractMyValues(*residual_from_dis, residual, lm);
    Core::LinAlg::Matrix<num_dof_per_ele<celltype>, 1> displ_inc(false);
    for (int i = 0; i < num_dof_per_ele<celltype>; ++i) displ_inc(i) = residual[i];

    return displ_inc;
  }

  /*!
   * @brief Updates the enhanced strains scalar increment
   *
   * @tparam celltype, eastype
   * @param displ_inc(in) : displacement increment delta_D_{i+1}
   * @param eas_iteration_data(in) : EAS matrices and vectors from iteration i
   */
  template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype>
  void update_alpha_increment(const Core::LinAlg::Matrix<num_dof_per_ele<celltype>, 1>& displ_inc,
      Discret::ELEMENTS::EasIterationData<celltype, eastype>& eas_iteration_data)
  {
    // the enhanced strains scalar increment is computed to:
    // delta_alpha_{i+1} = - invKaa_{i} (s_{i} + Kad_{i} delta_D_{i+1})

    // init as enhancement vector s_{i}
    Core::LinAlg::Matrix<STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas, 1> tmp(
        eas_iteration_data.s_);

    // addition of Kad_{i} delta_D_{i+1}
    tmp.multiply_tn(1.0, eas_iteration_data.Kda_, displ_inc, 1.0);

    // multiplication with (- invKaa_{i})
    eas_iteration_data.alpha_inc_.multiply(-1.0, eas_iteration_data.invKaa_, tmp);
  }

  /*!
   * @brief Updates the enhanced strain scalars, alpha_inc and alpha in the iteration data
   * accordingly
   *
   * @tparam celltype, eastype
   * @param eas_iteration_data(in/out) : EAS matrices and vectors
   * @param discretization(in) : reference to the discretization
   * @param lm(in) : Location vector of the element, i.e., global dof numbers of elemental dofs
   */
  template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype>
  void update_alpha(Discret::ELEMENTS::EasIterationData<celltype, eastype>& eas_iteration_data,
      const Core::FE::Discretization& discretization, const std::vector<int>& lm,
      const double step_length = 1.0)
  {
    // residual displacement at the previous step
    Core::LinAlg::Matrix<num_dof_per_ele<celltype>, 1> displ_inc(false);
    displ_inc = get_displacement_increment<celltype>(discretization, lm);

    // compute the enhanced strain scalar increment delta_alpha
    update_alpha_increment<celltype, eastype>(displ_inc, eas_iteration_data);

    // update alpha_i with the increment delta_alpha such that alpha_{i+1} = alpha_{i} + delta_alpha
    eas_iteration_data.alpha_.update(step_length, eas_iteration_data.alpha_inc_, 1.0);
  }

  /*!
   * @brief Correct alpha in a line search step by adapting the step length
   *
   * @tparam celltype
   * @tparam eastype
   * @param eas_iteration_data (in/out) : EAS iteration data
   * @param new_step_length (in) : new step length
   * @param old_step_length (in) : old step length
   */
  template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype>
  void correct_alpha(Discret::ELEMENTS::EasIterationData<celltype, eastype>& eas_iteration_data,
      const double new_step_length, const double old_step_length)
  {
    eas_iteration_data.alpha_.update(
        new_step_length - old_step_length, eas_iteration_data.alpha_inc_, 1.0);
  }

  /*!
   * @brief Compute the matrix M which is the element-wise matrix of the shape functions for the
   * enhanced strains in the parameter space
   *
   * @tparam celltype, eastype
   * @param xi(in) : coordinate in the parameter space
   * @return Core::LinAlg::Matrix<num_str, num_eas> : enhanced strains shape function matrix in
   * parameter space
   */
  template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype>
  Core::LinAlg::Matrix<num_str<celltype>, STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas>
  evaluate_eas_shape_functions_parameter_space(const Core::LinAlg::Matrix<num_dim<celltype>, 1>& xi)
  {
    Core::LinAlg::Matrix<num_str<celltype>, STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas> M(
        true);

    switch (eastype)
    {
      /* easmild is the EAS interpolation of 9 modes, based on
      **            r 0 0   0 0 0 0 0 0
      **            0 s 0   0 0 0 0 0 0
      **    M =     0 0 t   0 0 0 0 0 0
      **            0 0 0   r s 0 0 0 0
      **            0 0 0   0 0 s t 0 0
      **            0 0 0   0 0 0 0 r t
      */
      case STR::ELEMENTS::EasType::eastype_h8_9:
      {
        M(0, 0) = xi(0);
        M(1, 1) = xi(1);
        M(2, 2) = xi(2);
        M(3, 3) = xi(0);
        M(3, 4) = xi(1);
        M(4, 5) = xi(1);
        M(4, 6) = xi(2);
        M(5, 7) = xi(0);
        M(5, 8) = xi(2);

        break;
      }
      /* easfull is the EAS interpolation of 21 modes, based on
      **            r 0 0   0 0 0 0 0 0   0  0  0  0  0  0   rs rt 0  0  0  0
      **            0 s 0   0 0 0 0 0 0   0  0  0  0  0  0   0  0  rs st 0  0
      **    M =     0 0 t   0 0 0 0 0 0   0  0  0  0  0  0   0  0  0  0  rt st
      **            0 0 0   r s 0 0 0 0   rt st 0  0  0  0   0  0  0  0  0  0
      **            0 0 0   0 0 s t 0 0   0  0  rs rt 0  0   0  0  0  0  0  0
      **            0 0 0   0 0 0 0 r t   0  0  0  0  rs st  0  0  0  0  0  0
      */
      case STR::ELEMENTS::EasType::eastype_h8_21:
      {
        M(0, 0) = xi(0);
        M(0, 15) = xi(0) * xi(1);
        M(0, 16) = xi(0) * xi(2);
        M(1, 1) = xi(1);
        M(1, 17) = xi(0) * xi(1);
        M(1, 18) = xi(1) * xi(2);
        M(2, 2) = xi(2);
        M(2, 19) = xi(0) * xi(2);
        M(2, 20) = xi(1) * xi(2);
        M(3, 3) = xi(0);
        M(3, 4) = xi(1);
        M(3, 9) = xi(0) * xi(2);
        M(3, 10) = xi(1) * xi(2);
        M(4, 5) = xi(1);
        M(4, 6) = xi(2);
        M(4, 11) = xi(0) * xi(1);
        M(4, 12) = xi(0) * xi(2);
        M(5, 7) = xi(0);
        M(5, 8) = xi(2);
        M(5, 13) = xi(0) * xi(1);
        M(5, 14) = xi(1) * xi(2);

        break;
      }
      /* eassosh8 is the EAS interpolation for the Solid-Shell with t=thickness dir.
      ** consisting of 7 modes, based on
      **            r 0 0   0 0 0  0
      **            0 s 0   0 0 0  0
      **    M =     0 0 t   0 0 rt st
      **            0 0 0   r s 0  0
      **            0 0 0   0 0 0  0
      **            0 0 0   0 0 0  0
      */
      case STR::ELEMENTS::EasType::eastype_sh8_7:
      {
        /* eassosh8 is the EAS interpolation for the Solid-Shell with t=thickness dir.
        ** consisting of 7 modes, based on
        **            r 0 0   0 0 0  0
        **            0 s 0   0 0 0  0
        **    M =     0 0 t   0 0 rt st
        **            0 0 0   r s 0  0
        **            0 0 0   0 0 0  0
        **            0 0 0   0 0 0  0
        */
        M(0, 0) = xi(0);
        M(1, 1) = xi(1);
        M(2, 2) = xi(2);
        M(2, 5) = xi(0) * xi(2);
        M(2, 6) = xi(1) * xi(2);
        M(3, 3) = xi(0);
        M(3, 4) = xi(1);

        break;
      }

      default:
        FOUR_C_THROW("unknown EAS type");
        break;
    }
    return M;
  }

  /*!
   * @brief Map the matrix M in the parameter space to Mtilde in the material configuration and
   * return Mtilde
   *
   * @tparam celltype, eastype
   * @param detJ(in) : Jacobi determinant at Gauss point
   * @param centroid_transformation(in) : transformation matrix T0^{-T} and Jacobi determinant at
   * element centroid
   * @param M(in) : matrix M in the parameter space
   * @return Core::LinAlg::Matrix<num_str, num_eas> : matrix Mtilde in the material configuration
   */
  template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype>
  Core::LinAlg::Matrix<num_str<celltype>, STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas>
  map_eas_shape_functions_to_material_config(const double detJ,
      const CentroidTransformation<celltype>& centroid_transformation,
      const Core::LinAlg::Matrix<num_str<celltype>,
          STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas>& M)
  {
    Core::LinAlg::Matrix<num_str<celltype>, STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas>
        Mtilde;

    // Mtilde = detJ0/detJ T0^{-T} M
    Mtilde.multiply(centroid_transformation.detJ0_ / detJ, centroid_transformation.T0invT_, M);

    return Mtilde;
  }

  /*!
   * @brief Evaluate the element-wise matrix of the shape functions for the enhanced strains in the
   * parameter space Mtilde. Therefore set up M (in the material configuration) and map M to Mtilde
   * via T0^{-T}.
   *
   * @tparam celltype, eastype
   * @param detJ(in) : Jacobi determinant at Gauss point
   * @param centroid_transformation(in) : transformation matrix T0^{-T} and Jacobi determinant at
   * element centroid
   * @param xi(in) : coordinate in the parameter space
   * @return Core::LinAlg::Matrix<num_str, num_eas> : matrix Mtilde in the material configuration
   */
  template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype>
  Core::LinAlg::Matrix<num_str<celltype>, STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas>
  evaluate_eas_shape_functions_material_config(const double detJ,
      const CentroidTransformation<celltype>& centroid_transformation,
      const Core::LinAlg::Matrix<num_dim<celltype>, 1>& xi)
  {
    Core::LinAlg::Matrix<num_str<celltype>, STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas> M(
        evaluate_eas_shape_functions_parameter_space<celltype, eastype>(xi));
    Core::LinAlg::Matrix<num_str<celltype>, STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas>
        Mtilde = map_eas_shape_functions_to_material_config<celltype, eastype>(
            detJ, centroid_transformation, M);
    return Mtilde;
  }

  /*!
   * @brief Add the enhanced assumed Green-Lagrange strains E^{enh} = Mtilde alpha to the
   * conventional Green-Lagrange strains E^{u}
   *
   * Background: Choose deformation gradient F as sum of displacement-based F^{u} and enhanced
   * gradient F^{enh}. Considering F_0 the deformation gradient evaluated at the element centroid,
   * F^{enh} is computed to F^{enh} = F_0^{u} Mtilde alpha.
   *
   * @tparam celltype, eastype
   * @param gl_strain(in) : Green-Lagrange strains E^{u}
   * @param Mtilde(in) : matrix Mtilde in the material configuration
   * @param alpha(in) : enhanced strain scalars
   * @return Core::LinAlg::Matrix<num_str, 1>  : enhanced Green-Lagrange strains E^{enh}
   */
  template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype>
  Core::LinAlg::Matrix<num_str<celltype>, 1> evaluate_enhanced_assumed_gl_strains(
      const Core::LinAlg::Matrix<num_str<celltype>, 1>& gl_strain,
      const Core::LinAlg::Matrix<num_str<celltype>,
          STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas>& Mtilde,
      const Core::LinAlg::Matrix<STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas, 1>& alpha)
  {
    Core::LinAlg::Matrix<num_str<celltype>, 1> enhanced_gl_strain(gl_strain);
    enhanced_gl_strain.multiply(1.0, Mtilde, alpha, 1.0);
    return enhanced_gl_strain;
  }

  /*!
   * @brief Evaluate the enhanced assumed Green-Lagrange strains E^{enh}

   * @tparam celltype, eastype
   * @param displacement_based_mapping(in) : displacement-based spatial mapping
   * @param Mtilde(in) : matrix Mtilde in the material configuration
   * @param alpha(in) : enhanced strain scalars
   * @return Core::LinAlg::Matrix<num_str, 1> : Enhanced Green-Lagrange strains E^{enh}
   */
  template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype>
  Core::LinAlg::Matrix<num_str<celltype>, 1> evaluate_enhanced_assumed_gl_strains(
      const Discret::ELEMENTS::SpatialMaterialMapping<celltype>& displacement_based_mapping,
      const Core::LinAlg::Matrix<num_str<celltype>,
          STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas>& Mtilde,
      const Core::LinAlg::Matrix<STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas, 1>& alpha)
  {
    const Core::LinAlg::Matrix<num_dim<celltype>, num_dim<celltype>>
        displacement_based_cauchygreen =
            Discret::ELEMENTS::evaluate_cauchy_green<celltype>(displacement_based_mapping);

    const Core::LinAlg::Matrix<num_str<celltype>, 1> gl_strain =
        Discret::ELEMENTS::evaluate_green_lagrange_strain(displacement_based_cauchygreen);

    return evaluate_enhanced_assumed_gl_strains<celltype, eastype>(gl_strain, Mtilde, alpha);
  }

  /*!
   * @brief Compute the enhanced deformation gradient F^{enh}

   * @tparam dim
   * @param defgrd_disp(in) : displacement-based deformation gradient F^{u}
   * @param enhanced_gl_strain(in) : enhanced Green-Lagrange strains E^{enh}
   * @return Core::LinAlg::Matrix<dim, dim> : enhanced deformation gradient F^{enh}
   */
  template <unsigned dim>
  Core::LinAlg::Matrix<dim, dim> evaluate_consistent_defgrd(
      const Core::LinAlg::Matrix<dim, dim>& defgrd_disp,
      const Core::LinAlg::Matrix<dim*(dim + 1) / 2, 1>& enhanced_gl_strain)
  {
    Core::LinAlg::Matrix<dim, dim> R;       // rotation tensor
    Core::LinAlg::Matrix<dim, dim> U_enh;   // enhanced right stretch tensor
    Core::LinAlg::Matrix<dim, dim> U_disp;  // displacement-based right stretch tensor
    Core::LinAlg::Matrix<dim, dim> EW;      // temporarily store eigenvalues
    Core::LinAlg::Matrix<dim, dim> tmp;     // temporary matrix for matrix matrix matrix products
    Core::LinAlg::Matrix<dim, dim> tmp2;    // temporary matrix for matrix matrix matrix products

    // calculate modified right stretch tensor
    if (dim != 3) FOUR_C_THROW("stop: this currently only works for 3D");
    for (unsigned i = 0; i < dim; i++) U_enh(i, i) = 2. * enhanced_gl_strain(i) + 1.;
    U_enh(0, 1) = enhanced_gl_strain(dim);
    U_enh(1, 0) = enhanced_gl_strain(dim);
    U_enh(1, 2) = enhanced_gl_strain(4);
    U_enh(2, 1) = enhanced_gl_strain(4);
    U_enh(0, 2) = enhanced_gl_strain(5);
    U_enh(2, 0) = enhanced_gl_strain(5);

    Core::LinAlg::SYEV(U_enh, EW, U_enh);
    for (unsigned i = 0; i < dim; ++i) EW(i, i) = sqrt(EW(i, i));
    tmp.multiply(U_enh, EW);
    tmp2.multiply_nt(tmp, U_enh);
    U_enh.update(tmp2);

    // calculate displacement-based right stretch tensor
    U_disp.multiply_tn(defgrd_disp, defgrd_disp);

    Core::LinAlg::SYEV(U_disp, EW, U_disp);
    for (unsigned i = 0; i < dim; ++i) EW(i, i) = sqrt(EW(i, i));
    tmp.multiply(U_disp, EW);
    tmp2.multiply_nt(tmp, U_disp);
    U_disp.update(tmp2);

    // compose consistent deformation gradient
    U_disp.invert();
    R.multiply(defgrd_disp, U_disp);

    Core::LinAlg::Matrix<dim, dim> defgrd_enh;
    defgrd_enh.multiply(R, U_enh);
    return defgrd_enh;
  }

  /*!
   * @brief Integrate the EAS stiffness matrices
   *
   * @tparam celltype, eastype
   * @param stress(in) : 2. Piola Kirchhoff stress tensor and material tangent
   * @param Mtilde(in) : matrix Mtilde in the material configuration
   * @param Bop(in) : B-operator
   * @param integration_factor(in) : integration factor (Gauss point weight times Jacobi
   * determinant)
   * @param eas_iteration_data(in/out) : EAS matrices and vectors
   */
  template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype>
  void integrate_eas(const Discret::ELEMENTS::Stress<celltype>& stress,
      const Core::LinAlg::Matrix<num_str<celltype>,
          STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas>& Mtilde,
      const Core::LinAlg::Matrix<num_str<celltype>, num_dof_per_ele<celltype>>& Bop,
      const double integration_factor,
      Discret::ELEMENTS::EasIterationData<celltype, eastype>& eas_iteration_data)
  {
    // integrate Kaa: Kaa += (Mtilde^T . cmat . Mtilde) * detJ * w(gp)
    // IMPORTANT: We save this in invKaa_ here since after the loop over all Gauss points, we
    // invert the matrix. At this point, this is still Kaa and NOT invKaa.
    Core::LinAlg::Matrix<num_str<celltype>, STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas> cmatM(
        true);
    cmatM.multiply(stress.cmat_, Mtilde);
    eas_iteration_data.invKaa_.multiply_tn(integration_factor, Mtilde, cmatM, 1.);

    // integrate Kda: Kda += (B^T . cmat . Mtilde) * detJ * w(gp)
    eas_iteration_data.Kda_.multiply_tn(integration_factor, Bop, cmatM, 1.);

    // integrate s: s += (Mtilde^T . S) * detJ * w(gp)
    eas_iteration_data.s_.multiply_tn(integration_factor, Mtilde, stress.pk2_, 1.);
  }

  /*!
   * @brief Add EAS internal force contribution of one Gauss point
   *
   * The EAS internal force contribution is $f_{eas} = - K_{da} K_{aa}^{-1} s$.
   *
   * @tparam celltype : Cell type
   * @param minusKdainvKaa(in) : matrix product $- K_{da} K_{aa}^{-1}$
   * @param s(in) : enhancement vector s
   * @param force(in/out) : internal force vector where the contribution is added to
   */
  template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype>
  void add_eas_internal_force(const Core::LinAlg::Matrix<num_dof_per_ele<celltype>,
                                  STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas>& minusKdainvKaa,
      const Core::LinAlg::Matrix<STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas, 1>& s,
      Core::LinAlg::Matrix<num_dof_per_ele<celltype>, 1>& force_vector)
  {
    force_vector.multiply_nn(1.0, minusKdainvKaa, s, 1.0);
  }

  /*!
   * @brief Add EAS stiffness matrix contribution of one Gauss point
   *
   * The EAS stiffness matrix contribution is $- K_{da} K_{aa}^{-1} K_{da}^T$.
   *
   * @tparam celltype : Cell type
   * @param minusKdainvKaa(in) : matrix product $- K_{da} K_{aa}^{-1}$
   * @param Kda(in) : EAS stiffness matrix part K_{da}
   * @param stiffness_matrix(in/out) : stiffness matrix where the local contribution is added to
   */
  template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype>
  void add_eas_stiffness_matrix(
      const Core::LinAlg::Matrix<num_dof_per_ele<celltype>,
          STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas>& minusKdainvKaa,
      const Core::LinAlg::Matrix<num_dof_per_ele<celltype>,
          STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas>& Kda,
      Core::LinAlg::Matrix<num_dof_per_ele<celltype>, num_dof_per_ele<celltype>>& stiffness_matrix)
  {
    stiffness_matrix.multiply_nt(1.0, minusKdainvKaa, Kda, 1.0);
  }

  template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype>
  struct EASKinematics
  {
    static constexpr int num_str = Core::FE::dim<celltype> * (Core::FE::dim<celltype> + 1) / 2;
    static constexpr int num_dof_per_ele = Core::FE::num_nodes<celltype> * Core::FE::dim<celltype>;
    Core::LinAlg::Matrix<num_str, num_dof_per_ele> b_op{};
    Core::LinAlg::Matrix<num_str, 1> enhanced_gl{};
    Core::LinAlg::Matrix<Core::FE::dim<celltype>, Core::FE::dim<celltype>>
        enhanced_deformation_gradient{};
    Core::LinAlg::Matrix<num_str, STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas> m_tilde{};
  };

  template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype,
      Inpar::STR::KinemType kinematic_type>
  EASKinematics<celltype, eastype> evaluate_eas_kinematics(
      const Discret::ELEMENTS::ElementNodes<celltype> nodal_coordinates,
      const CentroidTransformation<celltype>& centeroid_transformation,
      const Core::LinAlg::Matrix<Core::FE::dim<celltype>, 1>& xi,
      const Discret::ELEMENTS::JacobianMapping<celltype>& jacobian_mapping,
      const Discret::ELEMENTS::EasIterationData<celltype, eastype>& eas_iteration_data)
  {
    EASKinematics<celltype, eastype> eas_kinematics{};

    if constexpr (kinematic_type == Inpar::STR::KinemType::nonlinearTotLag)
    {
      const Discret::ELEMENTS::SpatialMaterialMapping<celltype>
          displacement_based_spatial_material_mapping =
              evaluate_spatial_material_mapping(jacobian_mapping, nodal_coordinates);

      eas_kinematics.b_op = Discret::ELEMENTS::evaluate_strain_gradient(
          jacobian_mapping, displacement_based_spatial_material_mapping);

      eas_kinematics.m_tilde = evaluate_eas_shape_functions_material_config<celltype, eastype>(
          jacobian_mapping.determinant_, centeroid_transformation, xi);

      eas_kinematics.enhanced_gl = evaluate_enhanced_assumed_gl_strains<celltype, eastype>(
          displacement_based_spatial_material_mapping, eas_kinematics.m_tilde,
          eas_iteration_data.alpha_);

      eas_kinematics.enhanced_deformation_gradient = evaluate_consistent_defgrd(
          displacement_based_spatial_material_mapping.deformation_gradient_,
          eas_kinematics.enhanced_gl);
    }
    else if constexpr (kinematic_type == Inpar::STR::KinemType::linear)
    {
      eas_kinematics.b_op = Discret::ELEMENTS::evaluate_linear_strain_gradient(jacobian_mapping);

      eas_kinematics.m_tilde = evaluate_eas_shape_functions_material_config<celltype, eastype>(
          jacobian_mapping.determinant_, centeroid_transformation, xi);

      Core::LinAlg::Matrix<num_str<celltype>, 1> gl_strain_displacement_based =
          evaluate_linear_gl_strain(nodal_coordinates, eas_kinematics.b_op);

      eas_kinematics.enhanced_gl = evaluate_enhanced_assumed_gl_strains<celltype, eastype>(
          gl_strain_displacement_based, eas_kinematics.m_tilde, eas_iteration_data.alpha_);

      eas_kinematics.enhanced_deformation_gradient =
          Core::LinAlg::IdentityMatrix<Core::FE::dim<celltype>>();
    }
    else
    {
      FOUR_C_THROW("Unknown kinematic type!");
    }

    return eas_kinematics;
  }
}  // namespace

template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype,
    Inpar::STR::KinemType kinematic_type>
Discret::ELEMENTS::SolidEleCalcEas<celltype, eastype, kinematic_type>::SolidEleCalcEas()
    : stiffness_matrix_integration_(
          create_gauss_integration<celltype>(get_gauss_rule_stiffness_matrix<celltype>())),
      mass_matrix_integration_(
          create_gauss_integration<celltype>(get_gauss_rule_mass_matrix<celltype>()))
{
}

template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype,
    Inpar::STR::KinemType kinematic_type>
void Discret::ELEMENTS::SolidEleCalcEas<celltype, eastype, kinematic_type>::pack(
    Core::Communication::PackBuffer& data) const
{
  constexpr int num_dof_per_element = Core::FE::num_nodes<celltype> * Core::FE::dim<celltype>;
  Core::Communication::ParObject::add_to_pack<STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas, 1>(
      data, eas_iteration_data_.alpha_inc_);
  Core::Communication::ParObject::add_to_pack<STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas, 1>(
      data, eas_iteration_data_.alpha_);
  Core::Communication::ParObject::add_to_pack<STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas, 1>(
      data, eas_iteration_data_.s_);
  Core::Communication::ParObject::add_to_pack<STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas,
      STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas>(data, eas_iteration_data_.invKaa_);
  Core::Communication::ParObject::add_to_pack<num_dof_per_element,
      STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas>(data, eas_iteration_data_.Kda_);
  Core::Communication::ParObject::add_to_pack(data, old_step_length_);
};

template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype,
    Inpar::STR::KinemType kinematic_type>
void Discret::ELEMENTS::SolidEleCalcEas<celltype, eastype, kinematic_type>::unpack(
    std::vector<char>::size_type& position, const std::vector<char>& data)
{
  Core::Communication::ParObject::extract_from_pack(position, data, eas_iteration_data_.alpha_inc_);
  Core::Communication::ParObject::extract_from_pack(position, data, eas_iteration_data_.alpha_);
  Core::Communication::ParObject::extract_from_pack(position, data, eas_iteration_data_.s_);
  Core::Communication::ParObject::extract_from_pack(position, data, eas_iteration_data_.invKaa_);
  Core::Communication::ParObject::extract_from_pack(position, data, eas_iteration_data_.Kda_);
  Core::Communication::ParObject::extract_from_pack(position, data, old_step_length_);
};

template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype,
    Inpar::STR::KinemType kinematic_type>
void Discret::ELEMENTS::SolidEleCalcEas<celltype, eastype,
    kinematic_type>::evaluate_nonlinear_force_stiffness_mass(const Core::Elements::Element& ele,
    Mat::So3Material& solid_material, const Core::FE::Discretization& discretization,
    const std::vector<int>& lm, Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseVector* force_vector,
    Core::LinAlg::SerialDenseMatrix* stiffness_matrix, Core::LinAlg::SerialDenseMatrix* mass_matrix)
{
  // Create views to SerialDenseMatrices
  std::optional<Core::LinAlg::Matrix<num_dof_per_ele_, num_dof_per_ele_>> stiff = {};
  std::optional<Core::LinAlg::Matrix<num_dof_per_ele_, num_dof_per_ele_>> mass = {};
  std::optional<Core::LinAlg::Matrix<num_dof_per_ele_, 1>> force = {};
  if (stiffness_matrix != nullptr) stiff.emplace(*stiffness_matrix, true);
  if (mass_matrix != nullptr) mass.emplace(*mass_matrix, true);
  if (force_vector != nullptr) force.emplace(*force_vector, true);

  const ElementNodes<celltype> nodal_coordinates =
      evaluate_element_nodes<celltype>(ele, discretization, lm);

  bool equal_integration_mass_stiffness =
      compare_gauss_integration(mass_matrix_integration_, stiffness_matrix_integration_);

  CentroidTransformation<celltype> centroid_transformation =
      evaluate_centroid_transformation<celltype>(nodal_coordinates);

  if (!ele.IsParamsInterface())
  {
    // Update alpha only in old time integration scheme
    update_alpha<celltype, eastype>(eas_iteration_data_, discretization, lm);
  }

  // clear for integration
  eas_iteration_data_.invKaa_.clear();
  eas_iteration_data_.Kda_.clear();
  eas_iteration_data_.s_.clear();

  evaluate_centroid_coordinates_and_add_to_parameter_list<celltype>(nodal_coordinates, params);

  double element_mass = 0.0;
  double element_volume = 0.0;
  ForEachGaussPoint<celltype>(nodal_coordinates, stiffness_matrix_integration_,
      [&](const Core::LinAlg::Matrix<num_dim_, 1>& xi,
          const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
          const JacobianMapping<celltype>& jacobian_mapping, double integration_factor, int gp)
      {
        const EASKinematics<celltype, eastype> kinematic_quantitites =
            evaluate_eas_kinematics<celltype, eastype, kinematic_type>(nodal_coordinates,
                centroid_transformation, xi, jacobian_mapping, eas_iteration_data_);

        evaluate_gp_coordinates_and_add_to_parameter_list<celltype>(
            nodal_coordinates, shape_functions, params);

        const Stress<celltype> stress = evaluate_material_stress<celltype>(solid_material,
            kinematic_quantitites.enhanced_deformation_gradient, kinematic_quantitites.enhanced_gl,
            params, gp, ele.Id());

        integrate_eas<celltype, eastype>(stress, kinematic_quantitites.m_tilde,
            kinematic_quantitites.b_op, integration_factor, eas_iteration_data_);

        if (force.has_value())
        {
          add_internal_force_vector(kinematic_quantitites.b_op, stress, integration_factor, *force);
        }

        if (stiff.has_value())
        {
          add_elastic_stiffness_matrix(
              kinematic_quantitites.b_op, stress, integration_factor, *stiff);
          add_geometric_stiffness_matrix(
              jacobian_mapping.N_XYZ_, stress, integration_factor, *stiff);
        }

        if (mass.has_value())
        {
          if (equal_integration_mass_stiffness)
          {
            add_mass_matrix(shape_functions, integration_factor, solid_material.Density(gp), *mass);
          }
          else
          {
            element_mass += solid_material.Density(gp) * integration_factor;
            element_volume += integration_factor;
          }
        }
      });

  // invert Kaa with solver. eas_iteration_data_.invKaa_ then is Kaa^{-1}
  solve_for_inverse_ignoring_errors(eas_iteration_data_.invKaa_);

  // compute the product (- Kda Kaa^{-1}) which is later needed for force and stiffness update
  Core::LinAlg::Matrix<num_dof_per_ele_, STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas>
      minusKdainvKaa(true);
  minusKdainvKaa.multiply_nn(-1.0, eas_iteration_data_.Kda_, eas_iteration_data_.invKaa_);

  if (force.has_value())
  {
    add_eas_internal_force<celltype, eastype>(minusKdainvKaa, eas_iteration_data_.s_, *force);
  }

  if (stiff.has_value())
  {
    add_eas_stiffness_matrix<celltype, eastype>(minusKdainvKaa, eas_iteration_data_.Kda_, *stiff);
  }

  if (mass.has_value() && !equal_integration_mass_stiffness)
  {
    // integrate mass matrix
    FOUR_C_ASSERT(element_mass > 0, "It looks like the element mass is 0.0");
    ForEachGaussPoint<celltype>(nodal_coordinates, mass_matrix_integration_,
        [&](const Core::LinAlg::Matrix<num_dim_, 1>& xi,
            const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
            const JacobianMapping<celltype>& jacobian_mapping, double integration_factor, int gp) {
          add_mass_matrix(
              shape_functions, integration_factor, element_mass / element_volume, *mass);
        });
  }
}

template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype,
    Inpar::STR::KinemType kinematic_type>
void Discret::ELEMENTS::SolidEleCalcEas<celltype, eastype, kinematic_type>::Recover(
    Core::Elements::Element& ele, const Core::FE::Discretization& discretization,
    const std::vector<int>& lm, Teuchos::ParameterList& params)
{
  STR::ELEMENTS::ParamsInterface& params_interface =
      *Teuchos::rcp_dynamic_cast<STR::ELEMENTS::ParamsInterface>(ele.ParamsInterfacePtr());

  const double step_length = params_interface.get_step_length();

  if (params_interface.is_default_step())
  {
    params_interface.sum_into_my_previous_sol_norm(NOX::Nln::StatusTest::quantity_eas,
        STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas, &eas_iteration_data_.alpha_(0, 0),
        ele.Owner());

    // Update alpha
    update_alpha(eas_iteration_data_, discretization, lm, step_length);
  }
  else
  {
    correct_alpha(eas_iteration_data_, step_length, old_step_length_);
  }

  // store old step length
  old_step_length_ = step_length;

  params_interface.sum_into_my_update_norm(NOX::Nln::StatusTest::quantity_eas,
      STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas, &eas_iteration_data_.alpha_inc_(0, 0),
      &eas_iteration_data_.alpha_(0, 0), step_length, ele.Owner());
}

template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype,
    Inpar::STR::KinemType kinematic_type>
void Discret::ELEMENTS::SolidEleCalcEas<celltype, eastype, kinematic_type>::Update(
    const Core::Elements::Element& ele, Mat::So3Material& solid_material,
    const Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Teuchos::ParameterList& params)
{
  const ElementNodes<celltype> nodal_coordinates =
      evaluate_element_nodes<celltype>(ele, discretization, lm);
  CentroidTransformation<celltype> centroid_transformation =
      evaluate_centroid_transformation<celltype>(nodal_coordinates);

  // No need to update alpha here. Update is called to copy states from t_{n+1} to t_{n} after the
  // time step and output. Hence, there are no more Newton iterations that would require an update
  // of alpha

  evaluate_centroid_coordinates_and_add_to_parameter_list<celltype>(nodal_coordinates, params);

  ForEachGaussPoint<celltype>(nodal_coordinates, stiffness_matrix_integration_,
      [&](const Core::LinAlg::Matrix<num_dim_, 1>& xi,
          const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
          const JacobianMapping<celltype>& jacobian_mapping, double integration_factor, int gp)
      {
        const EASKinematics<celltype, eastype> kinematic_quantitites =
            evaluate_eas_kinematics<celltype, eastype, kinematic_type>(nodal_coordinates,
                centroid_transformation, xi, jacobian_mapping, eas_iteration_data_);

        evaluate_gp_coordinates_and_add_to_parameter_list<celltype>(
            nodal_coordinates, shape_functions, params);

        solid_material.update(
            kinematic_quantitites.enhanced_deformation_gradient, gp, params, ele.Id());
      });

  solid_material.update();
}

template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype,
    Inpar::STR::KinemType kinematic_type>
void Discret::ELEMENTS::SolidEleCalcEas<celltype, eastype, kinematic_type>::calculate_stress(
    const Core::Elements::Element& ele, Mat::So3Material& solid_material, const StressIO& stressIO,
    const StrainIO& strainIO, const Core::FE::Discretization& discretization,
    const std::vector<int>& lm, Teuchos::ParameterList& params)
{
  std::vector<char>& serialized_stress_data = stressIO.mutable_data;
  std::vector<char>& serialized_strain_data = strainIO.mutable_data;
  Core::LinAlg::SerialDenseMatrix stress_data(stiffness_matrix_integration_.NumPoints(), num_str_);
  Core::LinAlg::SerialDenseMatrix strain_data(stiffness_matrix_integration_.NumPoints(), num_str_);

  const ElementNodes<celltype> nodal_coordinates =
      evaluate_element_nodes<celltype>(ele, discretization, lm);

  CentroidTransformation<celltype> centroid_transformation =
      evaluate_centroid_transformation<celltype>(nodal_coordinates);

  evaluate_centroid_coordinates_and_add_to_parameter_list<celltype>(nodal_coordinates, params);

  ForEachGaussPoint<celltype>(nodal_coordinates, stiffness_matrix_integration_,
      [&](const Core::LinAlg::Matrix<num_dim_, 1>& xi,
          const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
          const JacobianMapping<celltype>& jacobian_mapping, double integration_factor, int gp)
      {
        const EASKinematics<celltype, eastype> kinematic_quantitites =
            evaluate_eas_kinematics<celltype, eastype, kinematic_type>(nodal_coordinates,
                centroid_transformation, xi, jacobian_mapping, eas_iteration_data_);

        evaluate_gp_coordinates_and_add_to_parameter_list<celltype>(
            nodal_coordinates, shape_functions, params);

        const Stress<celltype> stress = evaluate_material_stress<celltype>(solid_material,
            kinematic_quantitites.enhanced_deformation_gradient, kinematic_quantitites.enhanced_gl,
            params, gp, ele.Id());

        assemble_strain_type_to_matrix_row<celltype>(kinematic_quantitites.enhanced_gl,
            kinematic_quantitites.enhanced_deformation_gradient, strainIO.type, strain_data, gp);
        assemble_stress_type_to_matrix_row(kinematic_quantitites.enhanced_deformation_gradient,
            stress, stressIO.type, stress_data, gp);
      });

  serialize(stress_data, serialized_stress_data);
  serialize(strain_data, serialized_strain_data);
}

template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype,
    Inpar::STR::KinemType kinematic_type>
double
Discret::ELEMENTS::SolidEleCalcEas<celltype, eastype, kinematic_type>::calculate_internal_energy(
    const Core::Elements::Element& ele, Mat::So3Material& solid_material,
    const Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Teuchos::ParameterList& params)
{
  double intenergy = 0.0;
  const ElementNodes<celltype> nodal_coordinates =
      evaluate_element_nodes<celltype>(ele, discretization, lm);

  CentroidTransformation<celltype> centroid_transformation =
      evaluate_centroid_transformation<celltype>(nodal_coordinates);

  ForEachGaussPoint<celltype>(nodal_coordinates, stiffness_matrix_integration_,
      [&](const Core::LinAlg::Matrix<num_dim_, 1>& xi,
          const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
          const JacobianMapping<celltype>& jacobian_mapping, double integration_factor, int gp)
      {
        const EASKinematics<celltype, eastype> kinematic_quantitites =
            evaluate_eas_kinematics<celltype, eastype, kinematic_type>(nodal_coordinates,
                centroid_transformation, xi, jacobian_mapping, eas_iteration_data_);

        double psi = 0.0;
        solid_material.StrainEnergy(kinematic_quantitites.enhanced_gl, psi, gp, ele.Id());

        intenergy += psi * integration_factor;
      });

  return intenergy;
}

template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype,
    Inpar::STR::KinemType kinematic_type>
void Discret::ELEMENTS::SolidEleCalcEas<celltype, eastype, kinematic_type>::setup(
    Mat::So3Material& solid_material, Input::LineDefinition* linedef)
{
  solid_material.setup(stiffness_matrix_integration_.NumPoints(), linedef);
}

template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype,
    Inpar::STR::KinemType kinematic_type>
void Discret::ELEMENTS::SolidEleCalcEas<celltype, eastype, kinematic_type>::material_post_setup(
    const Core::Elements::Element& ele, Mat::So3Material& solid_material)
{
  Teuchos::ParameterList params{};

  // Check if element has fiber nodes, if so interpolate fibers to Gauss Points and add to params
  InterpolateFibersToGaussPointsAndAddToParameterList<celltype>(
      stiffness_matrix_integration_, ele, params);

  // Call post_setup of material
  solid_material.post_setup(params, ele.Id());
}

template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype,
    Inpar::STR::KinemType kinematic_type>
void Discret::ELEMENTS::SolidEleCalcEas<celltype, eastype,
    kinematic_type>::initialize_gauss_point_data_output(const Core::Elements::Element& ele,
    const Mat::So3Material& solid_material,
    STR::MODELEVALUATOR::GaussPointDataOutputManager& gp_data_output_manager) const
{
  FOUR_C_ASSERT(ele.IsParamsInterface(),
      "This action type should only be called from the new time integration framework!");

  ask_and_add_quantities_to_gauss_point_data_output(
      stiffness_matrix_integration_.NumPoints(), solid_material, gp_data_output_manager);
}

template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype,
    Inpar::STR::KinemType kinematic_type>
void Discret::ELEMENTS::SolidEleCalcEas<celltype, eastype,
    kinematic_type>::evaluate_gauss_point_data_output(const Core::Elements::Element& ele,
    const Mat::So3Material& solid_material,
    STR::MODELEVALUATOR::GaussPointDataOutputManager& gp_data_output_manager) const
{
  FOUR_C_ASSERT(ele.IsParamsInterface(),
      "This action type should only be called from the new time integration framework!");

  collect_and_assemble_gauss_point_data_output<celltype>(
      stiffness_matrix_integration_, solid_material, ele, gp_data_output_manager);
}

template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype,
    Inpar::STR::KinemType kinematic_type>
void Discret::ELEMENTS::SolidEleCalcEas<celltype, eastype, kinematic_type>::reset_to_last_converged(
    const Core::Elements::Element& ele, Mat::So3Material& solid_material)
{
  solid_material.reset_step();
}

template <Core::FE::CellType celltype, STR::ELEMENTS::EasType eastype,
    Inpar::STR::KinemType kinematic_type>
void Discret::ELEMENTS::SolidEleCalcEas<celltype, eastype, kinematic_type>::for_each_gauss_point(
    const Core::Elements::Element& ele, Mat::So3Material& solid_material,
    const Core::FE::Discretization& discretization, const std::vector<int>& lm,
    const std::function<void(Mat::So3Material&, double, int)>& integrator) const
{
  const ElementNodes<celltype> nodal_coordinates =
      evaluate_element_nodes<celltype>(ele, discretization, lm);

  ForEachGaussPoint(nodal_coordinates, stiffness_matrix_integration_,
      [&](const Core::LinAlg::Matrix<DETAIL::num_dim<celltype>, 1>& xi,
          const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
          const JacobianMapping<celltype>& jacobian_mapping, double integration_factor, int gp)
      { integrator(solid_material, integration_factor, gp); });
}

// template classes
template class Discret::ELEMENTS::SolidEleCalcEas<Core::FE::CellType::hex8,
    STR::ELEMENTS::EasType::eastype_h8_9, Inpar::STR::KinemType::nonlinearTotLag>;
template class Discret::ELEMENTS::SolidEleCalcEas<Core::FE::CellType::hex8,
    STR::ELEMENTS::EasType::eastype_h8_21, Inpar::STR::KinemType::nonlinearTotLag>;
template class Discret::ELEMENTS::SolidEleCalcEas<Core::FE::CellType::hex8,
    STR::ELEMENTS::EasType::eastype_h8_9, Inpar::STR::KinemType::linear>;
template class Discret::ELEMENTS::SolidEleCalcEas<Core::FE::CellType::hex8,
    STR::ELEMENTS::EasType::eastype_h8_21, Inpar::STR::KinemType::linear>;

static_assert(
    Discret::ELEMENTS::IsPackable<Discret::ELEMENTS::SolidEleCalcEas<Core::FE::CellType::hex8,
        STR::ELEMENTS::EasType::eastype_h8_9, Inpar::STR::KinemType::nonlinearTotLag>*>,
    "EAS needs to implement the method pack(Core::Communication::PackBuffer&) to be able to store "
    "history data!");
static_assert(
    Discret::ELEMENTS::IsUnpackable<Discret::ELEMENTS::SolidEleCalcEas<Core::FE::CellType::hex8,
        STR::ELEMENTS::EasType::eastype_h8_9, Inpar::STR::KinemType::nonlinearTotLag>*>,
    "EAS needs to implement the method unpack(std::size_t, std::vector<char>&) to be able to store "
    "history data!");
static_assert(
    Discret::ELEMENTS::IsPackable<Discret::ELEMENTS::SolidEleCalcEas<Core::FE::CellType::hex8,
        STR::ELEMENTS::EasType::eastype_h8_9, Inpar::STR::KinemType::linear>*>,
    "EAS needs to implement the method pack(Core::Communication::PackBuffer&) to be able to store "
    "history data!");
static_assert(
    Discret::ELEMENTS::IsUnpackable<Discret::ELEMENTS::SolidEleCalcEas<Core::FE::CellType::hex8,
        STR::ELEMENTS::EasType::eastype_h8_9, Inpar::STR::KinemType::linear>*>,
    "EAS needs to implement the method unpack(std::size_t, std::vector<char>&) to be able to store "
    "history data!");
FOUR_C_NAMESPACE_CLOSE