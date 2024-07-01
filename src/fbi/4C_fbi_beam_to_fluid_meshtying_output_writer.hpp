/*----------------------------------------------------------------------*/
/*! \file

\brief Object to handle beam to fluid meshtying output creation.

\level 2

*/


#ifndef FOUR_C_FBI_BEAM_TO_FLUID_MESHTYING_OUTPUT_WRITER_HPP
#define FOUR_C_FBI_BEAM_TO_FLUID_MESHTYING_OUTPUT_WRITER_HPP


#include "4C_config.hpp"

#include "4C_io_visualization_parameters.hpp"

#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN


// Forward declarations.
namespace Adapter
{
  class FBIConstraintenforcer;
}
namespace BEAMINTERACTION
{
  class BeamToSolidVisualizationOutputWriterBase;
}
namespace FBI
{
  class BeamToFluidMeshtyingVtkOutputParams;
}
namespace STR::TimeInt
{
  class ParamsRuntimeOutput;
}

namespace BEAMINTERACTION
{
  /**
   * \brief This class manages and creates all visualization output for beam to solid volume
   * meshtying.
   */
  class BeamToFluidMeshtyingVtkOutputWriter
  {
   public:
    /**
     * \brief Constructor
     */
    BeamToFluidMeshtyingVtkOutputWriter();

    /**
     * \brief empty Destructor
     */
    virtual ~BeamToFluidMeshtyingVtkOutputWriter() = default;

    /**
     * \brief Initialize the object.
     */
    void init();

    /**
     * \brief Setup the output writer base and the desired field data.
     * @param visualization_output_params (in) RCP to parameter container for global visualization
     * output options.
     * @param output_params_ptr (in) RCP to parameter container for beam to solid output.
     */
    void setup(const Core::IO::VisualizationParameters& visualization_params,
        Teuchos::RCP<const STR::TimeInt::ParamsRuntimeOutput> visualization_output_params,
        Teuchos::RCP<const FBI::BeamToFluidMeshtyingVtkOutputParams> output_params_ptr);

    /**
     * \brief Setup time step output creation, and call WriteOutputData.
     * @param beam_contact (in) Pointer to the beam contact sub model evaluator. This is a raw
     * pointer since this function is called from within the sub model evaluator, which does not
     * (and probably can not) have a RCP to itself.
     */
    void write_output_runtime(const Teuchos::RCP<Adapter::FBIConstraintenforcer>& couplingenforcer,
        int i_step, double time) const;

    /**
     * \brief Setup post iteration output creation, and call WriteOutputData.
     * @param beam_contact (in) Pointer to the beam contact sub model evaluator. This is a raw
     * pointer since this function is called from within the sub model evaluator, which does not
     * (and probably can not) have a RCP to itself.
     * @param i_iteration (in) current number of iteration.
     */
    void write_output_runtime_iteration(
        const Teuchos::RCP<Adapter::FBIConstraintenforcer>& couplingenforcer, int i_iteration,
        int i_step, double time) const;

   private:
    /**
     * \brief Gather all output data after for beam to solid volume mesh tying and write the files
     * to disc.
     * @param beam_contact (in) Pointer to the beam contact sub model evaluator.
     * @param i_step (in) Number of this visualization step (does not have to be continuous, e.g. in
     * iteration visualization).
     * @param time (in) Scalar time value for this visualization step.
     */
    void write_output_beam_to_fluid_mesh_tying(
        const Teuchos::RCP<Adapter::FBIConstraintenforcer>& couplingenforcer, int i_step,
        double time) const;

    /**
     * \brief Checks the init and setup status.
     */
    void check_init_setup() const;

    /**
     * \brief Checks the init status.
     */
    void check_init() const;

   private:
    //! Flag if object is initialized.
    bool isinit_;

    //! Flag if object is set up.
    bool issetup_;

    //! Parameter container for output.
    Teuchos::RCP<const FBI::BeamToFluidMeshtyingVtkOutputParams> output_params_ptr_;

    //! Pointer to the output writer, which handles the actual output data for this object.
    Teuchos::RCP<BEAMINTERACTION::BeamToSolidVisualizationOutputWriterBase> output_writer_base_ptr_;

    //! visualization parameters
    Core::IO::VisualizationParameters visualization_params_;
  };

}  // namespace BEAMINTERACTION

FOUR_C_NAMESPACE_CLOSE

#endif