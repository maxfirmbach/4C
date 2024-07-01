/*-----------------------------------------------------------*/
/*! \file

\brief extracting maps of fluid discretizations


\level 1

*/
/*-----------------------------------------------------------*/

#include "4C_fluid_utils_mapextractor.hpp"

#include "4C_fem_condition_selector.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FLD::UTILS::MapExtractor::setup(
    const Core::FE::Discretization& dis, bool withpressure, bool overlapping, const int nds_master)
{
  const int ndim = Global::Problem::Instance()->NDim();
  Core::Conditions::MultiConditionSelector mcs;
  mcs.SetOverlapping(overlapping);  // defines if maps can overlap
  mcs.AddSelector(Teuchos::rcp(
      new Core::Conditions::NDimConditionSelector(dis, "FSICoupling", 0, ndim + withpressure)));
  mcs.AddSelector(Teuchos::rcp(new Core::Conditions::NDimConditionSelector(
      dis, "FREESURFCoupling", 0, ndim + withpressure)));
  mcs.AddSelector(Teuchos::rcp(new Core::Conditions::NDimConditionSelector(
      dis, "StructAleCoupling", 0, ndim + withpressure)));
  mcs.AddSelector(Teuchos::rcp(
      new Core::Conditions::NDimConditionSelector(dis, "Mortar", 0, ndim + withpressure)));
  mcs.AddSelector(Teuchos::rcp(new Core::Conditions::NDimConditionSelector(
      dis, "ALEUPDATECoupling", 0, ndim + withpressure)));
  mcs.SetupExtractor(dis, *dis.dof_row_map(nds_master), *this);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FLD::UTILS::MapExtractor::setup(
    Teuchos::RCP<const Epetra_Map>& additionalothermap, const FLD::UTILS::MapExtractor& extractor)
{
  // build the new othermap
  std::vector<Teuchos::RCP<const Epetra_Map>> othermaps;
  othermaps.push_back(additionalothermap);
  othermaps.push_back(extractor.other_map());

  if (Core::LinAlg::MultiMapExtractor::intersect_maps(othermaps)->NumGlobalElements() > 0)
    FOUR_C_THROW("Failed to add dofmap of foreign discretization to other_map. Detected overlap.");

  Teuchos::RCP<const Epetra_Map> mergedothermap =
      Core::LinAlg::MultiMapExtractor::merge_maps(othermaps);

  // the vector of maps for the new map extractor consists of othermap at position 0
  // followed by the maps of conditioned DOF
  std::vector<Teuchos::RCP<const Epetra_Map>> maps;
  // append the merged other map at first position
  maps.push_back(mergedothermap);

  // append the condition maps subsequently
  for (int i = 1; i < extractor.num_maps(); ++i) maps.push_back(extractor.Map(i));

  // merge
  Teuchos::RCP<const Epetra_Map> fullmap = Core::LinAlg::MultiMapExtractor::merge_maps(maps);

  Core::LinAlg::MultiMapExtractor::setup(*fullmap, maps);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<std::set<int>> FLD::UTILS::MapExtractor::conditioned_element_map(
    const Core::FE::Discretization& dis) const
{
  Teuchos::RCP<std::set<int>> condelements =
      Core::Conditions::conditioned_element_map(dis, "FSICoupling");
  Teuchos::RCP<std::set<int>> condelements2 =
      Core::Conditions::conditioned_element_map(dis, "FREESURFCoupling");
  Teuchos::RCP<std::set<int>> condelements3 =
      Core::Conditions::conditioned_element_map(dis, "StructAleCoupling");
  Teuchos::RCP<std::set<int>> condelements4 =
      Core::Conditions::conditioned_element_map(dis, "Mortar");
  Teuchos::RCP<std::set<int>> condelements5 =
      Core::Conditions::conditioned_element_map(dis, "ALEUPDATECoupling");
  std::copy(condelements2->begin(), condelements2->end(),
      std::inserter(*condelements, condelements->begin()));
  std::copy(condelements3->begin(), condelements3->end(),
      std::inserter(*condelements, condelements->begin()));
  std::copy(condelements4->begin(), condelements4->end(),
      std::inserter(*condelements, condelements->begin()));
  std::copy(condelements5->begin(), condelements5->end(),
      std::inserter(*condelements, condelements->begin()));
  return condelements;
}

void FLD::UTILS::VolumetricFlowMapExtractor::setup(const Core::FE::Discretization& dis)
{
  const int ndim = Global::Problem::Instance()->NDim();
  Core::Conditions::MultiConditionSelector mcs;
  mcs.SetOverlapping(true);  // defines if maps can overlap
  mcs.AddSelector(Teuchos::rcp(
      new Core::Conditions::NDimConditionSelector(dis, "VolumetricSurfaceFlowCond", 0, ndim)));
  mcs.SetupExtractor(dis, *dis.dof_row_map(), *this);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FLD::UTILS::KSPMapExtractor::setup(const Core::FE::Discretization& dis)
{
  Core::Conditions::MultiConditionSelector mcs;
  mcs.AddSelector(
      Teuchos::rcp(new Core::Conditions::ConditionSelector(dis, "KrylovSpaceProjection")));
  mcs.SetupExtractor(dis, *dis.dof_row_map(), *this);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<std::set<int>> FLD::UTILS::KSPMapExtractor::conditioned_element_map(
    const Core::FE::Discretization& dis) const
{
  Teuchos::RCP<std::set<int>> condelements =
      Core::Conditions::conditioned_element_map(dis, "KrylovSpaceProjection");
  return condelements;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FLD::UTILS::VelPressExtractor::setup(const Core::FE::Discretization& dis)
{
  const int ndim = Global::Problem::Instance()->NDim();
  Core::LinAlg::CreateMapExtractorFromDiscretization(dis, ndim, *this);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FLD::UTILS::FsiMapExtractor::setup(const Core::FE::Discretization& dis)
{
  const int ndim = Global::Problem::Instance()->NDim();
  Core::Conditions::MultiConditionSelector mcs;
  mcs.AddSelector(
      Teuchos::rcp(new Core::Conditions::NDimConditionSelector(dis, "FSICoupling", 0, ndim)));
  mcs.SetupExtractor(dis, *dis.dof_row_map(), *this);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FLD::UTILS::FsiMapExtractor::setup(Teuchos::RCP<const Epetra_Map>& additionalothermap,
    const FLD::UTILS::FsiMapExtractor& extractor)
{
  // build the new othermap
  std::vector<Teuchos::RCP<const Epetra_Map>> othermaps;
  othermaps.push_back(additionalothermap);
  othermaps.push_back(extractor.other_map());

  if (Core::LinAlg::MultiMapExtractor::intersect_maps(othermaps)->NumGlobalElements() > 0)
    FOUR_C_THROW("Failed to add dofmap of foreign discretization to other_map. Detected overlap.");

  Teuchos::RCP<const Epetra_Map> mergedothermap =
      Core::LinAlg::MultiMapExtractor::merge_maps(othermaps);

  // the vector of maps for the new map extractor consists of othermap at position 0
  // followed by the maps of conditioned DOF
  std::vector<Teuchos::RCP<const Epetra_Map>> maps;
  // append the merged other map at first position
  maps.push_back(mergedothermap);

  // append the condition maps subsequently
  for (int i = 1; i < extractor.num_maps(); ++i) maps.push_back(extractor.Map(i));

  // merge
  Teuchos::RCP<const Epetra_Map> fullmap = Core::LinAlg::MultiMapExtractor::merge_maps(maps);

  Core::LinAlg::MultiMapExtractor::setup(*fullmap, maps);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FLD::UTILS::XFluidFluidMapExtractor::setup(const Epetra_Map& fullmap,
    Teuchos::RCP<const Epetra_Map> fluidmap, Teuchos::RCP<const Epetra_Map> xfluidmap)
{
  std::vector<Teuchos::RCP<const Epetra_Map>> maps;
  maps.push_back(fluidmap);
  maps.push_back(xfluidmap);
  MultiMapExtractor::setup(fullmap, maps);
}

FOUR_C_NAMESPACE_CLOSE